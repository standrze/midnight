import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import ModelRunnerProtocol
import Testing

@testable import ModelRunnerCore

@Suite("Passive activation recording", .serialized)
struct InspectorRecordingTests {
    @Test("Request defaults and recording selection remain bounded")
    func requestBounds() throws {
        let request = try JSONDecoder().decode(InspectorRecordingRequest.self, from: Data(#"{"layers":[0]}"#.utf8))
        #expect(request.maxTokens == 32)
        #expect(request.maxCaptureBytes == 4 * 1_048_576)
        #expect(request.sites == [.layerInput, .afterAttention])
        let model = descriptor()
        #expect(try PendingInspectorRecording.validate(request: request, descriptor: model) > 0)
        for invalid in [
            InspectorRecordingRequest(layers: []), .init(layers: [0, 0]), .init(layers: [99]),
            .init(layers: [0], maxTokens: 0), .init(layers: [0], maxTokens: 65),
            .init(layers: [0], maxCaptureBytes: 1), .init(model: "other", layers: [0]),
            .init(layers: [0], sites: []), .init(layers: [0], sites: [.layerInput, .layerInput]),
        ] {
            #expect(throws: ModelInspectionError.self) {
                try PendingInspectorRecording.validate(request: invalid, descriptor: model)
            }
        }
    }

    @Test("Concurrent requests consume a single arm, and cancellation permits a new recording")
    func claimsOnce() async throws {
        let store = InspectorRecordingStore()
        let session = try store.arm(model: descriptor())
        #expect(throws: InspectorRecordingError.self) { try store.arm(model: descriptor()) }
        let successes = await withTaskGroup(of: Bool.self) { group in
            for _ in 0..<20 { group.addTask { store.claim(id: session.id) } }
            var count = 0
            for await claimed in group where claimed { count += 1 }
            return count
        }
        #expect(successes == 1)
        #expect(store.status(id: session.id)?.status == "recording")
        #expect(store.cancel(id: session.id)?.status == "cancelled")
        let next = try store.arm(model: descriptor())
        store.finish(id: session.id, status: "completed", message: nil)
        #expect(store.status(id: session.id)?.status == "cancelled")
        #expect(store.status(id: next.id)?.status == "armed")
        #expect(!store.claim(id: session.id))
    }

    @Test("Retention evicts completed sessions while preserving the current arm")
    func boundedStore() throws {
        let store = InspectorRecordingStore()
        let first = try store.arm(model: descriptor())
        store.cancel(id: first.id)
        var latest = first
        for _ in 0..<4 {
            latest = try store.arm(model: descriptor())
            store.cancel(id: latest.id)
        }
        #expect(store.status(id: first.id) == nil)
        #expect(store.status(id: latest.id)?.status == "cancelled")
    }

    @Test("Ordinary greedy iteration preserves tokens, bounded samples, cache positions, and exact norm identities")
    func capturesNormalIterator() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            let original = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            let prompt = Array(repeating: Int32(2), count: 20)
            var parameters = GenerateParameters(maxTokens: 5, temperature: 0)
            parameters.prefill.stepSize = 8
            let prefix = MLXArray([Int32(1), 3, 4, 5, 6, 7, 8]).reshaped(1, 7)
            let baselineCache = try model.newCache(parameters: parameters)
            eval(model(prefix, cache: baselineCache))
            let baselineIterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(prompt)), model: model, cache: baselineCache,
                parameters: parameters)
            let baseline = Array(IteratorSequence(baselineIterator))
            let store = InspectorRecordingStore()
            let capture = try makeCapture(model: model, store: store, maxTokens: 3)
            try capture.install(model: model)
            defer { capture.restore() }
            // This cache-building forward intentionally precedes processor.prompt.
            // It must not be counted as part of the newly submitted suffix.
            let cache = try model.newCache(parameters: parameters)
            eval(model(prefix, cache: cache))
            let iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray(prompt)), model: model, cache: cache,
                parameters: parameters, components: capture.components(.init()))
            let output = Array(IteratorSequence(iterator))
            #expect(output == baseline)
            capture.record(metrics: metrics(prompt: 27, generated: 5, cached: 7))
            capture.restore()
            capture.finish(tokenizer: RecordingTestTokenizer())
            let session = try #require(store.status(id: capture.recording.id))
            #expect(session.status == "completed")
            #expect(session.cachedPromptTokenCount == 7)
            let trace = try #require(session.trace)
            #expect(trace.tokens.map(\.position) == [26, 27, 28, 29])
            #expect(trace.tokens.map(\.id) == [2] + Array(output.prefix(3)))
            #expect(trace.layers.count == 4)
            let components = try #require(trace.components)
            #expect(components.count == 4)
            #expect(Set(components.map(\.site)) == ["attention_output", "feed_forward_output"])
            #expect(components.allSatisfy { $0.samples.map(\.tokenIndex) == [26, 27, 28, 29] })
            #expect(components.allSatisfy { $0.samples.allSatisfy { $0.rms.isFinite && $0.rms >= 0 } })
            #expect(trace.layers.allSatisfy { $0.samples.map(\.tokenIndex) == [26, 27, 28, 29] })
            #expect(trace.generatedTokenCount == 5)
            #expect(trace.capture?.unobservedDecodePositions == [])
            #expect(try JSONEncoder().encode(trace).count <= capture.recording.request.maxCaptureBytes)
            let restored = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            for (path, module) in original { #expect(restored[path] === module) }
        }
    }

    @Test("EOS lookahead is excluded from the replay window")
    func trimsUnemittedToken() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            let store = InspectorRecordingStore()
            let capture = try makeCapture(model: model, store: store, maxTokens: 3)
            try capture.install(model: model)
            defer { capture.restore() }
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(1), 2])), model: model,
                parameters: GenerateParameters(maxTokens: 3, temperature: 0),
                components: capture.components(.init()))
            _ = iterator.next()
            _ = iterator.next()
            capture.record(metrics: metrics(prompt: 2, generated: 1, stop: "stop"))
            capture.restore()
            capture.finish(tokenizer: RecordingTestTokenizer())
            let trace = try #require(store.status(id: capture.recording.id)?.trace)
            #expect(trace.tokens.count == 2)
            #expect(trace.layers.allSatisfy { $0.samples.map(\.tokenIndex) == [1, 2] })
            #expect(trace.capture?.unobservedDecodePositions == [1, 2])
        }
    }

    @Test("Cancelling observation leaves model iteration alive and restores originals")
    func cancelObservationOnly() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            let store = InspectorRecordingStore()
            let capture = try makeCapture(model: model, store: store, maxTokens: 3)
            let original = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            try capture.install(model: model)
            defer { capture.restore() }
            var iterator = try TokenIterator(
                input: LMInput(tokens: MLXArray([Int32(1), 2])), model: model,
                parameters: GenerateParameters(maxTokens: 3, temperature: 0),
                components: capture.components(.init()))
            #expect(iterator.next() != nil)
            store.cancel(id: capture.recording.id)
            #expect(iterator.next() != nil)
            #expect(iterator.next() != nil)
            #expect(iterator.next() == nil)
            capture.restore()
            capture.finish(tokenizer: RecordingTestTokenizer())
            #expect(store.status(id: capture.recording.id)?.status == "cancelled")
            let restored = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            for (path, module) in original { #expect(restored[path] === module) }
        }
    }

    @Test("Malformed observations fail recording while the pass-through logits remain exact")
    func failsWithoutChangingLogits() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            let store = InspectorRecordingStore()
            let capture = try makeCapture(model: model, store: store, maxTokens: 3)
            try capture.install(model: model)
            defer { capture.restore() }
            let processor = InspectorRecordingProcessor(capture: capture)
            processor.prompt(MLXArray([Int32(1), 2]))
            capture.observe(path: "invalid", input: MLXArray([Float(1)]))
            let logits = MLXArray([Float(1), 2, 3]).reshaped(1, 3)
            let unchanged = processor.process(logits: logits)
            #expect(allClose(logits, unchanged, rtol: 0, atol: 0).item(Bool.self))
            #expect(store.status(id: capture.recording.id)?.status == "failed")
        }
    }

    @Test("Dense output capture distinguishes an inactive attention projection")
    func componentIsolation() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            let path = "model.layers.0.self_attn.o_proj"
            let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            let attention = try #require(leaves[path] as? Linear)
            try model.update(
                parameters: ModuleParameters.unflattened([
                    (path + ".weight", MLXArray.zeros(attention.weight.shape))
                ]), verify: [.noUnusedKeys])
            let capture = try makeCapture(model: model, store: InspectorRecordingStore(), maxTokens: 1)
            try capture.install(model: model)
            defer { capture.restore() }
            capture.begin(prompt: MLXArray([Int32(1), 2]))
            eval(model(MLXArray([Int32(1), 2]).reshaped(1, 2), cache: nil))
            capture.flush()
            capture.record(metrics: metrics(prompt: 2, generated: 0))
            capture.restore()
            capture.finish(tokenizer: RecordingTestTokenizer())
            let trace = try #require(capture.recording.store.status(id: capture.recording.id)?.trace)
            let components = try #require(trace.components)
            #expect(components.first { $0.index == 0 && $0.site == "attention_output" }?.samples.first?.rms == 0)
            #expect(
                (components.first { $0.index == 0 && $0.site == "feed_forward_output" }?.samples.first?.rms ?? 0) > 0)
        }
    }

    @Test(
        "Native MoE captures routed IDs and shared outputs across chunked prefill without changing logits",
        arguments: ["laguna", "laguna_compiled", "gpt_oss"])
    func expertRoutes(_ variant: String) throws {
        let kind = variant == "laguna_compiled" ? "laguna" : variant
        try Device.withDefaultDevice(.cpu) {
            try LagunaRuntimeTuning.$useCompiledBlockTail.withValue(false) {
                try LagunaRuntimeTuning.$useCompiledMoEFusion.withValue(variant == "laguna_compiled") {
                    let lagunaJSON =
                        #"{"model_type":"laguna","vocab_size":64,"hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":64,"max_position_embeddings":256,"sliding_window":8,"layer_types":["full_attention","sliding_attention"],"mlp_layer_types":["dense","sparse"],"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":128,"shared_expert_intermediate_size":128}"#
                    let gptJSON =
                        #"{"model_type":"gpt_oss","vocab_size":64,"hidden_size":128,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":64,"rms_norm_eps":0.00001,"sliding_window":8,"num_local_experts":4,"num_experts_per_tok":2}"#
                    let model: any LLMModel
                    if kind == "laguna" {
                        model = LagunaModel(
                            try JSONDecoder().decode(LagunaConfiguration.self, from: Data(lagunaJSON.utf8)))
                    } else {
                        model = GPTOSSModel(
                            try JSONDecoder().decode(GPTOSSConfiguration.self, from: Data(gptJSON.utf8)))
                    }
                    if kind == "gpt_oss" {
                        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
                        var weights: [(String, MLXArray)] = []
                        for (path, module) in leaves where path.hasSuffix(".mlp.router") {
                            let router = try #require(module as? Linear)
                            weights.append((path + ".weight", MLXArray.zeros(router.weight.shape)))
                            weights.append((path + ".bias", MLXArray([Float(0), 1, 2, 3])))
                        }
                        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
                    }
                    let original = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
                    let metadata = ModelInspectionMetadata(
                        modelType: kind, hiddenSize: 128, expertCount: 4, expertsPerToken: 2)
                    let descriptor = ModelInspection.descriptor(
                        model: model, id: kind, metadata: metadata, contextLength: 128)
                    let request = InspectorRecordingRequest(layers: [0, 1], maxTokens: 2)
                    let store = InspectorRecordingStore()
                    let session = try store.arm(model: descriptor)
                    #expect(store.claim(id: session.id))
                    let capture = InspectorRecordingCapture(
                        recording: .init(
                            id: session.id, request: request,
                            descriptor: descriptor, store: store,
                            estimatedBytes: try PendingInspectorRecording.validate(
                                request: request, descriptor: descriptor)))
                    let baselineCache = try model.newCache(parameters: GenerateParameters())
                    let chunks = [8, 8, 4, 1]
                    let expected = chunks.map { length -> [Float] in
                        let output = model(
                            MLXArray(Array(repeating: Int32(2), count: length)).reshaped(1, length),
                            cache: baselineCache)
                        eval(output)
                        return output.asArray(Float.self)
                    }
                    try capture.install(model: model)
                    defer { capture.restore() }
                    capture.begin(prompt: MLXArray(Array(repeating: Int32(2), count: 20)))
                    let cache = try model.newCache(parameters: GenerateParameters())
                    for (index, length) in chunks.enumerated() {
                        let output = model(
                            MLXArray(Array(repeating: Int32(2), count: length)).reshaped(1, length), cache: cache)
                        eval(output)
                        capture.flush()
                        #expect(output.asArray(Float.self) == expected[index])
                    }
                    capture.sampled(MLXArray(Int32(2)))
                    capture.record(metrics: metrics(prompt: 20, generated: 1))
                    capture.restore()
                    capture.finish(tokenizer: RecordingTestTokenizer())
                    let recorded = try #require(store.status(id: session.id))
                    #expect(recorded.status == "completed")
                    let trace = try #require(recorded.trace)
                    let routes = try #require(trace.expertRouting)
                    #expect(routes.count == (kind == "laguna" ? 1 : 2))
                    for route in routes {
                        #expect(route.expertCount == 4)
                        #expect(route.samples.map(\.tokenIndex) == [19, 20])
                        #expect(
                            route.samples.allSatisfy {
                                $0.expertIDs.count == 2 && Set($0.expertIDs).count == 2
                                    && abs($0.weights.reduce(0, +) - 1) < 0.02
                            })
                    }
                    if kind == "laguna" {
                        #expect(
                            trace.components?.contains { $0.index == 1 && $0.site == "shared_expert_output" } == true)
                    }
                    let restored = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
                    for (path, module) in original { #expect(restored[path] === module) }
                }
            }
        }
    }

    @Test("Quantized projection observation delegates exact original math")
    func quantizedOutput() throws {
        try Device.withDefaultDevice(.cpu) {
            let original = QuantizedLinear(64, 64, groupSize: 32, bits: 4)
            let input = MLXArray.ones([1, 2, 64])
            var observed: MLXArray?
            let wrapper = InspectorPassiveLinear(original: original) { observed = $0 }
            let expected = original(input)
            let actual = wrapper(input)
            let sample = try #require(observed)
            eval(expected, actual, sample)
            #expect(actual.asArray(Float.self) == expected.asArray(Float.self))
            #expect(sample.asArray(Float.self) == actual.asArray(Float.self))
        }
    }

    private func makeModel() -> Mistral3TextModel {
        Mistral3TextModel(
            Mistral3TextConfiguration(
                hiddenSize: 64, hiddenLayers: 2, intermediateSize: 128, attentionHeads: 4,
                rmsNormEps: 0.00001, vocabularySize: 64, kvHeads: 2,
                layerTypes: ["full_attention", "sliding_attention"], slidingWindow: 64))
    }

    private func descriptor(_ model: Mistral3TextModel? = nil) -> InspectorModel {
        if let model {
            return ModelInspection.descriptor(
                model: model, id: "test", metadata: ModelInspectionMetadata(modelType: "ministral3"),
                contextLength: 128)
        }
        return InspectorModel(
            id: "test", modelType: "ministral3", runtimeType: "fixture", layerCount: 1, hiddenSize: 64,
            storedElementCount: 64, weightBytes: 256, traceSupported: true, traceReason: nil,
            layers: [
                InspectorLayer(
                    index: 0, path: "model.layers.0", kind: "layer", storedElementCount: 64,
                    weightBytes: 256, modules: [])
            ])
    }

    private func makeCapture(
        model: Mistral3TextModel, store: InspectorRecordingStore, maxTokens: Int
    ) throws -> InspectorRecordingCapture {
        let descriptor = descriptor(model)
        let request = InspectorRecordingRequest(layers: [0, 1], maxTokens: maxTokens)
        let estimate = try PendingInspectorRecording.validate(request: request, descriptor: descriptor)
        let session = try store.arm(model: descriptor)
        #expect(store.claim(id: session.id))
        return InspectorRecordingCapture(
            recording: PendingInspectorRecording(
                id: session.id, request: request, descriptor: descriptor, store: store, estimatedBytes: estimate))
    }

    private func metrics(prompt: Int, generated: Int, cached: Int = 0, stop: String = "length")
        -> LocalModelRunnerMetrics
    {
        LocalModelRunnerMetrics(
            promptTokenCount: prompt, prefilledPromptTokenCount: prompt - cached, cachedPromptTokenCount: cached,
            generationTokenCount: generated, promptTokensPerSecond: 1, tokensPerSecond: 1, stopReason: stop)
    }
}

private struct RecordingTestTokenizer: MLXLMCommon.Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1, 2] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { tokenIds.map { "t\($0)" }.joined() }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { "t\(id)" }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1, 2] }
}
