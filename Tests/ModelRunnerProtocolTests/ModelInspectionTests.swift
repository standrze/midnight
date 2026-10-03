import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import ModelRunnerProtocol
import Testing

@testable import ModelRunnerCore

@Suite("Model inspection", .serialized)
struct ModelInspectionTests {
    @Test("Questions and output limits are bounded before inference")
    func boundsRequests() throws {
        #expect(
            try ModelInspection.maximumTokens(
                .init(
                    question: "Hello", layers: [0],
                    tokenPositions: .init(prefill: [-1], decode: []))) == 32)
        #expect(
            try ModelInspection.maximumTokens(
                .init(
                    question: "Hello", maxTokens: 0, layers: [0],
                    tokenPositions: .init(prefill: [-1], decode: []))) == 0)
        for request in [
            InspectorTraceRequest(question: "  "), .init(question: "Hello", maxTokens: -1),
            .init(question: "Hello", maxTokens: 65), .init(question: String(repeating: "x", count: 16_385)),
        ] {
            #expect(throws: ModelInspectionError.self) { try ModelInspection.maximumTokens(request) }
        }
    }

    @Test("Metadata preserves nested attention and expert configuration")
    func decodesMetadata() {
        let metadata = ModelInspectionMetadata.decode(
            Data(
                #"{"model_type":"mistral3","text_config":{"model_type":"ministral3","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"num_local_experts":8,"num_experts_per_tok":2,"layer_types":["full_attention","sliding_attention"]}}"#
                    .utf8))
        #expect(metadata.modelType == "ministral3")
        #expect(metadata.hiddenSize == 64)
        #expect(metadata.attentionHeads == 4)
        #expect(metadata.kvHeads == 2)
        #expect(metadata.expertCount == 8)
        #expect(metadata.expertsPerToken == 2)
        #expect(metadata.layerTypes == ["full_attention", "sliding_attention"])
        #expect(
            InspectionLayerAddress.parse("language_model.model.layers.12.mlp.gate_proj")
                == InspectionLayerAddress(index: 12, path: "language_model.model.layers.12"))
        #expect(InspectionLayerAddress.parse("lm_head") == nil)
    }

    @Test("Harmony semantic stops require confirmed tokens and the GPT-OSS runtime")
    func validatesHarmonyStops() {
        let tokenizer = InspectionTestTokenizer(
            controlTokens: [
                "<|ghissue|>": 11, "<|fim_suffix|>": 12,
                "<unk>": 0,
            ], unknownToken: "<unk>")
        #expect(ModelInspection.semanticStopTokenIDs(tokenizer: tokenizer, isGPTOSS: true) == [11, 12])
        #expect(ModelInspection.semanticStopTokenIDs(tokenizer: tokenizer, isGPTOSS: false).isEmpty)
        let renamed = InspectionTestTokenizer(controlTokens: ["<|call|>": 21, "<|return|>": 22])
        #expect(ModelInspection.semanticStopTokenIDs(tokenizer: renamed, isGPTOSS: true) == [21, 22])
    }

    @Test("Loaded topology describes packed tensors without logical parameter inflation")
    func describesQuantizedTopology() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            quantize(model: model, groupSize: 32, bits: 4)
            let descriptor = descriptor(model)
            #expect(descriptor.layerCount == 2)
            #expect(descriptor.traceSupported)
            #expect(descriptor.hiddenSize == 64)
            #expect(descriptor.traceCapabilities?.requiresExplicitLayers == true)
            #expect(descriptor.traceCapabilities?.requiresExplicitTokenPositions == true)
            #expect(descriptor.traceCapabilities?.observationPoints.map(\.site) == [.layerInput, .afterAttention])
            #expect(descriptor.traceCapabilities?.rawDtype == "float32")
            #expect(descriptor.storedElementCount == model.parameters().flattened().reduce(0) { $0 + $1.1.size })
            #expect(descriptor.weightBytes == model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes })
            #expect(descriptor.layers[0].modules.contains { $0.kind == "QuantizedLinear" })
            let quantized = descriptor.layers[0].modules.filter { $0.kind == "QuantizedLinear" }
            #expect(quantized.allSatisfy { $0.quantization == .init(format: "affine", bits: 4, groupSize: 32) })
            let data = try JSONEncoder().encode(descriptor)
            #expect(try JSONDecoder().decode(InspectorModel.self, from: data) == descriptor)
        }
    }

    @Test("Residual summaries measure known tensor values")
    func measuresResidualValues() throws {
        try Device.withDefaultDevice(.cpu) {
            let recorder = InspectionActivationRecorder()
            recorder.register(path: "model.layers.0.input_layernorm", layer: 0, site: "layer_input")
            recorder.observe(
                path: "model.layers.0.input_layernorm", input: MLXArray([Float(3), 4, 0, -2]).reshaped(1, 2, 2))
            try recorder.flush()
            let samples = try recorder.finalize(tokenCount: 2)[0].samples
            #expect(abs(samples[0].rms - sqrt(12.5)) < 0.00001)
            #expect(samples[0].maxAbs == 4)
            #expect(abs(samples[1].rms - sqrt(2)) < 0.00001)
            #expect(samples[1].maxAbs == 2)
            #expect(samples[0].channels.count == 16)
            #expect(samples[0].channels[0] == 3)
            #expect(samples[0].channels[15] == 4)
            #expect(throws: ModelInspectionError.self) { try recorder.finalize(tokenCount: 3) }
        }
    }

    @Test("Inspection restores exact norms and preserves quantized model logits")
    func restoresModel() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            quantize(model: model, groupSize: 32, bits: 4)
            let description = descriptor(model)
            let tokens = MLXArray([Int32(1), 2]).reshaped(1, 2)
            let baseline = model(tokens, cache: nil)
            eval(baseline)
            let originals = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            let recorder = InspectionActivationRecorder()
            let installation = try InspectionNormInstallation(
                model: model, layers: description.layers, recorder: recorder)
            let observed = model(tokens, cache: nil)
            eval(observed)
            try recorder.flush()
            installation.restore()
            #expect(allClose(baseline, observed, rtol: 0, atol: 0).item(Bool.self))
            #expect(try recorder.finalize(tokenCount: 2).count == 4)
            let restored = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            #expect(restored.count == originals.count)
            for (path, original) in originals {
                #expect(restored[path] === original)
            }
            let after = model(tokens, cache: nil)
            eval(after)
            #expect(allClose(baseline, after, rtol: 0, atol: 0).item(Bool.self))
        }
    }

    @Test("Invalid installation leaves the original graph untouched")
    func rejectsIncompleteInstallation() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            var layers = descriptor(model).layers
            layers[1].path = "model.layers.99"
            let originals = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            #expect(throws: ModelInspectionError.self) {
                try InspectionNormInstallation(
                    model: model, layers: layers,
                    recorder: InspectionActivationRecorder())
            }
            let restored = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            #expect(restored.count == originals.count)
            for (path, original) in originals {
                #expect(restored[path] === original)
            }
        }
    }

    @Test("Restoring the graph releases observers and their temporary buffers")
    func releasesObservers() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            weak var releasedRecorder: InspectionActivationRecorder?
            do {
                let recorder = InspectionActivationRecorder(
                    plan: try plan(
                        model, promptTokenCount: 2,
                        maximumTokens: 0, mode: .both))
                releasedRecorder = recorder
                let installation = try InspectionNormInstallation(
                    model: model,
                    layers: descriptor(model).layers, recorder: recorder)
                let output = model(MLXArray([Int32(1), 2]).reshaped(1, 2), cache: nil)
                eval(output)
                try recorder.flush()
                installation.restore()
                installation.restore()  // Cleanup is safe if invoked more than once.
            }
            #expect(releasedRecorder == nil)
            #expect(!model.namedModules().contains { $0.1 is InspectionRecordingNorm })
        }
    }

    @Test("Cancellation restores observers before ordinary inference resumes", arguments: [0, 1])
    func restoresAfterCancellation(maximumTokens: Int) async throws {
        try await Task {
            try Device.withDefaultDevice(.cpu) {
                let model = makeModel()
                let tokens = MLXArray([Int32(1), 2]).reshaped(1, 2)
                let baseline = model(tokens, cache: nil)
                eval(baseline)
                let originals = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
                let context = ModelContext(
                    configuration: ModelConfiguration(id: "inspection-test"),
                    model: model, processor: InspectionTestProcessor(),
                    tokenizer: InspectionTestTokenizer(cancelOnDecode: true))
                #expect(throws: CancellationError.self) {
                    try ModelInspection.trace(
                        context: context, descriptor: descriptor(model),
                        question: "test", promptTokens: [1, 2], maximumTokens: maximumTokens,
                        capturePlan: plan(model, promptTokenCount: 2, maximumTokens: maximumTokens, mode: .both))
                }
                let restored = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
                #expect(restored.count == originals.count)
                for (path, original) in originals {
                    #expect(restored[path] === original)
                }
                let after = model(tokens, cache: nil)
                eval(after)
                #expect(allClose(baseline, after, rtol: 0, atol: 0).item(Bool.self))
            }
        }.value
    }

    @Test("Short decode traces every displayed token and leaves the graph unchanged")
    func tracesTokenSequence() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            quantize(model: model, groupSize: 32, bits: 4)
            let before = descriptor(model)
            let context = ModelContext(
                configuration: ModelConfiguration(id: "inspection-test"), model: model,
                processor: InspectionTestProcessor(), tokenizer: InspectionTestTokenizer())
            let trace = try ModelInspection.trace(
                context: context, descriptor: before,
                question: "test", promptTokens: [1, 2, 3], maximumTokens: 2,
                capturePlan: plan(model, promptTokenCount: 3, maximumTokens: 2))
            #expect(trace.promptTokenCount == 3)
            #expect(trace.generatedTokenCount == 2)
            #expect(trace.tokens.count == 5)
            #expect(trace.predictions?.map(\.afterTokenIndex) == [2, 3, 4])
            #expect(
                trace.predictions?.allSatisfy { p in
                    p.candidates.count == 5
                        && p.candidates.allSatisfy {
                            $0.probability.isFinite && $0.probability >= 0 && $0.probability <= 1
                        }
                        && zip(p.candidates, p.candidates.dropFirst()).allSatisfy { $0.probability >= $1.probability }
                } == true)
            #expect(trace.layers.count == 4)
            #expect(trace.layers.allSatisfy { $0.samples.map(\.tokenIndex) == [0, 1, 2, 3, 4] })
            #expect(descriptor(model) == before)
            #expect(trace.stopReason == "length")
            #expect(trace.outputKind == "raw")
        }
    }

    @Test("A failed model evaluation also restores original normalization modules")
    func restoresAfterFailure() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            model.update(
                parameters: model.parameters().mapValues { MLXArray.full($0.shape, values: MLXArray(Float.nan)) })
            let original = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            let context = ModelContext(
                configuration: ModelConfiguration(id: "inspection-test"), model: model,
                processor: InspectionTestProcessor(), tokenizer: InspectionTestTokenizer())
            #expect(throws: ModelInspectionError.self) {
                try ModelInspection.trace(
                    context: context, descriptor: descriptor(model),
                    question: "test", promptTokens: [1, 2], maximumTokens: 1,
                    capturePlan: plan(model, promptTokenCount: 2, maximumTokens: 1, mode: .raw))
            }
            let restored = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            #expect(restored.count == original.count)
            for (path, module) in original {
                #expect(restored[path] === module)
            }
        }
    }

    @Test("Selection and encoded memory bounds fail before observers are installed")
    func validatesSelectionBudget() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            let original = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            let invalid: [InspectorTraceRequest] = [
                .init(question: "test"),
                .init(question: "test", layers: [], tokenPositions: .init(prefill: [0], decode: [])),
                .init(question: "test", layers: [0, 0], tokenPositions: .init(prefill: [0], decode: [])),
                .init(question: "test", layers: [2], tokenPositions: .init(prefill: [0], decode: [])),
                .init(question: "test", layers: [0], sites: [], tokenPositions: .init(prefill: [0], decode: [])),
                .init(question: "test", layers: [0], tokenPositions: .init(prefill: [], decode: [])),
                .init(question: "test", layers: [0], tokenPositions: .init(prefill: [0, -3], decode: [])),
                .init(question: "test", layers: [0], tokenPositions: .init(prefill: [-4], decode: [])),
                .init(question: "test", layers: [0], tokenPositions: .init(prefill: [3], decode: [])),
                .init(question: "test", layers: [0], tokenPositions: .init(prefill: [0], decode: [1])),
                .init(
                    question: "test", layers: [0], tokenPositions: .init(prefill: [0], decode: []), capture: .raw,
                    maxCaptureBytes: 1),
                .init(
                    question: "test", layers: [0], tokenPositions: .init(prefill: [0], decode: []), model: "different"),
            ]
            for request in invalid {
                #expect(throws: ModelInspectionError.self) {
                    try ModelInspection.capturePlan(
                        request: request, descriptor: descriptor(model),
                        promptTokenCount: 3, maximumTokens: 1)
                }
            }
            let valid = try plan(
                model, promptTokenCount: 3, maximumTokens: 1, layers: [1],
                sites: [.layerInput], positions: .init(prefill: [-1], decode: [0]), mode: .raw)
            #expect(valid.tokenPositions == .init(prefill: [2], decode: [0]))
            #expect(valid.absolutePositions == [2, 3])
            #expect(valid.estimatedCaptureBytes <= ModelInspection.captureByteLimit)
            #expect(valid.estimatedMemoryBytes > valid.estimatedCaptureBytes)
            let restored = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            for (path, module) in original {
                #expect(restored[path] === module)
            }
        }
    }

    @Test("Selected raw tensors preserve float bits, positions, token IDs, and summaries")
    func capturesSelectedRawValues() throws {
        try Device.withDefaultDevice(.cpu) {
            let description = InspectorModel(
                id: "tiny", modelType: "test", runtimeType: "test",
                layerCount: 1, hiddenSize: 4, storedElementCount: 0, weightBytes: 0,
                traceSupported: true, traceReason: nil,
                layers: [
                    .init(
                        index: 0, path: "model.layers.0", kind: "test", storedElementCount: 0, weightBytes: 0,
                        modules: [])
                ])
            let capture = try ModelInspection.capturePlan(
                request: .init(
                    question: "test", maxTokens: 1,
                    layers: [0], sites: [.layerInput], tokenPositions: .init(prefill: [-1], decode: [0]), capture: .both
                ),
                descriptor: description, promptTokenCount: 3, maximumTokens: 1)
            let recorder = InspectionActivationRecorder(plan: capture)
            let path = "model.layers.0.input_layernorm"
            recorder.register(path: path, layer: 0, site: "layer_input")
            recorder.observe(
                path: path, input: MLXArray([Float(9), 9, 9, 9, 8, 8, 8, 8, 3, 4, -0.0, -2]).reshaped(1, 3, 4))
            try recorder.flush()
            recorder.startPosition = 3
            recorder.observe(path: path, input: MLXArray([Float(1), -1, 2, -2]).reshaped(1, 1, 4))
            try recorder.flush()
            let summary = try recorder.finalize(tokenCount: 4)
            #expect(summary[0].samples.map(\.tokenIndex) == [2, 3])
            #expect(abs(summary[0].samples[0].rms - sqrt(7.25)) < 0.00001)
            let tensors = try #require(try recorder.rawTensors(tokenIDs: [11, 12, 13, 14]))
            let tensor = try #require(tensors.first)
            #expect(tensors.count == 1)
            #expect(tensor.shape == [2, 4])
            #expect(tensor.tokenIndices == [2, 3])
            #expect(tensor.tokenIDs == [13, 14])
            #expect(tensor.encoding == "base64" && tensor.dtype == "float32" && tensor.byteOrder == "little")
            let actual = try rawFloats(tensor)
            let expected: [Float] = [3, 4, -0.0, -2, 1, -1, 2, -2]
            #expect(actual.map(\.bitPattern) == expected.map(\.bitPattern))
            #expect(try JSONDecoder().decode(InspectorRawTensor.self, from: JSONEncoder().encode(tensor)) == tensor)
        }
    }

    @Test("Only selected wrappers are installed and raw observation preserves exact logits")
    func selectedRawPreservesLogits() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            quantize(model: model, groupSize: 32, bits: 4)
            let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
            let baseline = model(tokens, cache: nil)
            eval(baseline)
            let capture = try plan(
                model, promptTokenCount: 3, maximumTokens: 0, layers: [1],
                sites: [.afterAttention], positions: .init(prefill: [-1], decode: []), mode: .raw)
            let recorder = InspectionActivationRecorder(plan: capture)
            let installation = try InspectionNormInstallation(
                model: model, layers: capture.layers,
                recorder: recorder, sites: capture.sites)
            let wrappers = model.namedModules().filter { $0.1 is InspectionRecordingNorm }
            #expect(wrappers.count == 1)
            #expect(wrappers.first?.0 == "model.layers.1.post_attention_layernorm")
            let observed = model(tokens, cache: nil)
            eval(observed)
            try recorder.flush()
            installation.restore()
            #expect(allClose(baseline, observed, rtol: 0, atol: 0).item(Bool.self))
            #expect(try recorder.finalize(tokenCount: 3).isEmpty)
            let tensors = try #require(try recorder.rawTensors(tokenIDs: [1, 2, 3]))
            #expect(tensors.count == 1)
            #expect(tensors[0].shape == [1, 64] && tensors[0].tokenIndices == [2])
            let after = model(tokens, cache: nil)
            eval(after)
            #expect(allClose(baseline, after, rtol: 0, atol: 0).item(Bool.self))
            #expect(!model.namedModules().contains { $0.1 is InspectionRecordingNorm })
        }
    }

    @Test("Prefill-only capture does no decode and resolves the last prompt token")
    func capturesPrefillOnly() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            let context = ModelContext(
                configuration: ModelConfiguration(id: "test"), model: model,
                processor: InspectionTestProcessor(), tokenizer: InspectionTestTokenizer())
            let capture = try plan(
                model, promptTokenCount: 3, maximumTokens: 0, layers: [0],
                sites: [.layerInput], positions: .init(prefill: [-1], decode: []), mode: .raw)
            let trace = try ModelInspection.trace(
                context: context, descriptor: descriptor(model), question: "test",
                promptTokens: [1, 2, 3], maximumTokens: 0, capturePlan: capture)
            #expect(trace.answer.isEmpty && trace.generatedTokenCount == 0)
            #expect(trace.stopReason == "prefill_only")
            #expect(trace.tensors?.first?.tokenIndices == [2])
            #expect(trace.capture?.tokenPositions.prefill == [2])
            #expect(trace.capture?.unobservedDecodePositions.isEmpty == true)
            #expect(try JSONDecoder().decode(InspectorTrace.self, from: JSONEncoder().encode(trace)) == trace)
        }
    }

    @Test("Early stop explicitly reports unobserved decode positions")
    func reportsEarlyStop() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            let context = ModelContext(
                configuration: ModelConfiguration(id: "test", eosTokenIds: Set(0..<64)),
                model: model, processor: InspectionTestProcessor(), tokenizer: InspectionTestTokenizer())
            let capture = try plan(
                model, promptTokenCount: 2, maximumTokens: 2, layers: [0],
                sites: [.layerInput], positions: .init(prefill: [], decode: [0, 1]), mode: .raw)
            let trace = try ModelInspection.trace(
                context: context, descriptor: descriptor(model), question: "test",
                promptTokens: [1, 2], maximumTokens: 2, capturePlan: capture)
            #expect(trace.stopReason == "stop" && trace.generatedTokenCount == 0)
            #expect(trace.capture?.unobservedDecodePositions == [0, 1])
            #expect(trace.tensors?.first?.shape == [0, 64])
            #expect(trace.tensors?.first?.data == "")
        }
    }

    @Test("Oversized tokenizer metadata fails and restores exact ordinary logits")
    func restoresAfterMetadataBudgetFailure() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = makeModel()
            let tokens = MLXArray([Int32(1), 2]).reshaped(1, 2)
            let baseline = model(tokens, cache: nil)
            eval(baseline)
            let original = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            let context = ModelContext(
                configuration: ModelConfiguration(id: "test"), model: model,
                processor: InspectionTestProcessor(),
                tokenizer: InspectionTestTokenizer(oversizedLabels: true))
            #expect(throws: ModelInspectionError.self) {
                try ModelInspection.trace(
                    context: context, descriptor: descriptor(model), question: "test",
                    promptTokens: [1, 2], maximumTokens: 0,
                    capturePlan: plan(model, promptTokenCount: 2, maximumTokens: 0, mode: .raw))
            }
            let restored = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            for (path, module) in original {
                #expect(restored[path] === module)
            }
            let after = model(tokens, cache: nil)
            eval(after)
            #expect(allClose(baseline, after, rtol: 0, atol: 0).item(Bool.self))
        }
    }

    private func rawFloats(_ tensor: InspectorRawTensor) throws -> [Float] {
        let bytes = try #require(Data(base64Encoded: tensor.data))
        #expect(bytes.count == tensor.shape.reduce(1, *) * 4)
        return bytes.withUnsafeBytes { buffer in
            stride(from: 0, to: buffer.count, by: 4).map {
                Float(bitPattern: UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: $0, as: UInt32.self)))
            }
        }
    }

    private func makeModel() -> Mistral3TextModel {
        Mistral3TextModel(
            Mistral3TextConfiguration(
                hiddenSize: 64, hiddenLayers: 2,
                intermediateSize: 128, attentionHeads: 4, rmsNormEps: 0.00001,
                vocabularySize: 64, kvHeads: 2, layerTypes: ["full_attention", "sliding_attention"], slidingWindow: 32))
    }

    private func plan(
        _ model: Mistral3TextModel, promptTokenCount: Int, maximumTokens: Int,
        layers: [Int] = [0, 1], sites: [InspectorObservationSite]? = nil,
        positions: InspectorTokenPositions? = nil, mode: InspectorCaptureMode = .summary,
        maxCaptureBytes: Int? = nil
    ) throws -> InspectionCapturePlan {
        try ModelInspection.capturePlan(
            request: .init(
                question: "test", maxTokens: maximumTokens,
                layers: layers, sites: sites,
                tokenPositions: positions
                    ?? .init(prefill: Array(0..<promptTokenCount), decode: Array(0..<maximumTokens)),
                capture: mode, maxCaptureBytes: maxCaptureBytes), descriptor: descriptor(model),
            promptTokenCount: promptTokenCount, maximumTokens: maximumTokens)
    }

    private func descriptor(_ model: Mistral3TextModel) -> InspectorModel {
        ModelInspection.descriptor(
            model: model, id: "test",
            metadata: ModelInspectionMetadata(
                modelType: "ministral3", layerTypes: ["full_attention", "sliding_attention"]), contextLength: 128)
    }
}

private struct InspectionTestTokenizer: MLXLMCommon.Tokenizer {
    var controlTokens: [String: Int] = [:]
    // Trace decodes candidate labels after its first observed model evaluation.
    // Cancelling here exercises cleanup with live observation buffers, rather
    // than cancellation before inference has started.
    var cancelOnDecode = false
    var oversizedLabels = false
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? = nil
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1, 2, 3] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        if cancelOnDecode {
            withUnsafeCurrentTask { $0?.cancel() }
        }
        if oversizedLabels {
            return String(repeating: "x", count: 256 * 1024 + 1)
        }
        return tokenIds.map { "t\($0)" }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? {
        controlTokens[token] ?? unknownToken.flatMap { controlTokens[$0] }
    }
    func convertIdToToken(_ id: Int) -> String? { controlTokens.first { $0.value == id }?.key ?? "t\(id)" }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1, 2, 3] }
}

private struct InspectionTestProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput { LMInput(tokens: MLXArray([Int32(1), 2, 3])) }
}
