import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import ModelRunnerProtocol
import Tokenizers

/// An inspection request is invalid, unsupported, or has an unusable model graph.
public enum ModelInspectionError: LocalizedError, Equatable, Sendable {
    case invalidRequest(String)
    case unsupported(String)
    case invalidGraph(String)

    /// User-facing explanation for this error.
    public var errorDescription: String? {
        switch self {
        case .invalidRequest(let message), .unsupported(let message), .invalidGraph(let message): message
        }
    }
}

struct ModelInspectionMetadata: Sendable {
    var modelType = "unknown"
    var hiddenSize: Int?
    var attentionHeads: Int?
    var kvHeads: Int?
    var expertCount: Int?
    var expertsPerToken: Int?
    var configuredContextLength: Int?
    var activation: String?
    var layerTypes: [String] = []
    var mlpLayerTypes: [String] = []

    static func load(modelPath: String) -> Self {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: modelPath).appendingPathComponent("config.json"))
        else {
            return Self()
        }
        return decode(data)
    }

    static func decode(_ data: Data) -> Self {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return Self()
        }
        let config = (root["text_config"] as? [String: Any]) ?? root
        return Self(
            modelType: (config["model_type"] as? String) ?? (root["model_type"] as? String) ?? "unknown",
            hiddenSize: (config["hidden_size"] as? Int) ?? (config["n_embd"] as? Int),
            attentionHeads: (config["num_attention_heads"] as? Int) ?? (config["n_head"] as? Int),
            kvHeads: config["num_key_value_heads"] as? Int,
            expertCount: (config["num_local_experts"] as? Int) ?? (config["num_experts"] as? Int),
            expertsPerToken: config["num_experts_per_tok"] as? Int,
            configuredContextLength: (config["max_position_embeddings"] as? Int) ?? (config["n_positions"] as? Int),
            activation: (config["hidden_act"] as? String) ?? (config["activation_function"] as? String),
            layerTypes: (config["layer_types"] as? [String]) ?? [],
            mlpLayerTypes: (config["mlp_layer_types"] as? [String]) ?? [])
    }
}

struct InspectionLayerAddress: Equatable {
    var index: Int
    var path: String

    static func parse(_ path: String) -> Self? {
        let parts = path.split(separator: ".").map(String.init)
        guard parts.count >= 2 else {
            return nil
        }
        for index in 0..<(parts.count - 1) {
            if ["layers", "h", "blocks"].contains(parts[index]), let number = Int(parts[index + 1]), number >= 0 {
                return Self(index: number, path: parts[0...(index + 1)].joined(separator: "."))
            }
        }
        return nil
    }
}

/// Resolved entirely before observers or raw activation buffers are installed.
struct InspectionCapturePlan: Sendable {
    let layers: [InspectorLayer]
    let sites: [InspectorObservationSite]
    let tokenPositions: InspectorTokenPositions
    let absolutePositions: Set<Int>
    let mode: InspectorCaptureMode
    let hiddenSize: Int
    let maxCaptureBytes: Int
    let estimatedCaptureBytes: Int
    let estimatedMemoryBytes: Int

    func report(generatedTokenCount: Int) -> InspectorCaptureReport {
        InspectorCaptureReport(
            mode: mode, layers: layers.map(\.index), sites: sites,
            tokenPositions: tokenPositions,
            unobservedDecodePositions: tokenPositions.decode.filter { $0 >= generatedTokenCount },
            maxCaptureBytes: maxCaptureBytes, estimatedCaptureBytes: estimatedCaptureBytes)
    }
}

private struct InspectionTextBudget {
    private var remaining = 256 * 1024

    mutating func consume(_ text: String) throws -> String {
        let count = text.utf8.count
        guard count <= remaining else {
            throw ModelInspectionError.invalidGraph(
                "Inspection token labels, predictions, and answer exceed the 256 KiB text budget.")
        }
        remaining -= count
        return text
    }
}

enum ModelInspection {
    static let promptTokenLimit = 256
    static let outputTokenLimit = 64
    static let channelBinCount = 16
    static let prefillChunkSize = 16
    static let selectedLayerLimit = 128
    static let captureByteLimit = 16 * 1_048_576
    static let measurement =
        "Measured pre-normalization residual activations: layer input and after attention. RMS and maximum absolute value summarize each token's hidden vector; channels are 16 contiguous channel-group RMS values. These are not attention probabilities, feature meanings, or explanations."

    static func maximumTokens(_ request: InspectorTraceRequest) throws -> Int {
        guard !request.question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ModelInspectionError.invalidRequest("Enter a question to inspect.")
        }
        // Bound work before chat-template/tokenizer allocation as well as after it.
        guard request.question.utf8.count <= 16_384 else {
            throw ModelInspectionError.invalidRequest("Inspector questions must be shorter than 16 KiB.")
        }
        let result = request.maxTokens ?? 32
        guard (0...outputTokenLimit).contains(result) else {
            throw ModelInspectionError.invalidRequest("Inspector maxTokens must be between 0 and \(outputTokenLimit).")
        }
        try validateSelection(request, maximumTokens: result)
        return result
    }

    private static func validateSelection(_ request: InspectorTraceRequest, maximumTokens: Int) throws {
        guard let layers = request.layers, !layers.isEmpty, layers.count <= selectedLayerLimit,
            layers.allSatisfy({ $0 >= 0 }), Set(layers).count == layers.count
        else {
            throw ModelInspectionError.invalidRequest(
                "Select 1...\(selectedLayerLimit) unique nonnegative layer indices explicitly.")
        }
        let sites = request.sites ?? InspectorObservationSite.allCases
        guard !sites.isEmpty, Set(sites).count == sites.count else {
            throw ModelInspectionError.invalidRequest(
                "Select at least one supported observation site without duplicates.")
        }
        guard let positions = request.tokenPositions,
            positions.prefill.count <= promptTokenLimit, positions.decode.count <= maximumTokens,
            !positions.prefill.isEmpty || !positions.decode.isEmpty,
            Set(positions.prefill).count == positions.prefill.count,
            Set(positions.decode).count == positions.decode.count,
            positions.prefill.allSatisfy({ (-promptTokenLimit..<promptTokenLimit).contains($0) }),
            positions.decode.allSatisfy({ (0..<maximumTokens).contains($0) })
        else {
            throw ModelInspectionError.invalidRequest(
                "Select tokenPositions explicitly: unique prefill positions (negative values count from the prompt end) and decode positions from 0 up to maxTokens-1. Select at least one position; maxTokens:0 requires an empty decode selection."
            )
        }
        let budget = request.maxCaptureBytes ?? captureByteLimit
        guard (1...captureByteLimit).contains(budget) else {
            throw ModelInspectionError.invalidRequest("maxCaptureBytes must be between 1 and \(captureByteLimit).")
        }
    }

    static func capturePlan(
        request: InspectorTraceRequest, descriptor: InspectorModel,
        promptTokenCount: Int, maximumTokens: Int
    ) throws -> InspectionCapturePlan {
        guard (0...outputTokenLimit).contains(maximumTokens) else {
            throw ModelInspectionError.invalidRequest("Inspector maxTokens must be between 0 and \(outputTokenLimit).")
        }
        try validateSelection(request, maximumTokens: maximumTokens)
        guard descriptor.traceSupported else {
            throw ModelInspectionError.unsupported(descriptor.traceReason ?? "Activation capture is unavailable.")
        }
        if let requestedModel = request.model, requestedModel != descriptor.id {
            throw ModelInspectionError.invalidRequest("The requested inspection model is not the loaded model.")
        }
        guard (1...promptTokenLimit).contains(promptTokenCount),
            (0...outputTokenLimit).contains(maximumTokens),
            let width = descriptor.hiddenSize, (1...131_072).contains(width)
        else {
            throw ModelInspectionError.invalidRequest(
                "The rendered prompt or residual width exceeds supported inspection bounds.")
        }
        let byIndex = Dictionary(uniqueKeysWithValues: descriptor.layers.map { ($0.index, $0) })
        let selected = try request.layers!.sorted().map { index in
            guard let layer = byIndex[index] else {
                throw ModelInspectionError.invalidRequest("Layer \(index) is not available in this model.")
            }
            return layer
        }
        let prefill = request.tokenPositions!.prefill.map { $0 < 0 ? promptTokenCount + $0 : $0 }.sorted()
        guard prefill.allSatisfy({ (0..<promptTokenCount).contains($0) }), Set(prefill).count == prefill.count else {
            throw ModelInspectionError.invalidRequest(
                "Selected prefill positions must resolve to distinct positions in the rendered prompt (\(promptTokenCount) tokens)."
            )
        }
        let positions = InspectorTokenPositions(prefill: prefill, decode: request.tokenPositions!.decode.sorted())
        let sites = (request.sites ?? InspectorObservationSite.allCases).sorted { $0.rawValue < $1.rawValue }
        let mode = request.capture ?? .summary
        let rows = positions.prefill.count + positions.decode.count
        let pointCount = selected.count * sites.count
        // Conservative encoded-payload bound: 512 bytes per summary row, raw
        // base64 expansion including worst-case escaped slashes in JSON,
        // token indices/IDs, and per-tensor JSON metadata.
        let rawBytes = mode == .summary ? 0 : try boundedProduct([rows, width, pointCount, 4])
        let summaryBytes = mode == .raw ? 0 : try boundedProduct([rows, pointCount, 512])
        let encodedBytes = try boundedSum([
            summaryBytes, ((rawBytes + 2) / 3) * 8,
            try boundedProduct([pointCount, rows * 48 + 1024]),
        ])
        let budget = request.maxCaptureBytes ?? captureByteLimit
        guard encodedBytes <= budget else {
            throw ModelInspectionError.invalidRequest(
                "Selected activations need up to \(encodedBytes) capture bytes, exceeding maxCaptureBytes \(budget). Select fewer layers, sites, or token positions."
            )
        }
        // Include summaries/raw CPU storage, encoding copies, selected temporary
        // graphs, wrapper weights, and the pre-existing minimum scratch allowance.
        let scratch = try boundedProduct([width, prefillChunkSize, pointCount, mode == .raw ? 8 : 32])
        let memory = try boundedSum([
            max(8 * 1_048_576, scratch), rawBytes * 2, encodedBytes * 3,
            // Text metadata has its own 256 KiB decoded cap. This allowance covers
            // worst-case JSON escaping/copies plus token/prediction object overhead.
            8 * 1_048_576,
        ])
        return InspectionCapturePlan(
            layers: selected, sites: sites, tokenPositions: positions,
            absolutePositions: Set(positions.prefill + positions.decode.map { promptTokenCount + $0 }),
            mode: mode, hiddenSize: width, maxCaptureBytes: budget,
            estimatedCaptureBytes: encodedBytes, estimatedMemoryBytes: memory)
    }

    private static func boundedProduct(_ values: [Int]) throws -> Int {
        try values.reduce(1) { product, value in
            let (result, overflow) = product.multipliedReportingOverflow(by: value)
            guard !overflow, result >= 0 else {
                throw ModelInspectionError.invalidRequest("The requested capture is too large.")
            }
            return result
        }
    }

    private static func boundedSum(_ values: [Int]) throws -> Int {
        try values.reduce(0) { sum, value in
            let (result, overflow) = sum.addingReportingOverflow(value)
            guard !overflow, result >= 0 else {
                throw ModelInspectionError.invalidRequest("The requested capture is too large.")
            }
            return result
        }
    }

    static func descriptor(
        model: any LanguageModel, id: String,
        metadata: ModelInspectionMetadata, contextLength: Int
    ) -> InspectorModel {
        let leaves = model.leafModules().flattened().sorted { $0.0 < $1.0 }
        let parameters = model.parameters().flattened()
        var grouped: [String: [(String, Module)]] = [:]
        for (path, module) in leaves {
            guard let address = InspectionLayerAddress.parse(path) else {
                continue
            }
            grouped[address.path, default: []].append((path, module))
        }
        let layers = grouped.keys.sorted {
            let left = InspectionLayerAddress.parse($0)!
            let right = InspectionLayerAddress.parse($1)!
            return left.index == right.index ? left.path < right.path : left.index < right.index
        }.map { path in
            let index = InspectionLayerAddress.parse(path)!.index
            let layerParameters = parameters.filter { $0.0.hasPrefix(path + ".") }
            let modules = grouped[path, default: []].map { modulePath, module in
                let tensors = module.parameters().flattened().sorted { $0.0 < $1.0 }
                let quantized = module as? any Quantized
                return InspectorModule(
                    path: modulePath, kind: String(describing: type(of: module)),
                    storedElementCount: tensors.reduce(0) { $0 + $1.1.size },
                    parameters: tensors.map {
                        InspectorParameter(name: $0.0, shape: $0.1.shape, dtype: String(describing: $0.1.dtype))
                    },
                    quantization: quantized.map {
                        InspectorQuantization(format: $0.mode.rawValue, bits: $0.bits, groupSize: $0.groupSize)
                    })
            }
            var kind: [String] = []
            if metadata.layerTypes.indices.contains(index) {
                kind.append(metadata.layerTypes[index])
            }
            if metadata.mlpLayerTypes.indices.contains(index) {
                kind.append(metadata.mlpLayerTypes[index])
            }
            if kind.isEmpty {
                kind = ["transformer"]
            }
            return InspectorLayer(
                index: index, path: path, kind: kind.joined(separator: " · "),
                storedElementCount: layerParameters.reduce(0) { $0 + $1.1.size },
                weightBytes: layerParameters.reduce(0) { $0 + $1.1.nbytes }, modules: modules)
        }
        let supportReason = traceSupportReason(model: model, leaves: leaves, layers: layers)
        let inferredWidth = leaves.first { $0.0.hasSuffix(".input_layernorm") }
            .flatMap { ($0.1 as? RMSNorm)?.weight.dim(0) }
        let width = inferredWidth ?? metadata.hiddenSize
        var recordingComponents = [
            (".self_attn.o_proj", "attention_output"),
            (".mlp.down_proj", "feed_forward_output"),
            (".mlp.shared_expert.down_proj", "shared_expert_output"),
        ].filter { suffix, _ in leaves.contains { $0.0.hasSuffix(suffix) && $0.1 is Linear } }.map { $0.1 }
        if (model as? LagunaModel)?.fusedGateUpSiluSparseLayerCount ?? 0 > 0 || model is GPTOSSModel {
            recordingComponents.append("expert_routing")
        }
        let capabilities =
            supportReason == nil && width != nil
            ? InspectorTraceCapabilities(
                observationPoints: [
                    .init(
                        site: .layerInput,
                        description: "Residual input to the decoder layer, before input RMS normalization.",
                        hiddenSize: width!),
                    .init(
                        site: .afterAttention,
                        description: "Residual after attention addition, before post-attention RMS normalization.",
                        hiddenSize: width!),
                ], maxSelectedLayers: min(selectedLayerLimit, layers.count), maxPrefillTokens: promptTokenLimit,
                maxDecodeTokens: outputTokenLimit, maxCaptureBytes: captureByteLimit,
                recordingComponents: recordingComponents) : nil
        return InspectorModel(
            id: id, modelType: metadata.modelType, runtimeType: String(describing: type(of: model)),
            layerCount: layers.count, hiddenSize: width,
            storedElementCount: parameters.reduce(0) { $0 + $1.1.size },
            weightBytes: parameters.reduce(0) { $0 + $1.1.nbytes }, traceSupported: supportReason == nil,
            traceReason: supportReason, layers: layers, attentionHeads: metadata.attentionHeads,
            kvHeads: metadata.kvHeads, expertCount: metadata.expertCount,
            expertsPerToken: metadata.expertsPerToken, contextLength: contextLength,
            configuredContextLength: metadata.configuredContextLength,
            activation: model is LFM2Model ? "SwiGLU (native LFM2 runtime)" : metadata.activation,
            traceCapabilities: capabilities)
    }

    private static func traceSupportReason(
        model: any LanguageModel, leaves: [(String, Module)],
        layers: [InspectorLayer]
    ) -> String? {
        guard model is LagunaModel || model is Mistral3TextModel || model is GPTOSSModel else {
            return
                "Live activation capture currently supports native Laguna, Mistral3/Ministral3, and GPT-OSS runtimes. Loaded architecture inspection remains available."
        }
        guard !layers.isEmpty, Set(layers.map(\.index)) == Set(0..<layers.count) else {
            return "This runtime does not expose a complete, contiguous decoder layer stack."
        }
        let norms = Dictionary(
            uniqueKeysWithValues: leaves.compactMap { path, module -> (String, RMSNorm)? in
                guard let norm = module as? RMSNorm else {
                    return nil
                }
                return (path, norm)
            })
        guard
            layers.allSatisfy({
                norms[$0.path + ".input_layernorm"] != nil
                    && norms[$0.path + ".post_attention_layernorm"] != nil
            })
        else {
            return "This runtime does not expose both supported residual observation sites for every layer."
        }
        return nil
    }

    /// All MLX references stay inside the model container's serialized closure.
    /// This uses a fresh cache, synchronous greedy decode, and a bounded scratch graph.
    static func trace(
        context: ModelContext, descriptor: InspectorModel, question: String,
        promptTokens: [Int], maximumTokens: Int, capturePlan: InspectionCapturePlan
    ) throws -> InspectorTrace {
        guard descriptor.traceSupported else {
            throw ModelInspectionError.unsupported(descriptor.traceReason ?? "Activation capture is unavailable.")
        }
        guard !promptTokens.isEmpty, promptTokens.count <= promptTokenLimit,
            (0...outputTokenLimit).contains(maximumTokens)
        else {
            throw ModelInspectionError.invalidRequest(
                "The rendered inspector prompt must contain 1...\(promptTokenLimit) tokens. Shorten the question.")
        }
        let model = context.model
        let recorder = InspectionActivationRecorder(plan: capturePlan)
        let installation = try InspectionNormInstallation(
            model: model, layers: capturePlan.layers,
            recorder: recorder, sites: capturePlan.sites)
        // Installation also restores partially installed hooks if an update throws.
        defer { installation.restore() }
        defer { StreamOrDevice.default.stream.synchronize() }
        let cache = try model.newCache(parameters: GenerateParameters(maxTokens: maximumTokens, temperature: 0))
        var stopTokens = context.configuration.eosTokenIds
        if let eos = context.tokenizer.eosTokenId {
            stopTokens.insert(eos)
        }
        if let unknown = context.tokenizer.unknownTokenId {
            stopTokens.insert(unknown)
        }
        for text in context.configuration.extraEOSTokens {
            if let token = context.tokenizer.convertTokenToId(text) {
                stopTokens.insert(token)
            }
        }
        stopTokens.formUnion(semanticStopTokenIDs(tokenizer: context.tokenizer, isGPTOSS: model is GPTOSSModel))
        var predictions: [InspectorPrediction] = []
        var textBudget = InspectionTextBudget()
        _ = try textBudget.consume(question)
        var nextToken: Int?
        for start in stride(from: 0, to: promptTokens.count, by: prefillChunkSize) {
            try Task.checkCancellation()
            let end = min(start + prefillChunkSize, promptTokens.count)
            nextToken = try forward(
                model: model, tokens: Array(promptTokens[start..<end]), cache: cache,
                recorder: recorder, start: start, tokenizer: context.tokenizer,
                predictions: &predictions, textBudget: &textBudget)
        }
        var generated: [Int] = []
        var stopReason = maximumTokens == 0 ? "prefill_only" : "length"
        for _ in 0..<maximumTokens {
            try Task.checkCancellation()
            guard let token = nextToken else {
                throw ModelInspectionError.invalidGraph("The model returned no next token.")
            }
            if stopTokens.contains(token) {
                stopReason = "stop"
                break
            }
            generated.append(token)
            // Evaluate even the last emitted token so each displayed answer token has
            // real input activations. The extra prediction is discarded at the limit.
            nextToken = try forward(
                model: model, tokens: [token], cache: cache, recorder: recorder,
                start: promptTokens.count + generated.count - 1, tokenizer: context.tokenizer,
                predictions: &predictions, textBudget: &textBudget)
        }
        try Task.checkCancellation()
        let allTokens = promptTokens + generated
        let tokens = try allTokens.enumerated().map { position, token in
            InspectorToken(
                id: token, text: try textBudget.consume(context.tokenizer.decode(tokenIds: [token])),
                position: position,
                phase: position < promptTokens.count ? "prompt" : "answer")
        }
        return InspectorTrace(
            model: descriptor.id, question: question,
            answer: try textBudget.consume(context.tokenizer.decode(tokenIds: generated, skipSpecialTokens: false)),
            measurement: measurement, tokens: tokens, layers: try recorder.finalize(tokenCount: allTokens.count),
            stopReason: stopReason, maxTokens: maximumTokens,
            promptTokenCount: promptTokens.count, generatedTokenCount: generated.count, outputKind: "raw",
            predictions: predictions,
            tensors: try recorder.rawTensors(tokenIDs: allTokens),
            capture: capturePlan.report(generatedTokenCount: generated.count))
    }

    /// Converted Harmony checkpoints may omit generation_config.json. Confirm
    /// each spelling with the tokenizer so an unknown-token fallback never
    /// invents a stop ID; support both published and renamed control spellings.
    static func semanticStopTokenIDs(tokenizer: any MLXLMCommon.Tokenizer, isGPTOSS: Bool) -> Set<Int> {
        guard isGPTOSS else {
            return []
        }
        return Set(
            ["<|call|>", "<|return|>", "<|ghissue|>", "<|fim_suffix|>"].compactMap { spelling in
                guard let id = tokenizer.convertTokenToId(spelling), id != tokenizer.unknownTokenId,
                    tokenizer.convertIdToToken(id) == spelling
                        || tokenizer.decode(tokenIds: [id], skipSpecialTokens: false) == spelling
                else {
                    return nil
                }
                return id
            })
    }

    private static func forward(
        model: any LanguageModel, tokens: [Int], cache: [KVCache],
        recorder: InspectionActivationRecorder, start: Int, tokenizer: any MLXLMCommon.Tokenizer,
        predictions: inout [InspectorPrediction], textBudget: inout InspectionTextBudget
    ) throws -> Int {
        try autoreleasepool {
            recorder.startPosition = start
            let logits = model(MLXArray(tokens).reshaped(1, tokens.count), cache: cache)
            guard logits.ndim == 3, logits.dim(0) == 1, logits.dim(1) > 0,
                (1...1_048_576).contains(logits.dim(2))
            else {
                throw ModelInspectionError.invalidGraph("The model returned an unsupported logits shape.")
            }
            let last = logits[0, -1, 0...].asType(.float32)
            let valid = MLX.any(MLX.isFinite(last))
            let next = MLX.argMax(MLX.which(MLX.isFinite(last), last, MLXArray(-Float.infinity)))
            try recorder.flush(evaluating: [next, valid] + cache.flatMap(\.state))
            guard valid.item(Bool.self) else {
                throw ModelInspectionError.invalidGraph("The model produced non-finite logits during inspection.")
            }
            let probabilities = MLX.softmax(MLX.which(MLX.isFinite(last), last, MLXArray(-Float.infinity))).asArray(
                Float.self)
            var top: [(Int, Float)] = []
            for (id, probability) in probabilities.enumerated() {
                if top.count < 5 || probability > top.last!.1 {
                    top.append((id, probability))
                    top.sort { $0.1 == $1.1 ? $0.0 < $1.0 : $0.1 > $1.1 }
                    if top.count > 5 {
                        top.removeLast()
                    }
                }
            }
            predictions.append(
                InspectorPrediction(
                    afterTokenIndex: start + tokens.count - 1,
                    candidates: try top.map {
                        InspectorTokenProbability(
                            id: $0.0,
                            text: try textBudget.consume(tokenizer.decode(tokenIds: [$0.0], skipSpecialTokens: false)),
                            probability: $0.1)
                    }))
            return next.item(Int.self)
        }
    }
}

/// Wraps the original normalization operation exactly; never reinitializes or
/// copies trained weights. Wrappers are installed only during an inspection.
final class InspectionRecordingNorm: RMSNorm {
    private let original: RMSNorm
    private let path: String
    private let recorder: InspectionActivationRecorder

    init(original: RMSNorm, path: String, recorder: InspectionActivationRecorder) {
        self.original = original
        self.path = path
        self.recorder = recorder
        super.init(dimensions: original.weight.dim(0), eps: original.eps)
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        recorder.observe(path: path, input: input)
        return original(input)
    }
}

final class InspectionNormInstallation {
    private let model: Module
    private var originals: [(String, Module)] = []

    init(
        model: Module, layers: [InspectorLayer], recorder: InspectionActivationRecorder,
        sites: [InspectorObservationSite] = InspectorObservationSite.allCases
    ) throws {
        self.model = model
        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        var replacements: [String: Module] = [:]
        for layer in layers {
            for site in sites {
                let suffix = site == .layerInput ? "input_layernorm" : "post_attention_layernorm"
                let path = layer.path + "." + suffix
                guard let norm = leaves[path] as? RMSNorm else {
                    throw ModelInspectionError.invalidGraph("Missing supported normalization at \(path).")
                }
                recorder.register(path: path, layer: layer.index, site: site.rawValue)
                replacements[path] = InspectionRecordingNorm(original: norm, path: path, recorder: recorder)
            }
        }
        // MLX's unflattened array updates require all intermediate layer indices.
        // Include original decoder norm references for unselected layers/sites so a request
        // for layer 4 does not build a sparse array with holes at layers 0...3.
        // Only selected entries receive wrappers; other norm identities are unchanged.
        originals = leaves.compactMap { path, module in
            guard module is RMSNorm, let address = InspectionLayerAddress.parse(path),
                path == address.path + ".input_layernorm" || path == address.path + ".post_attention_layernorm"
            else {
                return nil
            }
            return (path, module)
        }.sorted { $0.0 < $1.0 }
        do {
            try model.update(
                modules: ModuleChildren.unflattened(
                    originals.map { path, module in
                        (path, replacements[path] ?? module)
                    }), verify: [.noUnusedKeys])
        } catch {
            restore()
            throw ModelInspectionError.invalidGraph(
                "Unable to install activation observers: \(error.localizedDescription)")
        }
    }

    func restore() {
        guard !originals.isEmpty else {
            return
        }
        // Every path already referred to this exact assignable module before the
        // transaction. Updating with no validation cannot fail for these paths.
        model.update(modules: ModuleChildren.unflattened(originals))
        originals.removeAll()
    }
}

final class InspectionActivationRecorder {
    struct Pending {
        var path: String
        var tokenIndices: [Int]
        var width: Int
        var rms: MLXArray?
        var maxAbs: MLXArray?
        var channels: MLXArray?
        var raw: MLXArray?

        var arrays: [MLXArray] { [rms, maxAbs, channels, raw].compactMap { $0 } }
    }

    var startPosition = 0
    private let mode: InspectorCaptureMode
    private let selectedPositions: Set<Int>?
    private let expectedWidth: Int?
    private var layers: [String: InspectorActivationLayer] = [:]
    private var pending: [Pending] = []
    private var executedPaths = Set<String>()
    private var capturedIndices: [String: [Int]] = [:]
    private var widths: [String: Int] = [:]
    private var rawData: [String: Data] = [:]
    private var failure: String?

    init(plan: InspectionCapturePlan? = nil) {
        mode = plan?.mode ?? .summary
        selectedPositions = plan?.absolutePositions
        expectedWidth = plan?.hiddenSize
    }

    func register(path: String, layer: Int, site: String) {
        guard layers[path] == nil else {
            failure = "Duplicate activation observation site at \(path)."
            return
        }
        layers[path] = InspectorActivationLayer(index: layer, path: path, site: site, samples: [])
        capturedIndices[path] = []
        if mode != .summary {
            rawData[path] = Data()
        }
    }

    func observe(path: String, input: MLXArray) {
        guard failure == nil else {
            return
        }
        guard layers[path] != nil, executedPaths.insert(path).inserted,
            input.ndim == 3, input.dim(0) == 1,
            input.dim(1) > 0, input.dim(1) <= ModelInspection.prefillChunkSize, input.dim(2) > 0,
            expectedWidth == nil || input.dim(2) == expectedWidth
        else {
            failure = "Unsupported or repeated residual activation geometry at \(path)."
            return
        }
        widths[path] = input.dim(2)
        let localIndices = (0..<input.dim(1)).filter {
            selectedPositions?.contains(startPosition + $0) ?? true
        }
        guard !localIndices.isEmpty else {
            return
        }
        // Slice before casting or reducing. Unselected rows create no observation
        // tensors and never cross to the CPU.
        let x = MLX.stacked(localIndices.map { input[0, $0, 0...] }, axis: 0).asType(.float32)
        let width = input.dim(2)
        var item = Pending(path: path, tokenIndices: localIndices.map { startPosition + $0 }, width: width)
        if mode != .raw {
            let squared = MLX.square(x)
            let bins = (0..<ModelInspection.channelBinCount).map { bin in
                let start = min(width - 1, bin * width / ModelInspection.channelBinCount)
                let end = max(start + 1, (bin + 1) * width / ModelInspection.channelBinCount)
                return MLX.sqrt(squared[0..., start..<end].mean(axis: -1))
            }
            item.rms = MLX.sqrt(squared.mean(axis: -1))
            item.maxAbs = MLX.abs(x).max(axis: -1)
            item.channels = MLX.stacked(bins, axis: -1)
        }
        if mode != .summary {
            item.raw = x
        }
        pending.append(item)
    }

    func flush(evaluating additionalArrays: [MLXArray] = []) throws {
        // Failed evaluation must not keep lazy observation graphs alive until a
        // later request; the trace's synchronization/restoration still runs.
        defer {
            pending.removeAll(keepingCapacity: true)
            executedPaths.removeAll(keepingCapacity: true)
        }
        if let failure {
            throw ModelInspectionError.invalidGraph(failure)
        }
        guard executedPaths == Set(layers.keys) else {
            throw ModelInspectionError.invalidGraph("Not every selected observation site executed exactly once.")
        }
        try Task.checkCancellation()
        try MLX.checkedEval(additionalArrays + pending.flatMap(\.arrays))
        for item in pending {
            try Task.checkCancellation()
            let previous = capturedIndices[item.path, default: []]
            guard item.tokenIndices.allSatisfy({ !previous.contains($0) }),
                previous.last.map({ $0 < item.tokenIndices[0] }) ?? true
            else {
                throw ModelInspectionError.invalidGraph(
                    "Activation token positions repeated or moved backwards at \(item.path).")
            }
            if let rmsArray = item.rms, let maxArray = item.maxAbs, let channelArray = item.channels {
                let rms = rmsArray.asArray(Float.self)
                let maxAbs = maxArray.asArray(Float.self)
                let bins = channelArray.asArray(Float.self)
                guard (rms + maxAbs + bins).allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                    throw ModelInspectionError.invalidGraph("Non-finite activation summary at \(item.path).")
                }
                for (row, tokenIndex) in item.tokenIndices.enumerated() {
                    let start = row * ModelInspection.channelBinCount
                    layers[item.path]!.samples.append(
                        InspectorActivationSample(
                            tokenIndex: tokenIndex, rms: rms[row], maxAbs: maxAbs[row],
                            channels: Array(bins[start..<(start + ModelInspection.channelBinCount)])))
                }
            }
            if let raw = item.raw {
                let values = raw.asArray(Float.self)
                guard values.allSatisfy(\.isFinite) else {
                    throw ModelInspectionError.invalidGraph("Non-finite raw activation at \(item.path).")
                }
                let words = values.map { $0.bitPattern.littleEndian }
                words.withUnsafeBytes { rawData[item.path]!.append(contentsOf: $0) }
            }
            capturedIndices[item.path, default: []].append(contentsOf: item.tokenIndices)
        }
    }

    private func validateCoverage(tokenCount: Int) throws {
        guard (0...(ModelInspection.promptTokenLimit + ModelInspection.outputTokenLimit)).contains(tokenCount) else {
            throw ModelInspectionError.invalidGraph("Activation token count exceeds inspection bounds.")
        }
        let expected = selectedPositions.map { $0.filter { $0 < tokenCount }.sorted() } ?? Array(0..<tokenCount)
        guard failure == nil, !layers.isEmpty, pending.isEmpty, executedPaths.isEmpty,
            capturedIndices.values.allSatisfy({ $0 == expected })
        else {
            throw ModelInspectionError.invalidGraph(
                "Activation observations did not cover the selected token sequence.")
        }
    }

    func finalize(tokenCount: Int) throws -> [InspectorActivationLayer] {
        try validateCoverage(tokenCount: tokenCount)
        guard mode != .raw else {
            return []
        }
        return orderedLayers
    }

    func rawTensors(tokenIDs: [Int]) throws -> [InspectorRawTensor]? {
        guard mode != .summary else {
            return nil
        }
        try validateCoverage(tokenCount: tokenIDs.count)
        return try orderedLayers.map { layer in
            try Task.checkCancellation()
            let indices = capturedIndices[layer.path, default: []]
            guard let width = widths[layer.path] ?? expectedWidth,
                let bytes = rawData[layer.path], bytes.count == indices.count * width * 4,
                let site = InspectorObservationSite(rawValue: layer.site)
            else {
                throw ModelInspectionError.invalidGraph("Raw tensor geometry is inconsistent at \(layer.path).")
            }
            return InspectorRawTensor(
                layerIndex: layer.index, path: layer.path, site: site,
                shape: [indices.count, width], tokenIndices: indices, tokenIDs: indices.map { tokenIDs[$0] },
                data: bytes.base64EncodedString())
        }
    }

    private var orderedLayers: [InspectorActivationLayer] {
        layers.values.sorted {
            $0.index == $1.index ? $0.path < $1.path : $0.index < $1.index
        }
    }
}
