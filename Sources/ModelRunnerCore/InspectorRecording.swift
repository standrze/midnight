import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import ModelRunnerProtocol
import Synchronization

/// The listener already has an armed or active recording.
public enum InspectorRecordingError: LocalizedError, Sendable {
    case busy

    public var errorDescription: String? {
        "A recording is already armed or recording. Cancel it before arming another."
    }
}

/// A listener-owned, bounded recording store. Polling and cancellation never wait for model execution.
public final class InspectorRecordingStore: Sendable {
    private struct State {
        var sessions: [String: InspectorRecordingSession] = [:]
        var order: [String] = []
        var active: String?
    }

    private let state = Mutex(State())

    /// Retains at most four recordings; each trace has a separately validated payload budget.
    public init() {}

    /// Returns a snapshot, or nil when the ID is unknown or was evicted.
    public func status(id: String) -> InspectorRecordingSession? {
        state.withLock { $0.sessions[id] }
    }

    /// Cancels observation only; the other client's generation continues normally.
    @discardableResult
    public func cancel(id: String) -> InspectorRecordingSession? {
        state.withLock { state in
            guard var session = state.sessions[id] else { return nil }
            if session.status == "armed" || session.status == "recording" {
                session.status = "cancelled"
                session.message = "Recording cancelled. The conversation continues."
                state.sessions[id] = session
                if state.active == id { state.active = nil }
            }
            return session
        }
    }

    func arm(model: InspectorModel) throws -> InspectorRecordingSession {
        try state.withLock { state in
            guard state.active == nil else { throw InspectorRecordingError.busy }
            while state.order.count >= 4 {
                state.sessions.removeValue(forKey: state.order.removeFirst())
            }
            let session = InspectorRecordingSession(
                id: UUID().uuidString, status: "armed", createdAt: Date().timeIntervalSince1970,
                message: "Waiting for the next text response on this model.", model: model)
            state.sessions[session.id] = session
            state.order.append(session.id)
            state.active = session.id
            return session
        }
    }

    func claim(id: String) -> Bool {
        state.withLock { state in
            guard state.active == id, var session = state.sessions[id], session.status == "armed" else {
                return false
            }
            session.status = "recording"
            session.message = "Recording activations from an external conversation."
            state.sessions[id] = session
            return true
        }
    }

    func isRecording(id: String) -> Bool {
        state.withLock { $0.sessions[id]?.status == "recording" }
    }

    func finish(
        id: String, status: String, message: String?, trace: InspectorTrace? = nil,
        cachedPromptTokenCount: Int? = nil
    ) {
        state.withLock { state in
            guard var session = state.sessions[id], session.status == "recording" else { return }
            session.status = status
            session.message = message
            session.trace = trace
            session.cachedPromptTokenCount = cachedPromptTokenCount
            state.sessions[id] = session
            if state.active == id { state.active = nil }
        }
    }
}

struct PendingInspectorRecording: Sendable {
    let id: String
    let request: InspectorRecordingRequest
    let descriptor: InspectorModel
    let store: InspectorRecordingStore
    let estimatedBytes: Int

    static func validate(request: InspectorRecordingRequest, descriptor: InspectorModel) throws -> Int {
        guard descriptor.traceSupported, descriptor.hiddenSize.map({ (1...131_072).contains($0) }) == true else {
            throw ModelInspectionError.unsupported(descriptor.traceReason ?? "Activation recording is unavailable.")
        }
        guard request.model == nil || request.model == descriptor.id else {
            throw ModelInspectionError.invalidRequest("The recording model is not the loaded model.")
        }
        let available = Set(descriptor.layers.map(\.index))
        guard (1...128).contains(request.layers.count), Set(request.layers).count == request.layers.count,
            request.layers.allSatisfy(available.contains), !request.sites.isEmpty,
            Set(request.sites).count == request.sites.count, (1...64).contains(request.maxTokens),
            (1...16 * 1_048_576).contains(request.maxCaptureBytes)
        else {
            throw ModelInspectionError.invalidRequest(
                "Select 1...128 available distinct layers, distinct observation sites, maxTokens 1...64, and maxCaptureBytes 1...16777216."
            )
        }
        // Conservative summary JSON plus paths, token labels, and scalar metadata.
        // No full prompt, full response, logits, or raw activation vectors are retained.
        let points = request.layers.count * request.sites.count
        let bytes =
            points * ((request.maxTokens + 1) * 512 + 4096)
            + request.layers.count * ((request.maxTokens + 1) * 4096 + 8192) + 512 * 1024
        guard bytes <= request.maxCaptureBytes else {
            throw ModelInspectionError.invalidRequest(
                "Recording needs up to \(bytes) bytes; select fewer layers or tokens, or increase maxCaptureBytes.")
        }
        return bytes
    }
}

/// MLX state is confined to the request's serialized execution and joined producer.
/// Only the store and final metrics cross tasks; neither contains unevaluated arrays.
final class InspectorRecordingCapture: @unchecked Sendable {
    private struct Pending {
        let path: String
        let positions: [Int]
        let rms: MLXArray
        let maximum: MLXArray
        let channels: MLXArray
    }

    private struct PendingRouting {
        let path: String
        let positions: [Int]
        let ids: MLXArray
        let weights: MLXArray
    }

    let recording: PendingInspectorRecording
    private var model: Module?
    private var originals: [(String, Module)] = []
    private var layers: [String: InspectorActivationLayer] = [:]
    private var componentLayers: [String: InspectorActivationLayer] = [:]
    private var routes: [String: InspectorExpertRouting] = [:]
    private var pendingRoutes: [PendingRouting] = []
    private var counts: [String: Int] = [:]
    private var pending: [Pending] = []
    private var inputCount: Int?
    private var promptToken: Int?
    private var sampledTokens: [Int] = []
    private var failure: String?
    private let metrics = Mutex<LocalModelRunnerMetrics?>(nil)

    init(recording: PendingInspectorRecording) { self.recording = recording }

    func components(_ original: GenerationComponents) -> GenerationComponents {
        original.appendingLogitProcessor { InspectorRecordingProcessor(capture: self) }
    }

    func record(metrics: LocalModelRunnerMetrics) { self.metrics.withLock { $0 = metrics } }

    func install(model: Module) throws {
        self.model = model
        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let selected = Set(recording.request.layers)
        var replacements: [String: Module] = [:]
        for layer in recording.descriptor.layers where selected.contains(layer.index) {
            for site in recording.request.sites {
                let suffix = site == .layerInput ? "input_layernorm" : "post_attention_layernorm"
                let path = layer.path + "." + suffix
                guard let norm = leaves[path] as? RMSNorm else {
                    throw ModelInspectionError.invalidGraph("Missing recording normalization at \(path).")
                }
                layers[path] = InspectorActivationLayer(
                    index: layer.index, path: path, site: site.rawValue, samples: [])
                counts[path] = 0
                replacements[path] = InspectorPassiveNorm(original: norm, path: path, capture: self)
            }
        }
        // Observe exact projection outputs, including quantized originals. No weights
        // or inference values are replaced by measurements.
        for layer in recording.descriptor.layers where selected.contains(layer.index) {
            for (suffix, site) in [
                ("self_attn.o_proj", "attention_output"),
                ("mlp.down_proj", "feed_forward_output"),
                ("mlp.shared_expert.down_proj", "shared_expert_output"),
            ] {
                let path = layer.path + "." + suffix
                guard let original = leaves[path] as? Linear else { continue }
                componentLayers[path] = .init(index: layer.index, path: path, site: site, samples: [])
                counts[path] = 0
                replacements[path] = InspectorPassiveLinear(original: original) { [self] output in
                    observe(path: path, input: output)
                }
            }
            if model is GPTOSSModel {
                let path = layer.path + ".mlp.router"
                guard let original = leaves[path] as? Linear,
                    let count = recording.descriptor.expertCount,
                    let topK = recording.descriptor.expertsPerToken,
                    (1...4096).contains(count), (1...min(64, count)).contains(topK)
                else {
                    throw ModelInspectionError.invalidGraph("Missing bounded GPT-OSS router metadata.")
                }
                routes[path] = .init(index: layer.index, path: path, expertCount: count, samples: [])
                counts[path] = 0
                replacements[path] = InspectorPassiveLinear(original: original) { [self] output in
                    guard let window = window(path: path, input: output, width: count) else { return }
                    // Identical to the pinned GPT-OSS router: partition, selected
                    // score gather, then precise softmax over only the selected K.
                    let ids = MLX.argPartition(window.values, kth: -topK, axis: -1)[.ellipsis, (-topK)...]
                    let scores = MLX.takeAlong(window.values, ids, axis: -1)
                    pendingRoutes.append(
                        .init(
                            path: path, positions: window.positions, ids: ids,
                            weights: MLX.softmax(scores, axis: -1, precise: true)))
                }
            }
        }
        if let laguna = model as? LagunaModel {
            laguna.setRecordingRouter { [self] index, ids, weights, count in
                guard selected.contains(index),
                    let layer = recording.descriptor.layers.first(where: { $0.index == index })
                else { return }
                let path = layer.path + ".mlp.gate"
                if routes[path] == nil {
                    guard (1...4096).contains(count), ids.ndim == 3, (1...64).contains(ids.dim(2)) else {
                        fail(message: "Unsupported expert routing geometry.")
                        return
                    }
                    routes[path] = .init(index: index, path: path, expertCount: count, samples: [])
                    counts[path] = 0
                }
                guard ids.shape == weights.shape,
                    let window = window(path: path, input: ids, width: ids.dim(-1))
                else { return }
                let start = window.positions[0] - (counts[path]! - ids.dim(1))
                pendingRoutes.append(
                    .init(
                        path: path, positions: window.positions, ids: window.values,
                        weights: weights[0, start..<(start + window.positions.count), 0...]))
            }
        }
        // Preserve all relevant leaves, including unselected array children. Restoration
        // returns the exact original modules, not reconstructed weights.
        let suffixes = [
            "input_layernorm", "post_attention_layernorm", "self_attn.o_proj",
            "mlp.down_proj", "mlp.shared_expert.down_proj", "mlp.router",
        ]
        originals = leaves.compactMap { path, module in
            guard let layer = InspectionLayerAddress.parse(path),
                suffixes.contains(where: { path == layer.path + "." + $0 }),
                module is RMSNorm || module is Linear
            else { return nil }
            return (path, module)
        }.sorted { $0.0 < $1.0 }
        do {
            try model.update(
                modules: ModuleChildren.unflattened(originals.map { ($0.0, replacements[$0.0] ?? $0.1) }),
                verify: [.noUnusedKeys])
        } catch {
            restore()
            throw error
        }
    }

    /// Called only after the generation producer has joined and synchronized.
    func restore() {
        (model as? LagunaModel)?.setRecordingRouter(nil)
        if let model, !originals.isEmpty { model.update(modules: ModuleChildren.unflattened(originals)) }
        originals.removeAll()
        model = nil
        pending.removeAll()
        pendingRoutes.removeAll()
    }

    func fail(_ error: Error) {
        fail(message: error.localizedDescription)
    }

    private func fail(message: String) {
        failure = message
        pending.removeAll()
        pendingRoutes.removeAll()
        recording.store.finish(id: recording.id, status: "failed", message: message)
    }

    private var active: Bool {
        failure == nil && recording.store.isRecording(id: recording.id)
    }

    func begin(prompt: MLXArray) {
        guard active else { return }
        guard inputCount == nil, prompt.ndim == 1, prompt.size > 0 else {
            fail(message: "Recording encountered an unsupported or repeated prompt.")
            return
        }
        inputCount = prompt.size
        let last = prompt[-1]
        do {
            try MLX.checkedEval(last)
            promptToken = last.item(Int.self)
        } catch { fail(error) }
    }

    private func window(path: String, input: MLXArray, width: Int) -> (positions: [Int], values: MLXArray)? {
        // Prefix-cache construction happens before processor.prompt.
        guard active, let inputCount else { return nil }
        guard input.ndim == 3, input.dim(0) == 1, input.dim(1) > 0,
            input.dim(2) == width, let start = counts[path]
        else {
            fail(message: "Recording encountered unsupported geometry at \(path).")
            return nil
        }
        counts[path] = start + input.dim(1)
        let first = max(start, inputCount - 1)
        let end = min(start + input.dim(1), inputCount + recording.request.maxTokens)
        guard first < end else { return nil }
        return (Array(first..<end), input[0, (first - start)..<(end - start), 0...])
    }

    func observe(path: String, input: MLXArray) {
        guard let window = window(path: path, input: input, width: recording.descriptor.hiddenSize ?? 0) else { return }
        let positions = window.positions
        let x = window.values.asType(.float32)
        let width = input.dim(2)
        let squared = MLX.square(x)
        let bins = (0..<ModelInspection.channelBinCount).map { bin in
            let lower = min(width - 1, bin * width / ModelInspection.channelBinCount)
            let upper = max(lower + 1, (bin + 1) * width / ModelInspection.channelBinCount)
            return MLX.sqrt(squared[0..., lower..<upper].mean(axis: -1))
        }
        pending.append(
            Pending(
                path: path, positions: positions, rms: MLX.sqrt(squared.mean(axis: -1)),
                maximum: MLX.abs(x).max(axis: -1), channels: MLX.stacked(bins, axis: -1)))
    }

    func flush() {
        defer {
            pending.removeAll(keepingCapacity: true)
            pendingRoutes.removeAll(keepingCapacity: true)
        }
        guard active else { return }
        do {
            try MLX.checkedEval(
                pending.flatMap { [$0.rms, $0.maximum, $0.channels] }
                    + pendingRoutes.flatMap { [$0.ids, $0.weights] })
            for item in pending {
                let rms = item.rms.asArray(Float.self)
                let maxima = item.maximum.asArray(Float.self)
                let bins = item.channels.asArray(Float.self)
                guard (rms + maxima + bins).allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                    fail(message: "Recording encountered non-finite residual values.")
                    return
                }
                for (row, position) in item.positions.enumerated() {
                    let start = row * ModelInspection.channelBinCount
                    let sample = InspectorActivationSample(
                        tokenIndex: position, rms: rms[row], maxAbs: maxima[row],
                        channels: Array(bins[start..<(start + ModelInspection.channelBinCount)]))
                    if layers[item.path] != nil {
                        layers[item.path]!.samples.append(sample)
                    } else {
                        componentLayers[item.path]!.samples.append(sample)
                    }
                }
            }
            for item in pendingRoutes {
                let ids = item.ids.asArray(Int.self)
                let weights = item.weights.asType(.float32).asArray(Float.self)
                let topK = item.ids.dim(-1)
                for (row, position) in item.positions.enumerated() {
                    let range = (row * topK)..<((row + 1) * topK)
                    let selected = Array(ids[range])
                    let values = Array(weights[range])
                    guard Set(selected).count == topK,
                        selected.allSatisfy({ $0 >= 0 && $0 < routes[item.path]!.expertCount }),
                        values.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }),
                        abs(values.reduce(0, +) - 1) < 0.02
                    else {
                        fail(message: "Invalid expert routing weights or identifiers.")
                        return
                    }
                    routes[item.path]!.samples.append(.init(tokenIndex: position, expertIDs: selected, weights: values))
                }
            }
        } catch { fail(error) }
    }

    func sampled(_ token: MLXArray) {
        guard active, sampledTokens.count < recording.request.maxTokens else { return }
        guard token.size == 1 else {
            fail(message: "Recording requires single-sequence token sampling.")
            return
        }
        do {
            try MLX.checkedEval(token)
            sampledTokens.append(token.item(Int.self))
        } catch { fail(error) }
    }

    func finish(tokenizer: any MLXLMCommon.Tokenizer, error: Error? = nil) {
        defer {
            // A retained ChatSession may retain the old component factory until
            // reconfigured. It must hold no model arrays or working observations.
            pending.removeAll()
            pendingRoutes.removeAll()
            componentLayers.removeAll()
            routes.removeAll()
            layers.removeAll()
            counts.removeAll()
            sampledTokens.removeAll()
        }
        guard active else { return }
        if let error {
            recording.store.finish(
                id: recording.id, status: error is CancellationError ? "cancelled" : "failed",
                message: error.localizedDescription)
            return
        }
        guard let info = metrics.withLock({ $0 }), let inputCount, let promptToken,
            info.promptTokenCount >= inputCount
        else {
            fail(message: "Recording completed without valid token coordinates.")
            return
        }
        if info.stopReason == "cancelled" {
            recording.store.finish(id: recording.id, status: "cancelled", message: "The conversation was cancelled.")
            return
        }
        let count = min(recording.request.maxTokens, info.generationTokenCount)
        guard sampledTokens.count >= count else {
            fail(message: "Recording did not observe all emitted token identifiers.")
            return
        }
        let offset = info.promptTokenCount - inputCount
        let expected = Array((inputCount - 1)..<(inputCount + count))
        var result = layers.values.sorted { $0.index == $1.index ? $0.path < $1.path : $0.index < $1.index }
        for index in result.indices {
            // TokenIterator can evaluate EOS or one queued token beyond the
            // output accepted by a stop parser. Only emitted tokens are replayed.
            result[index].samples.removeAll { $0.tokenIndex >= inputCount + count }
            guard result[index].samples.map(\.tokenIndex) == expected else {
                fail(message: "Recording observation coverage did not match the emitted token sequence.")
                return
            }
            for sample in result[index].samples.indices { result[index].samples[sample].tokenIndex += offset }
        }
        var components = componentLayers.values.sorted {
            $0.index == $1.index ? $0.path < $1.path : $0.index < $1.index
        }
        var routing = routes.values.sorted { $0.index < $1.index }
        for index in components.indices {
            components[index].samples.removeAll { $0.tokenIndex >= inputCount + count }
            guard components[index].samples.map(\.tokenIndex) == expected else {
                fail(message: "Incomplete component observation coverage.")
                return
            }
            for row in components[index].samples.indices { components[index].samples[row].tokenIndex += offset }
        }
        for index in routing.indices {
            routing[index].samples.removeAll { $0.tokenIndex >= inputCount + count }
            guard routing[index].samples.map(\.tokenIndex) == expected else {
                fail(message: "Incomplete expert routing coverage.")
                return
            }
            for row in routing[index].samples.indices { routing[index].samples[row].tokenIndex += offset }
        }
        let generated = Array(sampledTokens.prefix(count))
        var textBytes = 0
        let tokens = ([promptToken] + generated).enumerated().map { index, id in
            let text = tokenizer.decode(tokenIds: [id], skipSpecialTokens: false)
            textBytes += text.utf8.count
            return InspectorToken(
                id: id, text: text, position: info.promptTokenCount - 1 + index,
                phase: index == 0 ? "prompt" : "answer")
        }
        let answer = tokenizer.decode(tokenIds: generated, skipSpecialTokens: false)
        guard textBytes + answer.utf8.count <= 64 * 1024 else {
            fail(message: "Recording token labels exceeded the 64 KiB text budget.")
            return
        }
        let trace = InspectorTrace(
            model: recording.descriptor.id, question: "", answer: answer,
            measurement: ModelInspection.measurement, tokens: tokens, layers: result,
            stopReason: info.stopReason, maxTokens: recording.request.maxTokens,
            promptTokenCount: info.promptTokenCount, generatedTokenCount: info.generationTokenCount,
            outputKind: "recorded",
            capture: InspectorCaptureReport(
                mode: .summary, layers: recording.request.layers.sorted(), sites: recording.request.sites,
                tokenPositions: InspectorTokenPositions(
                    prefill: [info.promptTokenCount - 1], decode: Array(0..<recording.request.maxTokens)),
                unobservedDecodePositions: Array(count..<recording.request.maxTokens),
                maxCaptureBytes: recording.request.maxCaptureBytes,
                estimatedCaptureBytes: recording.estimatedBytes),
            components: components, expertRouting: routing)
        guard let encoded = try? JSONEncoder().encode(trace), encoded.count <= recording.request.maxCaptureBytes else {
            fail(message: "Recording exceeded its encoded capture budget.")
            return
        }
        recording.store.finish(
            id: recording.id, status: "completed",
            message: "Recorded the final prompt token and \(count) generated tokens.",
            trace: trace, cachedPromptTokenCount: info.cachedPromptTokenCount)
    }
}

/// Observes residual inputs while delegating the exact original normalization.
final class InspectorPassiveNorm: RMSNorm {
    private let original: RMSNorm
    private let path: String
    private let capture: InspectorRecordingCapture

    init(original: RMSNorm, path: String, capture: InspectorRecordingCapture) {
        self.original = original
        self.path = path
        self.capture = capture
        super.init(dimensions: original.weight.dim(0), eps: original.eps)
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        capture.observe(path: path, input: input)
        return original(input)
    }
}

/// Preserves sampler and parser behavior; captured values never modify logits.
struct InspectorRecordingProcessor: LogitProcessor {
    let capture: InspectorRecordingCapture

    func prompt(_ prompt: MLXArray) { capture.begin(prompt: prompt) }

    func process(logits: MLXArray) -> MLXArray {
        capture.flush()
        return logits
    }

    func didSample(token: MLXArray) { capture.sampled(token) }
}

/// Delegates to the original linear implementation, preserving packed quantized math.
final class InspectorPassiveLinear: Linear {
    private let original: Linear
    private let observer: (MLXArray) -> Void

    init(original: Linear, observer: @escaping (MLXArray) -> Void) {
        self.original = original
        self.observer = observer
        super.init(weight: original.weight, bias: original.bias)
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let output = original(input)
        observer(output)
        return output
    }
}
