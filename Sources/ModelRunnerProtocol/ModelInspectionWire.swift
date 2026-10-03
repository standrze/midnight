import Foundation

/// Describes tensors in the loaded runtime, including packed quantized storage.
/// `storedElementCount` deliberately does not estimate logical model parameters.
public struct InspectorParameter: Codable, Equatable, Sendable {
    public var name: String
    public var shape: [Int]
    public var dtype: String

    /// Describes a stored tensor by name, shape, and dtype.
    public init(name: String, shape: [Int], dtype: String) {
        self.name = name
        self.shape = shape
        self.dtype = dtype
    }
}

/// A loaded module's path, stored tensor count, parameters, and quantization.
public struct InspectorModule: Codable, Equatable, Sendable {
    public var path: String
    public var kind: String
    public var storedElementCount: Int
    public var parameters: [InspectorParameter]
    public var quantization: InspectorQuantization?

    /// Describes one module and its stored parameters and quantization.
    public init(
        path: String, kind: String, storedElementCount: Int, parameters: [InspectorParameter],
        quantization: InspectorQuantization? = nil
    ) {
        self.path = path
        self.kind = kind
        self.storedElementCount = storedElementCount
        self.parameters = parameters
        self.quantization = quantization
    }
}

/// Read from the loaded module's Quantized conformance, never its filename.
public struct InspectorQuantization: Codable, Equatable, Sendable {
    public var format: String
    public var bits: Int
    public var groupSize: Int
    /// Records quantization format, bit width, and group size.
    public init(format: String, bits: Int, groupSize: Int) {
        self.format = format
        self.bits = bits
        self.groupSize = groupSize
    }
}

/// One inspected layer with stored elements and estimated weight bytes.
public struct InspectorLayer: Codable, Equatable, Sendable {
    public var index: Int
    public var path: String
    public var kind: String
    public var storedElementCount: Int
    public var weightBytes: Int
    public var modules: [InspectorModule]

    /// Describes one layer with stored element and byte counts.
    public init(
        index: Int, path: String, kind: String, storedElementCount: Int,
        weightBytes: Int, modules: [InspectorModule]
    ) {
        self.index = index
        self.path = path
        self.kind = kind
        self.storedElementCount = storedElementCount
        self.weightBytes = weightBytes
        self.modules = modules
    }
}

/// Architecture and runtime metadata for the currently loaded model.
///
/// Optional dimensions and capabilities are omitted when the runtime cannot
/// report them; `weightBytes` measures storage rather than logical parameters.
public struct InspectorModel: Codable, Equatable, Sendable {
    public var modelCard: ModelCard? = nil
    public var id: String
    public var modelType: String
    public var runtimeType: String
    public var layerCount: Int
    public var hiddenSize: Int?
    public var attentionHeads: Int?
    public var kvHeads: Int?
    public var expertCount: Int?
    public var expertsPerToken: Int?
    /// Active runtime context capacity, in token positions.
    public var contextLength: Int?
    /// Context capacity reported by the checkpoint configuration, in token positions.
    public var configuredContextLength: Int?
    public var activation: String?
    public var storedElementCount: Int
    public var weightBytes: Int
    public var traceSupported: Bool
    public var traceReason: String?
    public var traceCapabilities: InspectorTraceCapabilities?
    /// Managed model lifecycle generation associated with this inspection snapshot.
    public var runtimeGeneration: UInt64?
    public var layers: [InspectorLayer]

    /// Describes the loaded model, its layers, and available trace capabilities.
    public init(
        id: String, modelType: String, runtimeType: String, layerCount: Int,
        hiddenSize: Int?, storedElementCount: Int, weightBytes: Int,
        traceSupported: Bool, traceReason: String?, layers: [InspectorLayer],
        attentionHeads: Int? = nil, kvHeads: Int? = nil, expertCount: Int? = nil,
        expertsPerToken: Int? = nil, contextLength: Int? = nil,
        configuredContextLength: Int? = nil, activation: String? = nil,
        traceCapabilities: InspectorTraceCapabilities? = nil, runtimeGeneration: UInt64? = nil
    ) {
        self.id = id
        self.modelType = modelType
        self.runtimeType = runtimeType
        self.layerCount = layerCount
        self.hiddenSize = hiddenSize
        self.storedElementCount = storedElementCount
        self.weightBytes = weightBytes
        self.traceSupported = traceSupported
        self.traceReason = traceReason
        self.layers = layers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.expertCount = expertCount
        self.expertsPerToken = expertsPerToken
        self.contextLength = contextLength
        self.configuredContextLength = configuredContextLength
        self.activation = activation
        self.traceCapabilities = traceCapabilities
        self.runtimeGeneration = runtimeGeneration
    }
}

/// Where a residual activation is sampled within a transformer layer.
public enum InspectorObservationSite: String, Codable, CaseIterable, Sendable {
    case layerInput = "layer_input"
    case afterAttention = "after_attention"
}

/// Selects summary statistics, raw tensors, or both for an inspection trace.
public enum InspectorCaptureMode: String, Codable, Sendable {
    case summary, raw, both
}

/// Positions refer to actual forward-pass inputs, including chat-template tokens.
/// Negative prefill positions count from the end; -1 selects the last prompt token.
/// `decode[0]` is the first emitted token, evaluated after the prompt has completed.
public struct InspectorTokenPositions: Codable, Equatable, Sendable {
    public var prefill: [Int]
    public var decode: [Int]

    /// Selects prefill and generated token positions for capture.
    public init(prefill: [Int], decode: [Int]) {
        self.prefill = prefill
        self.decode = decode
    }
}

/// An available observation site and its residual hidden size.
public struct InspectorObservationPoint: Codable, Equatable, Sendable {
    public var site: InspectorObservationSite
    public var description: String
    public var hiddenSize: Int

    /// Describes an observation site and its hidden-vector width.
    public init(site: InspectorObservationSite, description: String, hiddenSize: Int) {
        self.site = site
        self.description = description
        self.hiddenSize = hiddenSize
    }
}

/// Limits apply to this dedicated operation, never to ordinary inference.
public struct InspectorTraceCapabilities: Codable, Equatable, Sendable {
    public var version: Int
    public var observationPoints: [InspectorObservationPoint]
    public var captureModes: [InspectorCaptureMode]
    public var requiresExplicitLayers: Bool
    public var requiresExplicitTokenPositions: Bool
    public var maxSelectedLayers: Int
    public var maxPrefillTokens: Int
    public var maxDecodeTokens: Int
    public var maxCaptureBytes: Int
    public var rawEncoding: String
    public var rawDtype: String
    public var rawByteOrder: String
    /// Passive recording summaries; these do not add raw tensor sites to /trace.
    public var recordingComponents: [String]?

    /// Sets trace capture limits and supported sites for the loaded runtime.
    public init(
        observationPoints: [InspectorObservationPoint], maxSelectedLayers: Int,
        maxPrefillTokens: Int, maxDecodeTokens: Int, maxCaptureBytes: Int, recordingComponents: [String]? = nil
    ) {
        version = 1
        self.observationPoints = observationPoints
        captureModes = [.summary, .raw, .both]
        requiresExplicitLayers = true
        requiresExplicitTokenPositions = true
        self.maxSelectedLayers = maxSelectedLayers
        self.maxPrefillTokens = maxPrefillTokens
        self.maxDecodeTokens = maxDecodeTokens
        self.maxCaptureBytes = maxCaptureBytes
        rawEncoding = "base64"
        rawDtype = "float32"
        rawByteOrder = "little"
        self.recordingComponents = recordingComponents
    }
}

/// Selects a question and bounded activation capture from the loaded runtime.
///
/// Layers, sites, and token positions refer to the rendered model input;
/// unsupported or excessive selections are rejected before capture.
public struct InspectorTraceRequest: Codable, Equatable, Sendable {
    public var model: String?
    /// Required managed model lifecycle generation, when supplied.
    public var runtimeGeneration: UInt64?
    public var question: String
    /// Maximum generated token IDs to inspect.
    public var maxTokens: Int?
    public var layers: [Int]?
    public var sites: [InspectorObservationSite]?
    public var tokenPositions: InspectorTokenPositions?
    public var capture: InspectorCaptureMode?
    public var maxCaptureBytes: Int?

    /// Requests bounded trace capture for a question and selected model.
    public init(
        question: String, maxTokens: Int? = nil, layers: [Int]? = nil,
        sites: [InspectorObservationSite]? = nil, tokenPositions: InspectorTokenPositions? = nil,
        capture: InspectorCaptureMode? = nil, maxCaptureBytes: Int? = nil,
        model: String? = nil, runtimeGeneration: UInt64? = nil
    ) {
        self.question = question
        self.maxTokens = maxTokens
        self.layers = layers
        self.sites = sites
        self.tokenPositions = tokenPositions
        self.capture = capture
        self.maxCaptureBytes = maxCaptureBytes
        self.model = model
        self.runtimeGeneration = runtimeGeneration
    }
}

/// An exact selected residual tensor, converted to float32 and copied to the CPU.
/// Row-major data has shape [tokenIndices.count, hiddenSize]; token indices are absolute.
public struct InspectorRawTensor: Codable, Equatable, Sendable {
    public var layerIndex: Int
    public var path: String
    public var site: InspectorObservationSite
    public var encoding: String
    public var dtype: String
    public var byteOrder: String
    public var shape: [Int]
    public var tokenIndices: [Int]
    public var tokenIDs: [Int]
    public var data: String

    /// Records a copied tensor and its absolute token positions.
    public init(
        layerIndex: Int, path: String, site: InspectorObservationSite, shape: [Int],
        tokenIndices: [Int], tokenIDs: [Int], data: String
    ) {
        self.layerIndex = layerIndex
        self.path = path
        self.site = site
        encoding = "base64"
        dtype = "float32"
        byteOrder = "little"
        self.shape = shape
        self.tokenIndices = tokenIndices
        self.tokenIDs = tokenIDs
        self.data = data
    }
}

/// Records the selected capture settings and estimated byte use.
public struct InspectorCaptureReport: Codable, Equatable, Sendable {
    public var mode: InspectorCaptureMode
    public var layers: [Int]
    public var sites: [InspectorObservationSite]
    public var tokenPositions: InspectorTokenPositions
    /// Selected generated positions not reached because generation ended first.
    public var unobservedDecodePositions: [Int]
    public var maxCaptureBytes: Int
    public var estimatedCaptureBytes: Int

    /// Reports selected capture settings and estimated byte use.
    public init(
        mode: InspectorCaptureMode, layers: [Int], sites: [InspectorObservationSite],
        tokenPositions: InspectorTokenPositions, unobservedDecodePositions: [Int],
        maxCaptureBytes: Int, estimatedCaptureBytes: Int
    ) {
        self.mode = mode
        self.layers = layers
        self.sites = sites
        self.tokenPositions = tokenPositions
        self.unobservedDecodePositions = unobservedDecodePositions
        self.maxCaptureBytes = maxCaptureBytes
        self.estimatedCaptureBytes = estimatedCaptureBytes
    }
}

/// A token observed during prompt processing or generation, with its position.
public struct InspectorToken: Codable, Equatable, Sendable {
    public var id: Int
    public var text: String
    /// Zero-based index in the combined prompt and generated-token sequence.
    public var position: Int
    public var phase: String

    /// Records one token with its position and generation phase.
    public init(id: Int, text: String, position: Int, phase: String) {
        self.id = id
        self.text = text
        self.position = position
        self.phase = phase
    }
}

/// Residual magnitude statistics for one observed token position.
public struct InspectorActivationSample: Codable, Equatable, Sendable {
    public var tokenIndex: Int
    public var rms: Float
    public var maxAbs: Float
    /// RMS of contiguous hidden-channel groups, not individual neurons.
    public var channels: [Float]

    /// Records residual magnitude statistics for one token position.
    public init(tokenIndex: Int, rms: Float, maxAbs: Float, channels: [Float]) {
        self.tokenIndex = tokenIndex
        self.rms = rms
        self.maxAbs = maxAbs
        self.channels = channels
    }
}

/// Activation samples from one layer and observation site.
public struct InspectorActivationLayer: Codable, Equatable, Sendable {
    public var index: Int
    public var path: String
    public var site: String
    public var samples: [InspectorActivationSample]

    /// Groups activation samples by layer path and observation site.
    public init(index: Int, path: String, site: String, samples: [InspectorActivationSample]) {
        self.index = index
        self.path = path
        self.site = site
        self.samples = samples
    }
}

/// Generated answer, token sequence, and selected activation measurements.
public struct InspectorTrace: Codable, Equatable, Sendable {
    public var predictions: [InspectorPrediction]?
    public var model: String
    public var question: String
    public var answer: String
    /// Inspection preserves protocol delimiters; bounded output may precede a final answer.
    public var outputKind: String?
    public var measurement: String
    public var tokens: [InspectorToken]
    public var layers: [InspectorActivationLayer]
    public var stopReason: String
    /// Requested maximum generated token IDs.
    public var maxTokens: Int
    /// Token IDs in the rendered prompt, including chat-template tokens.
    public var promptTokenCount: Int
    /// Token IDs emitted before generation stopped.
    public var generatedTokenCount: Int
    public var tensors: [InspectorRawTensor]?
    public var capture: InspectorCaptureReport?
    /// Passive recordings only: actual attention, dense MLP, and shared-expert output summaries.
    public var components: [InspectorActivationLayer]?
    /// Per-token selected expert IDs and normalized routing weights; not causal importance.
    public var expertRouting: [InspectorExpertRouting]?

    /// Builds a trace from generated text, tokens, and selected measurements.
    public init(
        model: String, question: String, answer: String, measurement: String,
        tokens: [InspectorToken], layers: [InspectorActivationLayer], stopReason: String,
        maxTokens: Int, promptTokenCount: Int, generatedTokenCount: Int, outputKind: String? = nil,
        predictions: [InspectorPrediction]? = nil, tensors: [InspectorRawTensor]? = nil,
        capture: InspectorCaptureReport? = nil, components: [InspectorActivationLayer]? = nil,
        expertRouting: [InspectorExpertRouting]? = nil
    ) {
        self.model = model
        self.question = question
        self.answer = answer
        self.measurement = measurement
        self.tokens = tokens
        self.layers = layers
        self.stopReason = stopReason
        self.maxTokens = maxTokens
        self.promptTokenCount = promptTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.outputKind = outputKind
        self.predictions = predictions
        self.tensors = tensors
        self.capture = capture
        self.components = components
        self.expertRouting = expertRouting
    }
}

/// Candidate next-token probabilities after an observed token.
public struct InspectorPrediction: Codable, Equatable, Sendable {
    /// Prediction after processing this token, not the probability of this token.
    public var afterTokenIndex: Int
    public var candidates: [InspectorTokenProbability]
    /// Records candidate probabilities after an observed token.
    public init(afterTokenIndex: Int, candidates: [InspectorTokenProbability]) {
        self.afterTokenIndex = afterTokenIndex
        self.candidates = candidates
    }
}

/// One candidate token and its model probability.
public struct InspectorTokenProbability: Codable, Equatable, Sendable {
    public var id: Int
    public var text: String
    public var probability: Float
    /// Records one candidate token and its probability.
    public init(id: Int, text: String, probability: Float) {
        self.id = id
        self.text = text
        self.probability = probability
    }
}

/// A bounded per-token router selection, in the router's native selected order.
public struct InspectorExpertRoutingSample: Codable, Equatable, Sendable {
    public var tokenIndex: Int
    public var expertIDs: [Int]
    public var weights: [Float]

    /// Weights correspond one-to-one to IDs and are normalized by the model's router.
    public init(tokenIndex: Int, expertIDs: [Int], weights: [Float]) {
        self.tokenIndex = tokenIndex
        self.expertIDs = expertIDs
        self.weights = weights
    }
}

/// Selected routed experts for one decoder layer. Shared experts are measured separately.
public struct InspectorExpertRouting: Codable, Equatable, Sendable {
    public var index: Int
    public var path: String
    public var expertCount: Int
    public var samples: [InspectorExpertRoutingSample]

    /// Records actual routing; no inference of causal necessity is made.
    public init(index: Int, path: String, expertCount: Int, samples: [InspectorExpertRoutingSample]) {
        self.index = index
        self.path = path
        self.expertCount = expertCount
        self.samples = samples
    }
}
