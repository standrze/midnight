import Foundation

/// Describes tensors in the loaded runtime, including packed quantized storage.
/// `storedElementCount` deliberately does not estimate logical model parameters.
public struct InspectorParameter: Codable, Equatable, Sendable {
  public var name: String
  public var shape: [Int]
  public var dtype: String

  public init(name: String, shape: [Int], dtype: String) {
    self.name = name; self.shape = shape; self.dtype = dtype
  }
}

public struct InspectorModule: Codable, Equatable, Sendable {
  public var path: String
  public var kind: String
  public var storedElementCount: Int
  public var parameters: [InspectorParameter]
  public var quantization: InspectorQuantization?

  public init(path: String, kind: String, storedElementCount: Int, parameters: [InspectorParameter],
              quantization: InspectorQuantization? = nil) {
    self.path = path; self.kind = kind; self.storedElementCount = storedElementCount
    self.parameters = parameters
    self.quantization = quantization
  }
}

/// Read from the loaded module's Quantized conformance, never its filename.
public struct InspectorQuantization: Codable, Equatable, Sendable {
  public var format: String
  public var bits: Int
  public var groupSize: Int
  public init(format: String, bits: Int, groupSize: Int) {
    self.format = format; self.bits = bits; self.groupSize = groupSize
  }
}

public struct InspectorLayer: Codable, Equatable, Sendable {
  public var index: Int
  public var path: String
  public var kind: String
  public var storedElementCount: Int
  public var weightBytes: Int
  public var modules: [InspectorModule]

  public init(index: Int, path: String, kind: String, storedElementCount: Int,
              weightBytes: Int, modules: [InspectorModule]) {
    self.index = index; self.path = path; self.kind = kind
    self.storedElementCount = storedElementCount; self.weightBytes = weightBytes
    self.modules = modules
  }
}

public struct InspectorModel: Codable, Equatable, Sendable {
  public var id: String
  public var modelType: String
  public var runtimeType: String
  public var layerCount: Int
  public var hiddenSize: Int?
  public var attentionHeads: Int?
  public var kvHeads: Int?
  public var expertCount: Int?
  public var expertsPerToken: Int?
  public var contextLength: Int?
  public var configuredContextLength: Int?
  public var activation: String?
  public var storedElementCount: Int
  public var weightBytes: Int
  public var traceSupported: Bool
  public var traceReason: String?
  public var layers: [InspectorLayer]

  public init(id: String, modelType: String, runtimeType: String, layerCount: Int,
              hiddenSize: Int?, storedElementCount: Int, weightBytes: Int,
              traceSupported: Bool, traceReason: String?, layers: [InspectorLayer],
              attentionHeads: Int? = nil, kvHeads: Int? = nil, expertCount: Int? = nil,
              expertsPerToken: Int? = nil, contextLength: Int? = nil,
              configuredContextLength: Int? = nil, activation: String? = nil) {
    self.id = id; self.modelType = modelType; self.runtimeType = runtimeType
    self.layerCount = layerCount; self.hiddenSize = hiddenSize
    self.storedElementCount = storedElementCount; self.weightBytes = weightBytes
    self.traceSupported = traceSupported; self.traceReason = traceReason; self.layers = layers
    self.attentionHeads = attentionHeads; self.kvHeads = kvHeads; self.expertCount = expertCount
    self.expertsPerToken = expertsPerToken; self.contextLength = contextLength
    self.configuredContextLength = configuredContextLength; self.activation = activation
  }
}

public struct InspectorTraceRequest: Codable, Equatable, Sendable {
  public var question: String
  public var maxTokens: Int?

  public init(question: String, maxTokens: Int? = nil) {
    self.question = question; self.maxTokens = maxTokens
  }
}

public struct InspectorToken: Codable, Equatable, Sendable {
  public var id: Int
  public var text: String
  public var position: Int
  public var phase: String

  public init(id: Int, text: String, position: Int, phase: String) {
    self.id = id; self.text = text; self.position = position; self.phase = phase
  }
}

public struct InspectorActivationSample: Codable, Equatable, Sendable {
  public var tokenIndex: Int
  public var rms: Float
  public var maxAbs: Float
  /// RMS of contiguous hidden-channel groups, not individual neurons.
  public var channels: [Float]

  public init(tokenIndex: Int, rms: Float, maxAbs: Float, channels: [Float]) {
    self.tokenIndex = tokenIndex; self.rms = rms; self.maxAbs = maxAbs
    self.channels = channels
  }
}

public struct InspectorActivationLayer: Codable, Equatable, Sendable {
  public var index: Int
  public var path: String
  public var site: String
  public var samples: [InspectorActivationSample]

  public init(index: Int, path: String, site: String, samples: [InspectorActivationSample]) {
    self.index = index; self.path = path; self.site = site; self.samples = samples
  }
}

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
  public var maxTokens: Int
  public var promptTokenCount: Int
  public var generatedTokenCount: Int

  public init(model: String, question: String, answer: String, measurement: String,
              tokens: [InspectorToken], layers: [InspectorActivationLayer], stopReason: String,
              maxTokens: Int, promptTokenCount: Int, generatedTokenCount: Int, outputKind: String? = nil,
              predictions: [InspectorPrediction]? = nil) {
    self.model = model; self.question = question; self.answer = answer
    self.measurement = measurement; self.tokens = tokens; self.layers = layers
    self.stopReason = stopReason; self.maxTokens = maxTokens
    self.promptTokenCount = promptTokenCount; self.generatedTokenCount = generatedTokenCount
    self.outputKind = outputKind
    self.predictions = predictions
  }
}

public struct InspectorPrediction: Codable, Equatable, Sendable {
  /// Prediction after processing this token, not the probability of this token.
  public var afterTokenIndex: Int
  public var candidates: [InspectorTokenProbability]
  public init(afterTokenIndex: Int, candidates: [InspectorTokenProbability]) {
    self.afterTokenIndex = afterTokenIndex; self.candidates = candidates
  }
}

public struct InspectorTokenProbability: Codable, Equatable, Sendable {
  public var id: Int
  public var text: String
  public var probability: Float
  public init(id: Int, text: String, probability: Float) {
    self.id = id; self.text = text; self.probability = probability
  }
}
