import Foundation
import CoreFoundation

/// Process-wide cache policy. Compression is deliberately opt-in.
public struct LongContextOptions: Equatable, Sendable {
  public enum Compression: String, CaseIterable, Sendable {
    case none, affine8, affine4, turbo8v4
  }

  public let contextLength: Int?
  public let prefillStepSize: Int
  public let compression: Compression

  public init(contextLength: Int? = nil, prefillStepSize: Int = 512,
              kvCompression: String = "none") throws {
    if let contextLength, !(1...16_777_216).contains(contextLength) {
      throw RequestAdmissionError.configuration("context length must be between 1 and 16777216")
    }
    guard (1...8192).contains(prefillStepSize) else {
      throw RequestAdmissionError.configuration("prefill step size must be between 1 and 8192")
    }
    guard let compression = Compression(rawValue: kvCompression) else {
      throw RequestAdmissionError.configuration("KV compression must be none, affine8, affine4, or turbo8v4")
    }
    self.contextLength = contextLength
    self.prefillStepSize = prefillStepSize
    self.compression = compression
  }
}

public enum RequestAdmissionError: Error, LocalizedError, Equatable, Sendable {
  case configuration(String)
  case contextExceeded(prompt: Int, output: Int, limit: Int)
  case memoryExceeded(required: Int, available: Int)

  public var errorDescription: String? {
    switch self {
    case .configuration(let detail): "Invalid long-context configuration: \(detail)."
    case .contextExceeded(let prompt, let output, let limit):
      "The request needs \(prompt) prompt + \(output) output tokens; the context limit is \(limit)."
    case .memoryExceeded(let required, let available):
      "Estimated request memory \(required) bytes exceeds the \(available)-byte budget. Reduce context/output length or increase the configured memory budget."
    }
  }
}

/// Conservative admission estimate, not a promise about backend peak allocation.
/// Full-precision KV is budgeted even with compression: conversion can temporarily
/// retain both representations. Native sliding windows remain bounded.
public struct ModelMemoryProfile: Sendable {
  public let contextLength: Int
  public let hasKnownGeometry: Bool
  public let bytesPerLayerToken: Int
  public let layerWindows: [Int?]
  public let hiddenSize: Int
  public let intermediateSize: Int
  public let vocabularySize: Int

  public init(configuration data: Data, options: LongContextOptions) throws {
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw RequestAdmissionError.configuration("model config must be a JSON object")
    }
    let text = root["text_config"] as? [String: Any] ?? root
    func positive(_ key: String) -> Int? {
      guard let number = text[key] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
            number.doubleValue.isFinite, number.doubleValue.rounded() == number.doubleValue,
            number.doubleValue > 0, number.doubleValue <= 16_777_216 else { return nil }
      return number.intValue
    }
    let declaredContext = positive("max_position_embeddings") ?? positive("n_positions")
      ?? positive("max_sequence_length") ?? positive("seq_length")
    if let requested = options.contextLength, let declaredContext, requested > declaredContext {
      throw RequestAdmissionError.configuration("requested context exceeds model limit \(declaredContext)")
    }
    contextLength = options.contextLength ?? declaredContext ?? 4096
    hiddenSize = positive("hidden_size") ?? positive("d_model") ?? 4096
    intermediateSize = positive("intermediate_size") ?? Self.multiply(hiddenSize, 4)
    vocabularySize = positive("vocab_size") ?? 131_072
    let layers = positive("num_hidden_layers") ?? positive("n_layer")
    let heads = positive("num_attention_heads") ?? positive("n_head")
    let kvHeads = positive("num_key_value_heads") ?? heads
    let hidden = hiddenSize
    let dimension = positive("head_dim") ?? heads.map { max(1, hidden / $0) }
    hasKnownGeometry = layers != nil && kvHeads != nil && dimension != nil
    // FP32 is conservative across model dtypes, fused/cast paths, and missing metadata.
    bytesPerLayerToken = Self.multiply(8, Self.multiply(kvHeads ?? 32, dimension ?? 128))
    let count = min(layers ?? 64, 4096)
    guard layers == nil || layers! <= 4096 else {
      throw RequestAdmissionError.configuration("unsupported layer count")
    }
    let types = text["layer_types"] as? [String]
    let window = positive("sliding_window")
    // Missing or unfamiliar topology is conservatively treated as full attention.
    layerWindows = (0..<count).map { index in
      if let types, types.count == count, types[index] == "sliding_attention" { return window }
      return nil
    }
  }

  public func kvBytes(tokens: Int) -> Int {
    // Include growth-block slack and the transient incoming prefill rows in callers.
    layerWindows.reduce(0) { total, window in
      Self.add(total, Self.multiply(bytesPerLayerToken, min(max(0, tokens), window ?? Int.max)))
    }
  }

  public func workspaceBytes(prefillStepSize: Int) -> Int {
    let width = Self.add(vocabularySize, Self.add(Self.multiply(hiddenSize, 8), intermediateSize))
    return max(512 * 1_048_576, Self.multiply(Self.multiply(prefillStepSize, width), 8))
  }

  public func validateContext(prompt: Int, output: Int) throws {
    guard prompt >= 0, output > 0, Self.add(prompt, output) <= contextLength else {
      throw RequestAdmissionError.contextExceeded(prompt: prompt, output: output, limit: contextLength)
    }
  }

  public func requestBytes(prompt: Int, output: Int, prefillStepSize: Int, residentBytes: Int) -> Int {
    let tokens = Self.add(Self.add(prompt, output), 256)
    // Every layer may transiently hold the incoming prefill block as well as its cache.
    let incoming = Self.multiply(Self.multiply(layerWindows.count, bytesPerLayerToken), prefillStepSize)
    return Self.add(residentBytes, Self.add(kvBytes(tokens: tokens),
      Self.add(incoming, workspaceBytes(prefillStepSize: prefillStepSize))))
  }

  public static func add(_ a: Int, _ b: Int) -> Int {
    let (value, overflow) = a.addingReportingOverflow(b)
    return overflow ? Int.max : value
  }
  public static func multiply(_ a: Int, _ b: Int) -> Int {
    let (value, overflow) = a.multipliedReportingOverflow(by: b)
    return overflow ? Int.max : value
  }
}
