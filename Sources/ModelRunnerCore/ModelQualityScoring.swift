import MLX
import MLXLMCommon
import MLXNN

/// Teacher-forced scoring diagnostics; no logits are retained after a chunk.
public struct ModelQualityNLLScore: Sendable {
  public var nllSum: Double
  public var scoredTokenCount: Int
  public var chunkCount: Int
  public var maximumLogitsTokenCount: Int
  public var finalCacheOffsets: [Int]
}

public enum ModelQualityScoringError: Error {
  case insufficientTokens
  case invalidPrefillStepSize
}

public enum ModelQualityScoring {
  /// A zero step preserves the original all-at-once, cache:nil scoring path.
  /// Positive steps create a fresh native cache for this sample and evaluate
  /// every next-token target, including the target across each chunk boundary.
  public static func scoreNLL(
    tokens: [Int], model: any LanguageModel, prefillStepSize: Int = 0
  ) throws -> ModelQualityNLLScore {
    guard tokens.count >= 2 else { throw ModelQualityScoringError.insufficientTokens }
    guard (0...8_192).contains(prefillStepSize) else {
      throw ModelQualityScoringError.invalidPrefillStepSize
    }
    let inputCount = tokens.count - 1
    let stepSize = prefillStepSize == 0 ? inputCount : prefillStepSize
    let cache: [KVCache]? = prefillStepSize == 0 ? nil : try model.newCache(parameters: nil)
    var total = 0.0
    var scoredTokenCount = 0
    var chunkCount = 0
    var maximumLogitsTokenCount = 0

    for start in stride(from: 0, to: inputCount, by: stepSize) {
      let end = min(start + stepSize, inputCount)
      let count = end - start
      let inputs = MLXArray(Array(tokens[start..<end])).reshaped(1, count)
      // The last input in this chunk predicts the next chunk's first input.
      let targets = MLXArray(Array(tokens[(start + 1)..<(end + 1)])).reshaped(1, count)
      let logits = model(inputs, cache: cache).asType(.float32)
      #if os(Linux)
      let lossSum = MLXNN.crossEntropy(logits: logits, targets: targets).sum()
      #else
      let lossSum = MLXFast.crossEntropy(logits: logits, targets: targets).sum()
      #endif
      MLX.eval(lossSum)
      if let cache { MLX.eval(cache) }
      total += Double(lossSum.item(Float.self))
      scoredTokenCount += count
      chunkCount += 1
      maximumLogitsTokenCount = max(maximumLogitsTokenCount, count)
    }

    return ModelQualityNLLScore(
      nllSum: total, scoredTokenCount: scoredTokenCount, chunkCount: chunkCount,
      maximumLogitsTokenCount: maximumLogitsTokenCount,
      finalCacheOffsets: cache?.map(\.offset) ?? [])
  }
}
