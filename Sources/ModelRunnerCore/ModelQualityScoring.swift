import MLX
import MLXLMCommon
import MLXNN

/// Teacher-forced scoring diagnostics; no logits are retained after a chunk.
public struct ModelQualityNLLScore: Sendable {
    /// Sum of next-token negative log likelihoods, in natural-log units (nats).
    public var nllSum: Double
    /// Number of next-token targets included in `nllSum`.
    public var scoredTokenCount: Int
    public var chunkCount: Int
    /// Largest number of token positions represented by logits in one chunk.
    public var maximumLogitsTokenCount: Int
    /// Token-position offsets of the native KV caches after scoring.
    public var finalCacheOffsets: [Int]
    /// Optional fixed-input choices; these describe the same reference prefixes in every run.
    public var tokenDiagnostics: [ModelQualityTokenDiagnostic]?
}

/// Compact diagnostics for one teacher-forced next-token prediction.
public struct ModelQualityTokenDiagnostic: Codable, Sendable {
    public var position: Int
    public var referenceTokenID: Int
    public var winnerTokenID: Int
    public var runnerUpTokenID: Int
    public var winnerLogit: Float
    public var runnerUpLogit: Float
    public var winnerMargin: Float
    public var referenceNLL: Double

    enum CodingKeys: String, CodingKey {
        case position
        case referenceTokenID = "reference_token_id"
        case winnerTokenID = "winner_token_id"
        case runnerUpTokenID = "runner_up_token_id"
        case winnerLogit = "winner_logit"
        case runnerUpLogit = "runner_up_logit"
        case winnerMargin = "winner_margin"
        case referenceNLL = "reference_nll"
    }
}

/// Too few tokens or an invalid prefill step for teacher-forced scoring.
public enum ModelQualityScoringError: Error {
    case insufficientTokens
    case invalidPrefillStepSize
    case diagnosticsRequireSingleTokenStep
}

/// Computes teacher-forced negative log likelihood in bounded token chunks.
public enum ModelQualityScoring {
    /// A zero step preserves the original all-at-once, cache:nil scoring path.
    /// Positive steps create a fresh native cache for this sample and evaluate
    /// every next-token target, including the target across each chunk boundary.
    public static func scoreNLL(
        tokens: [Int], model: any LanguageModel, prefillStepSize: Int = 0,
        tokenDiagnostics: Bool = false
    ) throws -> ModelQualityNLLScore {
        guard tokens.count >= 2 else {
            throw ModelQualityScoringError.insufficientTokens
        }
        guard (0...8_192).contains(prefillStepSize) else {
            throw ModelQualityScoringError.invalidPrefillStepSize
        }
        guard !tokenDiagnostics || prefillStepSize == 1 else {
            throw ModelQualityScoringError.diagnosticsRequireSingleTokenStep
        }
        let inputCount = tokens.count - 1
        let stepSize = prefillStepSize == 0 ? inputCount : prefillStepSize
        let cache: [KVCache]? = prefillStepSize == 0 ? nil : try model.newCache(parameters: nil)
        var total = 0.0
        var scoredTokenCount = 0
        var chunkCount = 0
        var maximumLogitsTokenCount = 0
        var diagnostics: [ModelQualityTokenDiagnostic]? = tokenDiagnostics ? [] : nil

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
            if let cache {
                MLX.eval(cache)
            }
            let nll = Double(lossSum.item(Float.self))
            total += nll
            if tokenDiagnostics {
                let row = logits[0, 0, 0...]
                let winner = argMax(row)
                let indices = MLXArray(0..<Int32(row.size))
                let alternatives = which(indices .== winner, MLXArray(-Float.infinity), row)
                let runnerUp = argMax(alternatives)
                let winnerLogit = row[winner]
                let runnerUpLogit = row[runnerUp]
                eval(winner, runnerUp, winnerLogit, runnerUpLogit)
                let first = winnerLogit.item(Float.self)
                let second = runnerUpLogit.item(Float.self)
                diagnostics?.append(
                    ModelQualityTokenDiagnostic(
                        position: start + 1, referenceTokenID: tokens[start + 1],
                        winnerTokenID: winner.item(Int.self), runnerUpTokenID: runnerUp.item(Int.self),
                        winnerLogit: first, runnerUpLogit: second, winnerMargin: first - second,
                        referenceNLL: nll))
            }
            scoredTokenCount += count
            chunkCount += 1
            maximumLogitsTokenCount = max(maximumLogitsTokenCount, count)
        }

        return ModelQualityNLLScore(
            nllSum: total, scoredTokenCount: scoredTokenCount, chunkCount: chunkCount,
            maximumLogitsTokenCount: maximumLogitsTokenCount,
            finalCacheOffsets: cache?.map(\.offset) ?? [], tokenDiagnostics: diagnostics)
    }
}
