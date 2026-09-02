import ArgumentParser
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import ModelQualityCore
import ModelRunnerCore
import ModelRunnerProtocol
import Tokenizers

private struct ScoringPayload: Sendable {
  var samples: [ModelQualityCorpusSample]
  var maximumTokensPerSample: Int
}

private struct ScoredCorpus: Sendable {
  var samples: [ModelQualitySampleResult]
  var tokenIDFingerprint: String
}

private struct QualityBenchmarkReport: Encodable, Sendable {
  var format = 1
  var status = "measured"
  var metric = "teacher_forced_next_token_nll"
  var createdAt: String
  var modelPath: String
  var modelType: String?
  var corpusPath: String
  var corpusFingerprint: String
  var tokenIDFingerprint: String
  var backend: String
  var device: String
  var addSpecialTokens = true
  var maximumTokensPerSample: Int
  var sampleCount: Int
  var scoredTokenCount: Int
  var nllSum: Double
  var tokenWeightedNLL: Double
  var perplexity: Double
  var mlxPeakMemoryBytes: Int
  var elapsedSeconds: Double
  var samples: [ModelQualitySampleResult]

  enum CodingKeys: String, CodingKey {
    case format, status, metric, backend, device, perplexity, samples
    case createdAt = "created_at"
    case modelPath = "model_path"
    case modelType = "model_type"
    case corpusPath = "corpus_path"
    case corpusFingerprint = "corpus_fingerprint"
    case tokenIDFingerprint = "token_id_fingerprint"
    case addSpecialTokens = "add_special_tokens"
    case maximumTokensPerSample = "maximum_tokens_per_sample"
    case sampleCount = "sample_count"
    case scoredTokenCount = "scored_token_count"
    case nllSum = "nll_sum"
    case tokenWeightedNLL = "token_weighted_nll"
    case mlxPeakMemoryBytes = "mlx_peak_memory_bytes"
    case elapsedSeconds = "elapsed_seconds"
  }
}

private enum QualityBenchmarkError: Error, LocalizedError {
  case invalidInput(String)
  case insufficientTokens(sampleID: String, count: Int)

  var errorDescription: String? {
    switch self {
    case .invalidInput(let detail):
      "Invalid quality benchmark input: \(detail)"
    case .insufficientTokens(let sampleID, let count):
      "Quality sample '\(sampleID)' encoded to \(count) token(s); at least 2 are required."
    }
  }
}

@main
private struct ModelQualityBenchmark: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "model-runner-quality-bench",
    abstract:
      "Measure teacher-forced NLL/perplexity for one local MLX checkpoint per process."
  )

  @Argument(help: "Local MLX checkpoint directory.")
  var model: String

  @Argument(help: "Deterministic JSONL corpus path.")
  var corpus: String

  @Argument(help: "JSON report path.")
  var output: String

  @Option(help: "Maximum encoded tokens evaluated per sample (2...2048).")
  var maxTokensPerSample = 512

  @Flag(help: "Run on CPU instead of the default MLX device.")
  var cpu = false

  @Flag(help: "Replace an existing report file.")
  var overwrite = false

  mutating func validate() throws {
    guard (2...2_048).contains(maxTokensPerSample) else {
      throw ValidationError("--max-tokens-per-sample must be in 2...2048.")
    }
  }

  mutating func run() async throws {
    let modelURL = localURL(model, isDirectory: true)
    let corpusURL = localURL(corpus)
    let outputURL = localURL(output)
    try validateInputs(modelURL: modelURL, corpusURL: corpusURL, outputURL: outputURL)

    let samples = try ModelQualityCore.loadCorpus(from: corpusURL)
    let payload = ScoringPayload(
      samples: samples,
      maximumTokensPerSample: maxTokensPerSample
    )
    let resourceLimits = try MLXResourceLimits.resolve(
      for: cpu ? .cpu : benchmarkEngine,
      physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory
    )
    let startedAt = ContinuousClock.now

    let runBenchmark: @Sendable () async throws -> ScoredCorpus = {
      Memory.peakMemory = 0
      try MLXResourceGuard.apply(resourceLimits)
      let container = try await #huggingFaceLoadModelContainer(
        configuration: ModelConfiguration(directory: modelURL)
      )
      return try await container.perform(values: payload) { context, payload in
        context.model.train(false)
        var measurements = [ModelQualitySampleResult]()
        var tokenSequences = [ModelQualityTokenSequence]()
        measurements.reserveCapacity(payload.samples.count)
        tokenSequences.reserveCapacity(payload.samples.count)

        for (index, sample) in payload.samples.enumerated() {
          let originalTokens = context.tokenizer.encode(
            text: sample.text,
            addSpecialTokens: true
          )
          let evaluatedTokens = try ModelQualityCore.boundedTokens(
            originalTokens,
            maximumCount: payload.maximumTokensPerSample
          )
          guard evaluatedTokens.count >= 2 else {
            throw QualityBenchmarkError.insufficientTokens(
              sampleID: sample.id,
              count: evaluatedTokens.count
            )
          }

          let nllSum = scoreNLL(tokens: evaluatedTokens, model: context.model)
          let fingerprint = ModelQualityCore.tokenIDFingerprint(evaluatedTokens)
          let measurement = try ModelQualitySampleResult(
            id: sample.id,
            category: sample.category,
            originalTokenCount: originalTokens.count,
            evaluatedTokenCount: evaluatedTokens.count,
            tokenIDFingerprint: fingerprint,
            nllSum: nllSum
          )
          measurements.append(measurement)
          tokenSequences.append(
            ModelQualityTokenSequence(sampleID: sample.id, tokenIDs: evaluatedTokens)
          )
          Memory.clearCache()
          print(
            "sample \(index + 1)/\(payload.samples.count) \(sample.id): "
              + String(format: "NLL %.6f, perplexity %.6f", measurement.nll, measurement.perplexity)
          )
        }

        return ScoredCorpus(
          samples: measurements,
          tokenIDFingerprint: ModelQualityCore.combinedTokenIDFingerprint(tokenSequences)
        )
      }
    }

    let scored: ScoredCorpus
    if cpu {
      scored = try await Device.withDefaultDevice(.cpu, runBenchmark)
    } else {
      scored = try await runBenchmark()
    }
    let peakMemory = Memory.peakMemory
    let elapsedSeconds = seconds(startedAt.duration(to: .now))
    let summary = try ModelQualityCore.summarize(scored.samples)
    let report = QualityBenchmarkReport(
      createdAt: ISO8601DateFormatter().string(from: Date()),
      modelPath: modelURL.path,
      modelType: checkpointModelType(modelURL),
      corpusPath: corpusURL.path,
      corpusFingerprint: ModelQualityCore.corpusFingerprint(samples),
      tokenIDFingerprint: scored.tokenIDFingerprint,
      backend: backendName,
      device: cpu ? "cpu" : (Device.defaultDevice().deviceType?.rawValue ?? "unknown"),
      maximumTokensPerSample: maxTokensPerSample,
      sampleCount: summary.sampleCount,
      scoredTokenCount: summary.scoredTokenCount,
      nllSum: summary.nllSum,
      tokenWeightedNLL: summary.tokenWeightedNLL,
      perplexity: summary.perplexity,
      mlxPeakMemoryBytes: peakMemory,
      elapsedSeconds: elapsedSeconds,
      samples: scored.samples
    )

    try FileManager.default.createDirectory(
      at: outputURL.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(report).write(to: outputURL, options: .atomic)
    print(
      String(
        format: "token-weighted NLL %.6f, perplexity %.6f across %d scored tokens",
        summary.tokenWeightedNLL,
        summary.perplexity,
        summary.scoredTokenCount
      )
    )
    print("Wrote \(outputURL.path)")
  }

  private func validateInputs(modelURL: URL, corpusURL: URL, outputURL: URL) throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: modelURL.path, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      throw QualityBenchmarkError.invalidInput(
        "model directory does not exist: \(modelURL.path)"
      )
    }

    isDirectory = false
    guard FileManager.default.fileExists(atPath: corpusURL.path, isDirectory: &isDirectory),
      !isDirectory.boolValue
    else {
      throw QualityBenchmarkError.invalidInput(
        "corpus file does not exist: \(corpusURL.path)"
      )
    }
    guard corpusURL != outputURL else {
      throw QualityBenchmarkError.invalidInput("output must not replace the corpus")
    }
    if FileManager.default.fileExists(atPath: outputURL.path), !overwrite {
      throw QualityBenchmarkError.invalidInput(
        "output already exists (pass --overwrite to replace it): \(outputURL.path)"
      )
    }
  }
}

private func scoreNLL(tokens: [Int], model: any LanguageModel) -> Double {
  let inputCount = tokens.count - 1
  let inputs = MLXArray(Array(tokens.dropLast())).reshaped(1, inputCount)
  let targets = MLXArray(Array(tokens.dropFirst())).reshaped(1, inputCount)
  let logits = model(inputs, cache: nil).asType(.float32)
  let lossSum = MLXFast.crossEntropy(logits: logits, targets: targets).sum()
  MLX.eval(lossSum)
  return Double(lossSum.item(Float.self))
}

private func localURL(_ path: String, isDirectory: Bool = false) -> URL {
  let expanded = NSString(string: path).expandingTildeInPath
  return URL(fileURLWithPath: expanded, isDirectory: isDirectory).standardizedFileURL
}

private func checkpointModelType(_ modelURL: URL) -> String? {
  let configURL = modelURL.appendingPathComponent("config.json")
  guard let data = try? Data(contentsOf: configURL),
    let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
  else {
    return nil
  }
  if let modelType = object["model_type"] as? String {
    return modelType
  }
  return (object["text_config"] as? [String: Any])?["model_type"] as? String
}

private func seconds(_ duration: Duration) -> Double {
  let components = duration.components
  return Double(components.seconds)
    + Double(components.attoseconds) / 1_000_000_000_000_000_000
}

private var backendName: String {
  #if MLX_METAL_BACKEND
    "metal"
  #elseif MLX_CUDA_BACKEND
    "cuda"
  #elseif MLX_CPU_BACKEND
    "cpu"
  #else
    "unknown"
  #endif
}

private var benchmarkEngine: ModelEngine {
  #if MLX_METAL_BACKEND
    .metal
  #elseif MLX_CUDA_BACKEND
    .cuda
  #else
    .cpu
  #endif
}
