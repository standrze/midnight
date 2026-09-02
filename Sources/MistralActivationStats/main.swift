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

private let statisticSuffix = ".input_second_moment"
private let statisticsFormat = "mistral_activation_stats_v1"

private struct CheckpointDescriptor: Sendable {
  var modelType: String
  var textModelType: String
  var hiddenLayerCount: Int
  var tiedWordEmbeddings: Bool
  var architectures: [String]
  var isQuantized: Bool
}

private struct CollectionPayload: Sendable {
  var samples: [ModelQualityCorpusSample]
  var segmentTokenLimit: Int
  var maximumTotalTokens: Int
  var expectedModulePaths: Set<String>
}

private struct CollectedSample: Encodable, Sendable {
  var id: String
  var category: String?
  var originalTokenCount: Int
  var observedTokenCount: Int
  var segmentCount: Int
  var truncated: Bool
  var tokenIDFingerprint: String

  enum CodingKeys: String, CodingKey {
    case id, category, truncated
    case originalTokenCount = "original_token_count"
    case observedTokenCount = "observed_token_count"
    case segmentCount = "segment_count"
    case tokenIDFingerprint = "token_id_fingerprint"
  }
}

private struct CollectedModule: Sendable {
  var path: String
  var tensorKey: String
  var inputWidth: Int
  var observedPositionCount: Int
  var values: [Float]
  var minimum: Float
  var maximum: Float
  var mean: Double
}

private struct CollectedStatistics: Sendable {
  var device: String
  var sourceWeightDType: String
  var tokenIDFingerprint: String
  var observedTokenCount: Int
  var segmentCount: Int
  var modules: [CollectedModule]
  var samples: [CollectedSample]
}

private struct ModuleReport: Encodable, Sendable {
  var path: String
  var tensorKey: String
  var inputWidth: Int
  var observedPositionCount: Int
  var minimumSecondMoment: Float
  var maximumSecondMoment: Float
  var meanSecondMoment: Double

  enum CodingKeys: String, CodingKey {
    case path
    case tensorKey = "tensor_key"
    case inputWidth = "input_width"
    case observedPositionCount = "observed_position_count"
    case minimumSecondMoment = "minimum_second_moment"
    case maximumSecondMoment = "maximum_second_moment"
    case meanSecondMoment = "mean_second_moment"
  }
}

private struct ActivationStatisticsReport: Encodable, Sendable {
  var format = 1
  var status = "measured"
  var algorithm = "input_channel_second_moment"
  var createdAt: String
  var sourceModel: String
  var modelType: String
  var textModelType: String
  var architectures: [String]
  var sourceWeightDType: String
  var sourceConfigFingerprint: String
  var sourceIndexFingerprint: String?
  var corpusPath: String
  var corpusFingerprint: String
  var tokenIDFingerprint: String
  var outputPath: String
  var backend: String
  var device: String
  var addSpecialTokens = true
  var segmentation = "contiguous_nonoverlapping_independent"
  var segmentTokenLimit: Int
  var maximumTotalTokens: Int?
  var corpusSampleCount: Int
  var observedSampleCount: Int
  var observedSegmentCount: Int
  var observedTokenCount: Int
  var moduleCount: Int
  var mlxPeakMemoryBytes: Int
  var elapsedSeconds: Double
  var modules: [ModuleReport]
  var samples: [CollectedSample]

  enum CodingKeys: String, CodingKey {
    case format, status, algorithm, architectures, backend, device, segmentation, modules, samples
    case createdAt = "created_at"
    case sourceModel = "source_model"
    case modelType = "model_type"
    case textModelType = "text_model_type"
    case sourceWeightDType = "source_weight_dtype"
    case sourceConfigFingerprint = "source_config_fingerprint"
    case sourceIndexFingerprint = "source_index_fingerprint"
    case corpusPath = "corpus_path"
    case corpusFingerprint = "corpus_fingerprint"
    case tokenIDFingerprint = "token_id_fingerprint"
    case outputPath = "output_path"
    case addSpecialTokens = "add_special_tokens"
    case segmentTokenLimit = "segment_token_limit"
    case maximumTotalTokens = "maximum_total_tokens"
    case corpusSampleCount = "corpus_sample_count"
    case observedSampleCount = "observed_sample_count"
    case observedSegmentCount = "observed_segment_count"
    case observedTokenCount = "observed_token_count"
    case moduleCount = "module_count"
    case mlxPeakMemoryBytes = "mlx_peak_memory_bytes"
    case elapsedSeconds = "elapsed_seconds"
  }
}

private enum ActivationStatisticsError: Error, LocalizedError {
  case invalidInput(String)
  case unsupportedCheckpoint(String)
  case invalidModelGraph(String)
  case collectionFailed(String)
  case invalidStatistics(String)

  var errorDescription: String? {
    switch self {
    case .invalidInput(let detail):
      "Invalid activation-statistics input: \(detail)"
    case .unsupportedCheckpoint(let detail):
      "Unsupported activation-statistics checkpoint: \(detail)"
    case .invalidModelGraph(let detail):
      "Invalid Mistral model graph: \(detail)"
    case .collectionFailed(let detail):
      "Activation-statistics collection failed: \(detail)"
    case .invalidStatistics(let detail):
      "Invalid activation statistics: \(detail)"
    }
  }
}

private final class ActivationRecorder {
  private struct Entry {
    var inputWidth: Int
    var sumSquares: MLXArray
    var positionCount: Int
  }

  private let expectedWidths: [String: Int]
  private var entries = [String: Entry]()
  private var firstFailure: String?

  init(expectedWidths: [String: Int]) {
    self.expectedWidths = expectedWidths
  }

  func observe(path: String, input: MLXArray) {
    guard firstFailure == nil else { return }
    guard let expectedWidth = expectedWidths[path] else {
      firstFailure = "unexpected recording path '\(path)'"
      return
    }
    guard input.ndim >= 1, input.dim(-1) == expectedWidth else {
      firstFailure =
        "\(path) received shape \(input.shape); expected final dimension \(expectedWidth)"
      return
    }

    var positionCount = 1
    for dimension in input.shape.dropLast() {
      let (next, overflow) = positionCount.multipliedReportingOverflow(by: dimension)
      guard !overflow else {
        firstFailure = "position count overflow for \(path) with shape \(input.shape)"
        return
      }
      positionCount = next
    }
    guard positionCount > 0 else {
      firstFailure = "\(path) received an empty activation tensor with shape \(input.shape)"
      return
    }

    let reductionAxes = Array(0..<max(0, input.ndim - 1))
    let squared = MLX.square(input.asType(.float32))
    let batchSum =
      reductionAxes.isEmpty
      ? squared
      : squared.sum(axes: reductionAxes)

    if var entry = entries[path] {
      let (nextCount, overflow) = entry.positionCount.addingReportingOverflow(positionCount)
      guard !overflow else {
        firstFailure = "accumulated position count overflow for \(path)"
        return
      }
      entry.sumSquares = entry.sumSquares + batchSum
      entry.positionCount = nextCount
      entries[path] = entry
    } else {
      entries[path] = Entry(
        inputWidth: expectedWidth,
        sumSquares: batchSum,
        positionCount: positionCount
      )
    }
  }

  func evaluatePending() throws {
    try throwRecordedFailure()
    guard !entries.isEmpty else {
      throw ActivationStatisticsError.collectionFailed(
        "the model forward pass recorded no Linear inputs")
    }
    do {
      try MLX.checkedEval(entries.values.map(\.sumSquares))
    } catch {
      throw ActivationStatisticsError.collectionFailed(
        "MLX could not evaluate accumulated moments: \(error.localizedDescription)")
    }
    try throwRecordedFailure()
  }

  func finalize(expectedPaths: Set<String>) throws -> [CollectedModule] {
    try evaluatePending()
    let observedPaths = Set(entries.keys)
    let missing = expectedPaths.subtracting(observedPaths)
    let unexpected = observedPaths.subtracting(expectedPaths)
    guard missing.isEmpty, unexpected.isEmpty else {
      throw ActivationStatisticsError.invalidStatistics(
        "module coverage mismatch; missing [\(missing.sorted().joined(separator: ", "))], "
          + "unexpected [\(unexpected.sorted().joined(separator: ", "))]")
    }

    var moments = [String: MLXArray]()
    for path in expectedPaths.sorted() {
      guard let entry = entries[path], entry.positionCount > 0 else {
        throw ActivationStatisticsError.invalidStatistics(
          "\(path) has no observed activation positions")
      }
      moments[path] = (entry.sumSquares / Float(entry.positionCount)).asType(.float32)
    }
    do {
      try MLX.checkedEval(Array(moments.values))
    } catch {
      throw ActivationStatisticsError.invalidStatistics(
        "MLX could not materialize final moments: \(error.localizedDescription)")
    }

    var result = [CollectedModule]()
    result.reserveCapacity(expectedPaths.count)
    for path in expectedPaths.sorted() {
      guard let entry = entries[path], let moment = moments[path] else {
        throw ActivationStatisticsError.invalidStatistics(
          "internal result is missing \(path)")
      }
      let values = moment.asArray(Float.self)
      guard values.count == entry.inputWidth else {
        throw ActivationStatisticsError.invalidStatistics(
          "\(path) produced \(values.count) values; expected \(entry.inputWidth)")
      }
      guard values.allSatisfy({ $0.isFinite && $0 >= 0 }),
        values.contains(where: { $0 > 0 })
      else {
        throw ActivationStatisticsError.invalidStatistics(
          "\(path) contains non-finite, negative, or entirely zero second moments")
      }
      guard let minimum = values.min(), let maximum = values.max() else {
        throw ActivationStatisticsError.invalidStatistics("\(path) is empty")
      }
      let mean = values.reduce(0.0) { $0 + Double($1) } / Double(values.count)
      guard mean.isFinite, mean > 0 else {
        throw ActivationStatisticsError.invalidStatistics(
          "\(path) has an invalid mean second moment")
      }
      result.append(
        CollectedModule(
          path: path,
          tensorKey: path + statisticSuffix,
          inputWidth: entry.inputWidth,
          observedPositionCount: entry.positionCount,
          values: values,
          minimum: minimum,
          maximum: maximum,
          mean: mean
        ))
    }
    return result
  }

  private func throwRecordedFailure() throws {
    if let firstFailure {
      throw ActivationStatisticsError.collectionFailed(firstFailure)
    }
  }
}

private final class ActivationRecordingLinear: Linear {
  private let recordingPath: String
  private let recorder: ActivationRecorder

  init(path: String, linear: Linear, recorder: ActivationRecorder) {
    self.recordingPath = path
    self.recorder = recorder
    super.init(weight: linear.weight, bias: linear.bias)
    train(linear.training)
  }

  override func callAsFunction(_ x: MLXArray) -> MLXArray {
    recorder.observe(path: recordingPath, input: x)
    return super.callAsFunction(x)
  }
}

@main
private struct MistralActivationStats: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "model-runner-mistral-activation-stats",
    abstract:
      "Collect deterministic BF16 Mistral per-Linear input-channel second moments for activation-weighted Q4 ScaleSearch."
  )

  @Argument(help: "Unquantized BF16 Mistral/Ministral MLX checkpoint directory.")
  var model: String

  @Argument(help: "Deterministic plain-text JSONL calibration corpus.")
  var corpus: String

  @Argument(help: "New .safetensors statistics output path.")
  var output: String

  @Option(
    name: .customLong("segment-tokens"),
    help: "Maximum tokens per independent contiguous model segment (1...2048)."
  )
  var segmentTokens = 512

  @Option(
    name: .customLong("maximum-total-tokens"),
    help: "Deterministic corpus-prefix token cap; zero means all tokens."
  )
  var maximumTotalTokens = 0

  @Flag(help: "Run on CPU instead of the default MLX device.")
  var cpu = false

  @Flag(help: "Atomically replace existing statistics and report files.")
  var overwrite = false

  mutating func validate() throws {
    guard (1...2_048).contains(segmentTokens) else {
      throw ValidationError("--segment-tokens must be in 1...2048.")
    }
    guard maximumTotalTokens >= 0 else {
      throw ValidationError("--maximum-total-tokens must be nonnegative.")
    }
  }

  mutating func run() async throws {
    let modelURL = localURL(model, isDirectory: true)
    let corpusURL = localURL(corpus)
    let outputURL = localURL(output)
    let reportURL = outputURL.deletingPathExtension().appendingPathExtension("json")
    try validateInputs(
      modelURL: modelURL,
      corpusURL: corpusURL,
      outputURL: outputURL,
      reportURL: reportURL
    )

    let configURL = modelURL.appendingPathComponent("config.json")
    let configData = try Data(contentsOf: configURL)
    let descriptor = try checkpointDescriptor(configData)
    try validateCheckpoint(descriptor, modelURL: modelURL)
    let configFingerprint = fnv1a64Fingerprint(configData)
    let indexURL = modelURL.appendingPathComponent("model.safetensors.index.json")
    let indexFingerprint = try optionalFingerprint(indexURL)
    let samples = try ModelQualityCore.loadCorpus(from: corpusURL)
    let corpusFingerprint = ModelQualityCore.corpusFingerprint(samples)
    let expectedPaths = expectedProjectionPaths(descriptor)
    let payload = CollectionPayload(
      samples: samples,
      segmentTokenLimit: segmentTokens,
      maximumTotalTokens: maximumTotalTokens,
      expectedModulePaths: expectedPaths
    )

    let resourceLimits = try MLXResourceLimits.resolve(
      for: cpu ? .cpu : collectionEngine,
      physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory
    )
    let startedAt = ContinuousClock.now

    let collect: @Sendable () async throws -> CollectedStatistics = {
      Memory.peakMemory = 0
      try MLXResourceGuard.apply(resourceLimits)
      let container = try await #huggingFaceLoadModelContainer(
        configuration: ModelConfiguration(directory: modelURL)
      )
      return try await container.perform(values: payload) { context, payload in
        context.model.train(false)
        let leaves = context.model.leafModules().flattened()
        let linearLeaves = leaves.compactMap { path, module -> (String, Linear)? in
          guard let linear = module as? Linear else { return nil }
          return (path, linear)
        }
        let actualPaths = Set(linearLeaves.map(\.0))
        let missing = payload.expectedModulePaths.subtracting(actualPaths)
        let unexpected = actualPaths.subtracting(payload.expectedModulePaths)
        guard missing.isEmpty, unexpected.isEmpty else {
          throw ActivationStatisticsError.invalidModelGraph(
            "Linear path mismatch; missing [\(missing.sorted().joined(separator: ", "))], "
              + "unexpected [\(unexpected.sorted().joined(separator: ", "))]")
        }

        var expectedWidths = [String: Int]()
        var weightDTypes = Set<DType>()
        var replacements = [(String, Module)]()
        replacements.reserveCapacity(linearLeaves.count)
        for (path, linear) in linearLeaves.sorted(by: { $0.0 < $1.0 }) {
          guard !(linear is Quantized) else {
            throw ActivationStatisticsError.invalidModelGraph(
              "\(path) is already quantized; collect statistics from the BF16 teacher")
          }
          guard linear.weight.ndim == 2, linear.weight.dim(-1) > 0 else {
            throw ActivationStatisticsError.invalidModelGraph(
              "\(path) has invalid weight shape \(linear.weight.shape)")
          }
          guard linear.weight.dtype == .bfloat16 else {
            throw ActivationStatisticsError.invalidModelGraph(
              "\(path) uses \(linear.weight.dtype), but this collector requires BF16 teacher weights"
            )
          }
          expectedWidths[path] = linear.weight.dim(-1)
          weightDTypes.insert(linear.weight.dtype)
        }
        guard weightDTypes == Set([DType.bfloat16]) else {
          throw ActivationStatisticsError.invalidModelGraph(
            "expected one BF16 source dtype, found \(weightDTypes)")
        }

        let recorder = ActivationRecorder(expectedWidths: expectedWidths)
        for (path, linear) in linearLeaves.sorted(by: { $0.0 < $1.0 }) {
          replacements.append(
            (path, ActivationRecordingLinear(path: path, linear: linear, recorder: recorder)))
        }
        try context.model.update(
          modules: ModuleChildren.unflattened(replacements),
          verify: [.noUnusedKeys]
        )
        context.model.train(false)

        var observedTokens = 0
        var observedSegments = 0
        var sampleReports = [CollectedSample]()
        var tokenSequences = [ModelQualityTokenSequence]()
        sampleReports.reserveCapacity(payload.samples.count)

        for (sampleIndex, sample) in payload.samples.enumerated() {
          let encoded = context.tokenizer.encode(text: sample.text, addSpecialTokens: true)
          guard !encoded.isEmpty else {
            throw ActivationStatisticsError.collectionFailed(
              "sample '\(sample.id)' encoded to no tokens")
          }
          let remaining =
            payload.maximumTotalTokens == 0
            ? encoded.count
            : max(0, payload.maximumTotalTokens - observedTokens)
          if remaining == 0 { break }
          let selected = Array(encoded.prefix(remaining))
          var sampleSegments = 0

          for offset in stride(from: 0, to: selected.count, by: payload.segmentTokenLimit) {
            let end = min(offset + payload.segmentTokenLimit, selected.count)
            let segment = Array(selected[offset..<end])
            let inputs = MLXArray(segment).reshaped(1, segment.count)
            _ = context.model(inputs, cache: nil)
            try recorder.evaluatePending()
            Memory.clearCache()

            let segmentID = "\(sample.id)#\(sampleSegments)"
            tokenSequences.append(
              ModelQualityTokenSequence(sampleID: segmentID, tokenIDs: segment))
            sampleSegments += 1
            observedSegments += 1
            let (nextObservedTokens, overflow) = observedTokens.addingReportingOverflow(
              segment.count)
            guard !overflow else {
              throw ActivationStatisticsError.collectionFailed(
                "observed token count overflow")
            }
            observedTokens = nextObservedTokens
          }

          sampleReports.append(
            CollectedSample(
              id: sample.id,
              category: sample.category,
              originalTokenCount: encoded.count,
              observedTokenCount: selected.count,
              segmentCount: sampleSegments,
              truncated: selected.count < encoded.count,
              tokenIDFingerprint: ModelQualityCore.tokenIDFingerprint(selected)
            ))
          print(
            "sample \(sampleIndex + 1)/\(payload.samples.count) \(sample.id): "
              + "\(selected.count) tokens in \(sampleSegments) segment(s)"
          )
          if payload.maximumTotalTokens > 0,
            observedTokens >= payload.maximumTotalTokens
          {
            break
          }
        }

        guard observedTokens > 0, observedSegments > 0 else {
          throw ActivationStatisticsError.collectionFailed(
            "the configured corpus prefix contains no tokens")
        }
        let modules = try recorder.finalize(expectedPaths: payload.expectedModulePaths)
        let moduleCounts = Set(modules.map(\.observedPositionCount))
        guard moduleCounts == Set([observedTokens]) else {
          throw ActivationStatisticsError.invalidStatistics(
            "dense Mistral modules observed inconsistent position counts \(moduleCounts.sorted()); "
              + "expected \(observedTokens)")
        }

        return CollectedStatistics(
          device: Device.defaultDevice().deviceType?.rawValue ?? "unknown",
          sourceWeightDType: "bfloat16",
          tokenIDFingerprint: ModelQualityCore.combinedTokenIDFingerprint(tokenSequences),
          observedTokenCount: observedTokens,
          segmentCount: observedSegments,
          modules: modules,
          samples: sampleReports
        )
      }
    }

    let collected: CollectedStatistics
    if cpu {
      collected = try await Device.withDefaultDevice(.cpu, collect)
    } else {
      collected = try await collect()
    }
    let elapsedSeconds = seconds(startedAt.duration(to: .now))
    let peakMemory = Memory.peakMemory

    var arrays = [String: MLXArray]()
    arrays.reserveCapacity(collected.modules.count)
    for module in collected.modules {
      arrays[module.tensorKey] = MLXArray(module.values)
    }
    try MLX.checkedEval(Array(arrays.values))

    var metadata = [
      "format": statisticsFormat,
      "algorithm": "input_channel_second_moment",
      "model_type": descriptor.modelType,
      "source_model": modelURL.path,
      "source_config_fingerprint": configFingerprint,
      "corpus_fingerprint": corpusFingerprint,
      "token_id_fingerprint": collected.tokenIDFingerprint,
      "observed_token_count": String(collected.observedTokenCount),
      "module_count": String(collected.modules.count),
      "segment_token_limit": String(segmentTokens),
      "add_special_tokens": "true",
      "dtype": "float32",
    ]
    if let indexFingerprint {
      metadata["source_index_fingerprint"] = indexFingerprint
    }
    try writeSafetensorsAtomically(arrays, metadata: metadata, to: outputURL)

    let moduleReports = collected.modules.map {
      ModuleReport(
        path: $0.path,
        tensorKey: $0.tensorKey,
        inputWidth: $0.inputWidth,
        observedPositionCount: $0.observedPositionCount,
        minimumSecondMoment: $0.minimum,
        maximumSecondMoment: $0.maximum,
        meanSecondMoment: $0.mean
      )
    }
    let report = ActivationStatisticsReport(
      createdAt: ISO8601DateFormatter().string(from: Date()),
      sourceModel: modelURL.path,
      modelType: descriptor.modelType,
      textModelType: descriptor.textModelType,
      architectures: descriptor.architectures,
      sourceWeightDType: collected.sourceWeightDType,
      sourceConfigFingerprint: configFingerprint,
      sourceIndexFingerprint: indexFingerprint,
      corpusPath: corpusURL.path,
      corpusFingerprint: corpusFingerprint,
      tokenIDFingerprint: collected.tokenIDFingerprint,
      outputPath: outputURL.path,
      backend: backendName,
      device: collected.device,
      segmentTokenLimit: segmentTokens,
      maximumTotalTokens: maximumTotalTokens == 0 ? nil : maximumTotalTokens,
      corpusSampleCount: samples.count,
      observedSampleCount: collected.samples.count,
      observedSegmentCount: collected.segmentCount,
      observedTokenCount: collected.observedTokenCount,
      moduleCount: collected.modules.count,
      mlxPeakMemoryBytes: peakMemory,
      elapsedSeconds: elapsedSeconds,
      modules: moduleReports,
      samples: collected.samples
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(report).write(to: reportURL, options: .atomic)

    print(
      "Collected \(collected.modules.count) module statistics across "
        + "\(collected.observedTokenCount) tokens in \(collected.segmentCount) segment(s)."
    )
    print("Wrote \(outputURL.path)")
    print("Wrote \(reportURL.path)")
  }

  private func validateInputs(
    modelURL: URL,
    corpusURL: URL,
    outputURL: URL,
    reportURL: URL
  ) throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: modelURL.path, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      throw ActivationStatisticsError.invalidInput(
        "model directory does not exist: \(modelURL.path)")
    }
    isDirectory = false
    guard FileManager.default.fileExists(atPath: corpusURL.path, isDirectory: &isDirectory),
      !isDirectory.boolValue
    else {
      throw ActivationStatisticsError.invalidInput(
        "corpus file does not exist: \(corpusURL.path)")
    }
    guard outputURL.pathExtension == "safetensors" else {
      throw ActivationStatisticsError.invalidInput(
        "output must use the .safetensors extension")
    }
    guard outputURL != corpusURL, reportURL != corpusURL else {
      throw ActivationStatisticsError.invalidInput(
        "output and report paths must not replace the corpus")
    }
    let modelPrefix = modelURL.path.hasSuffix("/") ? modelURL.path : modelURL.path + "/"
    guard !outputURL.path.hasPrefix(modelPrefix), !reportURL.path.hasPrefix(modelPrefix) else {
      throw ActivationStatisticsError.invalidInput(
        "statistics must not be written inside the model directory")
    }
    if !overwrite {
      for url in [outputURL, reportURL]
      where FileManager.default.fileExists(atPath: url.path) {
        throw ActivationStatisticsError.invalidInput(
          "output already exists (pass --overwrite to replace it): \(url.path)")
      }
    }
  }
}

private func checkpointDescriptor(_ data: Data) throws -> CheckpointDescriptor {
  guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
    let modelType = json["model_type"] as? String
  else {
    throw ActivationStatisticsError.invalidInput(
      "config.json has no string model_type")
  }
  let text = (json["text_config"] as? [String: Any]) ?? json
  guard let hiddenLayerCount = (text["num_hidden_layers"] as? NSNumber)?.intValue,
    hiddenLayerCount > 0
  else {
    throw ActivationStatisticsError.invalidInput(
      "config.json has no positive num_hidden_layers")
  }
  let textModelType = (text["model_type"] as? String) ?? modelType
  let tiedWordEmbeddings =
    (text["tie_word_embeddings"] as? Bool)
    ?? (json["tie_word_embeddings"] as? Bool)
    ?? false
  let architectures = json["architectures"] as? [String] ?? []
  let isQuantized =
    json["quantization"].map { !($0 is NSNull) } ?? false
    || (json["quantization_config"].map { !($0 is NSNull) } ?? false)
  return CheckpointDescriptor(
    modelType: modelType,
    textModelType: textModelType,
    hiddenLayerCount: hiddenLayerCount,
    tiedWordEmbeddings: tiedWordEmbeddings,
    architectures: architectures,
    isQuantized: isQuantized
  )
}

private func validateCheckpoint(
  _ descriptor: CheckpointDescriptor,
  modelURL: URL
) throws {
  let supportedTypes = Set(["mistral3", "ministral3"])
  guard supportedTypes.contains(descriptor.modelType),
    supportedTypes.contains(descriptor.textModelType)
  else {
    throw ActivationStatisticsError.unsupportedCheckpoint(
      "expected Mistral3/Ministral3, found top-level '\(descriptor.modelType)' and text '\(descriptor.textModelType)'"
    )
  }
  guard !descriptor.isQuantized else {
    throw ActivationStatisticsError.unsupportedCheckpoint(
      "\(modelURL.path) declares quantization; use its unquantized BF16 teacher")
  }
}

private func expectedProjectionPaths(_ descriptor: CheckpointDescriptor) -> Set<String> {
  var result = Set<String>()
  for layer in 0..<descriptor.hiddenLayerCount {
    let prefix = "model.layers.\(layer)"
    for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
      result.insert("\(prefix).self_attn.\(projection)")
    }
    for projection in ["gate_proj", "up_proj", "down_proj"] {
      result.insert("\(prefix).mlp.\(projection)")
    }
  }
  if !descriptor.tiedWordEmbeddings {
    result.insert("lm_head")
  }
  return result
}

private func writeSafetensorsAtomically(
  _ arrays: [String: MLXArray],
  metadata: [String: String],
  to outputURL: URL
) throws {
  let fileManager = FileManager.default
  let parent = outputURL.deletingLastPathComponent()
  try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
  let temporaryURL = parent.appendingPathComponent(
    ".\(outputURL.lastPathComponent).\(UUID().uuidString).partial.safetensors")
  defer { try? fileManager.removeItem(at: temporaryURL) }
  try MLX.save(arrays: arrays, metadata: metadata, url: temporaryURL)
  if fileManager.fileExists(atPath: outputURL.path) {
    _ = try fileManager.replaceItemAt(outputURL, withItemAt: temporaryURL)
  } else {
    try fileManager.moveItem(at: temporaryURL, to: outputURL)
  }
}

private func optionalFingerprint(_ url: URL) throws -> String? {
  guard FileManager.default.fileExists(atPath: url.path) else { return nil }
  return fnv1a64Fingerprint(try Data(contentsOf: url))
}

private func fnv1a64Fingerprint(_ data: Data) -> String {
  var value: UInt64 = 0xcbf2_9ce4_8422_2325
  for byte in data {
    value ^= UInt64(byte)
    value &*= 0x100_0000_01b3
  }
  let hex = String(value, radix: 16)
  return "fnv1a64:" + String(repeating: "0", count: 16 - hex.count) + hex
}

private func localURL(_ path: String, isDirectory: Bool = false) -> URL {
  let expanded = NSString(string: path).expandingTildeInPath
  return URL(fileURLWithPath: expanded, isDirectory: isDirectory).standardizedFileURL
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

private var collectionEngine: ModelEngine {
  #if MLX_METAL_BACKEND
    .metal
  #elseif MLX_CUDA_BACKEND
    .cuda
  #else
    .cpu
  #endif
}
