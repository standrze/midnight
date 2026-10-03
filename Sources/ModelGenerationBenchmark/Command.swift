import ArgumentParser
import Foundation
import MLX
import ModelQualityCore
import ModelRunnerCore
import ModelRunnerProtocol

@main
struct ModelGenerationBenchmark: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "model-runner-generation-bench",
    abstract: "Generate independent greedy responses to a public JSONL corpus with one model load.",
    discussion: "Records require id, category and prompt; optional metadata is ignored. Never supply answers or tests. Each prompt is one user turn with the checkpoint's chat template and no added system text. No prompt truncation or cache reuse is allowed. Token-limit stops are recorded explicitly. Reports are atomically saved after every sample; errors stop the run and preserve completed results. Existing output paths are refused. Timings are diagnostics, not a controlled speed comparison."
  )

  @Argument(help: "Local MLX checkpoint directory.") var model: String
  @Argument(help: "Public JSONL input file, with no answers or tests.") var corpus: String
  @Argument(help: "New JSON report path; existing files are never resumed or replaced.") var output: String
  @Option(help: "Execution engine: auto, metal, cuda, or cpu.") var engine = "auto"
  @Option(help: "Maximum output tokens per sample (1...32768); EOS may stop earlier.") var tokens = 512
  @Option(help: "Prompt plus reserved output context ceiling; cannot exceed model metadata.") var contextLength: Int?
  @Option(help: "Prefill chunk size (1...8192).") var prefillStepSize = 512
  @Option(help: "KV compression: none, affine8, affine4, or turbo8v4.") var kvCompression = "none"
  @Flag(help: "Validate public corpus/settings and write a validation report without loading a model.") var validateOnly = false
  @Flag(help: "Load once and prepare/admit all prompts without generating; record exact rendered token counts.") var prepareOnly = false

  mutating func validate() throws {
    guard (1...32_768).contains(tokens) else { throw ValidationError("--tokens must be in 1...32768") }
    guard !validateOnly || !prepareOnly else { throw ValidationError("choose only one of --validate-only and --prepare-only") }
    _ = try ModelEngine(argument: engine)
    _ = try LongContextOptions(contextLength: contextLength, prefillStepSize: prefillStepSize, kvCompression: kvCompression)
  }

  mutating func run() async throws {
    defer { clearModelRunnerMLXStreams() }
    let clock = ContinuousClock()
    let startedAt = clock.now
    let modelURL = generationURL(model)
    let corpusURL = generationURL(corpus)
    let outputURL = generationURL(output)
    guard !FileManager.default.fileExists(atPath: outputURL.path) else {
      throw GenerationBenchmarkError.invalidInput("output already exists: \(outputURL.path)")
    }
    try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    let environment = ProcessInfo.processInfo.environment
    let recordedEnvironmentKeys = [
      "MODEL_RUNNER_MLX_MEMORY_LIMIT_GIB", "MODEL_RUNNER_HOST_RESERVE_GIB",
      "MODEL_RUNNER_MLX_CACHE_LIMIT_MIB", "MODEL_RUNNER_LAGUNA_FUSED_GATHER_SILU",
      "MODEL_RUNNER_WIRED_MEMORY", "MODEL_RUNNER_WIRED_TUNING_TOKENS",
      "MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER",
    ]
    let now = ISO8601DateFormatter().string(from: Date())
    var report = GenerationReport(
      createdAt: now, updatedAt: now, modelPath: modelURL.path, corpusPath: corpusURL.path,
      requestedEngine: engine, requestedTokens: tokens, requestedContextLength: contextLength,
      contextLength: contextLength, prefillStepSize: prefillStepSize, kvCompression: kvCompression,
      invocation: CommandLine.arguments,
      runtimeEnvironment: Dictionary(uniqueKeysWithValues: recordedEnvironmentKeys.compactMap { key in
        environment[key].map { (key, $0) }
      }))
    try writeGenerationReport(&report, to: outputURL, startedAt: startedAt, initial: true)
    var stage = "corpus_validation"
    do {
      let data = try Data(contentsOf: corpusURL)
      report.corpusBytes = data.count
      report.corpusSha256 = generationSHA256(data)
      let inputs = try readGenerationInputs(data)
      report.inputSampleCount = inputs.count
      report.corpusFingerprint = generationCorpusFingerprint(inputs)
      if validateOnly {
        report.samples = inputs.enumerated().map { index, input in
          var record = GenerationSample(sequence: index, id: input.id, category: input.category)
          record.status = "validated"
          record.promptSha256 = generationSHA256(Data(input.prompt.utf8))
          return record
        }
        report.status = "validated"
        try writeGenerationReport(&report, to: outputURL, startedAt: startedAt)
        print("Validated \(inputs.count) public records; model was not loaded. Report: \(outputURL.path)")
        return
      }
      stage = "model_load"
      report.status = "loading"
      report.modelConfigSha256 = generationSHA256(try Data(contentsOf: modelURL.appendingPathComponent("config.json")))
      let indexURL = modelURL.appendingPathComponent("model.safetensors.index.json")
      if FileManager.default.fileExists(atPath: indexURL.path) {
        report.modelIndexSha256 = generationSHA256(try Data(contentsOf: indexURL))
      }
      try writeGenerationReport(&report, to: outputURL, startedAt: startedAt)
      let loadStartedAt = clock.now
      let runner = try await LocalModelRunner(
        modelPath: modelURL.path, servedModelName: "generation-benchmark",
        engine: try ModelEngine(argument: engine), maximumTokens: tokens,
        longContext: try LongContextOptions(contextLength: contextLength,
          prefillStepSize: prefillStepSize, kvCompression: kvCompression))
      report.modelLoadMilliseconds = generationMilliseconds(loadStartedAt.duration(to: clock.now))
      report.modelLoadCount = 1
      report.engine = runner.engine.rawValue
      report.contextLength = runner.contextLength
      report.prefillStepSize = runner.prefillStepSize
      report.kvCompression = runner.kvCompression
      report.memoryLimitBytes = runner.memoryLimitBytes
      report.compiledLagunaBlockTail = runner.engine == .metal
      report.fusedLagunaRouterTopK = runner.engine == .metal
      report.status = prepareOnly ? "preparing" : "running"
      for (index, input) in inputs.enumerated() {
        stage = prepareOnly ? "prompt_preparation" : "generation"
        report.activeSampleId = input.id
        try writeGenerationReport(&report, to: outputURL, startedAt: startedAt)
        let record = await runSample(input, sequence: index, runner: runner)
        report.samples.append(record)
        if record.status == "error" {
          stage = record.failureStage ?? stage
          throw GenerationBenchmarkError.sampleFailed(input.id, record.error ?? "unknown error")
        }
        report.activeSampleId = nil
        try writeGenerationReport(&report, to: outputURL, startedAt: startedAt)
        print("[\(index + 1)/\(inputs.count)] \(input.id): prompt=\(record.promptTokenCount ?? 0), generated=\(record.generationTokenCount ?? 0), stop=\(record.stopReason ?? record.status)")
      }
      report.status = prepareOnly ? "prepared" : "completed"
      try writeGenerationReport(&report, to: outputURL, startedAt: startedAt)
      print("Saved \(report.status) report: \(outputURL.path)")
    } catch {
      report.status = "failed"
      report.failureStatus = stage
      report.error = error.localizedDescription
      do { try writeGenerationReport(&report, to: outputURL, startedAt: startedAt) }
      catch {
        FileHandle.standardError.write(Data("Could not save failure snapshot: \(error.localizedDescription)\n".utf8))
      }
      throw error
    }
  }

  private func runSample(
    _ input: GenerationInput, sequence: Int, runner: LocalModelRunner
  ) async -> GenerationSample {
    let clock = ContinuousClock()
    let startedAt = clock.now
    Memory.peakMemory = 0
    var record = GenerationSample(sequence: sequence, id: input.id, category: input.category)
    record.promptSha256 = generationSHA256(Data(input.prompt.utf8))
    var stage = "prompt_preparation"
    do {
      try Task.checkCancellation()
      let messages = [OpenAIMessage(role: "user", content: input.prompt)]
      let prepared = try await runner.preparePrompt(messages: messages, maximumTokens: tokens)
      record.promptPreparationMilliseconds = generationMilliseconds(startedAt.duration(to: clock.now))
      record.promptTokenCount = prepared.promptTokenCount
      record.promptTokenIdFingerprint = ModelQualityCore.tokenIDFingerprint(prepared.promptTokenIDs)
      guard prepared.promptTokenCount > 0 else {
        throw GenerationBenchmarkError.invalidResult("rendered prompt is empty")
      }
      if prepareOnly {
        record.status = "prepared"
        record.totalMilliseconds = generationMilliseconds(startedAt.duration(to: clock.now))
        record.peakMlxMemoryBytes = Memory.peakMemory
        return record
      }
      stage = "generation"
      // Pin the existing production defaults and disable the opt-in experiment,
      // independent of any ambient MODEL_RUNNER_LAGUNA_FUSED_GATHER_SILU value.
      let events = await LagunaRuntimeTuning.$useCompiledMoEFusion.withValue(true) {
        await LagunaRuntimeTuning.$useCompiledAttentionGate.withValue(true) {
          await LagunaRuntimeTuning.$useCompiledBlockTail.withValue(nil) {
            await LagunaRuntimeTuning.$useFusedRouterTopK.withValue(nil) {
              await LagunaRuntimeTuning.$useFusedGateUpSilu.withValue(false) {
                await runner.stream(
                  messages: messages, maximumTokens: tokens, temperature: 0, topP: 1,
                  enablePromptCache: false, enableSpeculativeDecoding: false,
                  preparedPrompt: prepared)
              }
            }
          }
        }
      }
      var metrics: LocalModelRunnerMetrics?
      for try await event in events {
        try Task.checkCancellation()
        switch event {
        case .content(let text):
          if !text.isEmpty && record.timeToFirstContentMilliseconds == nil {
            record.timeToFirstContentMilliseconds = generationMilliseconds(startedAt.duration(to: clock.now))
          }
          record.generatedText += text
        case .metrics(let value):
          guard metrics == nil else { throw GenerationBenchmarkError.invalidResult("multiple final metrics events") }
          metrics = value
        case .toolCall:
          throw GenerationBenchmarkError.invalidResult("unexpected tool call; this benchmark requests text only")
        }
      }
      guard let metrics else { throw GenerationBenchmarkError.invalidResult("stream ended without final metrics") }
      record.generationTokenCount = metrics.generationTokenCount
      record.stopReason = metrics.stopReason
      record.prefilledPromptTokenCount = metrics.prefilledPromptTokenCount
      record.cachedPromptTokenCount = metrics.cachedPromptTokenCount
      // Keep failure reports JSON-encodable even if malformed runtime timing
      // accompanies another accounting failure detected below.
      record.promptTokensPerSecond = metrics.promptTokensPerSecond.isFinite ? metrics.promptTokensPerSecond : nil
      record.tokensPerSecond = metrics.tokensPerSecond.isFinite ? metrics.tokensPerSecond : nil
      record.outputLimitReached = metrics.stopReason == "length" || metrics.generationTokenCount == tokens
      guard metrics.promptTokenCount == prepared.promptTokenCount,
        metrics.prefilledPromptTokenCount == prepared.promptTokenCount,
        metrics.cachedPromptTokenCount == 0 else {
        throw GenerationBenchmarkError.invalidResult("prompt accounting does not match the complete independently prepared prompt")
      }
      guard (0...tokens).contains(metrics.generationTokenCount),
        ["stop", "length"].contains(metrics.stopReason),
        metrics.stopReason != "length" || metrics.generationTokenCount == tokens else {
        throw GenerationBenchmarkError.invalidResult("unexpected stop reason or output-token count")
      }
      guard metrics.promptTokensPerSecond.isFinite, metrics.promptTokensPerSecond >= 0,
        metrics.tokensPerSecond.isFinite, metrics.tokensPerSecond >= 0 else {
        record.promptTokensPerSecond = nil
        record.tokensPerSecond = nil
        throw GenerationBenchmarkError.invalidResult("non-finite or negative timing metrics")
      }
      record.status = "completed"
    } catch {
      record.status = "error"
      record.failureStage = stage
      record.error = error.localizedDescription
    }
    record.totalMilliseconds = generationMilliseconds(startedAt.duration(to: clock.now))
    record.peakMlxMemoryBytes = Memory.peakMemory
    return record
  }
}
