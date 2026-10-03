import ArgumentParser
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
@_spi(GemmaCompiledTailTesting) @_spi(GemmaEncoder) import MLXLLM
import MLXLMCommon
import MLXNN
import ModelQualityCore
import ModelRunnerCore
import ModelRunnerProtocol
import Tokenizers

private struct ScoringPayload: Sendable {
    var samples: [ModelQualityCorpusSample]
    var maximumTokensPerSample: Int
    var prefillStepSize: Int
    var tokenDiagnostics: Bool
}

private struct ScoredCorpus: Sendable {
    var samples: [ModelQualitySampleResult]
    var tokenIDFingerprint: String
    var modelImplementation: String
    var cacheImplementations: [String]
    var gemma3CompiledTailTraceCounts: [Int]?
    var tokenDiagnostics: [SampleTokenDiagnostics]?
}

private struct SampleTokenDiagnostics: Encodable, Sendable {
    var id: String
    var rows: [ModelQualityTokenDiagnostic]
}

private struct QualityBenchmarkReport: Encodable, Sendable {
    var format = 1
    var status = "measured"
    var metric = "teacher_forced_next_token_nll"
    var createdAt: String
    var modelPath: String
    var modelType: String?
    var modelImplementation: String
    var cacheImplementations: [String]
    var gemma3CompiledTailTraceCounts: [Int]?
    var gemma3CompiledTailTraceCount: Int?
    var corpusPath: String
    var corpusFingerprint: String
    var tokenIDFingerprint: String
    var backend: String
    var device: String
    var addSpecialTokens = true
    var tokenization = "raw_text_add_special_tokens_no_chat_template"
    var runtimeEnvironment: [String: String]
    var maximumTokensPerSample: Int
    var prefillStepSize: Int
    var sampleCount: Int
    var scoredTokenCount: Int
    var nllSum: Double
    var tokenWeightedNLL: Double
    var perplexity: Double
    var mlxPeakMemoryBytes: Int
    var elapsedSeconds: Double
    var samples: [ModelQualitySampleResult]
    var tokenDiagnostics: [SampleTokenDiagnostics]?

    enum CodingKeys: String, CodingKey {
        case format, status, metric, backend, device, perplexity, samples, tokenization
        case createdAt = "created_at"
        case modelPath = "model_path"
        case modelType = "model_type"
        case modelImplementation = "model_implementation"
        case cacheImplementations = "cache_implementations"
        case gemma3CompiledTailTraceCounts = "gemma3_compiled_tail_trace_counts"
        case gemma3CompiledTailTraceCount = "gemma3_compiled_tail_trace_count"
        case runtimeEnvironment = "runtime_environment"
        case tokenDiagnostics = "token_diagnostics"
        case corpusPath = "corpus_path"
        case corpusFingerprint = "corpus_fingerprint"
        case tokenIDFingerprint = "token_id_fingerprint"
        case addSpecialTokens = "add_special_tokens"
        case maximumTokensPerSample = "maximum_tokens_per_sample"
        case prefillStepSize = "prefill_step_size"
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

    @Option(help: "Maximum encoded tokens per sample (2...2048, or up to 32768 with positive --prefill-step-size).")
    var maxTokensPerSample = 512

    @Option(help: "Scoring chunk size (1...8192); 0 keeps all-at-once scoring without a KV cache.")
    var prefillStepSize = 0

    @Flag(help: "Record each fixed-prefix winner, runner-up and margin; requires --prefill-step-size 1.")
    var tokenDiagnostics = false

    @Flag(help: "Run on CPU instead of the default MLX device.")
    var cpu = false

    @Flag(help: "Replace an existing report file.")
    var overwrite = false

    mutating func validate() throws {
        guard (0...8_192).contains(prefillStepSize) else {
            throw ValidationError("--prefill-step-size must be in 0...8192.")
        }
        let maximum = prefillStepSize == 0 ? 2_048 : 32_768
        guard (2...maximum).contains(maxTokensPerSample) else {
            throw ValidationError("--max-tokens-per-sample must be in 2...\(maximum) for this prefill setting.")
        }
        guard !tokenDiagnostics || prefillStepSize == 1 else {
            throw ValidationError("--token-diagnostics requires --prefill-step-size 1.")
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
            maximumTokensPerSample: maxTokensPerSample,
            prefillStepSize: prefillStepSize, tokenDiagnostics: tokenDiagnostics
        )
        let resourceLimits = try MLXResourceGuard.resolve(
            for: cpu ? .cpu : benchmarkEngine,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory
        )
        let startedAt = ContinuousClock.now

        let scored = try await score(modelURL: modelURL, payload: payload, resourceLimits: resourceLimits)
        let peakMemory = Memory.peakMemory
        let elapsedSeconds = seconds(startedAt.duration(to: .now))
        let (report, summary) = try makeReport(
            modelURL: modelURL, corpusURL: corpusURL, samples: samples, scored: scored,
            peakMemory: peakMemory, elapsedSeconds: elapsedSeconds)
        try writeReport(report, summary: summary, to: outputURL)
    }

    private func score(
        modelURL: URL, payload: ScoringPayload, resourceLimits: MLXResourceLimits
    ) async throws -> ScoredCorpus {
        let runBenchmark: @Sendable () async throws -> ScoredCorpus = {
            Memory.peakMemory = 0
            await LagunaModelRegistration.register()
            await TalkieModelRegistration.register()
            // The automatic registry tries VLM before LLM. Wrapped Gemma configs
            // must use the same text implementation as the production runner.
            let container: ModelContainer
            if checkpointModelType(modelURL) == "muse_glimmer" {
                try MLXResourceGuard.apply(resourceLimits)
                container = try await #huggingFaceLoadModelContainer(
                    configuration: ModelConfiguration(directory: modelURL))
            } else {
                try MLXResourceGuard.apply(resourceLimits)
                let boundedGemmaCache =
                    !cpu && benchmarkEngine == .metal
                    && ProcessInfo.processInfo.environment["MIDNIGHT_GEMMA4_BOUNDED_KV"] == "1"
                container = try await Gemma4RuntimeTuning.$useBoundedWindowCache.withValue(boundedGemmaCache) {
                    try await LLMModelFactory.shared.loadContainer(
                        from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                        configuration: ModelConfiguration(directory: modelURL))
                }
            }
            return try await container.perform(values: payload) { context, payload in
                context.model.train(false)
                let cacheImplementations = try Array(
                    Set(
                        context.model.newCache(parameters: nil).map {
                            String(reflecting: type(of: $0))
                        })
                ).sorted()
                var measurements = [ModelQualitySampleResult]()
                var tokenSequences = [ModelQualityTokenSequence]()
                var diagnostics: [SampleTokenDiagnostics]? = payload.tokenDiagnostics ? [] : nil
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

                    let score = try ModelQualityScoring.scoreNLL(
                        tokens: evaluatedTokens, model: context.model,
                        prefillStepSize: payload.prefillStepSize, tokenDiagnostics: payload.tokenDiagnostics)
                    if let rows = score.tokenDiagnostics {
                        diagnostics?.append(SampleTokenDiagnostics(id: sample.id, rows: rows))
                    }
                    let fingerprint = ModelQualityCore.tokenIDFingerprint(evaluatedTokens)
                    let measurement = try ModelQualitySampleResult(
                        id: sample.id,
                        category: sample.category,
                        originalTokenCount: originalTokens.count,
                        evaluatedTokenCount: evaluatedTokens.count,
                        tokenIDFingerprint: fingerprint,
                        nllSum: score.nllSum
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

                // Read CPU-only trace counters after scoring, while the loaded
                // model is still alive. A startup flag alone does not prove that
                // the shape/type/inference gates selected the compiled path.
                let gemma3TraceCounts = (context.model as? Gemma3TextModel)?.model.layers
                    .map(\.compiledTailTraceCount)
                return ScoredCorpus(
                    samples: measurements,
                    tokenIDFingerprint: ModelQualityCore.combinedTokenIDFingerprint(tokenSequences),
                    modelImplementation: String(reflecting: type(of: context.model)),
                    cacheImplementations: cacheImplementations,
                    gemma3CompiledTailTraceCounts: gemma3TraceCounts,
                    tokenDiagnostics: diagnostics
                )
            }
        }

        let scored: ScoredCorpus
        if cpu {
            scored = try await Device.withDefaultDevice(.cpu, runBenchmark)
        } else {
            scored = try await runBenchmark()
        }
        return scored
    }

    private func makeReport(
        modelURL: URL, corpusURL: URL, samples: [ModelQualityCorpusSample],
        scored: ScoredCorpus, peakMemory: Int, elapsedSeconds: Double
    ) throws -> (QualityBenchmarkReport, ModelQualitySummary) {
        let summary = try ModelQualityCore.summarize(scored.samples)
        let report = QualityBenchmarkReport(
            createdAt: ISO8601DateFormatter().string(from: Date()),
            modelPath: modelURL.path,
            modelType: checkpointModelType(modelURL),
            modelImplementation: scored.modelImplementation,
            cacheImplementations: scored.cacheImplementations,
            gemma3CompiledTailTraceCounts: scored.gemma3CompiledTailTraceCounts,
            gemma3CompiledTailTraceCount: scored.gemma3CompiledTailTraceCounts?.reduce(0, +),
            corpusPath: corpusURL.path,
            corpusFingerprint: ModelQualityCore.corpusFingerprint(samples),
            tokenIDFingerprint: scored.tokenIDFingerprint,
            backend: backendName,
            device: cpu ? "cpu" : (Device.defaultDevice().deviceType?.rawValue ?? "unknown"),
            runtimeEnvironment: ProcessInfo.processInfo.environment.filter {
                $0.key.hasPrefix("MLX_METAL_") || $0.key.hasPrefix("MIDNIGHT_METAL_")
                    || $0.key.hasPrefix("MIDNIGHT_GEMMA4_") || $0.key.hasPrefix("MIDNIGHT_GEMMA3_")
            },
            maximumTokensPerSample: maxTokensPerSample,
            prefillStepSize: prefillStepSize,
            sampleCount: summary.sampleCount,
            scoredTokenCount: summary.scoredTokenCount,
            nllSum: summary.nllSum,
            tokenWeightedNLL: summary.tokenWeightedNLL,
            perplexity: summary.perplexity,
            mlxPeakMemoryBytes: peakMemory,
            elapsedSeconds: elapsedSeconds,
            samples: scored.samples, tokenDiagnostics: scored.tokenDiagnostics
        )

        return (report, summary)
    }

    private func writeReport(
        _ report: QualityBenchmarkReport, summary: ModelQualitySummary, to outputURL: URL
    ) throws {
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
