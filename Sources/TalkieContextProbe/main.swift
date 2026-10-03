import ArgumentParser
import CorpusPreparationCore
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

private struct ProbeFailure: Error, LocalizedError {
    let detail: String
    init(_ detail: String) { self.detail = detail }
    var errorDescription: String? { detail }
}

private struct CacheMeasurement: Codable, Sendable {
    var processedTokens: Int
    var allocatedTokens: Int
    var layerCount: Int
    var storageBytes: Int
    var dtypes: [String]
    var floatingPointBytes: Int
}

private struct Checkpoint: Codable, Sendable {
    var phase: String
    var elapsedSeconds: Double
    var cache: CacheMeasurement
    var activeMemoryBytes: Int
    var peakMemoryBytes: Int
    var estimatedNextPeakBytes: Int
    var scoredTokens: Int
    var nllSum: Double?
    var intervalScoredTokens: Int
    var intervalNllSum: Double?
}

private struct ProbeReport: Codable, Sendable {
    var format = 2
    var status = "running"
    var limitation =
        "Experimental capacity/quality diagnostic; successful execution does not validate an extended context window. No RoPE scaling is applied."
    var createdAt = ISO8601DateFormatter().string(from: Date())
    var modelPath: String
    var modelConfigSHA256: String
    var invocation: [String]
    var workingDirectory: String
    var engine: String
    var device: String
    var deviceArchitecture: String?
    var promptSource: String
    var promptID: String?
    var inputMode: String
    var declaredContext: Int
    var extrapolatesDeclaredContext = false
    var actuallyProcessedBeyondDeclaredContext = false
    var ropeTheta: Float
    var kvBits: Int
    var kvGroupSize: Int
    var chunkSize: Int
    var maximumPromptTokens: Int
    var requestedOutputTokens: Int
    var maximumSeconds: Double
    var memoryLimitBytes: Int
    var allocationHeadroomBytes: Int
    var checkpointStoredBytes: Int
    var residentModelBytes = 0
    var promptTokens = 0
    var promptTokenFingerprint = ""
    var checkpoints = [Checkpoint]()
    var outputTokenIDs = [Int]()
    var outputText = ""
    var outputAccounting =
        "Emitted output tokens are recorded separately from evaluated cache positions. The final emitted token is not forwarded when the output limit is reached; stop markers are recorded separately and are not emitted or cached."
    var lastCompletedForwardTokens = 0
    var outputTokensInCache = 0
    var scoredTokenCount = 0
    var nllSum: Double?
    var finalCache: CacheMeasurement?
    var observedCacheOffsets = [Int]()
    var stopReason: String?
    var stopTokenID: Int?
    var elapsedSeconds = 0.0
    var peakMemoryBytes = 0
    var error: String?
}

private struct ProbeSettings: Sendable {
    var configuration: TalkieConfiguration
    var prompt: String
    var reportURL: URL
    var report: ProbeReport
    var rawInput: Bool
    var repeatInput: Bool
    var allowExtrapolation: Bool
    var scoreNLL: Bool
    var checkpointInterval: Int
    var limits: MLXResourceLimits
}

@main
private struct TalkieContextProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "model-runner-talkie-context-probe",
        abstract: "Bounded, experimental Talkie prefill/decode with KV compression from token zero.")

    @Argument(help: "Local Talkie MLX checkpoint directory.") var model: String
    @Argument(help: "Incrementally written JSON report path.") var output: String
    @Option(help: "UTF-8 prompt file; a JSONL file additionally requires --prompt-id.") var promptFile: String?
    @Option(help: "Select this id from a JSONL prompt file; only its prompt field is read.") var promptID: String?
    @Option(help: "Prompt used when --prompt-file is absent.") var prompt =
        "Describe a journey by railway through the countryside."
    @Option(help: "Maximum prompt tokens; ordinary prompts are never truncated (1...262144).") var maxTokens = 2_048
    @Option(help: "Maximum greedy output tokens (0...1024). Prompt plus output must not exceed 262144.")
    var outputTokens = 32
    @Option(help: "Affine KV bits: 2, 4, or 8; 0 selects uncompressed KV for a matched baseline.") var kvBits = 2
    @Option(help: "Affine KV group size: 32, 64, or 128, divisible into the model head dimension.") var kvGroupSize = 64
    @Option(help: "Synchronous prefill chunk size (1...64).") var chunkSize = 32
    @Option(help: "Write/report a checkpoint at least this often in prompt tokens (1...262144).")
    var checkpointInterval = 2_048
    @Option(help: "Monotonic elapsed-time budget after model loading; checked before every forward (1...86400).")
    var maxSeconds = 300.0
    @Option(help: "Unallocated headroom inside the normal MLX memory limit (1...16 GiB).") var headroomGib = 2.0
    @Flag(help: "Encode raw completion text instead of applying the checkpoint's user chat template.") var rawInput =
        false
    @Flag(help: "Repeat raw input tokens to exactly --max-tokens for a synthetic capacity test; requires --raw-input.")
    var repeatInput = false
    @Flag(help: "Explicitly permit this diagnostic to exceed the checkpoint's declared context. Does not change RoPE.")
    var allowExtrapolation = false
    @Flag(help: "Also record teacher-forced next-token NLL over the prompt and each checkpoint interval.")
    var scoreNll = false
    @Flag(help: "Use CPU, retaining the normal CPU allocator limit.") var cpu = false
    @Flag(help: "Replace an existing report file.") var overwrite = false

    mutating func validate() throws {
        guard (1...262_144).contains(maxTokens), (0...1_024).contains(outputTokens),
            [0, 2, 4, 8].contains(kvBits), [32, 64, 128].contains(kvGroupSize),
            (1...64).contains(chunkSize), (1...262_144).contains(checkpointInterval),
            maxSeconds.isFinite, (1...86_400).contains(maxSeconds),
            headroomGib.isFinite, (1...16).contains(headroomGib)
        else {
            throw ValidationError("Invalid token, KV, chunk, time, or memory-headroom bounds; see --help.")
        }
        guard !repeatInput || rawInput else {
            throw ValidationError("--repeat-input requires --raw-input.")
        }
        guard promptID == nil || promptFile != nil else {
            throw ValidationError("--prompt-id requires --prompt-file.")
        }
    }

    mutating func run() async throws {
        let modelURL = localURL(model)
        let reportURL = localURL(output)
        let (configurationData, configuration) = try loadConfiguration(from: modelURL)
        try validateReportDestination(reportURL, modelURL: modelURL)
        let text = try loadPrompt(reportURL: reportURL)
        let (limits, storedBytes, headroom) = try resourceBudget(modelURL: modelURL)
        let report = makeReport(
            modelURL: modelURL, configurationData: configurationData, configuration: configuration,
            limits: limits, storedBytes: storedBytes, headroom: headroom)
        let settings = ProbeSettings(
            configuration: configuration, prompt: text, reportURL: reportURL, report: report,
            rawInput: rawInput, repeatInput: repeatInput, allowExtrapolation: allowExtrapolation,
            scoreNLL: scoreNll, checkpointInterval: checkpointInterval, limits: limits)
        try await execute(
            modelURL: modelURL, reportURL: reportURL, report: report,
            settings: settings, limits: limits)
    }

    private func loadConfiguration(from modelURL: URL) throws -> (Data, TalkieConfiguration) {
        let configurationURL = modelURL.appendingPathComponent("config.json")
        let configurationData = try Data(contentsOf: configurationURL)
        let configuration = try JSONDecoder().decode(TalkieConfiguration.self, from: configurationData)
        let json = try JSONSerialization.jsonObject(with: configurationData) as? [String: Any]
        guard json?["model_type"] as? String == "talkie",
            configuration.headDimension.isMultiple(of: kvGroupSize)
        else {
            throw ProbeFailure(
                "Requires a Talkie checkpoint whose head dimension is divisible by the requested KV group size.")
        }
        return (configurationData, configuration)
    }

    private func validateReportDestination(_ reportURL: URL, modelURL: URL) throws {
        let configurationURL = modelURL.appendingPathComponent("config.json")
        guard reportURL != configurationURL, reportURL != modelURL,
            !reportURL.path.hasPrefix(modelURL.path + "/")
        else {
            throw ProbeFailure("Write the diagnostic report outside the model directory.")
        }
        if FileManager.default.fileExists(atPath: reportURL.path), !overwrite {
            throw ProbeFailure("Report exists; use --overwrite or choose a new path.")
        }
    }

    private func loadPrompt(reportURL: URL) throws -> String {
        var text = prompt
        if let promptFile {
            let inputURL = localURL(promptFile)
            guard inputURL != reportURL else {
                throw ProbeFailure("Report must not replace the prompt file.")
            }
            let data = try Data(contentsOf: inputURL)
            guard data.count <= 32 * 1_048_576, let contents = String(data: data, encoding: .utf8)
            else {
                throw ProbeFailure("Prompt must be UTF-8 and at most 32 MiB.")
            }
            if let promptID {
                struct Input: Decodable {
                    var id: String
                    var prompt: String
                }
                let inputs = try contents.split(whereSeparator: \.isNewline).map {
                    try JSONDecoder().decode(Input.self, from: Data($0.utf8))
                }
                let matches = inputs.filter { $0.id == promptID }
                guard matches.count == 1 else {
                    throw ProbeFailure("Expected exactly one JSONL prompt with id '\(promptID)'.")
                }
                text = matches[0].prompt
            } else {
                guard inputURL.pathExtension != "jsonl" else {
                    throw ProbeFailure("JSONL prompt files require --prompt-id.")
                }
                text = contents
            }
        }
        return text
    }

    private func resourceBudget(modelURL: URL) throws -> (MLXResourceLimits, Int, Int) {
        let limits = try MLXResourceGuard.resolve(
            for: cpu ? .cpu : CompiledMLXBackend.current.engine,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            recommendedWorkingSetBytes: cpu ? nil : GPU.maxRecommendedWorkingSetBytes())
        let shards = try FileManager.default.contentsOfDirectory(
            at: modelURL, includingPropertiesForKeys: [.fileSizeKey]
        )
        .filter { $0.pathExtension == "safetensors" }
        guard !shards.isEmpty else {
            throw ProbeFailure("No safetensors checkpoint files found.")
        }
        let storedBytes = try shards.reduce(0) { $0 + (try $1.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) }
        let headroom = Int(headroomGib * 1_073_741_824)
        guard storedBytes + headroom < limits.memoryLimitBytes else {
            throw ProbeFailure("Checkpoint payload plus reserved headroom exceeds the normal allocator limit.")
        }
        return (limits, storedBytes, headroom)
    }

    private func makeReport(
        modelURL: URL, configurationData: Data, configuration: TalkieConfiguration,
        limits: MLXResourceLimits, storedBytes: Int, headroom: Int
    ) -> ProbeReport {
        var architecture: String?
        #if os(macOS)
            architecture = cpu ? nil : GPU.deviceInfo().architecture
        #endif
        let report = ProbeReport(
            modelPath: modelURL.path, modelConfigSHA256: corpusSHA256(configurationData),
            invocation: ProcessInfo.processInfo.arguments, workingDirectory: FileManager.default.currentDirectoryPath,
            engine: cpu ? "cpu" : CompiledMLXBackend.current.engine.rawValue,
            device: cpu ? "cpu" : String(describing: Device.defaultDevice()), deviceArchitecture: architecture,
            promptSource: promptFile.map { localURL($0).path } ?? "--prompt",
            promptID: promptID,
            inputMode: repeatInput
                ? "synthetic_repeated_raw_tokens" : (rawInput ? "raw_completion" : "checkpoint_user_chat_template"),
            declaredContext: configuration.maxPositionEmbeddings, ropeTheta: configuration.ropeTheta,
            kvBits: kvBits, kvGroupSize: kvGroupSize, chunkSize: chunkSize, maximumPromptTokens: maxTokens,
            requestedOutputTokens: outputTokens, maximumSeconds: maxSeconds, memoryLimitBytes: limits.memoryLimitBytes,
            allocationHeadroomBytes: headroom, checkpointStoredBytes: storedBytes)
        return report
    }

    private func execute(
        modelURL: URL, reportURL: URL, report: ProbeReport,
        settings: ProbeSettings, limits: MLXResourceLimits
    ) async throws {
        try write(report, to: reportURL)
        let work: @Sendable () async throws -> Void = {
            try MLXResourceGuard.apply(limits)
            await TalkieModelRegistration.register()
            let container = try await LLMModelFactory.shared.loadContainer(
                from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                configuration: ModelConfiguration(directory: modelURL))
            try await container.perform(values: settings) { context, settings in
                guard context.model is TalkieModel else {
                    throw ProbeFailure("The loaded model is not the native Talkie implementation.")
                }
                context.model.train(false)
                try runProbe(model: context.model, tokenizer: context.tokenizer, settings: settings)
            }
        }
        do {
            if cpu {
                try await Device.withDefaultDevice(.cpu, work)
            } else {
                try await work()
            }
        } catch {
            // Preserve the most recent incremental measurements, including failures during model loading.
            var failed = (try? JSONDecoder().decode(ProbeReport.self, from: Data(contentsOf: reportURL))) ?? report
            failed.status = "stopped"
            failed.error = error.localizedDescription
            try write(failed, to: reportURL)
            throw error
        }
    }
}

private func runProbe(model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer, settings: ProbeSettings) throws {
    let c = settings.configuration
    var report = settings.report
    let started = ContinuousClock.now
    var processed = 0
    var totalNLL = 0.0
    var scoredTokens = 0
    var checkpointNLL = 0.0
    var checkpointScoredTokens = 0
    var lastCheckpoint = 0
    var scalarBytes = 4  // Conservatively assume FP32 until actual cache dtype is observed.
    var nextToken: Int?
    let cache: [KVCache] = (0..<c.hiddenLayers).map { _ in
        report.kvBits == 0 ? KVCacheSimple() : QuantizedKVCache(groupSize: report.kvGroupSize, bits: report.kvBits)
    }
    func captureFinalState() {
        report.lastCompletedForwardTokens = processed
        report.outputTokensInCache = max(0, processed - report.promptTokens)
        report.scoredTokenCount = scoredTokens
        report.nllSum = settings.scoreNLL ? totalNLL : nil
        report.actuallyProcessedBeyondDeclaredContext = processed > c.maxPositionEmbeddings
        report.observedCacheOffsets = cache.map(\.offset)
        // A failed forward can mutate only part of the cache. Do not claim that
        // mixed offsets represent a complete model state; retain only valid metadata.
        report.finalCache = try? inspect(
            cache, configuration: c, bits: report.kvBits,
            groupSize: report.kvGroupSize, processed: processed)
    }
    do {
        var tokens =
            settings.rawInput
            ? tokenizer.encode(text: settings.prompt, addSpecialTokens: false)
            : try tokenizer.applyChatTemplate(messages: [["role": "user", "content": settings.prompt]])
        guard !tokens.isEmpty else {
            throw ProbeFailure("Prompt encoded to zero tokens.")
        }
        if settings.repeatInput {
            let seed = tokens
            tokens = (0..<report.maximumPromptTokens).map { seed[$0 % seed.count] }
        }
        guard tokens.count <= report.maximumPromptTokens else {
            throw ProbeFailure(
                "Prompt has \(tokens.count) tokens, exceeding --max-tokens \(report.maximumPromptTokens); no truncation was performed."
            )
        }
        let totalRequested = tokens.count + report.requestedOutputTokens
        guard totalRequested <= 262_144 else {
            throw ProbeFailure("Prompt plus requested output exceeds this diagnostic's 262144-token bound.")
        }
        report.promptTokens = tokens.count
        report.promptTokenFingerprint = ModelQualityCore.tokenIDFingerprint(tokens)
        report.extrapolatesDeclaredContext = totalRequested > c.maxPositionEmbeddings
        guard !report.extrapolatesDeclaredContext || settings.allowExtrapolation else {
            throw ProbeFailure(
                "Prompt plus output exceeds declared context \(c.maxPositionEmbeddings); explicitly pass --allow-extrapolation for this diagnostic."
            )
        }
        Memory.clearCache()
        Memory.peakMemory = Memory.activeMemory
        report.residentModelBytes = Memory.activeMemory
        try write(report, to: settings.reportURL)

        func checkBudget(incoming: Int) throws -> Int {
            try Task.checkCancellation()
            guard ProbeBounds.seconds(started.duration(to: .now)) < report.maximumSeconds else {
                throw ProbeFailure("Monotonic elapsed-time budget reached before the next forward.")
            }
            let estimate = nextPeakEstimate(
                configuration: c, bits: report.kvBits, groupSize: report.kvGroupSize,
                scalarBytes: scalarBytes, processed: processed, incoming: incoming,
                activeMemory: Memory.activeMemory, residentModel: report.residentModelBytes)
            guard estimate + report.allocationHeadroomBytes < settings.limits.memoryLimitBytes else {
                throw ProbeFailure(
                    "Next forward estimated at \(estimate) bytes plus \(report.allocationHeadroomBytes) headroom exceeds allocator limit \(settings.limits.memoryLimitBytes)."
                )
            }
            return estimate
        }

        func record(phase: String, estimate: Int) throws {
            let measured = try inspect(
                cache, configuration: c, bits: report.kvBits,
                groupSize: report.kvGroupSize, processed: processed)
            scalarBytes = measured.floatingPointBytes
            report.elapsedSeconds = ProbeBounds.seconds(started.duration(to: .now))
            report.peakMemoryBytes = Memory.peakMemory
            captureFinalState()
            report.checkpoints.append(
                Checkpoint(
                    phase: phase, elapsedSeconds: report.elapsedSeconds, cache: measured,
                    activeMemoryBytes: Memory.activeMemory, peakMemoryBytes: Memory.peakMemory,
                    estimatedNextPeakBytes: estimate,
                    scoredTokens: scoredTokens, nllSum: settings.scoreNLL ? totalNLL : nil,
                    intervalScoredTokens: scoredTokens - checkpointScoredTokens,
                    intervalNllSum: settings.scoreNLL ? totalNLL - checkpointNLL : nil))
            checkpointNLL = totalNLL
            checkpointScoredTokens = scoredTokens
            try write(report, to: settings.reportURL)
            print(
                "\(phase) \(processed) tokens; KV \(measured.storageBytes) bytes; active \(Memory.activeMemory); peak \(Memory.peakMemory)"
            )
        }

        while processed < tokens.count {
            let end = min(processed + report.chunkSize, tokens.count)
            let estimate = try checkBudget(incoming: end - processed)
            try autoreleasepool {
                let logits = model(MLXArray(Array(tokens[processed..<end])).reshaped(1, end - processed), cache: cache)
                let last = logits[0, -1, 0...]
                let finite = all(isFinite(last))
                let selected = argMax(last)
                eval(finite, selected)
                guard finite.item(Bool.self) else {
                    throw ProbeFailure("Non-finite next-token logits.")
                }
                nextToken = selected.item(Int.self)
                if settings.scoreNLL {
                    let count = min(end, tokens.count - 1) - processed
                    if count > 0 {
                        let targets = MLXArray(Array(tokens[(processed + 1)...(processed + count)])).reshaped(1, count)
                        let loss = MLXNN.crossEntropy(
                            logits: logits[0..., 0..<count, 0...].asType(.float32), targets: targets
                        ).sum()
                        eval(loss)
                        let value = Double(loss.item(Float.self))
                        guard value.isFinite else {
                            throw ProbeFailure("Non-finite teacher-forced NLL.")
                        }
                        totalNLL += value
                        scoredTokens += count
                    }
                }
                // Complete every layer before submitting another chunk; no asynchronous prefill pipeline.
                eval(cache)
            }
            processed = end
            let actual = try inspect(
                cache, configuration: c, bits: report.kvBits, groupSize: report.kvGroupSize, processed: processed)
            scalarBytes = actual.floatingPointBytes
            if lastCheckpoint == 0 || processed - lastCheckpoint >= settings.checkpointInterval
                || processed == tokens.count
            {
                try record(phase: "prefill", estimate: estimate)
                lastCheckpoint = processed
            }
        }

        let stops = Set(
            ["<|endoftext|>", "<|end|>", "<|user|>", "<|assistant|>", "<|system|>"].compactMap {
                tokenizer.convertTokenToId($0)
            })
        report.stopReason = "output_limit"
        for index in 0..<report.requestedOutputTokens {
            guard let token = nextToken else {
                throw ProbeFailure("Prefill did not produce a next token.")
            }
            guard !stops.contains(token) else {
                report.stopReason = "stop_token"
                report.stopTokenID = token
                break
            }
            report.outputTokenIDs.append(token)
            report.outputText = tokenizer.decode(tokenIds: report.outputTokenIDs, skipSpecialTokens: true)
            if index + 1 == report.requestedOutputTokens {
                break
            }
            let estimate = try checkBudget(incoming: 1)
            try autoreleasepool {
                let logits = model(MLXArray([token]).reshaped(1, 1), cache: cache)[0, -1, 0...]
                let finite = all(isFinite(logits))
                let selected = argMax(logits)
                eval(finite, selected)
                eval(cache)
                guard finite.item(Bool.self) else {
                    throw ProbeFailure("Non-finite decode logits.")
                }
                nextToken = selected.item(Int.self)
            }
            processed += 1
            try record(phase: "decode", estimate: estimate)
        }
        report.status = "completed"
        report.elapsedSeconds = ProbeBounds.seconds(started.duration(to: .now))
        report.peakMemoryBytes = Memory.peakMemory
        captureFinalState()
        try write(report, to: settings.reportURL)
        print(report.outputText)
        print("Wrote \(settings.reportURL.path)")
    } catch {
        report.status = "stopped"
        report.error = error.localizedDescription
        report.stopReason = "diagnostic_limit_or_error"
        report.elapsedSeconds = ProbeBounds.seconds(started.duration(to: .now))
        report.peakMemoryBytes = Memory.peakMemory
        captureFinalState()
        try write(report, to: settings.reportURL)
        throw error
    }
}

private func inspect(
    _ cache: [KVCache], configuration c: TalkieConfiguration, bits: Int,
    groupSize: Int, processed: Int
) throws -> CacheMeasurement {
    var bytes = 0
    var allocated = 0
    var dtypeNames = Set<String>()
    var scalarBytes = 0
    guard cache.count == c.hiddenLayers else {
        throw ProbeFailure("Unexpected cache layer count.")
    }
    for layer in cache {
        guard layer.offset == processed else {
            throw ProbeFailure("Cache offsets diverged from processed tokens.")
        }
        if bits != 0 {
            guard let q = layer as? QuantizedKVCache, q.bits == bits, q.groupSize == groupSize, q.mode == .affine
            else {
                throw ProbeFailure("Every layer must use the requested affine cache before token zero.")
            }
        } else if !(layer is KVCacheSimple) {
            throw ProbeFailure("Unexpected full-precision cache implementation.")
        }
        let arrays = layer.innerState()
        guard arrays.count == (bits == 0 ? 2 : 6) else {
            throw ProbeFailure("Unexpected KV storage array count.")
        }
        for (index, array) in arrays.enumerated() {
            guard array.ndim == 4, array.dim(0) == 1, array.dim(1) == c.keyValueHeads, array.dim(2) >= processed
            else {
                throw ProbeFailure("Unexpected KV storage geometry.")
            }
            let packed = bits != 0 && (index == 0 || index == 3)
            let width =
                bits == 0 ? c.headDimension : (packed ? c.headDimension * bits / 32 : c.headDimension / groupSize)
            guard array.dim(3) == width else {
                throw ProbeFailure("Unexpected packed KV width.")
            }
            if packed {
                guard array.dtype == .uint32 else {
                    throw ProbeFailure("Packed KV storage must be uint32.")
                }
            } else {
                guard [.bfloat16, .float16, .float32].contains(array.dtype) else {
                    throw ProbeFailure("Unexpected KV scale/activation dtype.")
                }
                scalarBytes = max(scalarBytes, array.dtype == .float32 ? 4 : 2)
            }
            allocated = max(allocated, array.dim(2))
            bytes += array.nbytes
            dtypeNames.insert(String(describing: array.dtype))
        }
    }
    return CacheMeasurement(
        processedTokens: processed, allocatedTokens: allocated, layerCount: cache.count,
        storageBytes: bytes, dtypes: dtypeNames.sorted(), floatingPointBytes: scalarBytes)
}

private func nextPeakEstimate(
    configuration c: TalkieConfiguration, bits: Int, groupSize: Int,
    scalarBytes: Int, processed: Int, incoming: Int, activeMemory: Int, residentModel: Int
) -> Int {
    let next = processed + incoming
    let capacity = ProbeBounds.capacityUpperBound(after: next)
    let elementsPerLayerToken = 2 * c.keyValueHeads * c.headDimension
    let bytesPerLayerToken =
        bits == 0
        ? elementsPerLayerToken * scalarBytes
        : elementsPerLayerToken * bits / 8 + elementsPerLayerToken / groupSize * 2 * scalarBytes
    let nextKV = c.hiddenLayers * capacity * bytesPerLayerToken
    // Three score-sized buffers cover QK output, masked scores and softmax. Generic
    // quantized attention is not flash attention. Include one layer's KV relocation,
    // all incoming uncompressed K/V, logits/MLP workspace, and allocator headroom.
    let scoreWorkspace = 3 * c.attentionHeads * incoming * next * scalarBytes
    let relocation = capacity * bytesPerLayerToken
    let incomingKV = c.hiddenLayers * elementsPerLayerToken * incoming * scalarBytes
    let forwardWorkspace = max(
        512 * 1_048_576,
        incoming * (c.vocabularySize + c.intermediateSize * 4 + c.hiddenSize * 12) * 4)
    return max(activeMemory, residentModel + nextKV) + scoreWorkspace + relocation + incomingKV + forwardWorkspace
}

private func localURL(_ path: String) -> URL {
    URL(fileURLWithPath: NSString(string: path).expandingTildeInPath).standardizedFileURL.resolvingSymlinksInPath()
}

private func write(_ report: ProbeReport, to url: URL) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(report).write(to: url, options: .atomic)
}
