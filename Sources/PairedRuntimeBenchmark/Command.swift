import ArgumentParser
import Foundation
import MLX
import ModelRunnerCore
import ModelRunnerProtocol

@main
struct PairedRuntimeBenchmark: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "model-runner-paired-runtime-bench",
        abstract: "Record alternating native trials with two Gemma 3 270M checkpoints resident.")

    @Argument(help: "Local Gemma 3 270M checkpoint for arm A.")
    var modelA: String

    @Argument(help: "Local Gemma 3 270M checkpoint for arm B; may equal A for a control experiment.")
    var modelB: String

    @Argument(help: "New raw JSON report path; never overwrites an existing report.")
    var output: String

    @Option(help: "UTF-8 user prompt file, used unchanged for both arms.")
    var promptFile: String

    @Option(help: "Measured pairs, alternating AB then BA. Each arm requests 512 tokens.")
    var pairs = 8

    @Option(help: "Shared prompt plus output context ceiling.")
    var contextLength = 4096

    mutating func validate() throws {
        guard (1...64).contains(pairs) else {
            throw ValidationError("--pairs must be between 1 and 64.")
        }
        guard (513...32_768).contains(contextLength) else {
            throw ValidationError("--context-length must be between 513 and 32768.")
        }
    }

    mutating func run() async throws {
        #if !os(macOS)
            throw ValidationError("This prototype requires the Metal backend on macOS.")
        #else
            defer { clearModelRunnerMLXStreams() }
            let outputURL = canonicalURL(output)
            guard !FileManager.default.fileExists(atPath: outputURL.path) else {
                throw ValidationError("Output already exists: \(outputURL.path)")
            }
            let promptData = try Data(contentsOf: canonicalURL(promptFile))
            guard let prompt = String(data: promptData, encoding: .utf8),
                !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                promptData.count <= 1_048_576
            else {
                throw ValidationError("The prompt must be nonempty UTF-8 and at most 1 MiB.")
            }
            let messages = [OpenAIMessage(role: "user", content: prompt)]
            let options = try LongContextOptions(
                contextLength: contextLength, prefillStepSize: 512, kvCompression: "none")
            let checkpoints = try [
                CheckedCheckpoint(arm: "A", path: modelA, options: options),
                CheckedCheckpoint(arm: "B", path: modelB, options: options),
            ]
            let limits = try MLXResourceGuard.resolve(
                for: .metal, physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                recommendedWorkingSetBytes: GPU.maxRecommendedWorkingSetBytes())
            let storedCeiling = 2 * 1_073_741_824
            let memoryCeiling = min(limits.memoryLimitBytes, 4 * 1_073_741_824)
            let combinedStored = checkpoints.reduce(0) { ModelMemoryProfile.add($0, $1.record.storedWeightBytes) }
            guard combinedStored < storedCeiling else {
                throw ValidationError("Combined stored weights must be below the prototype's 2 GiB ceiling.")
            }
            let estimatedBeforeLoad = checkpoints.map {
                $0.profile.requestBytes(
                    prompt: contextLength - 512, output: 512, prefillStepSize: 512, residentBytes: combinedStored)
            }.max()!
            guard estimatedBeforeLoad < memoryCeiling else {
                throw ValidationError(
                    "Combined weights plus conservative request estimate exceeds \(memoryCeiling) bytes.")
            }
            var report = PairedRuntimeReport(
                pairs: pairs, contextLength: contextLength, prompt: prompt, promptUTF8SHA256: sha256(promptData),
                runtimeEnvironment: runtimeEnvironment(), combinedStoredWeightBytes: combinedStored,
                combinedStoredWeightCeilingBytes: storedCeiling, processMemoryCeilingBytes: memoryCeiling,
                allocatorMemoryLimitBytes: limits.memoryLimitBytes,
                estimatedMaximumRequestBytesBeforeLoad: estimatedBeforeLoad, models: checkpoints.map(\.record))
            try FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try writeReport(report, to: outputURL, initial: true)
            let clock = ContinuousClock()
            let experimentStart = clock.now
            // These two strong references remain alive for every warmup and trial.
            var runners: [LocalModelRunner] = []
            do {
                for (index, checkpoint) in checkpoints.enumerated() {
                    let before = Memory.activeMemory
                    Memory.peakMemory = 0
                    let loadStart = clock.now
                    let runner = try await LocalModelRunner(
                        modelPath: checkpoint.record.path, servedModelName: "paired-\(checkpoint.record.arm)",
                        engine: .metal, maximumTokens: 512, longContext: options)
                    await runner.waitUntilIdle()
                    runners.append(runner)
                    report.loads.append(
                        LoadRecord(
                            arm: checkpoint.record.arm, order: index + 1,
                            milliseconds: elapsedMilliseconds(loadStart.duration(to: clock.now)),
                            implementation: runner.modelImplementation, activeMemoryBeforeBytes: before,
                            activeMemoryAfterBytes: Memory.activeMemory, processPeakActiveMemoryBytes: Memory.peakMemory
                        ))
                    try writeReport(report, to: outputURL)
                }
                // Prompt rendering is checked exactly before warming either arm.
                // Every measured request still renders freshly inside its own timer.
                let preparedA = try await runners[0].preparePrompt(messages: messages, maximumTokens: 512)
                let preparedB = try await runners[1].preparePrompt(messages: messages, maximumTokens: 512)
                guard preparedA.promptTokenIDs == preparedB.promptTokenIDs else {
                    throw ValidationError("Arms render different prompt token IDs; comparison aborted before warmups.")
                }
                report.renderedPromptTokens = preparedA.promptTokenIDs
                report.combinedLoadedActiveMemoryBytes = Memory.activeMemory
                let estimatedAfterLoad = checkpoints.map {
                    $0.profile.requestBytes(
                        prompt: preparedA.promptTokenCount, output: 512, prefillStepSize: 512,
                        residentBytes: Memory.activeMemory)
                }.max()!
                report.estimatedRequestBytesAfterLoad = estimatedAfterLoad
                guard estimatedAfterLoad < memoryCeiling else {
                    throw ValidationError(
                        "Measured combined resident allocation plus request estimate exceeds \(memoryCeiling).")
                }
                try writeReport(report, to: outputURL)
                for phase in ["warmup", "measured"] {
                    let count = phase == "warmup" ? 3 : pairs
                    for pair in 1...count {
                        let order = pair.isMultiple(of: 2) ? [1, 0] : [0, 1]
                        let label = order == [0, 1] ? "AB" : "BA"
                        for (position, index) in order.enumerated() {
                            let record = await runTrial(
                                runner: runners[index], checkpoint: checkpoints[index], messages: messages,
                                expectedPromptIDs: report.renderedPromptTokens, phase: phase, pair: pair,
                                order: label, position: position + 1, sequence: report.trials.count + 1,
                                experimentStart: experimentStart, memoryCeiling: memoryCeiling)
                            report.trials.append(record)
                            // Persist every attempt, including early stops and failures.
                            try writeReport(report, to: outputURL)
                            if let error = record.error {
                                throw ValidationError(error)
                            }
                        }
                    }
                }
                report.status = "completed_raw_trials"
                try writeReport(report, to: outputURL)
            } catch {
                for runner in runners {
                    await runner.waitUntilIdle()
                }
                report.status = "aborted"
                report.error = error.localizedDescription
                try writeReport(report, to: outputURL)
                throw error
            }
            for runner in runners {
                await runner.waitUntilIdle()
            }
            print("Recorded \(report.trials.count) raw trials with two resident models: \(outputURL.path)")
        #endif
    }
}
