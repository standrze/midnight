import Foundation
import ModelRunnerProtocol
import Synchronization

#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

/// The subprocess owns PyTorch and its weights. All exchanges use a dedicated
/// serial queue so pipe I/O never blocks Swift's cooperative executor.
public final class VibeVoiceSpeechSynthesizer: LocalSpeechSynthesizing, @unchecked Sendable {
    public let servedModelName: String
    public private(set) var voiceCatalog = VoxtralVoiceCatalog(voices: [])
    public let supportedAudioFormats: Set<LocalSpeechAudioFormat> = [.wav, .pcm]
    public let supportsInstructions = false
    public let supportedSpeedRange: ClosedRange<Double> = 1...1
    public let supportsReferenceAudio = true
    public let requiresReferenceText = false
    private let workerConfiguration: VibeVoiceWorkerConfiguration
    private let responseTimeout: TimeInterval
    private let shutdownGracePeriod: TimeInterval
    private let queue = DispatchQueue(label: "midnight.vibevoice")
    private let lifetime = StreamProducerLifetime()
    private let worker = Mutex<VibeVoiceWorker?>(nil)

    /// Locates the configured Python, worker script, and voice metadata files.
    public static func validateRuntime() throws -> (python: URL, worker: URL, voices: URL) {
        let env = ProcessInfo.processInfo.environment
        let root = URL(fileURLWithPath: env["MIDNIGHT_VIBEVOICE_ROOT"] ?? FileManager.default.currentDirectoryPath)
        let python = URL(
            fileURLWithPath: env["MIDNIGHT_VIBEVOICE_PYTHON"]
                ?? root.appendingPathComponent("Optional/vibevoice-1.5b-env/bin/python").path)
        let worker = root.appendingPathComponent("Scripts/vibevoice-worker.py")
        guard FileManager.default.isExecutableFile(atPath: python.path),
            FileManager.default.isReadableFile(atPath: worker.path)
        else {
            throw VibeVoiceRuntimeError(
                "VibeVoice runtime is missing. Run bash Scripts/setup-vibevoice-1.5b.sh and set MIDNIGHT_VIBEVOICE_ROOT to this Midnight checkout."
            )
        }
        return (python, worker, root.appendingPathComponent("Samples/voxtral-english-voices"))
    }

    /// Loads the voices available through the configured VibeVoice worker.
    public static func availableVoices() throws -> VoxtralVoiceCatalog {
        let runtime = try validateRuntime()
        let presets = [
            ("female", "neutral-female.wav"), ("male", "neutral-male.wav"),
            ("cheerful-female", "cheerful-female.wav"),
        ]
        return makeVoiceCatalog(
            ["default", "clone"]
                + presets.compactMap { name, file in
                    FileManager.default.fileExists(atPath: runtime.voices.appendingPathComponent(file).path)
                        ? name : nil
                })
    }

    private static func makeVoiceCatalog(_ names: [String]) -> VoxtralVoiceCatalog {
        VoxtralVoiceCatalog(
            voices: names.enumerated().map { index, name in
                VoxtralPresetVoice(
                    id: name, apiID: name, name: name.capitalized, position: index,
                    languages: ["en"], gender: name.contains("female") ? "female" : name == "male" ? "male" : nil)
            })
    }

    /// Validates and starts the configured worker-backed VibeVoice runtime.
    public convenience init(
        modelPath: String, servedModelName: String, engine: ModelEngine, maximumTokens: Int
    ) async throws {
        let runtime = try Self.validateRuntime()
        let arguments = [
            runtime.worker.path, "--model", modelPath, "--device", engine == .cpu ? "cpu" : "mps",
            "--max-tokens", String(maximumTokens), "--voices", runtime.voices.path,
        ]
        var environment = ProcessInfo.processInfo.environment
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        try await self.init(
            servedModelName: servedModelName,
            workerExecutableURL: runtime.python,
            workerArguments: arguments,
            workerEnvironment: environment
        )
    }

    init(
        servedModelName: String,
        workerExecutableURL: URL,
        workerArguments: [String],
        workerEnvironment: [String: String],
        responseTimeout: TimeInterval = 180,
        shutdownGracePeriod: TimeInterval = 5
    ) async throws {
        self.servedModelName = servedModelName
        self.workerConfiguration = VibeVoiceWorkerConfiguration(
            executableURL: workerExecutableURL,
            arguments: workerArguments,
            environment: workerEnvironment
        )
        self.responseTimeout = responseTimeout
        self.shutdownGracePeriod = shutdownGracePeriod
        do {
            let ready = try await exchange(nil)
            guard ready.ready == true, let voices = ready.voices else {
                throw VibeVoiceRuntimeError("VibeVoice did not become ready")
            }
            voiceCatalog = Self.makeVoiceCatalog(voices)
        } catch {
            stopWorker()
            throw error
        }
    }

    deinit {
        stopWorker()
    }

    /// Waits for active speech streams and their worker process to exit.
    public func waitUntilIdle() async {
        await lifetime.waitUntilIdle()
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                stopWorker()
                continuation.resume()
            }
        }
    }

    /// Launches a worker-backed speech stream and reports audio and usage events.
    public func stream(request: LocalSpeechSynthesisRequest) async -> AsyncThrowingStream<
        LocalSpeechSynthesisEvent, Error
    > {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard supportedAudioFormats.contains(request.format), request.speed == 1,
                        request.instructions == nil
                    else {
                        throw VibeVoiceRuntimeError("VibeVoice supports WAV/PCM, speed 1, and no instructions")
                    }
                    guard voiceCatalog.voice(id: request.voiceID) != nil else {
                        throw VibeVoiceRuntimeError("Unknown VibeVoice voice")
                    }
                    try Task.checkCancellation()
                    let command = Command(
                        input: request.input, voice: request.voiceID, format: request.format.rawValue,
                        encoding: request.pcmEncoding.rawValue, reference: request.referenceAudio?.base64EncodedString()
                    )
                    let response = try await exchange(JSONEncoder().encode(command))
                    try Task.checkCancellation()
                    guard let encoded = response.audio, let data = Data(base64Encoded: encoded) else {
                        throw VibeVoiceRuntimeError("VibeVoice returned no audio")
                    }
                    continuation.yield(.audio(data))
                    continuation.yield(
                        .completed(
                            .init(
                                promptTokens: response.promptTokens ?? 0,
                                completionTokens: response.completionTokens ?? 0)))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            lifetime.track(task)
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private struct Command: Encodable {
        let input: String
        let voice: String
        let format: String
        let encoding: String
        let reference: String?
    }
    private struct Reply: Decodable, Sendable {
        var ready: Bool?
        var voices: [String]?
        var audio: String?
        var error: String?
        var promptTokens: Int?
        var completionTokens: Int?
    }

    private func exchange(_ command: Data?) async throws -> Reply {
        let cancellation = VibeVoiceExchangeCancellation()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                queue.async { [self] in
                    do {
                        continuation.resume(returning: try performExchange(command, cancellation: cancellation))
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        } onCancel: {
            cancellation.cancel()
        }
    }

    private func performExchange(
        _ command: Data?, cancellation: VibeVoiceExchangeCancellation
    ) throws -> Reply {
        guard !cancellation.isCancelled else {
            throw CancellationError()
        }

        let (worker, wasStarted) = try workerForExchange()
        guard cancellation.register(worker) else {
            discard(worker)
            throw CancellationError()
        }

        let timeout = DispatchWorkItem { cancellation.stop(worker) }
        DispatchQueue.global().asyncAfter(deadline: .now() + responseTimeout, execute: timeout)
        defer {
            timeout.cancel()
            cancellation.unregister(worker)
        }

        do {
            if wasStarted {
                let ready = try readReply(from: worker)
                guard ready.ready == true, ready.voices != nil else {
                    throw VibeVoiceRuntimeError("VibeVoice did not become ready")
                }
                if command == nil {
                    return ready
                }
            }

            guard var command else {
                throw VibeVoiceRuntimeError("VibeVoice worker is already running")
            }
            command.append(10)
            try worker.input.fileHandleForWriting.write(contentsOf: command)
            let reply = try readReply(from: worker)
            try cancellation.checkCancellation()
            if let error = reply.error {
                throw VibeVoiceRuntimeError(error)
            }
            return reply
        } catch {
            discard(worker)
            if cancellation.isCancelled {
                throw CancellationError()
            }
            throw error
        }
    }

    private func workerForExchange() throws -> (worker: VibeVoiceWorker, wasStarted: Bool) {
        if let current = worker.withLock({ $0 }), current.isRunning {
            return (current, false)
        }

        let previous = worker.withLock { current in
            defer { current = nil }
            return current
        }
        previous?.stopAndWait(gracePeriod: shutdownGracePeriod)

        let replacement = try VibeVoiceWorker(configuration: workerConfiguration)
        worker.withLock { $0 = replacement }
        return (replacement, true)
    }

    private func readReply(from worker: VibeVoiceWorker) throws -> Reply {
        var line = Data()
        while true {
            let chunk = worker.output.fileHandleForReading.availableData
            guard !chunk.isEmpty else {
                break
            }
            line.append(chunk)
            guard line.count <= 32 * 1024 * 1024 else {
                throw VibeVoiceRuntimeError("VibeVoice response exceeds limit")
            }
            if line.last == 10 {
                break
            }
        }
        guard !line.isEmpty else {
            throw VibeVoiceRuntimeError("VibeVoice worker exited or exceeded its response timeout")
        }
        return try JSONDecoder().decode(Reply.self, from: line)
    }

    private func discard(_ discardedWorker: VibeVoiceWorker) {
        let removed = worker.withLock { current -> VibeVoiceWorker? in
            guard current === discardedWorker else {
                return nil
            }
            current = nil
            return discardedWorker
        }
        removed?.stopAndWait(gracePeriod: shutdownGracePeriod)
    }

    private func stopWorker() {
        let stoppedWorker = worker.withLock { current -> VibeVoiceWorker? in
            defer { current = nil }
            return current
        }
        stoppedWorker?.stopAndWait(gracePeriod: shutdownGracePeriod)
    }
}

private struct VibeVoiceWorkerConfiguration: Sendable {
    let executableURL: URL
    let arguments: [String]
    let environment: [String: String]
}

/// Owns one worker process and its pipe endpoints. `stop()` may be called by a
/// cancellation handler while the serial exchange queue is blocked reading.
private final class VibeVoiceWorker: @unchecked Sendable {
    let process = Process()
    let input = Pipe()
    let output = Pipe()

    private let stopped = Mutex(false)

    init(configuration: VibeVoiceWorkerConfiguration) throws {
        process.executableURL = configuration.executableURL
        process.arguments = configuration.arguments
        process.environment = configuration.environment
        process.standardInput = input
        process.standardOutput = output
        process.standardError = FileHandle.standardError
        try process.run()
    }

    var isRunning: Bool {
        !stopped.withLock { $0 } && process.isRunning
    }

    func stop() {
        let shouldStop = stopped.withLock { stopped in
            guard !stopped else {
                return false
            }
            stopped = true
            return true
        }
        guard shouldStop else {
            return
        }

        if process.isRunning {
            process.terminate()
        }
        try? input.fileHandleForWriting.close()
        try? output.fileHandleForReading.close()
    }

    /// Joins the child so model memory is released before another worker starts.
    func stopAndWait(gracePeriod: TimeInterval) {
        stop()
        let deadline = Date().addingTimeInterval(max(0, gracePeriod))
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        if process.isRunning {
            #if canImport(Darwin) || canImport(Glibc)
                _ = kill(process.processIdentifier, SIGKILL)
            #else
                process.terminate()
            #endif
        }
        process.waitUntilExit()
    }

    deinit {
        stop()
    }
}

/// Bridges task cancellation to the process currently serving an exchange.
private final class VibeVoiceExchangeCancellation: Sendable {
    private struct State {
        var isCancelled = false
        var worker: VibeVoiceWorker?
    }

    private let state = Mutex(State())

    var isCancelled: Bool {
        state.withLock { $0.isCancelled }
    }

    func register(_ worker: VibeVoiceWorker) -> Bool {
        let isCancelled = state.withLock { state in
            guard !state.isCancelled else {
                return true
            }
            state.worker = worker
            return false
        }
        if isCancelled {
            worker.stop()
        }
        return !isCancelled
    }

    func unregister(_ worker: VibeVoiceWorker) {
        state.withLock { state in
            if state.worker === worker {
                state.worker = nil
            }
        }
    }

    func checkCancellation() throws {
        if isCancelled {
            throw CancellationError()
        }
    }

    func cancel() {
        let worker = state.withLock { state in
            state.isCancelled = true
            return state.worker
        }
        worker?.stop()
    }

    func stop(_ worker: VibeVoiceWorker) {
        let isRegistered = state.withLock { $0.worker === worker }
        if isRegistered {
            worker.stop()
        }
    }
}

private struct VibeVoiceRuntimeError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
