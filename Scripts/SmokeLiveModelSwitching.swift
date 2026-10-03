#!/usr/bin/env swift  // Run with: swift Scripts/SmokeLiveModelSwitching.swift /absolute/path/to/midnight MODEL_A MODEL_B  // MODEL_A must be a local text checkpoint. MODEL_B may be text or native speech.  // This launches and terminates its own runner. It never downloads checkpoints.
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif
#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

struct SmokeFailure: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

struct RuntimeState: Codable {
    struct Model: Codable { let id: String }
    struct Memory: Codable {
        let activeBytes: Int
        let cachedBytes: Int
        let peakBytes: Int
    }
    let instanceID: String?
    let processID: Int32
    let phase: String
    let operationID: String?
    let modelGeneration: UInt64
    let loadedModel: Model?
    let targetModel: String?
    let lastError: String?
    let memory: Memory
}

struct SmokeStep: Codable {
    let label: String
    let state: RuntimeState
}

final class LiveSwitchSmoke {
    let runner: URL
    let modelA: URL
    let modelB: URL
    let directory: URL
    let instanceID = "midnight-live-smoke-\(UUID().uuidString)"
    let process = Process()
    let session: URLSession
    let baseURL: URL
    let loadTimeout: TimeInterval
    var steps: [SmokeStep] = []
    var output: FileHandle?

    init(arguments: [String]) throws {
        guard arguments.count == 4 || arguments.count == 6 else {
            throw SmokeFailure(
                "Usage: swift Scripts/SmokeLiveModelSwitching.swift RUNNER MODEL_A MODEL_B [--timeout SECONDS]")
        }
        func path(_ value: String) -> URL {
            URL(fileURLWithPath: NSString(string: value).expandingTildeInPath).standardizedFileURL
        }
        runner = path(arguments[1])
        modelA = path(arguments[2])
        modelB = path(arguments[3])
        guard FileManager.default.isExecutableFile(atPath: runner.path) else {
            throw SmokeFailure("Runner is not executable: \(runner.path)")
        }
        for model in [modelA, modelB] {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: model.path, isDirectory: &isDirectory), isDirectory.boolValue
            else {
                throw SmokeFailure("Local checkpoint directory does not exist: \(model.path)")
            }
        }
        if arguments.count == 6 {
            guard arguments[4] == "--timeout", let value = Double(arguments[5]), value >= 10, value <= 3600 else {
                throw SmokeFailure("--timeout must be between 10 and 3600 seconds per load")
            }
            loadTimeout = value
        } else {
            loadTimeout = 600
        }
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("midnight-live-smoke-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        baseURL = URL(string: "http://127.0.0.1:\(try Self.availablePort())")!
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 10
        configuration.timeoutIntervalForResource = 180
        session = URLSession(configuration: configuration)
    }

    func run() async throws {
        defer {
            stopProcess()
            session.invalidateAndCancel()
            try? output?.close()
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            if let data = try? encoder.encode(steps) {
                try? data.write(to: directory.appendingPathComponent("snapshots.json"))
            }
            print("Artifacts: \(directory.path)")
        }
        let config = directory.appendingPathComponent("settings.json")
        try Data(#"{"mlxRunner":{}}"#.utf8).write(to: config)
        let log = directory.appendingPathComponent("runner.log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        output = try FileHandle(forWritingTo: log)
        process.executableURL = runner
        process.arguments = ["--idle", "--host", "127.0.0.1", "--port", String(baseURL.port!), "--config", config.path]
        var environment = ProcessInfo.processInfo.environment
        environment["MIDNIGHT_CONTROL_INSTANCE"] = instanceID
        environment["HF_HUB_OFFLINE"] = "1"
        process.environment = environment
        process.standardOutput = output
        process.standardError = output
        try process.run()
        print("Runner PID \(process.processIdentifier), \(baseURL.absoluteString)")
        let baseline = try await waitForStartup()
        try require(baseline.phase == "empty", "--idle did not start empty")
        record("initial-empty", baseline)

        let firstA = try await load(modelA, name: "smoke-A")
        record("A-ready", firstA)
        try await chat(model: "smoke-A")
        let (invalidStatus, _) = try await request(
            "POST", "/v1/runtime/load",
            json: [
                "model": directory.appendingPathComponent("missing-checkpoint").path
            ])
        try require(invalidStatus == 400, "Invalid preflight returned HTTP \(invalidStatus), expected 400")
        let preservedA = try await state()
        try require(
            preservedA.phase == "ready" && preservedA.loadedModel?.id == "smoke-A",
            "Invalid preflight disturbed A")
        try require(
            preservedA.modelGeneration == firstA.modelGeneration && preservedA.operationID == firstA.operationID,
            "Invalid preflight changed A's model identity")
        record("invalid-preflight-preserved-A", preservedA)

        let readyB = try await load(modelB, name: "smoke-B")
        try require(readyB.modelGeneration > firstA.modelGeneration, "B did not advance model generation")
        record("B-ready", readyB)
        let secondA = try await load(modelA, name: "smoke-A")
        try require(secondA.modelGeneration > readyB.modelGeneration, "Reloaded A did not advance model generation")
        record("A-ready-again", secondA)
        try await chat(model: "smoke-A")

        // A real directory passes the metadata preflight. An unsupported model
        // type fails asynchronously inside the loader after A has been released.
        let invalid = try makeUnsupportedCheckpoint()
        let (failureStatus, failureData) = try await request("POST", "/v1/runtime/load", json: ["model": invalid.path])
        try require(
            failureStatus == 202,
            "Unsupported architecture failed preflight instead of async loading: HTTP \(failureStatus) \(String(decoding: failureData, as: UTF8.self))"
        )
        let acceptedFailure = try JSONDecoder().decode(RuntimeState.self, from: failureData)
        let failed = try await waitFor(phase: "empty", operationID: acceptedFailure.operationID)
        try require(failed.lastError?.isEmpty == false, "Failed load did not expose an error")
        try require(failed.loadedModel == nil, "Failed replacement retained a loaded model")
        try memoryDropped(from: secondA, to: failed, baseline: baseline)
        record("async-failure-empty", failed)

        var unloadedStates: [RuntimeState] = [failed]
        for cycle in 1...2 {
            let recovered = try await load(modelA, name: "smoke-A")
            try require(recovered.lastError == nil, "Successful retry retained the previous error")
            record("recovery-\(cycle)-ready", recovered)
            let (unloadStatus, unloadData) = try await request("POST", "/v1/runtime/unload", json: [:])
            try require(unloadStatus == 202, "Unload returned HTTP \(unloadStatus)")
            let accepted = try JSONDecoder().decode(RuntimeState.self, from: unloadData)
            let empty = try await waitFor(phase: "empty", operationID: accepted.operationID)
            try require(empty.modelGeneration > recovered.modelGeneration, "Unload did not advance generation")
            try memoryDropped(from: recovered, to: empty, baseline: baseline)
            unloadedStates.append(empty)
            record("unload-\(cycle)-empty", empty)
        }
        let residuals = unloadedStates.map { $0.memory.activeBytes }
        try require(
            (residuals.max()! - residuals.min()!) <= 16 * 1_048_576,
            "MLX active memory grew across unload cycles: \(residuals)")
        let (emptyUnloadStatus, _) = try await request("POST", "/v1/runtime/unload", json: [:])
        try require(emptyUnloadStatus == 202, "Unloading an empty server should succeed")
        let (chatStatus, _) = try await request(
            "POST", "/v1/chat/completions",
            json: [
                "model": "smoke-A", "messages": [["role": "user", "content": "Hello"]],
            ])
        try require(chatStatus == 503, "Empty-server inference returned HTTP \(chatStatus), expected 503")
        print(
            "PASS: stable PID and instance; A → B → A; chat; preflight preservation; failed-load recovery; repeated unload memory release."
        )
        print("Memory figures are MLX allocator bytes. peakBytes is a high-water mark and is not expected to fall.")
    }

    private func load(_ model: URL, name: String) async throws -> RuntimeState {
        print("Loading \(name): \(model.path)")
        var body: [String: Any] = ["model": model.path, "name": name]
        let configData = try? Data(contentsOf: model.appendingPathComponent("config.json"))
        let config = configData.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        if config?["model_type"] as? String != "voxtral_tts" {
            body["maxTokens"] = 128
        }
        let (status, data) = try await request("POST", "/v1/runtime/load", json: body)
        try require(status == 202, "Load returned HTTP \(status): \(String(decoding: data, as: UTF8.self))")
        let accepted = try JSONDecoder().decode(RuntimeState.self, from: data)
        let loaded = try await waitFor(phase: "ready", operationID: accepted.operationID)
        try require(loaded.loadedModel?.id == name, "Ready descriptor does not describe \(name)")
        let (modelsStatus, modelsData) = try await request("GET", "/v1/models")
        let object = try JSONSerialization.jsonObject(with: modelsData) as? [String: Any]
        let names = (object?["data"] as? [[String: Any]])?.compactMap { $0["id"] as? String }
        try require(modelsStatus == 200 && names == [name], "/v1/models did not publish only \(name)")
        return loaded
    }

    private func chat(model: String) async throws {
        let (status, data) = try await request(
            "POST", "/v1/chat/completions",
            json: [
                "model": model, "messages": [["role": "user", "content": "Reply with a short greeting."]],
                "max_tokens": 16, "temperature": 0, "stream": false,
            ], timeout: 180)
        try require(status == 200, "Chat returned HTTP \(status): \(String(decoding: data, as: UTF8.self))")
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let choices = object?["choices"] as? [[String: Any]]
        let message = choices?.first?["message"] as? [String: Any]
        try require((message?["content"] as? String)?.isEmpty == false, "Chat returned no text")
        print("Chat succeeded: \(message?["content"] as? String ?? "")")
    }

    private func state() async throws -> RuntimeState {
        try require(process.isRunning, "Runner exited; see \(directory.appendingPathComponent("runner.log").path)")
        let (status, data) = try await request("GET", "/v1/runtime")
        try require(status == 200, "Runtime status returned HTTP \(status)")
        let value = try JSONDecoder().decode(RuntimeState.self, from: data)
        try require(value.processID == process.processIdentifier, "Runner PID changed")
        try require(value.instanceID == instanceID, "Runtime instance identity changed")
        return value
    }

    private func waitForStartup() async throws -> RuntimeState {
        let deadline = Date().addingTimeInterval(30)
        while Date() < deadline {
            if !process.isRunning {
                break
            }
            if let value = try? await state() {
                return value
            }
            try await Task.sleep(for: .milliseconds(200))
        }
        throw SmokeFailure(
            "Runner did not bind within 30 seconds; see \(directory.appendingPathComponent("runner.log").path)")
    }

    private func waitFor(phase: String, operationID: String?) async throws -> RuntimeState {
        let deadline = Date().addingTimeInterval(loadTimeout)
        while Date() < deadline {
            let value = try await state()
            if value.operationID == operationID && value.phase == phase {
                return value
            }
            if value.phase == "empty", let error = value.lastError, phase != "empty" {
                throw SmokeFailure("Model load failed: \(error)")
            }
            try await Task.sleep(for: .milliseconds(200))
        }
        throw SmokeFailure("Timed out waiting for \(phase); see \(directory.appendingPathComponent("runner.log").path)")
    }

    private func request(
        _ method: String, _ path: String, json: [String: Any]? = nil,
        timeout: TimeInterval = 10
    ) async throws -> (Int, Data) {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = method
        request.timeoutInterval = timeout
        if let json {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: json)
        }
        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else {
            throw SmokeFailure("Non-HTTP response")
        }
        return (response.statusCode, data)
    }

    private func makeUnsupportedCheckpoint() throws -> URL {
        let target = directory.appendingPathComponent("unsupported-checkpoint")
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try Data(
            #"{"model_type":"midnight_smoke_unsupported","architectures":["MidnightSmokeUnsupportedForCausalLM"],"max_position_embeddings":4096}"#
                .utf8
        )
        .write(to: target.appendingPathComponent("config.json"))
        try Data().write(to: target.appendingPathComponent("model.safetensors"))
        for name in [
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
            "chat_template.jinja",
        ] {
            let source = modelA.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: source.path) {
                try FileManager.default.createSymbolicLink(
                    at: target.appendingPathComponent(name), withDestinationURL: source)
            }
        }
        return target
    }

    private func memoryDropped(from loaded: RuntimeState, to empty: RuntimeState, baseline: RuntimeState) throws {
        try require(
            empty.memory.activeBytes <= baseline.memory.activeBytes + 64 * 1_048_576,
            "Unloaded MLX active memory remains high: \(empty.memory.activeBytes) bytes")
        try require(
            empty.memory.cachedBytes <= baseline.memory.cachedBytes + 1_048_576,
            "Unloaded MLX allocator cache was not cleared: \(empty.memory.cachedBytes) bytes")
        if loaded.memory.activeBytes > 64 * 1_048_576 {
            try require(
                empty.memory.activeBytes < loaded.memory.activeBytes / 4,
                "Unloading did not release most model allocations")
        }
    }

    private func record(_ label: String, _ state: RuntimeState) {
        steps.append(SmokeStep(label: label, state: state))
        print(
            "\(label): generation=\(state.modelGeneration) active=\(state.memory.activeBytes) cached=\(state.memory.cachedBytes) peak=\(state.memory.peakBytes)"
        )
    }

    private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
        if !condition() {
            throw SmokeFailure(message)
        }
    }

    private func stopProcess() {
        guard process.isRunning else {
            return
        }
        process.terminate()
        let deadline = Date().addingTimeInterval(3)
        while process.isRunning && Date() < deadline {
            usleep(20_000)
        }
        if process.isRunning {
            kill(process.processIdentifier, SIGKILL)
            let reapDeadline = Date().addingTimeInterval(2)
            while process.isRunning && Date() < reapDeadline {
                usleep(20_000)
            }
        }
    }

    private static func availablePort() throws -> UInt16 {
        #if os(Linux)
            let socketFD = socket(AF_INET, Int32(SOCK_STREAM.rawValue), 0)
        #else
            let socketFD = socket(AF_INET, SOCK_STREAM, 0)
        #endif
        guard socketFD >= 0 else {
            throw SmokeFailure("Could not create loopback port probe")
        }
        defer { close(socketFD) }
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(socketFD, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else {
            throw SmokeFailure("Could not bind loopback port probe")
        }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let inspected = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(socketFD, $0, &length) }
        }
        guard inspected == 0 else {
            throw SmokeFailure("Could not inspect loopback port probe")
        }
        return UInt16(bigEndian: address.sin_port)
    }
}

do {
    try await LiveSwitchSmoke(arguments: CommandLine.arguments).run()
} catch {
    FileHandle.standardError.write(Data("FAIL: \(error.localizedDescription)\n".utf8))
    exit(1)
}
