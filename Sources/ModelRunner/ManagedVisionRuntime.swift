import Foundation
import ModelRunnerProtocol

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif
#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

struct VisionHTTPResponse: Sendable {
    let status: Int
    let body: Data
}

protocol VisionModelServing: Sendable {
    func complete(_ body: Data) async throws -> VisionHTTPResponse
    func status() async throws -> VisionHTTPResponse
    func waitUntilIdle() async
    func shutdown() async
}

struct ManagedVisionConfiguration: Sendable {
    let executable: URL
    let memoryGiB: Double
    let contextLength: Int

    static func validate(
        directory: URL, configuration: Data,
        environment: [String: String]
    ) throws -> Self {
        #if !os(macOS)
            throw ModelLoadingError("Vision currently requires macOS Apple silicon; CUDA vision is not implemented.")
        #else
            guard let config = try JSONSerialization.jsonObject(with: configuration) as? [String: Any],
                let processor = try JSONSerialization.jsonObject(
                    with: Data(contentsOf: directory.appendingPathComponent("preprocessor_config.json")))
                    as? [String: Any],
                let vision = config["vision_config"] as? [String: Any],
                (vision["layers"] as? [Int])?.count == 5,
                vision["downsamples"] as? [Bool] == [true, true, true, true, true],
                vision["down_stride"] as? Int == 2, vision["down_patch_size"] as? Int == 7,
                vision["image_size"] as? Int == 1024,
                config["image_token_index"] == nil || config["image_token_index"] as? Int == -200,
                processor["processor_class"] as? String == "FastVLMProcessor",
                let crop = processor["crop_size"] as? [String: Int], crop["width"] == 1024, crop["height"] == 1024,
                let context = config["tokenizer_model_max_length"] as? Int, context > 0
            else {
                throw ModelLoadingError(
                    "Vision selection requires compatible FastVLM with its verified 1024-pixel, 256-image-position layout."
                )
            }
            guard let path = environment["MIDNIGHT_VISION_WORKER"], NSString(string: path).isAbsolutePath,
                FileManager.default.isExecutableFile(atPath: path)
            else {
                throw ModelLoadingError(
                    "Set MIDNIGHT_VISION_WORKER to the absolute path of the separately built midnight-vision-worker before selecting a vision model."
                )
            }
            let memoryText = environment["MIDNIGHT_VISION_MEMORY_GIB"] ?? "4"
            guard let memory = Double(memoryText), memory.isFinite, (1...256).contains(memory) else {
                throw ModelLoadingError("MIDNIGHT_VISION_MEMORY_GIB must be a finite value from 1 to 256.")
            }
            let files = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.fileSizeKey])
            let weightsBytes = try files.filter { $0.pathExtension == "safetensors" }.reduce(Int64(0)) {
                $0 + Int64(try $1.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
            }
            guard weightsBytes <= Int64(memory * 1024 * 1024 * 1024) - 256 * 1024 * 1024 else {
                throw ModelLoadingError(
                    "Vision memory limit must cover checkpoint weights plus at least 256 MiB of working space.")
            }
            return Self(
                executable: URL(fileURLWithPath: path).resolvingSymlinksInPath(),
                memoryGiB: memory, contextLength: min(8192, context))
        #endif
    }
}

/// Exists only after explicit vision selection. The text inference path neither
/// creates this process/client nor imports the optional vision backend.
actor ManagedVisionRuntime: VisionModelServing {
    private let process: Process
    private let directory: URL
    private let session: URLSession
    private let endpoint: URL
    private let token: String
    private let tokenLimit: GenerationTokenLimit
    private var stopping = false
    private var requests = 0
    private var idleWaiters: [CheckedContinuation<Void, Never>] = []

    private init(
        process: Process, directory: URL, endpoint: URL, token: String,
        tokenLimit: GenerationTokenLimit
    ) {
        self.process = process
        self.directory = directory
        self.endpoint = endpoint
        self.token = token
        self.tokenLimit = tokenLimit
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 180
        config.timeoutIntervalForResource = 180
        config.httpMaximumConnectionsPerHost = 2
        config.httpCookieStorage = nil
        config.urlCache = nil
        self.session = URLSession(configuration: config)
    }

    static func launch(
        configuration: ManagedVisionConfiguration, model: URL, name: String,
        managedRoot: URL, tokenLimit: GenerationTokenLimit,
        startupSeconds: Double = 120
    ) async throws -> ManagedVisionRuntime {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "midnight-vision-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        let ready = directory.appendingPathComponent("ready.json")
        let token = UUID().uuidString + UUID().uuidString
        let process = Process()
        process.executableURL = configuration.executable
        process.arguments = [
            "--model", model.path, "--served-model-name", name,
            "--memory-limit-gib", String(configuration.memoryGiB), "--port", "0",
            "--managed-root", managedRoot.path, "--ready-file", ready.path,
            "--control-token", token, "--parent-pid", String(ProcessInfo.processInfo.processIdentifier),
        ]
        process.currentDirectoryURL = directory
        process.standardInput = FileHandle.nullDevice
        // Diagnostics are bounded by the worker (no generated/request bodies).
        process.standardOutput = FileHandle.standardError
        process.standardError = FileHandle.standardError
        var environment = ProcessInfo.processInfo.environment
        environment["HF_HUB_OFFLINE"] = "1"
        process.environment = environment
        do {
            try process.run()
            let deadline = Date().addingTimeInterval(startupSeconds)
            while Date() < deadline {
                try Task.checkCancellation()
                guard process.isRunning else {
                    throw ModelLoadingError(
                        "Vision worker exited during model loading (status \(process.terminationStatus)).")
                }
                if let data = try? Data(contentsOf: ready), data.count < 4096,
                    let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                    value["token"] as? String == token, value["pid"] as? Int32 == process.processIdentifier,
                    value["model"] as? String == name, let port = value["port"] as? Int,
                    (1024...65535).contains(port)
                {
                    let runtime = ManagedVisionRuntime(
                        process: process, directory: directory,
                        endpoint: URL(string: "http://127.0.0.1:\(port)")!, token: token, tokenLimit: tokenLimit)
                    do {
                        let status = try await runtime.status()
                        let state = try JSONSerialization.jsonObject(with: status.body) as? [String: Any]
                        guard status.status == 200, state?["phase"] as? String == "ready",
                            state?["model"] as? String == name
                        else {
                            throw ModelLoadingError("Vision worker readiness check failed.")
                        }
                        return runtime
                    } catch {
                        await runtime.shutdown()
                        throw error
                    }
                }
                try await Task.sleep(for: .milliseconds(25))
            }
            throw ModelLoadingError("Vision worker model loading timed out.")
        } catch {
            await stop(process)
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    func complete(_ body: Data) async throws -> VisionHTTPResponse {
        guard !stopping, process.isRunning else {
            throw ModelLoadingError("The selected vision worker is unavailable. Reload the vision model.")
        }
        guard body.count <= 12 * 1024 * 1024 else {
            return Self.error(413, "request_too_large", "Vision request exceeds 12 MiB.")
        }
        var payload = body
        if var object = try? JSONSerialization.jsonObject(with: body) as? [String: Any] {
            let requested = object["max_tokens"] ?? object["max_completion_tokens"]
            if requested == nil {
                object["max_tokens"] = tokenLimit.defaultTokens
                payload = try JSONSerialization.data(withJSONObject: object)
            } else if let value = requested as? NSNumber, value.intValue > tokenLimit.configuredMaximum {
                return Self.error(
                    400, "invalid_token_limit",
                    "Requested vision output exceeds the loaded model's configured maximum of \(tokenLimit.configuredMaximum)."
                )
            }
        }
        requests += 1
        defer { finishRequest() }
        do {
            return try await send(path: "/v1/chat/completions", method: "POST", body: payload)
        } catch {
            // URLSession cancellation closes the image request; the worker then
            // cancels and joins its producer. Keep this model admitted until an
            // uncancelled drain confirms GPU work ended, or the process exits.
            if process.isRunning {
                let drain = Task.detached { [self] in
                    try await send(path: "/v1/vision/drain", method: "POST", body: Data(), timeout: 30)
                }
                if (try? await drain.value.status) != 200 {
                    await shutdown()
                }
            }
            throw error
        }
    }

    func status() async throws -> VisionHTTPResponse {
        guard !stopping, process.isRunning else {
            throw ModelLoadingError("The selected vision worker is unavailable. Reload the vision model.")
        }
        return try await send(path: "/v1/vision/status", method: "GET", body: nil, timeout: 5)
    }

    func waitUntilIdle() async {
        if requests > 0 {
            await withCheckedContinuation { idleWaiters.append($0) }
        }
    }

    func shutdown() async {
        stopping = true
        session.invalidateAndCancel()
        await Self.stop(process)
        try? FileManager.default.removeItem(at: directory)
    }

    private func finishRequest() {
        requests -= 1
        if requests == 0 {
            let waiters = idleWaiters
            idleWaiters = []
            for waiter in waiters {
                waiter.resume()
            }
        }
    }

    private func send(path: String, method: String, body: Data?, timeout: TimeInterval = 180) async throws
        -> VisionHTTPResponse
    {
        var request = URLRequest(url: endpoint.appendingPathComponent(path), timeoutInterval: timeout)
        request.httpMethod = method
        request.httpBody = body
        request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        #if os(macOS)
            let (bytes, response) = try await session.bytes(for: request)
            guard let response = response as? HTTPURLResponse,
                response.expectedContentLength <= 512 * 1024
            else {
                session.invalidateAndCancel()
                throw ModelLoadingError("Vision worker returned an invalid or oversized response.")
            }
            var data = Data()
            for try await byte in bytes {
                guard data.count < 512 * 1024 else {
                    session.invalidateAndCancel()
                    throw ModelLoadingError("Vision worker returned an oversized response.")
                }
                data.append(byte)
            }
            return VisionHTTPResponse(status: response.statusCode, body: data)
        #else
            throw ModelLoadingError("Native vision requires macOS; CUDA vision is not implemented.")
        #endif
    }

    private static func stop(_ process: Process) async {
        await Task.detached(priority: .utility) {
            guard process.isRunning else {
                return
            }
            process.terminate()
            let deadline = Date().addingTimeInterval(5)
            while process.isRunning && Date() < deadline {
                try? await Task.sleep(for: .milliseconds(25))
            }
            if process.isRunning {
                kill(process.processIdentifier, SIGKILL)
            }
            // Process exit, not just signal submission, is the resource boundary.
            while process.isRunning {
                try? await Task.sleep(for: .milliseconds(10))
            }
        }.value
    }

    private static func error(_ status: Int, _ code: String, _ message: String) -> VisionHTTPResponse {
        let body = try! JSONSerialization.data(withJSONObject: [
            "error": ["code": code, "message": message, "type": "invalid_request_error"]
        ])
        return VisionHTTPResponse(status: status, body: body)
    }
}
