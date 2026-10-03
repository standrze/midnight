import ArgumentParser
import ModelRunnerProtocol

/// Resolves launch configuration without loading weights or opening a listener.
struct ServerConfiguration {
    let host: String
    let port: Int
    let initialLoad: ModelLoadRequest?
    let settings: ModelStackSettings.MLXRunner?
    let engine: String?
    let verbose: Bool
    let consoleEnabled: Bool

    init(command: MidnightCommand, settings: ModelStackSettings.MLXRunner?, consoleEnabled: Bool? = nil) throws {
        self.settings = settings
        engine = command.listener.engine
        verbose = command.listener.verbose
        self.consoleEnabled = consoleEnabled ?? ServerConsole.shouldPresent(disabled: command.listener.noUI)
        if command.selection.idle, command.selection.model != nil {
            throw ValidationError("Use either --idle or --model")
        }
        let requestedModel = command.selection.idle ? nil : (command.selection.model ?? settings?.modelPath)
        if !command.selection.idle, requestedModel == nil, !self.consoleEnabled {
            throw ValidationError("Provide --model, set mlxRunner.modelPath in model-stack.local.json, or use --idle")
        }
        // Only listener settings and the execution-engine default live for the
        // whole process. Every later load resolves its own checkpoint settings.
        host = command.listener.host ?? settings?.host ?? "127.0.0.1"
        port = command.listener.port ?? settings?.port ?? 8080
        initialLoad = requestedModel.map {
            ModelLoadRequest(
                model: $0, name: command.selection.name, adapter: command.selection.adapter,
                adapterScale: command.selection.adapterScale,
                dflashModel: command.speculation.dflashModel, dflashBlockSize: command.speculation.dflashBlockSize,
                gemmaAssistantModel: command.speculation.gemmaAssistantModel,
                gemmaAssistantBlockSize: command.speculation.gemmaAssistantBlockSize,
                gemmaAssistantQuantizationBits: command.speculation.gemmaAssistantQuantizationBits,
                autoAssistant: command.speculation.noAutoAssistant ? false : nil,
                maxTokens: command.generation.maxTokens, contextLength: command.generation.contextLength,
                prefillStepSize: command.generation.prefillStepSize, kvCompression: command.generation.kvCompression,
                engine: command.listener.engine)
        }
    }

    @MainActor
    func run() async throws {
        let authentication = try APIKeyAuthentication.fromEnvironment()
        let loader = ModelLoader(settings: settings, defaultEngine: engine, verbose: verbose)
        let manager = ModelLifecycleManager(loader: loader, availability: try ModelAvailabilityStore.configured())
        let server = ModelHTTPServer(manager: manager, verbose: verbose, authentication: authentication)
        do {
            try await runListener(server: server, manager: manager)
        } catch {
            await manager.shutdown()
            throw error
        }
        await manager.shutdown()
    }

    @MainActor
    private func runListener(server: ModelHTTPServer, manager: ModelLifecycleManager) async throws {
        if consoleEnabled {
            let console = ServerConsole(manager: manager, host: host, port: port)
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask {
                    try await server.run(host: host, port: port) {
                        await console.didStartListening()
                        await startModel(manager: manager)
                    }
                }
                group.addTask { try await console.run() }
                defer { group.cancelAll() }
                _ = try await group.next()
            }
        } else {
            if verbose {
                print("Verbose request logging enabled (prompt and tool contents are redacted)")
            }
            try await server.run(host: host, port: port) {
                print("Listening: http://\(host):\(port)/v1")
                await startModel(manager: manager)
            }
        }
    }

    private func startModel(manager: ModelLifecycleManager) async {
        guard let initialLoad else {
            print("No model loaded. Load one through POST /v1/runtime/load.")
            return
        }
        do {
            _ = try await manager.load(initialLoad)
        } catch {
            // Keep status and lifecycle controls available after a bad
            // startup selection so the client can recover in this process.
            print("Initial model load failed: \(error.localizedDescription)")
        }
    }
}
