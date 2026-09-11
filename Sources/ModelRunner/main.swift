import ArgumentParser
import Foundation
import ModelRunnerCore
import ModelRunnerProtocol

@main
struct MidnightCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "midnight",
        abstract: "Midnight Runner — serve a local MLX model through OpenAI-compatible chat and local audio APIs.",
        version: "0.2.0-beta.5",
        subcommands: [DownloadCommand.self, AuthCommand.self]
    )

    @Option(name: .shortAndLong, help: "Model name in ~/.midnight/models or an MLX folder")
    var model: String?

    @Option(name: .long, help: "Model name exposed by the endpoint")
    var name: String?

    @Option(name: .long, help: "Path to an MLX LoRA adapter folder")
    var adapter: String?

    @Option(name: .long, help: "Override the LoRA adapter scale")
    var adapterScale: Float?

    @Option(
        name: .long,
        help: "Path to a Poolside Laguna DFlash drafter checkpoint (greedy decoding)"
    )
    var dflashModel: String?

    @Option(name: .long, help: "DFlash verification block size (2...checkpoint maximum)")
    var dflashBlockSize: Int?

    @Option(name: .long, help: "Address to listen on")
    var host: String?

    @Option(name: .shortAndLong, help: "Port to listen on")
    var port: Int?

    @Option(name: .long, help: "Default and hard maximum generated tokens per request")
    var maxTokens: Int?

    @Option(name: .long, help: "Maximum prompt plus output tokens (cannot exceed model context)")
    var contextLength: Int?

    @Option(name: .long, help: "Maximum tokens per prefill chunk (1...8192; default 512)")
    var prefillStepSize: Int?

    @Option(name: .long, help: "Experimental KV compression: none, affine8, affine4, turbo8v4")
    var kvCompression: String?

    @Option(name: .long, help: "Execution engine: auto, metal, cuda, or cpu")
    var engine: String?

    @Option(name: .shortAndLong, help: "Path to model-stack settings JSON")
    var config: String?

    @Flag(name: .long, help: "Log incoming requests, generation settings, and request outcomes")
    var verbose = false

    @Flag(name: .long, help: "Start the server with no model loaded")
    var idle = false

    @Flag(name: .long, help: "List models available under ~/.midnight/models and exit")
    var listModels = false

    @Flag(name: .long, help: "List models installed under ~/.midnight/models and exit")
    var list = false

    mutating func run() async throws {
        defer { clearModelRunnerMLXStreams() }

        if list || listModels {
            let directory = ModelCatalog.defaultDirectory()
            let models = ModelCatalog.availableModels(modelsDirectory: directory)
            print("Available models in \(directory.path):")
            if models.isEmpty {
                print("  (none)")
            } else {
                for model in models { print("  \(model)") }
            }
            return
        }
        let stackSettings = try ModelStackSettings.load(explicitPath: config)
        if idle, model != nil {
            throw ValidationError("Use either --idle or --model")
        }
        let requestedModel = idle ? nil : (model ?? stackSettings?.mlxRunner?.modelPath)
        if !idle, requestedModel == nil {
            throw ValidationError("Provide --model, set mlxRunner.modelPath in model-stack.local.json, or use --idle")
        }
        // Only listener settings and the execution-engine default live for the
        // whole process. Every later load resolves its own checkpoint settings.
        let listenHost = host ?? stackSettings?.mlxRunner?.host ?? "127.0.0.1"
        let listenPort = port ?? stackSettings?.mlxRunner?.port ?? 8080
        let loader = ModelLoader(settings: stackSettings?.mlxRunner,
                                 defaultEngine: engine, verbose: verbose)
        let manager = ModelLifecycleManager(loader: loader)
        let server = ModelHTTPServer(manager: manager, verbose: verbose)
        let initialLoad = requestedModel.map {
            ModelLoadRequest(model: $0, name: name, adapter: adapter, adapterScale: adapterScale,
                             dflashModel: dflashModel, dflashBlockSize: dflashBlockSize,
                             maxTokens: maxTokens, contextLength: contextLength,
                             prefillStepSize: prefillStepSize, kvCompression: kvCompression,
                             engine: engine)
        }
        if verbose {
            print("Verbose request logging enabled (prompt and tool contents are redacted)")
        }
        try await server.run(host: listenHost, port: listenPort) {
            print("Listening: http://\(listenHost):\(listenPort)/v1")
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
}
