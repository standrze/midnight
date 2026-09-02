import ArgumentParser
import Foundation
import ModelRunnerCore
import ModelRunnerProtocol

@main
struct MidnightCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "midnight",
        abstract: "Midnight Runner — serve a local MLX model through OpenAI-compatible chat and local audio APIs.",
        version: "0.1.0-beta.1"
    )

    @Option(name: .shortAndLong, help: "Model name in ~/.runner/models or an MLX folder")
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

    @Option(name: .long, help: "Execution engine: auto, metal, cuda, or cpu")
    var engine: String?

    @Option(name: .shortAndLong, help: "Path to model-stack settings JSON")
    var config: String?

    @Flag(name: .long, help: "Log incoming requests, generation settings, and request outcomes")
    var verbose = false

    @Flag(name: .long, help: "List models available under ~/.runner/models and exit")
    var listModels = false

    mutating func run() async throws {
        defer { clearModelRunnerMLXStreams() }

        if listModels {
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
        let fileSettings = stackSettings?.mlxRunner
        guard let requestedModel = model ?? fileSettings?.modelPath else {
            throw ValidationError("Provide --model or set mlxRunner.modelPath in model-stack.local.json")
        }
        let selection = ModelCatalog.resolveMLX(
            model: requestedModel,
            adapter: adapter,
            servedModelName: name ?? fileSettings?.servedModelName
        )
        let host = host ?? fileSettings?.host ?? "127.0.0.1"
        let port = port ?? fileSettings?.port ?? 8080
        let maxTokens = maxTokens ?? fileSettings?.maximumTokens ?? 512
        let dflashModel = dflashModel ?? fileSettings?.dflashModelPath
        let dflashBlockSize = dflashBlockSize ?? fileSettings?.dflashBlockSize
        let tokenLimit: GenerationTokenLimit
        do {
            tokenLimit = try GenerationTokenLimit(configuredMaximum: maxTokens)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
        let requestedEngine = try ModelEngine(argument: engine ?? fileSettings?.engine ?? "auto")
        let engine = try requestedEngine.resolve()
        if dflashBlockSize != nil, dflashModel == nil {
            throw ValidationError("--dflash-block-size requires --dflash-model")
        }

        if (try? VoxtralVoiceCatalog(modelDirectory: selection.modelPath)) != nil {
            guard selection.adapterPath == nil else {
                throw ValidationError("Voxtral TTS does not support a LoRA adapter")
            }
            guard adapterScale == nil else {
                throw ValidationError("--adapter-scale requires a chat model with --adapter")
            }
            guard dflashModel == nil else {
                throw ValidationError("--dflash-model requires a Laguna chat model")
            }
            guard dflashBlockSize == nil else {
                throw ValidationError("--dflash-block-size requires --dflash-model")
            }
            print("Loading \(selection.modelPath)…  engine=\(engine.rawValue)")
            let synthesizer = try await VoxtralTTSSynthesizer(
                modelPath: selection.modelPath,
                servedModelName: selection.servedModelName,
                engine: engine,
                maximumFrames: maxTokens,
                verbose: verbose
            )
            let server = ModelHTTPServer(
                servedModelName: selection.servedModelName,
                tokenLimit: tokenLimit,
                verbose: verbose,
                speechSynthesizer: synthesizer
            )
            print(
                "Ready: http://\(host):\(port)/v1  model=\(selection.servedModelName)  "
                    + "engine=\(engine.rawValue)"
            )
            print(
                "Native Voxtral speech generation is ready (24-kHz mono WAV or PCM; "
                    + "preset voices only)."
            )
            if verbose {
                print("Verbose request logging enabled (prompt and tool contents are redacted)")
            }
            try await server.run(host: host, port: port)
            return
        }

        print("Loading \(selection.modelPath)…  engine=\(engine.rawValue)")
        let runner = try await LocalModelRunner(
            modelPath: selection.modelPath,
            servedModelName: selection.servedModelName,
            engine: engine,
            maximumTokens: maxTokens,
            adapterPath: selection.adapterPath,
            adapterScale: adapterScale,
            dflashModelPath: dflashModel,
            dflashBlockSize: dflashBlockSize
        )
        let server = ModelHTTPServer(
            runner: runner,
            servedModelName: runner.servedModelName,
            tokenLimit: tokenLimit,
            verbose: verbose
        )
        print(
            "Ready: http://\(host):\(port)/v1  model=\(runner.servedModelName)  "
                + "engine=\(runner.engine.rawValue)"
        )
        if verbose {
            print("Verbose request logging enabled (prompt and tool contents are redacted)")
        }
        try await server.run(host: host, port: port)
    }
}
