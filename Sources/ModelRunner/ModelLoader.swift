import Foundation
import ModelRunnerCore
import ModelRunnerProtocol

/// Resolved metadata only: validation never allocates model weights or changes
/// the MLX device, so an invalid selection leaves the active model usable.
struct ValidatedModelConfiguration: Sendable {
    enum Backend: Sendable {
        case text
        case voxtral
        case chatterbox(ChatterboxSettings)
    }

    let selection: ResolvedModelSelection
    let engine: ModelEngine
    let tokenLimit: GenerationTokenLimit
    let adapterScale: Float?
    let dflashModel: String?
    let dflashBlockSize: Int?
    let longContext: LongContextOptions
    let backend: Backend

    var servedModelName: String { selection.servedModelName }
}

struct ModelLoader: Sendable {
    private let settings: ModelStackSettings.MLXRunner
    private let defaultEngine: String?
    private let verbose: Bool

    init(
        settings: ModelStackSettings.MLXRunner? = nil,
        defaultEngine: String? = nil,
        verbose: Bool = false
    ) {
        self.settings = settings ?? .empty
        self.defaultEngine = defaultEngine
        self.verbose = verbose
    }

    func validate(_ request: ModelLoadRequest) throws -> ValidatedModelConfiguration {
        guard !request.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ModelLoadingError("Select a nonempty model name or directory")
        }
        let initialSelection = ModelCatalog.resolveMLX(model: request.model, adapter: request.adapter)
        let fileSettings = try settings.resolving(for: initialSelection)
        let selection = ModelCatalog.resolveMLX(
            model: request.model, adapter: request.adapter,
            servedModelName: request.name ?? fileSettings.servedModelName)
        let modelDirectory = URL(fileURLWithPath: selection.modelPath, isDirectory: true)
        let chatterbox = try ChatterboxSettings.load(modelDirectory: selection.modelPath)
        let modelConfiguration = try Self.validateModelFolder(
            modelDirectory, isChatterbox: chatterbox != nil)
        let engine = try ModelEngine(
            argument: request.engine ?? defaultEngine ?? fileSettings.engine ?? "auto").resolve()
        let tokenLimit = try GenerationTokenLimit(
            configuredMaximum: request.maxTokens ?? fileSettings.maximumTokens ?? 512)
        let longContext = try LongContextOptions(
            contextLength: request.contextLength ?? fileSettings.contextLength,
            prefillStepSize: request.prefillStepSize ?? fileSettings.prefillStepSize ?? 512,
            kvCompression: request.kvCompression ?? fileSettings.kvCompression ?? "none")
        let dflashInput = request.dflashModel ?? fileSettings.dflashModelPath
        let dflashModel = dflashInput.map {
            ModelCatalog.resolveMLX(model: $0).modelPath
        }
        let dflashBlockSize = request.dflashBlockSize ?? fileSettings.dflashBlockSize
        if dflashBlockSize != nil, dflashModel == nil {
            throw ModelLoadingError("dflashBlockSize requires dflashModel")
        }
        if let scale = request.adapterScale {
            guard selection.adapterPath != nil else {
                throw ModelLoadingError("adapterScale requires an adapter")
            }
            guard scale.isFinite, scale >= 0 else {
                throw ModelLoadingError("Adapter scale must be finite and nonnegative")
            }
        }
        let hasContextOptions = request.contextLength != nil || request.prefillStepSize != nil
            || request.kvCompression != nil || fileSettings.contextLength != nil
            || fileSettings.prefillStepSize != nil || fileSettings.kvCompression != nil
        let backend: ValidatedModelConfiguration.Backend
        if let chatterbox {
#if os(macOS)
            guard engine == .metal else {
                throw ModelLoadingError("The native Chatterbox backend currently requires Metal on macOS")
            }
            guard selection.adapterPath == nil, request.adapterScale == nil,
                  dflashModel == nil, !hasContextOptions else {
                throw ModelLoadingError("Chatterbox does not support chat context, KV, LoRA, or DFlash options")
            }
            guard request.maxTokens == nil, fileSettings.maximumTokens == nil else {
                throw ModelLoadingError("Set max_tokens in chatterbox.json for Chatterbox's speech-token limit")
            }
            try Self.validateChatterboxPaths(chatterbox, directory: modelDirectory)
            backend = .chatterbox(chatterbox)
#else
            throw ModelLoadingError("The native Chatterbox backend currently requires macOS")
#endif
        } else if try VoxtralVoiceCatalog(modelDirectory: selection.modelPath) != nil {
            guard !hasContextOptions else {
                throw ModelLoadingError("Context and KV-cache options apply to text models only")
            }
            guard selection.adapterPath == nil, request.adapterScale == nil else {
                throw ModelLoadingError("Voxtral TTS does not support a LoRA adapter")
            }
            guard dflashModel == nil else {
                throw ModelLoadingError("DFlash requires a Laguna chat model")
            }
            backend = .voxtral
        } else {
            _ = try ModelMemoryProfile(configuration: modelConfiguration, options: longContext)
            if let adapter = selection.adapterPath {
                try Self.validateAdapter(directory: URL(fileURLWithPath: adapter, isDirectory: true))
            }
            if let dflashModel {
                guard longContext.compression == .none else {
                    throw ModelLoadingError("KV compression with DFlash is not supported")
                }
                let draftConfiguration = try Self.validateModelFolder(
                    URL(fileURLWithPath: dflashModel, isDirectory: true))
                try Self.validateDFlash(target: modelConfiguration, draft: draftConfiguration,
                                       blockSize: dflashBlockSize)
            }
            backend = .text
        }
        return ValidatedModelConfiguration(
            selection: selection, engine: engine, tokenLimit: tokenLimit,
            adapterScale: request.adapterScale, dflashModel: dflashModel,
            dflashBlockSize: dflashBlockSize, longContext: longContext, backend: backend)
    }

    func load(_ configuration: ValidatedModelConfiguration) async throws -> LoadedModel {
        let selection = configuration.selection
        let maximumTokens = configuration.tokenLimit.configuredMaximum
        print("Loading \(selection.modelPath)…  engine=\(configuration.engine.rawValue)")
        let loaded: LoadedModel
        switch configuration.backend {
        case .text:
            let runner = try await LocalModelRunner(
                modelPath: selection.modelPath,
                servedModelName: selection.servedModelName,
                engine: configuration.engine,
                maximumTokens: maximumTokens,
                adapterPath: selection.adapterPath,
                adapterScale: configuration.adapterScale,
                dflashModelPath: configuration.dflashModel,
                dflashBlockSize: configuration.dflashBlockSize,
                longContext: configuration.longContext)
            loaded = LoadedModel(runner: runner, servedModelName: selection.servedModelName,
                                 tokenLimit: configuration.tokenLimit)
        case .voxtral:
            let synthesizer = try await VoxtralTTSSynthesizer(
                modelPath: selection.modelPath, servedModelName: selection.servedModelName,
                engine: configuration.engine, maximumFrames: maximumTokens, verbose: verbose)
            loaded = LoadedModel(servedModelName: selection.servedModelName,
                                 tokenLimit: configuration.tokenLimit, speechSynthesizer: synthesizer)
        case .chatterbox(let settings):
#if os(macOS)
            let synthesizer = try await ChatterboxSpeechSynthesizer(
                modelPath: selection.modelPath, servedModelName: selection.servedModelName,
                settings: settings)
            loaded = LoadedModel(servedModelName: selection.servedModelName,
                                 tokenLimit: configuration.tokenLimit, speechSynthesizer: synthesizer)
#else
            throw ModelLoadingError("The native Chatterbox backend currently requires macOS")
#endif
        }
        return loaded
    }

    private static func validateModelFolder(_ directory: URL, isChatterbox: Bool = false) throws -> Data {
        try requireDirectory(directory)
        let config = directory.appendingPathComponent("config.json")
        // The native Chatterbox loader supports its built-in default config
        // when config.json is absent. Preserve that explicit-path workflow.
        let data = isChatterbox && !FileManager.default.fileExists(atPath: config.path)
            ? Data("{}".utf8) : try Data(contentsOf: config)
        guard try JSONSerialization.jsonObject(with: data) is [String: Any] else {
            throw ModelLoadingError("Model config must be a JSON object: \(config.path)")
        }
        let entries = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        guard entries.contains(where: {
            $0.pathExtension == "safetensors" && (!isChatterbox || $0.lastPathComponent != "conds.safetensors")
        }) else {
            throw ModelLoadingError("Model folder is missing safetensors weights: \(directory.path)")
        }
        return data
    }

    private static func validateAdapter(directory: URL) throws {
        try requireDirectory(directory)
        let config = directory.appendingPathComponent("adapter_config.json")
        let data = try Data(contentsOf: config)
        guard try JSONSerialization.jsonObject(with: data) is [String: Any] else {
            throw ModelLoadingError("Adapter config must be a JSON object: \(config.path)")
        }
        let weights = directory.appendingPathComponent("adapters.safetensors")
        guard FileManager.default.isReadableFile(atPath: weights.path) else {
            throw ModelLoadingError("Adapter weights are missing or unreadable: \(weights.path)")
        }
    }

    private static func validateDFlash(target: Data, draft: Data, blockSize: Int?) throws {
        let target = try JSONSerialization.jsonObject(with: target) as? [String: Any]
        let draft = try JSONSerialization.jsonObject(with: draft) as? [String: Any]
        guard target?["model_type"] as? String == "laguna",
              (draft?["architectures"] as? [String])?.contains("DFlashLagunaForCausalLM") == true,
              let block = draft?["dflash_config"] as? [String: Any],
              let maximum = block["block_size"] as? Int, maximum >= 2 else {
            throw ModelLoadingError("DFlash requires a Laguna target and a DFlashLagunaForCausalLM drafter")
        }
        if let blockSize, !(2...maximum).contains(blockSize) {
            throw ModelLoadingError("DFlash block size must be between 2 and \(maximum)")
        }
    }

    private static func validateChatterboxPaths(_ settings: ChatterboxSettings, directory: URL) throws {
        func resolve(_ path: String) -> URL {
            let expanded = NSString(string: path).expandingTildeInPath
            return NSString(string: expanded).isAbsolutePath
                ? URL(fileURLWithPath: expanded) : directory.appendingPathComponent(expanded)
        }
        if let tokenizer = settings.speechTokenizer {
            try requireDirectory(resolve(tokenizer))
        }
        for path in settings.voices.values {
            let voice = resolve(path)
            guard FileManager.default.isReadableFile(atPath: voice.path) else {
                throw ModelLoadingError("Reference voice audio is missing or unreadable: \(voice.path)")
            }
        }
        if let ffmpeg = settings.ffmpegPath {
            let path = NSString(string: ffmpeg).expandingTildeInPath
            guard FileManager.default.isExecutableFile(atPath: path) else {
                throw ModelLoadingError("Configured ffmpeg is not executable: \(path)")
            }
        }
    }

    private static func requireDirectory(_ directory: URL) throws {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw ModelLoadingError("Model directory does not exist: \(directory.path)")
        }
    }
}

private struct ModelLoadingError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
