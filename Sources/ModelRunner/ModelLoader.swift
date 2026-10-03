import Foundation
import ModelRunnerCore
import ModelRunnerProtocol

/// Resolved metadata only: validation never allocates model weights or changes
/// the MLX device, so an invalid selection leaves the active model usable.
struct ValidatedModelConfiguration: Sendable {
    enum Backend: Sendable {
        case text
        case voxtral
        case qwenTTS(supportsReferenceAudio: Bool)
        case vibeVoice
        case vision(ManagedVisionConfiguration)

        var capabilities: ModelCard.Capabilities {
            switch self {
            case .text: .init()
            case .voxtral: .init(audioOutput: true)
            case .qwenTTS(let supportsReferenceAudio):
                .init(audioInput: supportsReferenceAudio, audioOutput: true)
            case .vibeVoice: .init(audioInput: true, audioOutput: true)
            case .vision: .init(vision: true)
            }
        }
    }

    let selection: ResolvedModelSelection
    let engine: ModelEngine
    let tokenLimit: GenerationTokenLimit
    let adapterScale: Float?
    let dflashModel: String?
    let dflashBlockSize: Int?
    var gemmaAssistantModel: String? = nil
    var gemmaAssistantBlockSize: Int? = nil
    var gemmaAssistantQuantizationBits: Int? = nil
    let longContext: LongContextOptions
    let backend: Backend
    var loadRequest: ModelLoadRequest? = nil
    var modelCard: ModelCard? = nil
    var pendingAssistant: AssistantDiscovery.Selection? = nil

    var modality: String {
        switch backend {
        case .text: "text"
        case .voxtral: "voice"
        case .qwenTTS: "voice"
        case .vibeVoice: "voice"
        case .vision: "vision"
        }
    }

    var servedModelName: String { selection.servedModelName }
    var protectedDirectories: [URL] {
        let paths = [selection.modelPath, selection.adapterPath, dflashModel, gemmaAssistantModel].compactMap { $0 }.map
        {
            URL(fileURLWithPath: $0, isDirectory: true)
        }
        return paths
    }
}

struct ModelLoader: Sendable {
    private let settings: ModelStackSettings.MLXRunner
    private let defaultEngine: String?
    private let verbose: Bool
    private let downloads: ManagedModelDownloads
    private let assistantsDirectory: URL
    private let visionEnvironment: @Sendable () -> [String: String]

    private struct BackendResolution {
        let backend: ValidatedModelConfiguration.Backend
        let tokenLimit: GenerationTokenLimit
        let restoredContextLength: Int?
    }

    init(
        settings: ModelStackSettings.MLXRunner? = nil,
        defaultEngine: String? = nil,
        verbose: Bool = false,
        downloads: ManagedModelDownloads = ManagedModelDownloads(),
        assistantsDirectory: URL = AssistantDiscovery.defaultDirectory,
        visionEnvironment: @escaping @Sendable () -> [String: String] = { ProcessInfo.processInfo.environment }
    ) {
        self.settings = settings ?? .empty
        self.defaultEngine = defaultEngine
        self.verbose = verbose
        self.downloads = downloads
        self.assistantsDirectory = assistantsDirectory
        self.visionEnvironment = visionEnvironment
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
        let modelConfiguration = try Self.validateModelFolder(modelDirectory)
        try Self.validatePrimaryModelConfiguration(modelConfiguration)
        let engine = try ModelEngine(
            argument: request.engine ?? defaultEngine ?? fileSettings.engine ?? "auto"
        ).resolve()
        if ProcessInfo.processInfo.environment["MIDNIGHT_GEMMA4_EXPERT_GATE_UP"] == "1" {
            try GemmaExpertProjectionCompatibility.validate(
                configuration: modelConfiguration,
                engine: engine, hasAdapter: selection.adapterPath != nil)
        }
        let initialTokenLimit = try GenerationTokenLimit(
            configuredMaximum: request.maxTokens ?? fileSettings.maximumTokens ?? 512)
        let longContext = try LongContextOptions(
            contextLength: request.contextLength ?? fileSettings.contextLength,
            prefillStepSize: request.prefillStepSize ?? fileSettings.prefillStepSize ?? 512,
            kvCompression: request.kvCompression ?? fileSettings.kvCompression ?? "none")
        let card = try ModelCard.load(directory: selection.settingsDirectory)
        let autoAssistant = request.autoAssistant ?? fileSettings.autoAssistant ?? true
        let dflashInput = request.dflashModel ?? fileSettings.dflashModelPath
        var dflashModel = dflashInput.map {
            ModelCatalog.resolveMLX(model: $0).modelPath
        }
        let dflashBlockSize = request.dflashBlockSize ?? fileSettings.dflashBlockSize
        var gemmaAssistantModel = (request.gemmaAssistantModel ?? fileSettings.gemmaAssistantModelPath).map {
            ModelCatalog.resolveMLX(model: $0).modelPath
        }
        let gemmaAssistantBlockSize = request.gemmaAssistantBlockSize ?? fileSettings.gemmaAssistantBlockSize
        let gemmaAssistantBits = request.gemmaAssistantQuantizationBits ?? fileSettings.gemmaAssistantQuantizationBits
        var pendingAssistant: AssistantDiscovery.Selection?
        if autoAssistant, dflashModel == nil, gemmaAssistantModel == nil,
            engine == .metal, longContext.compression == .none, selection.adapterPath == nil
        {
            let type =
                (try JSONSerialization.jsonObject(with: modelConfiguration) as? [String: Any])?["model_type"] as? String
            if let assistant = try AssistantDiscovery(directory: assistantsDirectory).resolve(
                target: modelDirectory, settingsDirectory: URL(fileURLWithPath: selection.settingsDirectory),
                configuration: modelConfiguration, card: card,
                blockSize: type == "muse_glimmer" ? dflashBlockSize : gemmaAssistantBlockSize,
                quantizationBits: gemmaAssistantBits)
            {
                if assistant.repository != nil {
                    pendingAssistant = assistant
                } else {
                    switch assistant.kind {
                    case .gemma: gemmaAssistantModel = assistant.directory.path
                    case .dflash: dflashModel = assistant.directory.path
                    }
                }
            }
        }
        if dflashBlockSize != nil, dflashModel == nil, pendingAssistant?.kind != .dflash {
            throw ModelLoadingError("dflashBlockSize requires a DFlash assistant")
        }
        if gemmaAssistantModel == nil, pendingAssistant?.kind != .gemma,
            gemmaAssistantBlockSize != nil || gemmaAssistantBits != nil
        {
            throw ModelLoadingError("Gemma assistant options require gemmaAssistantModel")
        }
        if let gemmaAssistantModel {
            guard dflashModel == nil else {
                throw ModelLoadingError("Select one speculative assistant: Gemma or DFlash")
            }
            guard engine == .metal, longContext.compression == .none else {
                throw ModelLoadingError("Gemma assistant currently requires Metal with uncompressed KV caches")
            }
            let draftConfiguration = try Self.validateModelFolder(URL(fileURLWithPath: gemmaAssistantModel))
            _ = try GemmaAssistantCompatibility.validate(
                target: modelConfiguration,
                assistant: draftConfiguration, blockSize: gemmaAssistantBlockSize,
                quantizationBits: gemmaAssistantBits)
        }
        if let scale = request.adapterScale {
            guard selection.adapterPath != nil else {
                throw ModelLoadingError("adapterScale requires an adapter")
            }
            guard scale.isFinite, scale >= 0 else {
                throw ModelLoadingError("Adapter scale must be finite and nonnegative")
            }
        }
        let resolvedBackend = try resolveBackend(
            request: request, fileSettings: fileSettings, selection: selection,
            modelDirectory: modelDirectory, modelConfiguration: modelConfiguration,
            engine: engine, longContext: longContext, dflashModel: dflashModel,
            dflashBlockSize: dflashBlockSize, gemmaAssistantModel: gemmaAssistantModel,
            initialTokenLimit: initialTokenLimit
        )
        let backend = resolvedBackend.backend
        let tokenLimit = resolvedBackend.tokenLimit
        let restoredContextLength = resolvedBackend.restoredContextLength
        var validated = ValidatedModelConfiguration(
            selection: selection, engine: engine, tokenLimit: tokenLimit,
            adapterScale: request.adapterScale, dflashModel: dflashModel,
            dflashBlockSize: dflashBlockSize, longContext: longContext, backend: backend)
        validated.modelCard = (card ?? ModelCard(name: selection.servedModelName)).withCapabilities(
            backend.capabilities)
        validated.pendingAssistant = pendingAssistant
        let voices: VoxtralVoiceCatalog?
        switch backend {
        case .voxtral: voices = try VoxtralVoiceCatalog(modelDirectory: selection.modelPath)
        case .qwenTTS: voices = try Qwen3TTSSynthesizer.availableVoices(modelPath: selection.modelPath)
        case .vibeVoice: voices = try VibeVoiceSpeechSynthesizer.availableVoices()
        case .text, .vision: voices = nil
        }
        validated.modelCard = validated.modelCard?.withVoices(voices?.modelCardVoices)
        validated.gemmaAssistantModel = gemmaAssistantModel
        validated.gemmaAssistantBlockSize = gemmaAssistantBlockSize
        validated.gemmaAssistantQuantizationBits = gemmaAssistantBits
        // Adapter bundles resolve weights under base-model but own policy at
        // the bundle root. Preserve that selection directory when replaying an
        // automatic (nil) output limit, rather than reading another policy.
        validated.loadRequest = ModelLoadRequest(
            model: selection.settingsDirectory, name: selection.servedModelName,
            adapter: selection.adapterPath, adapterScale: request.adapterScale, dflashModel: dflashModel,
            dflashBlockSize: dflashBlockSize,
            gemmaAssistantModel: gemmaAssistantModel,
            gemmaAssistantBlockSize: gemmaAssistantBlockSize,
            gemmaAssistantQuantizationBits: gemmaAssistantBits,
            autoAssistant: autoAssistant,
            maxTokens: request.maxTokens ?? fileSettings.maximumTokens,
            contextLength: validated.modality == "text" ? restoredContextLength : nil,
            prefillStepSize: validated.modality == "text" ? longContext.prefillStepSize : nil,
            kvCompression: validated.modality == "text" ? longContext.compression.rawValue : nil,
            engine: engine.rawValue)
        return validated
    }

    private func resolveBackend(
        request: ModelLoadRequest, fileSettings: ModelStackSettings.MLXRunner,
        selection: ResolvedModelSelection, modelDirectory: URL, modelConfiguration: Data,
        engine: ModelEngine, longContext: LongContextOptions, dflashModel: String?, dflashBlockSize: Int?,
        gemmaAssistantModel: String?, initialTokenLimit: GenerationTokenLimit
    ) throws -> BackendResolution {
        var tokenLimit = initialTokenLimit
        let hasContextOptions =
            request.contextLength != nil || request.prefillStepSize != nil
            || request.kvCompression != nil || fileSettings.contextLength != nil
            || fileSettings.prefillStepSize != nil || fileSettings.kvCompression != nil
        let backend: ValidatedModelConfiguration.Backend
        var restoredContextLength = longContext.contextLength
        let modelType =
            (try JSONSerialization.jsonObject(with: modelConfiguration) as? [String: Any])?["model_type"] as? String
        if ["fastvlm", "llava_qwen2"].contains(modelType ?? "") {
            let vision = try ManagedVisionConfiguration.validate(
                directory: modelDirectory,
                configuration: modelConfiguration, environment: visionEnvironment())
            guard engine == .metal else {
                throw ModelLoadingError("Native vision currently requires Metal on macOS.")
            }
            guard selection.adapterPath == nil, request.adapterScale == nil, dflashModel == nil,
                !hasContextOptions
            else {
                throw ModelLoadingError(
                    "Vision does not support text context, KV compression, LoRA or DFlash overrides.")
            }
            let maximum = request.maxTokens ?? fileSettings.maximumTokens ?? 1024
            guard maximum <= 1024 else {
                throw ModelLoadingError("Vision maximum output must be between 1 and 1024 tokens.")
            }
            tokenLimit = try GenerationTokenLimit(configuredMaximum: maximum, defaultTokens: min(512, maximum))
            backend = .vision(vision)
        } else if modelType == "vibevoice" {
            guard !hasContextOptions, selection.adapterPath == nil, request.adapterScale == nil,
                dflashModel == nil, engine == .metal || engine == .cpu
            else {
                throw ModelLoadingError(
                    "VibeVoice supports Metal/MPS or CPU without text context, adapter or speculative options.")
            }
            guard let root = try JSONSerialization.jsonObject(with: modelConfiguration) as? [String: Any],
                root["text_config"] != nil, root["audio_config"] != nil
            else {
                throw ModelLoadingError(
                    "Convert Microsoft's original VibeVoice weights with Scripts/prepare-vibevoice-1.5b.py before loading."
                )
            }
            guard tokenLimit.configuredMaximum <= 4096 else {
                throw ModelLoadingError("VibeVoice maxTokens must be at most 4096")
            }
            _ = try VibeVoiceSpeechSynthesizer.validateRuntime()
            backend = .vibeVoice
        } else if modelType == "qwen3_tts" {
            guard !hasContextOptions else {
                throw ModelLoadingError("Context and KV-cache options apply to text models only")
            }
            guard selection.adapterPath == nil, request.adapterScale == nil, dflashModel == nil else {
                throw ModelLoadingError("Qwen3-TTS does not support adapters or speculative decoding")
            }
            guard engine == .metal else {
                throw ModelLoadingError("Qwen3-TTS currently requires Metal on macOS.")
            }
            let root = try JSONSerialization.jsonObject(with: modelConfiguration) as? [String: Any]
            let qwenModelType = (root?["tts_model_type"] as? String ?? "base").lowercased()
            backend = .qwenTTS(supportsReferenceAudio: qwenModelType == "base")
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
            let profile = try ModelMemoryProfile(configuration: modelConfiguration, options: longContext)
            restoredContextLength = profile.contextLength
            struct OutputMetadata: Decodable {
                // These names match the model's JSON metadata keys.
                // swift-format-ignore: AlwaysUseLowerCamelCase
                struct TextMetadata: Decodable { let max_output_tokens: Int? }
                // swift-format-ignore: AlwaysUseLowerCamelCase
                let max_output_tokens: Int?
                // swift-format-ignore: AlwaysUseLowerCamelCase
                let text_config: TextMetadata?
            }
            let metadata = try JSONDecoder().decode(OutputMetadata.self, from: modelConfiguration)
            let declaredOutput = metadata.text_config?.max_output_tokens ?? metadata.max_output_tokens
            tokenLimit = try .forTextModel(
                contextLength: profile.contextLength,
                configuredMaximum: request.maxTokens ?? fileSettings.maximumTokens,
                modelMaximum: declaredOutput)
            if let adapter = selection.adapterPath {
                try Self.validateAdapter(directory: URL(fileURLWithPath: adapter, isDirectory: true))
            }
            if let dflashModel {
                guard longContext.compression == .none else {
                    throw ModelLoadingError("KV compression with DFlash is not supported")
                }
                let draftConfiguration = try Self.validateModelFolder(
                    URL(fileURLWithPath: dflashModel, isDirectory: true))
                try Self.validateDFlash(
                    target: modelConfiguration, draft: draftConfiguration,
                    blockSize: dflashBlockSize)
            }
            backend = .text
        }
        if gemmaAssistantModel != nil {
            guard case .text = backend else {
                throw ModelLoadingError("Gemma assistant requires the text model backend")
            }
        }
        return BackendResolution(
            backend: backend, tokenLimit: tokenLimit, restoredContextLength: restoredContextLength
        )
    }

    func operation(for request: ModelLoadRequest) throws -> ModelLoadOperation {
        let configuration = try validate(request)
        if let assistant = configuration.pendingAssistant {
            return ModelLoadOperation(
                name: configuration.servedModelName,
                prepare: {
                    let target = try Self.validateModelFolder(URL(fileURLWithPath: configuration.selection.modelPath))
                    try await AssistantDownloader.install(
                        assistant, target: target,
                        blockSize: assistant.kind == .gemma
                            ? configuration.gemmaAssistantBlockSize : configuration.dflashBlockSize,
                        quantizationBits: configuration.gemmaAssistantQuantizationBits)
                    let prepared = try validate(request)
                    guard prepared.pendingAssistant == nil else {
                        throw ModelLoadingError("The downloaded assistant could not be resolved")
                    }
                    return ModelLoadOperation(name: prepared.servedModelName) { try await load(prepared) }
                }, load: { throw ModelLoadingError("Assistant preparation did not complete") })
        }
        return ModelLoadOperation(name: configuration.servedModelName) { try await load(configuration) }
    }

    func load(_ configuration: ValidatedModelConfiguration) async throws -> LoadedModel {
        guard configuration.pendingAssistant == nil else {
            throw ModelLoadingError("The named assistant must be downloaded before loading this model")
        }
        // Lock all managed paths before constructing any backend. Failure
        // releases these local leases; success transfers ownership to LoadedModel.
        let fileUsage = try downloads.acquireUsage(protectedDirectories: configuration.protectedDirectories)
        let selection = configuration.selection
        let maximumTokens = configuration.tokenLimit.configuredMaximum
        print("Loading \(selection.modelPath)…  engine=\(configuration.engine.rawValue)")
        if let assistant = configuration.gemmaAssistantModel ?? configuration.dflashModel {
            print("Speculative assistant: \(assistant)")
        }
        let loaded: LoadedModel
        switch configuration.backend {
        case .text:
            let runner = try await LocalModelRunner(
                modelPath: selection.modelPath,
                servedModelName: selection.servedModelName,
                engine: configuration.engine,
                maximumTokens: maximumTokens,
                defaultMaximumTokens: configuration.tokenLimit.defaultTokens,
                adapterPath: selection.adapterPath,
                adapterScale: configuration.adapterScale,
                dflashModelPath: configuration.dflashModel,
                dflashBlockSize: configuration.dflashBlockSize,
                gemmaAssistantModelPath: configuration.gemmaAssistantModel,
                gemmaAssistantBlockSize: configuration.gemmaAssistantBlockSize,
                gemmaAssistantQuantizationBits: configuration.gemmaAssistantQuantizationBits,
                longContext: configuration.longContext)
            loaded = LoadedModel(
                runner: runner, servedModelName: selection.servedModelName,
                tokenLimit: configuration.tokenLimit, protectedDirectories: configuration.protectedDirectories,
                fileUsage: fileUsage,
                loadRequest: configuration.loadRequest, modelCard: configuration.modelCard)
        case .voxtral:
            let synthesizer = try await VoxtralTTSSynthesizer(
                modelPath: selection.modelPath, servedModelName: selection.servedModelName,
                engine: configuration.engine, maximumFrames: maximumTokens, verbose: verbose)
            loaded = LoadedModel(
                servedModelName: selection.servedModelName,
                tokenLimit: configuration.tokenLimit, speechSynthesizer: synthesizer,
                protectedDirectories: configuration.protectedDirectories, fileUsage: fileUsage,
                loadRequest: configuration.loadRequest, modelCard: configuration.modelCard)
        case .qwenTTS:
            let synthesizer = try await Qwen3TTSSynthesizer(
                modelPath: selection.modelPath, servedModelName: selection.servedModelName,
                engine: configuration.engine)
            loaded = LoadedModel(
                servedModelName: selection.servedModelName,
                tokenLimit: configuration.tokenLimit, speechSynthesizer: synthesizer,
                protectedDirectories: configuration.protectedDirectories, fileUsage: fileUsage,
                loadRequest: configuration.loadRequest, modelCard: configuration.modelCard)
        case .vision(let vision):
            let runtime = try await ManagedVisionRuntime.launch(
                configuration: vision,
                model: URL(fileURLWithPath: selection.modelPath, isDirectory: true), name: selection.servedModelName,
                managedRoot: downloads.root, tokenLimit: configuration.tokenLimit)
            loaded = LoadedModel(
                servedModelName: selection.servedModelName, tokenLimit: configuration.tokenLimit,
                protectedDirectories: configuration.protectedDirectories, fileUsage: fileUsage,
                vision: runtime, visionContextLength: vision.contextLength,
                visionMemoryBytes: Int(vision.memoryGiB * 1024 * 1024 * 1024), loadRequest: configuration.loadRequest,
                modelCard: configuration.modelCard)
        case .vibeVoice:
            let synthesizer = try await VibeVoiceSpeechSynthesizer(
                modelPath: selection.modelPath,
                servedModelName: selection.servedModelName, engine: configuration.engine, maximumTokens: maximumTokens)
            loaded = LoadedModel(
                servedModelName: selection.servedModelName,
                tokenLimit: configuration.tokenLimit, speechSynthesizer: synthesizer,
                protectedDirectories: configuration.protectedDirectories, fileUsage: fileUsage,
                loadRequest: configuration.loadRequest, modelCard: configuration.modelCard)
        }
        return loaded
    }

    static func validateModelFolder(_ directory: URL) throws -> Data {
        try requireDirectory(directory)
        let config = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: config)
        guard try JSONSerialization.jsonObject(with: data) is [String: Any] else {
            throw ModelLoadingError("Model config must be a JSON object: \(config.path)")
        }
        let entries = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        guard
            entries.contains(where: {
                $0.pathExtension == "safetensors"
            })
        else {
            throw ModelLoadingError("Model folder is missing safetensors weights: \(directory.path)")
        }
        return data
    }

    private static func validatePrimaryModelConfiguration(_ data: Data) throws {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }
        let modelType = object["model_type"] as? String
        let architectures = object["architectures"] as? [String] ?? []
        guard
            modelType == "muse_glimmer_assistant"
                || architectures.contains("DFlashLagunaForCausalLM")
        else {
            return
        }
        throw ModelLoadingError(
            "Speculative assistant checkpoints cannot be served as standalone models")
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

    static func validateDFlash(target: Data, draft: Data, blockSize: Int?) throws {
        let targetObject = try JSONSerialization.jsonObject(with: target) as? [String: Any]
        let draftObject = try JSONSerialization.jsonObject(with: draft) as? [String: Any]
        if targetObject?["model_type"] as? String == "muse_glimmer",
            draftObject?["model_type"] as? String == "muse_glimmer_assistant"
        {
            do {
                _ = try MuseGlimmerDFlashCompatibility.validate(
                    target: target, assistant: draft, blockSize: blockSize)
                return
            } catch {
                throw ModelLoadingError(
                    "Muse DFlash target and assistant are incompatible: \(error.localizedDescription)")
            }
        }
        guard targetObject?["model_type"] as? String == "laguna",
            (draftObject?["architectures"] as? [String])?.contains("DFlashLagunaForCausalLM") == true,
            let block = draftObject?["dflash_config"] as? [String: Any],
            let maximum = block["block_size"] as? Int, maximum >= 2
        else {
            throw ModelLoadingError("DFlash requires a Laguna target and a DFlashLagunaForCausalLM drafter")
        }
        if let blockSize, !(2...maximum).contains(blockSize) {
            throw ModelLoadingError("DFlash block size must be between 2 and \(maximum)")
        }
    }

    private static func requireDirectory(_ directory: URL) throws {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw ModelLoadingError("Model directory does not exist: \(directory.path)")
        }
    }
}

struct ModelLoadingError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
