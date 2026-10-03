import Foundation
import ModelRunnerProtocol
import Testing

@testable import Midnight

@Suite("Persistent model loader preflight")
struct ModelLoaderTests {
    @Test("Installed catalog preflights metadata, deduplicates roots and skips incomplete entries")
    func installedCatalog() throws {
        try withFixture { directory in
            _ = try checkpoint("alpha", in: directory)
            _ = try checkpoint("beta", in: directory)
            let lagunaDraft = try checkpoint("laguna-draft", in: directory)
            try Data(
                #"{"model_type":"laguna","architectures":["DFlashLagunaForCausalLM"]}"#.utf8
            ).write(to: lagunaDraft.appendingPathComponent("config.json"))
            let museDraft = try checkpoint("muse-draft", in: directory)
            try Data(#"{"model_type":"muse_glimmer_assistant"}"#.utf8).write(
                to: museDraft.appendingPathComponent("config.json"))
            let incomplete = directory.appendingPathComponent("incomplete")
            try FileManager.default.createDirectory(at: incomplete, withIntermediateDirectories: true)
            try Data(#"{"model_type":"llama"}"#.utf8).write(to: incomplete.appendingPathComponent("config.json"))
            let loader = ModelLoader(defaultEngine: "cpu")
            let entries = loader.installedModels(roots: [directory, directory])
            #expect(entries.map { $0.descriptor.id } == ["alpha", "beta"])
            #expect(entries.allSatisfy { $0.request.model.hasPrefix(directory.path) })
            #expect(entries.allSatisfy { $0.descriptor.modality == "text" })
            #expect(throws: (any Error).self) {
                try loader.validate(ModelLoadRequest(model: lagunaDraft.path))
            }
            #expect(throws: (any Error).self) {
                try loader.validate(ModelLoadRequest(model: museDraft.path))
            }
        }
    }

    @Test("VibeVoice rejects unconverted weights and incompatible controls before starting its worker")
    func vibeVoicePreflight() throws {
        try withFixture { directory in
            let model = try checkpoint("vibevoice", in: directory)
            let configURL = model.appendingPathComponent("config.json")
            try Data(#"{"model_type":"vibevoice"}"#.utf8).write(to: configURL)
            let loader = ModelLoader(defaultEngine: "metal")
            #expect(throws: (any Error).self) {
                try loader.validate(ModelLoadRequest(model: model.path))
            }
            try Data(#"{"model_type":"vibevoice","text_config":{},"audio_config":{}}"#.utf8).write(to: configURL)
            for request in [
                ModelLoadRequest(model: model.path, contextLength: 2048),
                ModelLoadRequest(model: model.path, maxTokens: 4097),
                ModelLoadRequest(model: model.path, adapterScale: 0.5),
            ] {
                #expect(throws: (any Error).self) { try loader.validate(request) }
            }
        }
    }

    @Test("Qwen3-TTS discovery reports subtype-specific reference-audio support")
    func qwenTTSPreflight() throws {
        try withFixture { directory in
            let model = try checkpoint("qwen-tts", in: directory)
            let configURL = model.appendingPathComponent("config.json")
            let loader = ModelLoader(defaultEngine: "metal")

            try Data(#"{"model_type":"qwen3_tts","tts_model_type":"base"}"#.utf8).write(to: configURL)
            let base = try loader.validate(ModelLoadRequest(model: model.path))
            #expect(base.modelCard?.capabilities == .init(audioInput: true, audioOutput: true))
            #expect(base.modelCard?.voices?.first?.requiresReferenceAudio == true)

            try Data(
                #"{"model_type":"qwen3_tts","tts_model_type":"custom_voice","talker_config":{"spk_id":{"speaker":0}}}"#
                    .utf8
            ).write(to: configURL)
            let customVoice = try loader.validate(ModelLoadRequest(model: model.path))
            #expect(customVoice.modelCard?.capabilities == .init(audioInput: false, audioOutput: true))
            #expect(customVoice.modelCard?.voices?.first?.requiresReferenceAudio == false)
        }
    }

    @Test("Muse Glimmer text checkpoints are admitted through the native VLM runtime")
    func museGlimmerTextCheckpoint() throws {
        try withFixture { directory in
            let model = try checkpoint("muse", in: directory)
            let config: [String: Any] = [
                "model_type": "muse_glimmer",
                "text_config": [
                    "model_type": "muse_glimmer_text", "hidden_size": 64,
                    "intermediate_size": 128, "num_hidden_layers": 4,
                    "num_attention_heads": 2, "num_key_value_heads": 1,
                    "head_dim": 32, "vocab_size": 128,
                    "max_position_embeddings": 512, "sliding_window": 64,
                    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                ],
                "vision_config": [:],
            ]
            try JSONSerialization.data(withJSONObject: config).write(
                to: model.appendingPathComponent("config.json"))

            let result = try ModelLoader(defaultEngine: "cpu").validate(
                ModelLoadRequest(model: model.path))
            #expect(result.modality == "text")
            #expect(result.modelCard?.capabilities == ModelCard.Capabilities())
            #expect(result.dflashModel == nil)
            #expect(throws: (any Error).self) {
                try ModelLoader(defaultEngine: "cpu").validate(
                    ModelLoadRequest(model: model.path, dflashModel: model.path))
            }
        }
    }

    @Test("Editable model cards are metadata and never replace the served ID")
    func editableCard() throws {
        try withFixture { directory in
            let model = try checkpoint("folder-id", in: directory)
            let cardURL = model.appendingPathComponent(ModelCard.filename)
            try Data(
                #"{"name":"Talkie 1930","description":"Historical model","capabilities":{"vision":true,"audio_input":true,"audio_output":true}}"#
                    .utf8
            ).write(to: cardURL)
            let loader = ModelLoader(defaultEngine: "cpu")
            let first = try loader.validate(ModelLoadRequest(model: model.path, name: "stable-api-id"))
            #expect(first.servedModelName == "stable-api-id")
            #expect(first.modelCard?.name == "Talkie 1930")
            #expect(first.modelCard?.description == "Historical model")
            #expect(first.modelCard?.capabilities == ModelCard.Capabilities())
            #expect(
                try ModelCard.load(directory: model.path)?.capabilities
                    == ModelCard.Capabilities(vision: true, audioInput: true, audioOutput: true))
            try Data(#"{"name":"Edited display name"}"#.utf8).write(to: cardURL)
            let second = try loader.validate(ModelLoadRequest(model: model.path, name: "stable-api-id"))
            #expect(second.servedModelName == first.servedModelName)
            #expect(second.modelCard?.name == "Edited display name")
            #expect(second.modelCard?.capabilities == ModelCard.Capabilities())
            #expect(first.modelCard?.name == "Talkie 1930")
            #expect(first.modelCard?.description == "Historical model")
            #expect(first.modelCard?.capabilities == ModelCard.Capabilities())
            try Data(#"{"name":"   "}"#.utf8).write(to: cardURL)
            #expect(throws: (any Error).self) { try loader.validate(ModelLoadRequest(model: model.path)) }
        }
    }

    @Test("Gemma assistant preflight preserves options, protects files, and rejects incompatible loads")
    func gemmaAssistantConfiguration() throws {
        try withFixture { directory in
            let target = try checkpoint("gemma", in: directory)
            let assistant = try checkpoint("assistant", in: directory)
            let other = try checkpoint("other", in: directory)
            let text: [String: Any] = [
                "model_type": "gemma4_text", "hidden_size": 2816, "intermediate_size": 2112,
                "num_hidden_layers": 2, "num_attention_heads": 16, "num_key_value_heads": 8,
                "num_global_key_value_heads": 2, "head_dim": 256, "global_head_dim": 512,
                "sliding_window": 1024, "vocab_size": 262144, "attention_k_eq_v": true,
                "max_position_embeddings": 32768, "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0, "layer_types": ["sliding_attention", "full_attention"],
            ]
            var draft = text
            draft["hidden_size"] = 1024
            draft["num_kv_shared_layers"] = 2
            try JSONSerialization.data(withJSONObject: ["model_type": "gemma4", "text_config": text])
                .write(to: target.appendingPathComponent("config.json"))
            try JSONSerialization.data(withJSONObject: [
                "model_type": "gemma4_assistant", "text_config": draft,
                "backbone_hidden_size": 2816, "block_size": 4,
            ])
            .write(to: assistant.appendingPathComponent("config.json"))
            let request = ModelLoadRequest(
                model: target.path, gemmaAssistantModel: assistant.path,
                gemmaAssistantBlockSize: 4, gemmaAssistantQuantizationBits: 4)
            let loader = ModelLoader(defaultEngine: "metal")
            let result = try loader.validate(request)
            #expect(result.modality == "text")
            #expect(result.gemmaAssistantModel == assistant.path)
            #expect(result.gemmaAssistantBlockSize == 4)
            #expect(result.gemmaAssistantQuantizationBits == 4)
            #expect(result.protectedDirectories.contains { $0.path == assistant.path })
            let restored = try loader.validate(#require(result.loadRequest))
            #expect(restored.gemmaAssistantModel == assistant.path)
            #expect(restored.gemmaAssistantQuantizationBits == 4)
            #expect(try JSONDecoder().decode(ModelLoadRequest.self, from: JSONEncoder().encode(request)) == request)
            for bad in [
                ModelLoadRequest(model: target.path, gemmaAssistantBlockSize: 4),
                ModelLoadRequest(model: target.path, gemmaAssistantQuantizationBits: 4),
                ModelLoadRequest(
                    model: target.path, gemmaAssistantModel: assistant.path, gemmaAssistantQuantizationBits: 3),
                ModelLoadRequest(model: target.path, gemmaAssistantModel: assistant.path, gemmaAssistantBlockSize: 5),
                ModelLoadRequest(model: other.path, gemmaAssistantModel: assistant.path),
                ModelLoadRequest(model: target.path, dflashModel: assistant.path, gemmaAssistantModel: assistant.path),
                ModelLoadRequest(model: target.path, gemmaAssistantModel: assistant.path, engine: "cpu"),
            ] {
                #expect(throws: (any Error).self) { try loader.validate(bad) }
            }
            let settingsData = try JSONSerialization.data(withJSONObject: [
                "mlxRunner": [
                    "modelPath": target.path, "gemmaAssistantModelPath": assistant.path,
                    "gemmaAssistantBlockSize": 4, "gemmaAssistantQuantizationBits": 4,
                ]
            ])
            let settings = try JSONDecoder().decode(ModelStackSettings.self, from: settingsData)
            let configured = ModelLoader(settings: settings.mlxRunner, defaultEngine: "metal")
            #expect(try configured.validate(ModelLoadRequest(model: target.path)).gemmaAssistantQuantizationBits == 4)
            #expect(try configured.validate(ModelLoadRequest(model: other.path)).gemmaAssistantModel == nil)
        }
    }

    @Test("Text checkpoint metadata sets automatic and declared output limits")
    func automaticOutputPolicy() throws {
        try withFixture { directory in
            let model = try checkpoint("auto", in: directory)
            let loader = ModelLoader(defaultEngine: "cpu")
            let automatic = try loader.validate(ModelLoadRequest(model: model.path))
            #expect(automatic.tokenLimit.defaultTokens == 4096)
            #expect(automatic.tokenLimit.configuredMaximum == 32767)
            let small = try loader.validate(ModelLoadRequest(model: model.path, contextLength: 2048))
            #expect(small.tokenLimit.defaultTokens == 512)
            let configURL = model.appendingPathComponent("config.json")
            var config = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as! [String: Any]
            config["max_output_tokens"] = 1536
            try JSONSerialization.data(withJSONObject: config).write(to: configURL)
            let capped = try loader.validate(ModelLoadRequest(model: model.path))
            #expect(capped.tokenLimit.configuredMaximum == 1536)
            #expect(capped.tokenLimit.defaultTokens == 1536)
            for invalid in [true, "4096", 0] as [Any] {
                config["max_output_tokens"] = invalid
                try JSONSerialization.data(withJSONObject: config).write(to: configURL)
                #expect(throws: (any Error).self) { try loader.validate(ModelLoadRequest(model: model.path)) }
            }
        }
    }

    @Test("Voxtral TTS selects the native speech backend and rejects text-only controls")
    func voxtralPreflight() throws {
        try withFixture { directory in
            let model = try checkpoint("voxtral", in: directory)
            let configuration: [String: Any] = [
                "model_type": "voxtral_tts",
                "multimodal": [
                    "audio_tokenizer_args": ["voice": ["en_female": 0]]
                ],
            ]
            try JSONSerialization.data(withJSONObject: configuration).write(
                to: model.appendingPathComponent("config.json"))

            let loader = ModelLoader(defaultEngine: "cpu")
            let result = try loader.validate(ModelLoadRequest(model: model.path, maxTokens: 128))
            #expect(result.modality == "voice")
            #expect(result.modelCard?.capabilities == ModelCard.Capabilities(audioOutput: true))
            #expect(result.modelCard?.voices?.map(\.slug) == ["en_female"])
            #expect(result.modelCard?.voices?.first?.requiresReferenceAudio == false)
            #expect(result.tokenLimit.configuredMaximum == 128)
            #expect(result.loadRequest?.contextLength == nil)
            #expect(result.loadRequest?.prefillStepSize == nil)
            #expect(result.loadRequest?.kvCompression == nil)
            #expect(throws: (any Error).self) {
                try loader.validate(ModelLoadRequest(model: model.path, contextLength: 4096))
            }
            #expect(throws: (any Error).self) {
                try loader.validate(ModelLoadRequest(model: model.path, adapter: model.path))
            }
        }
    }

    @Test("Codestral and Devstral text metadata stay on the text backend")
    func mistralCodingPreflight() throws {
        try withFixture { directory in
            let loader = ModelLoader(defaultEngine: "cpu")
            for (name, type) in [("codestral", "mistral"), ("devstral", "ministral3")] {
                let model = try checkpoint(name, in: directory)
                try Data("{\"model_type\":\"\(type)\",\"max_position_embeddings\":32768}".utf8)
                    .write(to: model.appendingPathComponent("config.json"))
                #expect(try loader.validate(ModelLoadRequest(model: model.path)).modality == "text")
            }
        }
    }

    @Test("A new load resolves selected checkpoint settings without retaining earlier overrides")
    func freshSettingsOnEveryLoad() throws {
        try withFixture { directory in
            let first = try checkpoint("first", in: directory)
            let second = try checkpoint("second", in: directory)
            let settingsData = try JSONSerialization.data(withJSONObject: [
                "mlxRunner": [
                    "modelPath": first.path,
                    "servedModelName": "configured-first",
                    "maximumTokens": 1024,
                    "contextLength": 8192,
                    "prefillStepSize": 256,
                    "engine": "cpu",
                ]
            ])
            let settings = try JSONDecoder().decode(ModelStackSettings.self, from: settingsData)
            try Data(#"{"maximumTokens":96,"contextLength":2048}"#.utf8)
                .write(to: second.appendingPathComponent("midnight.json"))
            let loader = ModelLoader(settings: settings.mlxRunner)
            let firstLoad = try loader.validate(
                ModelLoadRequest(
                    model: first.path, name: "cli-first", maxTokens: 2048, prefillStepSize: 1024))
            #expect(firstLoad.servedModelName == "cli-first")
            #expect(firstLoad.tokenLimit.configuredMaximum == 2048)
            #expect(firstLoad.longContext.prefillStepSize == 1024)

            let secondLoad = try loader.validate(ModelLoadRequest(model: second.path))
            #expect(secondLoad.servedModelName == "second")
            #expect(secondLoad.tokenLimit.configuredMaximum == 96)
            #expect(secondLoad.longContext.contextLength == 2048)
            #expect(secondLoad.longContext.prefillStepSize == 512)
            #expect(secondLoad.engine == .cpu)
            #expect(secondLoad.selection.adapterPath == nil)
            #expect(secondLoad.dflashModel == nil)

            let firstAgain = try loader.validate(ModelLoadRequest(model: first.path))
            #expect(firstAgain.servedModelName == "configured-first")
            #expect(firstAgain.tokenLimit.configuredMaximum == 1024)
            #expect(firstAgain.longContext.prefillStepSize == 256)
        }
    }

    @Test("Missing model and adapter files and invalid limits fail before weights are loaded")
    func rejectsInvalidSelection() throws {
        try withFixture { directory in
            let model = try checkpoint("model", in: directory)
            let loader = ModelLoader(defaultEngine: "cpu")
            for request in [
                ModelLoadRequest(model: "   "),
                ModelLoadRequest(model: directory.appendingPathComponent("missing").path),
                ModelLoadRequest(model: model.path, adapter: directory.appendingPathComponent("missing-adapter").path),
                ModelLoadRequest(model: model.path, adapterScale: 1),
                ModelLoadRequest(model: model.path, maxTokens: 0),
                ModelLoadRequest(model: model.path, contextLength: 32769),
                ModelLoadRequest(model: model.path, prefillStepSize: 0),
                ModelLoadRequest(model: model.path, dflashBlockSize: 4),
                ModelLoadRequest(model: model.path, engine: "unknown"),
            ] {
                #expect(throws: (any Error).self) { try loader.validate(request) }
            }
            try FileManager.default.removeItem(at: model.appendingPathComponent("model.safetensors"))
            #expect(throws: (any Error).self) {
                try loader.validate(ModelLoadRequest(model: model.path))
            }
        }
    }

    @Test("Malformed checkpoint policy and incompatible drafter configuration are rejected")
    func rejectsPolicyAndDrafter() throws {
        try withFixture { directory in
            let model = try checkpoint("model", in: directory)
            let draft = try checkpoint("draft", in: directory)
            let loader = ModelLoader(defaultEngine: "cpu")
            #expect(throws: (any Error).self) {
                try loader.validate(ModelLoadRequest(model: model.path, dflashModel: draft.path))
            }
            try Data(#"{"maximumToken":512}"#.utf8)
                .write(to: model.appendingPathComponent("midnight.json"))
            #expect(throws: ModelStackSettingsError.self) {
                try loader.validate(ModelLoadRequest(model: model.path))
            }
        }
    }

    @Test("Load requests use camelCase and require a model")
    func wireFormat() throws {
        let request = try JSONDecoder().decode(
            ModelLoadRequest.self,
            from: Data(
                #"{"model":"local-model","maxTokens":128,"contextLength":4096,"dflashBlockSize":8}"#.utf8))
        #expect(request.model == "local-model")
        #expect(request.maxTokens == 128)
        #expect(request.contextLength == 4096)
        #expect(request.dflashBlockSize == 8)
        #expect(request.adapter == nil)
        #expect(try JSONDecoder().decode(ModelLoadRequest.self, from: JSONEncoder().encode(request)) == request)
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(ModelLoadRequest.self, from: Data("{}".utf8))
        }
    }

    private func withFixture(_ body: (URL) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("midnight-loader-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try body(directory)
    }

    private func checkpoint(_ name: String, in directory: URL) throws -> URL {
        let model = directory.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try Data(#"{"model_type":"llama","max_position_embeddings":32768}"#.utf8)
            .write(to: model.appendingPathComponent("config.json"))
        // Preflight reads metadata only. Deliberately unusable weights ensure
        // these tests never accidentally depend on model/GPU initialization.
        try Data().write(to: model.appendingPathComponent("model.safetensors"))
        return model
    }
}
