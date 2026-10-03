import Foundation
import ModelRunnerProtocol
import Testing

@testable import Midnight

@Suite("Automatic speculative assistants")
struct AssistantDiscoveryTests {
    @Test("Installed Gemma assistants resolve by compatibility and survive load-request replay")
    func installed() throws {
        try fixture { target, root, assistantData in
            let assistant = try checkpoint(
                root.appendingPathComponent("google--gemma-4-26B-A4B-it-assistant"), assistantData)
            let loader = ModelLoader(defaultEngine: "metal", assistantsDirectory: root)
            let result = try loader.validate(ModelLoadRequest(model: target.path))
            #expect(samePath(result.gemmaAssistantModel, assistant))
            #expect(result.protectedDirectories.contains { samePath($0.path, assistant) })
            #expect(try loader.validate(#require(result.loadRequest)).gemmaAssistantModel == assistant.path)
            #expect(
                try loader.validate(ModelLoadRequest(model: target.path, autoAssistant: false)).gemmaAssistantModel
                    == nil)
            #expect(try loader.validate(ModelLoadRequest(model: target.path, engine: "cpu")).gemmaAssistantModel == nil)

            // The QAT assistant shares dimensions but is not the original target's pairing.
            let qat = try checkpoint(
                root.appendingPathComponent("google--gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"), assistantData)
            #expect(samePath(try loader.validate(ModelLoadRequest(model: target.path)).gemmaAssistantModel, assistant))
            try Data(
                #"{"name":"Converted","source_model":"/cache/models--google--gemma-4-26B-A4B-it-qat-q4_0/snapshots/revision"}"#
                    .utf8
            )
            .write(to: target.appendingPathComponent(ModelCard.filename))
            #expect(samePath(try loader.validate(ModelLoadRequest(model: target.path)).gemmaAssistantModel, qat))
            let settings = try JSONDecoder().decode(
                ModelStackSettings.self,
                from: Data(#"{"mlxRunner":{"autoAssistant":false}}"#.utf8))
            let disabled = ModelLoader(settings: settings.mlxRunner, defaultEngine: "metal", assistantsDirectory: root)
            #expect(try disabled.validate(ModelLoadRequest(model: target.path)).gemmaAssistantModel == nil)
            #expect(
                samePath(
                    try disabled.validate(ModelLoadRequest(model: target.path, autoAssistant: true))
                        .gemmaAssistantModel, qat))
            let explicit = try checkpoint(root.appendingPathComponent("explicit"), assistantData)
            #expect(
                try loader.validate(ModelLoadRequest(model: target.path, gemmaAssistantModel: explicit.path))
                    .gemmaAssistantModel == explicit.path)
            #expect(
                try loader.validate(
                    ModelLoadRequest(model: target.path, gemmaAssistantModel: explicit.path, autoAssistant: false)
                ).gemmaAssistantModel == explicit.path)
        }
    }

    @Test("Cards choose installed assistants and mark missing repositories for download without network access")
    func cardReferences() throws {
        try fixture { target, root, assistantData in
            let loader = ModelLoader(defaultEngine: "metal", assistantsDirectory: root)
            let card = ModelCard(name: "Gemma", assistantModel: "publisher/matching-assistant")
            try JSONEncoder().encode(card).write(to: target.appendingPathComponent(ModelCard.filename))
            let missing = try loader.validate(ModelLoadRequest(model: target.path))
            #expect(missing.pendingAssistant?.repository == "publisher/matching-assistant")
            #expect(missing.gemmaAssistantModel == nil)
            #expect(missing.modelCard?.assistantModel == card.assistantModel)
            #expect(
                try loader.validate(ModelLoadRequest(model: target.path, autoAssistant: false)).pendingAssistant == nil)
            let assistant = try checkpoint(root.appendingPathComponent("publisher--matching-assistant"), assistantData)
            #expect(try loader.validate(ModelLoadRequest(model: target.path)).gemmaAssistantModel == assistant.path)
            try Data(#"{"model_type":"llama"}"#.utf8).write(to: assistant.appendingPathComponent("config.json"))
            #expect(throws: (any Error).self) { try loader.validate(ModelLoadRequest(model: target.path)) }
        }
    }

    @Test("Downloaded Hugging Face cards identify assistants; ambiguous references do not trigger downloads")
    func readmeReferences() throws {
        try fixture { target, root, _ in
            let readme = target.appendingPathComponent("README.md")
            try Data("ASSISTANT_MODEL_ID = \"google/gemma-4-26B-A4B-it-assistant\"\n".utf8).write(to: readme)
            let loader = ModelLoader(defaultEngine: "metal", assistantsDirectory: root)
            #expect(
                try loader.validate(ModelLoadRequest(model: target.path)).pendingAssistant?.repository
                    == "google/gemma-4-26B-A4B-it-assistant")
            try Data("[Assistant](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant)".utf8).write(to: readme)
            #expect(try loader.validate(ModelLoadRequest(model: target.path)).pendingAssistant != nil)
            try Data("assistant_model = \"a/one\"\nassistant_model = \"b/two\"".utf8).write(to: readme)
            #expect(try loader.validate(ModelLoadRequest(model: target.path)).pendingAssistant == nil)
        }
    }

    @Test("Incomplete and ambiguous installed candidates are skipped; embedded Muse assistants are discovered")
    func candidateSelection() throws {
        try fixture { target, root, assistantData in
            let one = try checkpoint(root.appendingPathComponent("one"), assistantData)
            _ = try checkpoint(root.appendingPathComponent("two"), assistantData)
            let loader = ModelLoader(defaultEngine: "metal", assistantsDirectory: root)
            #expect(try loader.validate(ModelLoadRequest(model: target.path)).gemmaAssistantModel == nil)
            try FileManager.default.removeItem(at: one.appendingPathComponent("weights.safetensors"))
            #expect(
                try loader.validate(ModelLoadRequest(model: target.path)).gemmaAssistantModel?.hasSuffix("/two") == true
            )

            var targetObject =
                try JSONSerialization.jsonObject(with: Data(contentsOf: target.appendingPathComponent("config.json")))
                as! [String: Any]
            targetObject["model_type"] = "muse_glimmer"
            try JSONSerialization.data(withJSONObject: targetObject).write(
                to: target.appendingPathComponent("config.json"))
            let muse = Data(
                #"{"model_type":"muse_glimmer_assistant","architectures":["MuseGlimmerAssistantModel"],"hidden_size":2816,"block_size":4,"mask_token_id":1,"sliding_window":1024,"target_layer_ids":[0,1],"num_hidden_layers":1,"layer_types":["sliding_attention"]}"#
                    .utf8)
            let embedded = try checkpoint(target.appendingPathComponent("drafter"), muse)
            let result = try loader.validate(ModelLoadRequest(model: target.path))
            #expect(result.dflashModel == embedded.path)
            #expect(result.gemmaAssistantModel == nil)
        }
    }

    @Test("Assistant staging publishes only compatible complete checkpoints and cleans up failures")
    func staging() async throws {
        let base = FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(
            UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: base) }
        let (targetData, assistantData) = try configs()
        let destination = base.appendingPathComponent("publisher--assistant")
        let selection = AssistantDiscovery.Selection(
            kind: .gemma, directory: destination, repository: "publisher/assistant")
        await #expect(throws: (any Error).self) {
            try await AssistantDownloader.install(
                selection, target: targetData, blockSize: nil, quantizationBits: nil,
                transfer: { _, staging, validate in
                    let invalid = Data(#"{"model_type":"llama"}"#.utf8)
                    try invalid.write(to: staging.appendingPathComponent("config.json"))
                    try validate(invalid)
                })
        }
        #expect(!FileManager.default.fileExists(atPath: destination.path))
        #expect(try FileManager.default.contentsOfDirectory(atPath: base.path).isEmpty)
        try await AssistantDownloader.install(
            selection, target: targetData, blockSize: nil, quantizationBits: nil,
            transfer: { _, staging, validate in
                try validate(assistantData)
                try assistantData.write(to: staging.appendingPathComponent("config.json"))
                try Data().write(to: staging.appendingPathComponent("weights.safetensors"))
            })
        #expect(FileManager.default.fileExists(atPath: destination.appendingPathComponent("config.json").path))
        // An already installed assistant does not redownload or overwrite files.
        try await AssistantDownloader.install(
            selection, target: targetData, blockSize: nil, quantizationBits: nil,
            transfer: { _, _, _ in throw ModelLoadingError("Unexpected transfer") })
    }

    private func fixture(_ body: (URL, URL, Data) throws -> Void) throws {
        let base = FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(
            UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: base) }
        let (targetData, assistantData) = try configs()
        let target = try checkpoint(base.appendingPathComponent("gemma-4-26B-A4B-it-midnight"), targetData)
        let root = base.appendingPathComponent("drafters")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try body(target, root, assistantData)
    }

    private func samePath(_ path: String?, _ expected: URL) -> Bool {
        guard let path else {
            return false
        }
        return URL(fileURLWithPath: path).resolvingSymlinksInPath().path == expected.resolvingSymlinksInPath().path
    }

    @discardableResult private func checkpoint(_ directory: URL, _ config: Data) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try config.write(to: directory.appendingPathComponent("config.json"))
        try Data().write(to: directory.appendingPathComponent("weights.safetensors"))
        return directory
    }

    private func configs() throws -> (Data, Data) {
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
        return (
            try JSONSerialization.data(withJSONObject: ["model_type": "gemma4", "text_config": text]),
            try JSONSerialization.data(withJSONObject: [
                "model_type": "gemma4_assistant", "text_config": draft,
                "backbone_hidden_size": 2816, "block_size": 4,
            ])
        )
    }
}
