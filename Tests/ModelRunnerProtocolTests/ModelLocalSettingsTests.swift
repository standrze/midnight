import Foundation
import ModelRunnerProtocol
import Testing

@Suite("Model-local runtime policy")
struct ModelLocalSettingsTests {
    func withModel(_ body: (URL) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("midnight-policy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try body(directory)
    }

    @Test("Local policy overrides matching stack settings and preserves other defaults")
    func overlay() throws {
        try withModel { directory in
            let stack = try JSONDecoder().decode(ModelStackSettings.self, from: Data("""
                {"mlxRunner":{"modelPath":"\(directory.path)","contextLength":8192,
                "maximumTokens":512,"prefillStepSize":256,"port":8088}}
                """.utf8))
            try Data(#"{"contextLength":32768,"maximumTokens":2048}"#.utf8)
                .write(to: directory.appendingPathComponent("midnight.json"))
            let resolved = try #require(stack.mlxRunner).resolving(
                for: ModelCatalog.resolveMLX(model: directory.path))
            #expect(resolved.contextLength == 32768)
            #expect(resolved.maximumTokens == 2048)
            #expect(resolved.prefillStepSize == 256)
            #expect(resolved.port == 8088)
            let profile = try ModelMemoryProfile(configuration: Data(#"{"max_position_embeddings":16384}"#.utf8),
                options: LongContextOptions(contextLength: 16384))
            #expect(throws: RequestAdmissionError.self) {
                try profile.validateContext(prompt: 16000, output: resolved.maximumTokens!)
            }
            #expect(throws: RequestAdmissionError.self) {
                try ModelMemoryProfile(configuration: Data(#"{"max_position_embeddings":16384}"#.utf8),
                    options: LongContextOptions(contextLength: resolved.contextLength))
            }
        }
    }

    @Test("Changing checkpoints drops the previous model policy, including output and drafter")
    func switching() throws {
        try withModel { directory in
            let stack = try JSONDecoder().decode(ModelStackSettings.self, from: Data(#"{"mlxRunner":{"modelPath":"/different/checkpoint","contextLength":32768,"maximumTokens":4096,"servedModelName":"old","dflashModelPath":"/old/draft","port":8088}}"#.utf8))
            let selection = ModelCatalog.resolveMLX(model: directory.path)
            let resolved = try #require(stack.mlxRunner).resolving(for: selection)
            #expect(resolved.contextLength == nil)
            #expect(resolved.maximumTokens == nil)
            #expect(resolved.servedModelName == nil)
            #expect(resolved.dflashModelPath == nil)
            #expect(resolved.port == 8088)
            #expect(try ModelStackSettings.MLXRunner.empty.resolving(for: selection).contextLength == nil)
        }
    }

    @Test("Bundle root policy is discovered, and malformed policies fail visibly")
    func bundlesAndErrors() throws {
        try withModel { directory in
            for child in ["base-model", "adapter"] {
                try FileManager.default.createDirectory(at: directory.appendingPathComponent(child), withIntermediateDirectories: true)
            }
            let selection = ModelCatalog.resolveMLX(model: directory.path)
            #expect(selection.settingsDirectory == directory.path)
            let file = directory.appendingPathComponent("midnight.json")
            try Data(#"{"contextLength":4096}"#.utf8).write(to: file)
            #expect(try ModelStackSettings.MLXRunner.empty.resolving(for: selection).contextLength == 4096)
            for invalid in [#"{"contextLenght":4096}"#, #"{"contextLength":"4096"}"#, "[]", "{"] {
                try Data(invalid.utf8).write(to: file)
                #expect(throws: ModelStackSettingsError.self) {
                    try ModelStackSettings.MLXRunner.empty.resolving(for: selection)
                }
            }
        }
    }
}
