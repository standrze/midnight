import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

/// Opt-in characterization of the native decoder used by the frozen quantization campaign.
@Suite("Gemma quote boundary diagnostic")
struct GemmaQuoteBoundaryDiagnosticTests {
    @Test(
        "Installed tokenizer replay distinguishes cleanup from model weights",
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_GEMMA_TOKENIZER_PROBE_MODEL"] != nil))
    func installedTokenizerReplay() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["MIDNIGHT_GEMMA_TOKENIZER_PROBE_MODEL"],
            let outputPath = environment["MIDNIGHT_GEMMA_TOKENIZER_PROBE_OUTPUT"]
        else { return }

        let source = URL(fileURLWithPath: modelPath, isDirectory: true)
        let manager = FileManager.default
        let temporary = manager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try manager.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: temporary) }
        try manager.createSymbolicLink(
            at: temporary.appendingPathComponent("tokenizer.json"),
            withDestinationURL: source.appendingPathComponent("tokenizer.json"))
        let original = try Data(contentsOf: source.appendingPathComponent("tokenizer_config.json"))
        var configuration = try #require(JSONSerialization.jsonObject(with: original) as? [String: Any])
        #expect(configuration["tokenizer_class"] as? String == "GemmaTokenizer")
        #expect(configuration["clean_up_tokenization_spaces"] == nil)
        let tokenIDs = [236782, 236789, 78676, 236789, 236764, 756, 3738, 236789, 236783]
        var records = [[String: Any]]()

        for cleanup in ["absent", "false", "true"] {
            configuration.removeValue(forKey: "clean_up_tokenization_spaces")
            if cleanup != "absent" { configuration["clean_up_tokenization_spaces"] = cleanup == "true" }
            let data = try JSONSerialization.data(withJSONObject: configuration, options: [.sortedKeys])
            try data.write(to: temporary.appendingPathComponent("tokenizer_config.json"))
            let native = try await AutoTokenizer.from(modelFolder: temporary)
            let tokenizer = #adaptHuggingFaceTokenizer(native)
            let batch = tokenizer.decode(tokenIds: tokenIDs)
            var stream = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            var chunks = [String]()
            for token in tokenIDs {
                stream.append(token: token)
                if let chunk = stream.next() { chunks.append(chunk) }
            }
            let streamed = chunks.joined()
            if cleanup != "true" {
                #expect(streamed == batch)
                #expect(batch == "{'outcome', ' source'}")
            } else {
                #expect(batch == "{'outcome','source'}")
                #expect(streamed == "{'outcome', ''source'}")
                #expect(streamed != batch)
            }
            records.append([
                "cleanup": cleanup, "token_ids": tokenIDs,
                "token_pieces": tokenIDs.map { tokenizer.convertIdToToken($0) ?? "<missing>" },
                "batch": batch, "stream": streamed, "chunks": chunks,
                "stream_matches_batch": streamed == batch,
            ])
        }
        let report: [String: Any] = [
            "scope": "actual native tokenizer and streaming adapter replay; no generated token capture",
            "model": modelPath, "records": records,
        ]
        let output = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try output.write(to: URL(fileURLWithPath: outputPath), options: [.withoutOverwriting])
    }
}
