import Foundation
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

@Suite("Gemma decoded whitespace defaults")
struct GemmaCleanupDefaultTests {
    @Test(arguments: ["GemmaTokenizer", "GemmaTokenizerFast"])
    func absentFlagPreservesCode(tokenizerClass: String) async throws {
        let tokenizer = try await fixture(tokenizerClass: tokenizerClass, cleanup: nil)
        let quoteIDs = [1, 2, 3, 2, 4, 5, 6, 2, 7]
        #expect(tokenizer.decode(tokenIds: quoteIDs) == "{'outcome', ' source'}")
        #expect(stream(quoteIDs, tokenizer: tokenizer) == tokenizer.decode(tokenIds: quoteIDs))
        let indentationAndUnicode = [10, 11, 12, 8, 9, 8, 4]
        #expect(tokenizer.decode(tokenIds: indentationAndUnicode) == "\n  café . ,")
        #expect(stream(indentationAndUnicode, tokenizer: tokenizer) == "\n  café . ,")
    }

    @Test(arguments: [true, false])
    func explicitFlagRemainsAuthoritative(cleanup: Bool) async throws {
        let tokenizer = try await fixture(tokenizerClass: "GemmaTokenizer", cleanup: cleanup)
        #expect(tokenizer.decode(tokenIds: [12, 8, 9]) == (cleanup ? "café." : "café ."))
    }

    @Test
    func otherFamilyRetainsItsDefault() async throws {
        let tokenizer = try await fixture(tokenizerClass: "GPT2Tokenizer", cleanup: nil)
        #expect(tokenizer.decode(tokenIds: [12, 8, 9]) == "café.")
    }

    private func stream(_ tokens: [Int], tokenizer: any MLXLMCommon.Tokenizer) -> String {
        var decoder = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        var text = ""
        for token in tokens {
            decoder.append(token: token)
            if let chunk = decoder.next() { text += chunk }
        }
        return text
    }

    private func fixture(tokenizerClass: String, cleanup: Bool?) async throws -> any MLXLMCommon.Tokenizer {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var config: [String: Any] = ["tokenizer_class": tokenizerClass, "unk_token": "<unk>"]
        if let cleanup { config["clean_up_tokenization_spaces"] = cleanup }
        let pieces = ["<unk>", "{", "'", "outcome", ",", "▁'", "▁source", "}", "▁", ".", "\n", "▁▁", "café"]
        let vocab = Dictionary(uniqueKeysWithValues: pieces.enumerated().map { ($0.element, $0.offset) })
        let data: [String: Any] = [
            "added_tokens": [],
            "model": ["type": "BPE", "vocab": vocab, "merges": [], "unk_token": "<unk>", "byte_fallback": false],
            "decoder": [
                "type": "Sequence",
                "decoders": [
                    ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
                    ["type": "ByteFallback"], ["type": "Fuse"],
                ],
            ],
        ]
        try JSONSerialization.data(withJSONObject: config).write(
            to: directory.appendingPathComponent("tokenizer_config.json"))
        try JSONSerialization.data(withJSONObject: data).write(to: directory.appendingPathComponent("tokenizer.json"))
        let native = try await AutoTokenizer.from(modelFolder: directory)
        return #adaptHuggingFaceTokenizer(native)
    }
}
