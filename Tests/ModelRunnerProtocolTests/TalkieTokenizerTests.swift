import Foundation
import Testing
import Tokenizers

@Suite("Talkie tokenizer compatibility")
struct TalkieTokenizerTests {
    private var fixtureDirectory: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/TalkieTokenizer", isDirectory: true)
    }

    @Test("The standalone Talkie template preserves role IDs and adds no BOS")
    func chatFraming() async throws {
        let tokenizer = try await AutoTokenizer.from(modelFolder: fixtureDirectory)
        let tokens = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "Hello"]])

        #expect(tokens == [65537, 72, 22882, 65536, 65538])
        #expect(tokenizer.bosTokenId == nil)
        #expect(tokenizer.eosTokenId == 65535)
        #expect(
            tokenizer.decode(tokens: tokens, skipSpecialTokens: false)
                == "<|user|>Hello<|end|><|assistant|>")
    }

    @Test("System messages and prior assistant turns use the checkpoint template")
    func multipleTurns() async throws {
        let tokenizer = try await AutoTokenizer.from(modelFolder: fixtureDirectory)
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "Hello"],
            ["role": "user", "content": "Hello"],
            ["role": "assistant", "content": "Hello"],
            ["role": "user", "content": "Hello"],
        ]
        let tokens = try tokenizer.applyChatTemplate(messages: messages)
        #expect(
            tokens == [
                65539, 72, 22882, 65536, 65537, 72, 22882, 65536,
                65538, 72, 22882, 65536, 65537, 72, 22882, 65536, 65538,
            ])
        let withoutGenerationPrompt = try tokenizer.applyChatTemplate(
            messages: messages, chatTemplate: nil, addGenerationPrompt: false,
            truncation: false, maxLength: nil, tools: nil)
        #expect(withoutGenerationPrompt == Array(tokens.dropLast()))
    }

    @Test("Unicode, punctuation, contractions and digits match the reference tokenizer")
    func unicodeBPEParity() async throws {
        let tokenizer = try await AutoTokenizer.from(modelFolder: fixtureDirectory)
        let text = "Café naïve résumé — 1930\nDon't change 123456."
        // Produced by tokenizers 0.22.2 using the full, unmodified DWQ tokenizer.
        let expected = [
            67, 1063, 1238, 6931, 46392, 316, 406, 10163, 417, 1238, 461,
            32, 5748, 48, 10, 17939, 2656, 32, 6276, 16040, 46,
        ]
        #expect(tokenizer.encode(text: text) == expected)
        #expect(tokenizer.decode(tokens: expected, skipSpecialTokens: false) == text)
    }
}
