import Foundation
import MLXHuggingFace
import MLXLMCommon
import ModelRunnerProtocol
import Testing
import Tokenizers

@Suite("Nimble publisher prompt parity")
struct DecisionPromptParityTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_RUN_NIMBLE_SMOKE"] == "1"))
    func publisherTokenIdentity() async throws {
        let root = URL(fileURLWithPath: "/Users/stephen/Documents/ChatGPT/midnight/benchmark-results/nimble-20261002")
        let request = try DecisionRequest.decode(Data(contentsOf: root.appendingPathComponent("request.json")))
        let contract = try JSONDecoder().decode(
            DecisionModelContract.self,
            from: Data(
                contentsOf: URL(
                    fileURLWithPath: "/Users/stephen/.midnight/models/nimble-9b/adapter-native/decision-model.json")))
        let expected = try JSONDecoder().decode(
            [String: [Int]].self, from: Data(contentsOf: root.appendingPathComponent("publisher-prompt-tokens.json")))
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: "/Users/stephen/.midnight/models/nimble-9b/base"))
        for field in request.names {
            let text = try DecisionPrompt.text(request: request, field: field, contract: contract)
            #expect(tokenizer.encode(text: text, addSpecialTokens: false) == expected[field])
        }
    }
}
