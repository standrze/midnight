import Foundation
import MLXHuggingFace
import Testing
import Tokenizers

@testable import MLXLMCommon

@Suite("Extra EOS vocabulary identity")
struct ExtraEOSTokenIdentityTests {
    private func tokenizer() async throws -> any MLXLMCommon.Tokenizer {
        let directory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/UnknownFallbackTokenizer", isDirectory: true)
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        return #adaptHuggingFaceTokenizer(tokenizer)
    }

    @Test("A real BPE unknown fallback does not introduce an accidental stop")
    func missingStringIsNotAnUnknownStop() async throws {
        let tokenizer = try await tokenizer()
        #expect(tokenizer.convertTokenToId("<turn|>") == 3)
        #expect(tokenizer.convertIdToToken(3) == "<unk>")
        let configuration = ModelConfiguration(id: "fixture", extraEOSTokens: ["<turn|>", "<end_of_turn>"])

        let stops = buildStopTokenIds(modelConfiguration: configuration, tokenizer: tokenizer)

        #expect(stops == [1, 106])
        #expect(!stops.contains(3))
        #expect(configuration.effectiveStopStrings == ["<turn|>", "<end_of_turn>"])
    }

    @Test("An explicitly named unknown token remains a legitimate EOS")
    func explicitlyNamedUnknownStop() async throws {
        let tokenizer = try await tokenizer()
        let configuration = ModelConfiguration(id: "fixture", extraEOSTokens: ["<unk>"])

        #expect(buildStopTokenIds(modelConfiguration: configuration, tokenizer: tokenizer) == [1, 3])
    }

    @Test("Configured numeric EOS IDs remain authoritative, including unknown")
    func configuredNumericStops() async throws {
        let tokenizer = try await tokenizer()
        var configuration = ModelConfiguration(id: "fixture", extraEOSTokens: ["<turn|>"])
        configuration.eosTokenIds = [3, 99]

        #expect(buildStopTokenIds(modelConfiguration: configuration, tokenizer: tokenizer) == [1, 3, 99])
    }

    @Test("A multi-token decoded stop string is retained without adding unknown EOS")
    func decodedStopStringsAreUnchanged() async throws {
        let tokenizer = try await tokenizer()
        var configuration = ModelConfiguration(id: "fixture", extraEOSTokens: ["a b"])
        configuration.stopStrings = ["a b", "b a"]
        #expect(tokenizer.convertTokenToId("a b") == 3)

        #expect(buildStopTokenIds(modelConfiguration: configuration, tokenizer: tokenizer) == [1])
        #expect(configuration.extraEOSTokens == ["a b"])
        #expect(configuration.effectiveStopStrings == ["a b", "b a"])
    }
}
