import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import ModelRunnerProtocol
import Testing
import Tokenizers

@testable import MLXLMCommon
@testable import ModelRunnerCore

/// Captures tokens through the ordinary native text loop without changing the checkpoint.
@Suite("Gemma generated token diagnostic")
struct GemmaGeneratedTokenDiagnosticTests {
    private struct Input: Decodable {
        let id: String
        let prompt: String
    }

    private struct Previous: Decodable {
        struct Sample: Decodable {
            let id: String
            let generatedText: String
            let promptTokenIDFingerprint: String

            enum CodingKeys: String, CodingKey {
                case id
                case generatedText = "generated_text"
                case promptTokenIDFingerprint = "prompt_token_id_fingerprint"
            }
        }
        let samples: [Sample]
    }

    private struct Capture: Codable, Sendable {
        let modelPath: String
        let memoryLimitBytes: Int
        let cacheLimitBytes: Int
        let taskID: String
        let promptTokenIDs: [Int]
        let promptFingerprint: String
        let generatedTokenIDs: [Int]
        let streamedContent: String
        let streamedReasoning: String
        let batchDecoded: String
        let naiveStreamDecoded: String
        let stopReason: String
        let matchesPriorStreamedContent: Bool
    }

    @Test(
        "Frozen failing task retains prompt identity and captures original token IDs",
        .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_GEMMA_GENERATED_PROBE_MODEL"] != nil))
    func captureFrozenTask() async throws {
        let environment = ProcessInfo.processInfo.environment
        let modelPath = try #require(environment["MIDNIGHT_GEMMA_GENERATED_PROBE_MODEL"])
        let corpusPath = try #require(environment["MIDNIGHT_GEMMA_GENERATED_PROBE_CORPUS"])
        let previousPath = try #require(environment["MIDNIGHT_GEMMA_GENERATED_PROBE_PREVIOUS"])
        let outputPath = try #require(environment["MIDNIGHT_GEMMA_GENERATED_PROBE_OUTPUT"])
        let taskID = environment["MIDNIGHT_GEMMA_GENERATED_PROBE_TASK"] ?? "cyber-remediation-code-03"
        #expect(!FileManager.default.fileExists(atPath: outputPath))
        let decoder = JSONDecoder()
        let inputs = try String(contentsOfFile: corpusPath, encoding: .utf8).split(separator: "\n").map {
            try decoder.decode(Input.self, from: Data($0.utf8))
        }
        let input = try #require(inputs.first { $0.id == taskID })
        let prior = try decoder.decode(Previous.self, from: Data(contentsOf: URL(fileURLWithPath: previousPath)))
        let sample = try #require(prior.samples.first { $0.id == taskID })
        let prompt = input.prompt
        let expectedFingerprint = sample.promptTokenIDFingerprint
        let expectedContent = sample.generatedText

        let capture = try await MLXPinnedRuntime.shared.run {
            try await Device.withDefaultDevice(.gpu) {
                let limits = try MLXResourceLimits.resolve(
                    for: .metal, physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                    environment: [:])
                try MLXResourceGuard.apply(limits)
                let root = URL(fileURLWithPath: modelPath)
                let container = try await LLMModelFactory.shared.loadContainer(
                    from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                    configuration: LocalModelChatConfiguration.make(directory: root, modelType: "gemma4"))
                let prepared = try await container.prepare(
                    input: UserInput(
                        prompt: .messages([
                            ["role": "user", "content": prompt]
                        ])))
                var promptIDs = prepared.text.tokens.asArray(Int32.self).map(Int.init)
                if Array(promptIDs.prefix(3)) == [2, 107, 105] { promptIDs.remove(at: 1) }
                let fingerprint = Self.fingerprint(promptIDs)
                // Refuse mismatched framing before any model generation.
                guard fingerprint == expectedFingerprint else {
                    throw CocoaError(.validationMissingMandatoryProperty)
                }
                var parameters = GenerateParameters(maxTokens: 2048, temperature: 0, topP: 1, topK: 64)
                parameters.prefill.stepSize = 512
                let fixedParameters = parameters
                let promptCount = promptIDs.count
                let components = GenerationComponents(
                    logitProcessorFactory: { SuppressTokenLogitProcessor(tokenID: 0) })
                let (stream, producer, tokenizer) = try await container.perform(
                    nonSendable: LMInput(tokens: MLXArray(promptIDs))
                ) { context, input in
                    let iterator = try TokenIterator(
                        input: input, model: context.model, parameters: fixedParameters, components: components)
                    let result = generateTaskRecordingTokens(
                        promptTokenCount: promptCount, modelConfiguration: context.configuration,
                        tokenizer: context.tokenizer, iterator: iterator)
                    return (result.0, result.1, context.tokenizer)
                }
                var content = ""
                var reasoning = ""
                var stopReason = "missing"
                for await event in stream {
                    switch event {
                    case .chunk(let text): content += text
                    case .reasoning(let text): reasoning += text
                    case .info(let info): stopReason = String(describing: info.stopReason)
                    case .toolCall, .rejectedToolCall:
                        Issue.record("Unexpected tool event in code diagnostic")
                    }
                }
                let tokens = await producer.value
                var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
                var rawStream = ""
                for token in tokens {
                    detokenizer.append(token: token)
                    if let chunk = detokenizer.next() { rawStream += chunk }
                }
                return Capture(
                    modelPath: modelPath, memoryLimitBytes: limits.memoryLimitBytes,
                    cacheLimitBytes: limits.cacheLimitBytes, taskID: taskID, promptTokenIDs: promptIDs,
                    promptFingerprint: fingerprint, generatedTokenIDs: tokens,
                    streamedContent: content, streamedReasoning: reasoning,
                    batchDecoded: tokenizer.decode(tokenIds: tokens), naiveStreamDecoded: rawStream,
                    stopReason: stopReason, matchesPriorStreamedContent: content == expectedContent)
            }
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(capture).write(to: URL(fileURLWithPath: outputPath), options: [.withoutOverwriting])
        #expect(capture.matchesPriorStreamedContent)
        #expect(capture.promptFingerprint == expectedFingerprint)
        #expect(!capture.generatedTokenIDs.isEmpty)
        #expect(capture.stopReason != "missing")
    }

    private static func fingerprint(_ tokens: [Int]) -> String {
        var value: UInt64 = 0xcbf2_9ce4_8422_2325
        for integer in [UInt64(tokens.count)] + tokens.map({ UInt64(bitPattern: Int64($0)) }) {
            for shift in stride(from: 0, through: 56, by: 8) {
                value ^= UInt64(UInt8(truncatingIfNeeded: integer >> UInt64(shift)))
                value &*= 0x100_0000_01b3
            }
        }
        return String(format: "fnv1a64:%016llx", value)
    }
}
