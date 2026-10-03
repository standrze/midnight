import Foundation
import MLX
import MLXLLM
import MLXNN
import Testing

@_spi(Testing) @testable import MLXLMCommon
@testable import ModelRunnerCore

@Suite("Adaptive MTP iterator correctness", .serialized)
struct MTPAdaptiveIteratorTests {
    @Test("A drafter capped at one position remains target-only")
    func singlePositionDrafter() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = ConstantTarget()
            let drafter = ConstantDrafter(maximumBlockSize: 1)
            var iterator = try MTPSpeculativeTokenIterator(
                input: LMInput(tokens: MLXArray(Array(repeating: 1, count: 6))),
                mainModel: target, drafter: drafter,
                parameters: GenerateParameters(maxTokens: 4, temperature: 0), blockSize: 8)
            #expect(iterator.blockSize == 1)
            #expect(iterator._nextAdaptiveDraftCountForTesting == nil)
            var tokens: [Int] = []
            while let token = iterator.next() {
                tokens.append(token)
            }
            #expect(tokens == [2, 2, 2, 2])
            #expect(drafter.widths.isEmpty)
        }
    }

    @Test("Early stop removes accepted lookahead after ring wrap", arguments: [7, 10, 17, 24])
    func acceptedLookahead(_ stopAfter: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let target = ConstantTarget()
            let drafter = ConstantDrafter()
            var iterator = try MTPSpeculativeTokenIterator(
                input: LMInput(tokens: MLXArray(Array(repeating: 1, count: 6))),
                mainModel: target, drafter: drafter,
                parameters: GenerateParameters(maxTokens: 40, temperature: 0), blockSize: 8)
            let clock = Clock()
            iterator._setAdaptiveDraftPolicyForTesting(.init(maximumBlockSize: 8), clock: clock.now)
            for _ in 0..<stopAfter {
                #expect(iterator.next() == 2)
            }
            let before = iterator.mainCacheStorage.processedTokenCount
            #expect(before > 6 + stopAfter, "This case must leave committed, unconsumed lookahead")
            iterator.finalizeGeneration()
            #expect(iterator.mainCacheStorage.processedTokenCount == 6 + stopAfter)
            #expect(iterator.mainCacheStorage.nativeAttentionOffsetsAreAligned)
            #expect(drafter.widths.contains(4))
            if stopAfter >= 17 {
                #expect(drafter.widths.contains(8))
            }
        }
    }

    @Test("Changing round widths preserves greedy tokens and final cache", arguments: [2, 5, 17, 40])
    func tokenAndCacheParity(_ stopAfter: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let (model, drafter) = try fixture()
            let input = LMInput(tokens: MLXArray(Array(1...6)))
            let parameters = GenerateParameters(maxTokens: 40, temperature: 0)
            var baseline = try TokenIterator(input: input, model: model, parameters: parameters)
            var expected: [Int] = []
            while expected.count < stopAfter, let token = baseline.next() {
                expected.append(token)
            }
            var iterator = try MTPSpeculativeTokenIterator(
                input: input, mainModel: model, drafter: drafter, parameters: parameters, blockSize: 8)
            let clock = Clock()
            iterator._setAdaptiveDraftPolicyForTesting(.init(maximumBlockSize: 8), clock: clock.now)
            var tokens: [Int] = []
            var widths: Set<Int> = []
            while tokens.count < stopAfter {
                let priorProposed = iterator.proposedCount
                guard let token = iterator.next() else {
                    break
                }
                tokens.append(token)
                let drafted = iterator.proposedCount - priorProposed
                if drafted > 0 {
                    widths.insert(drafted)
                }
            }
            iterator.finalizeGeneration()
            #expect(tokens == expected)
            #expect(iterator.blockSize == 8)
            #expect(widths.allSatisfy { (1...7).contains($0) })
            if stopAfter == 40 {
                #expect(widths.contains(1) && widths.contains(3) && widths.contains(7))
            }
            let committed = iterator.mainCacheStorage.processedTokenCount
            #expect((6 + tokens.count - 1...6 + tokens.count).contains(committed))
            #expect(iterator.mainCacheStorage.nativeAttentionOffsetsAreAligned)
        }
    }

    @Test("Processors and stochastic sampling bypass adaptation", arguments: [false, true])
    func unsupportedSampling(_ processor: Bool) throws {
        try Device.withDefaultDevice(.cpu) {
            let (model, drafter) = try fixture()
            var iterator = try MTPSpeculativeTokenIterator(
                input: LMInput(tokens: MLXArray(Array(1...6))), mainModel: model, drafter: drafter,
                parameters: GenerateParameters(maxTokens: 8, temperature: processor ? 0 : 0.8), blockSize: 4)
            let clock = Clock()
            iterator._setAdaptiveDraftPolicyForTesting(.init(maximumBlockSize: 4), clock: clock.now)
            if processor {
                iterator._setProcessorForTesting(IdentityProcessor())
            }
            while iterator.next() != nil {}
            #expect(clock.calls == 0)
            #expect(iterator._nextAdaptiveDraftCountForTesting == 1)
            #expect(iterator.proposedCount > 0)
        }
    }

    @Test("A one-token tail does not call the adaptive clock")
    func tailBudget() throws {
        try Device.withDefaultDevice(.cpu) {
            let (model, drafter) = try fixture()
            var iterator = try MTPSpeculativeTokenIterator(
                input: LMInput(tokens: MLXArray(Array(1...6))), mainModel: model, drafter: drafter,
                parameters: GenerateParameters(maxTokens: 2, temperature: 0), blockSize: 8)
            let clock = Clock()
            iterator._setAdaptiveDraftPolicyForTesting(.init(maximumBlockSize: 8), clock: clock.now)
            #expect(iterator.next() != nil)
            #expect(iterator.next() != nil)
            #expect(iterator.next() == nil)
            #expect(iterator.proposedCount == 0)
            #expect(clock.calls == 0)
        }
    }

    @Test("The production pad processor trains adaptive widths without batching nonfinite rows")
    func statelessProductionProcessor() throws {
        try Device.withDefaultDevice(.cpu) {
            let parameters = GenerateParameters(maxTokens: 40, temperature: 0)
            let components = GenerationComponents(logitProcessorFactory: { SuppressTokenLogitProcessor(tokenID: 0) })
            #expect(components.logitProcessor(parameters: parameters) is any MTPStatelessLogitProcessor)
            let input = LMInput(tokens: MLXArray(Array(repeating: 1, count: 6)))
            let target = ConstantTarget(nonfiniteEveryOtherRow: true)
            let drafter = ConstantDrafter()
            var baseline = try TokenIterator(
                input: input, model: target, parameters: parameters, components: components)
            var expected: [Int] = []
            while let token = baseline.next() {
                expected.append(token)
            }
            var iterator = try MTPSpeculativeTokenIterator(
                input: input, mainModel: target, drafter: drafter, parameters: parameters,
                blockSize: 8, components: components)
            let enabled = ProcessInfo.processInfo.environment["MIDNIGHT_MTP_ADAPTIVE_DRAFTS"] == "1"
            #expect((iterator._nextAdaptiveDraftCountForTesting != nil) == enabled)
            let clock = Clock()
            iterator._setAdaptiveDraftPolicyForTesting(.init(maximumBlockSize: 8), clock: clock.now)
            var actual: [Int] = []
            while let token = iterator.next() {
                actual.append(token)
            }
            iterator.finalizeGeneration()
            #expect(actual == expected)
            #expect(Set(actual) == Set([1, 2]), "Finite rows suppress the winning pad; nonfinite rows select EOS")
            #expect(drafter.widths.contains(2) && drafter.widths.contains(4) && drafter.widths.contains(8))
            #expect(clock.calls >= 12, "Sequential verification must finish timing and train every warmed width")
            #expect(clock.calls.isMultiple(of: 2))
            #expect(iterator.mainCacheStorage.nativeAttentionOffsetsAreAligned)
        }
    }

    @Test("Stateful and chained processors retain fixed widths", arguments: [0, 1, 2])
    func statefulProcessorGates(_ kind: Int) throws {
        try Device.withDefaultDevice(.cpu) {
            let components: GenerationComponents
            switch kind {
            case 0:
                components = GenerationComponents(logitProcessorFactory: { IdentityProcessor() })
            case 1:
                components = GenerationComponents(logitProcessorFactory: { SuppressTokenLogitProcessor(tokenID: 0) })
                    .appendingLogitProcessor { ForcedTokenPrefixLogitProcessor(tokenIDs: [2, 3]) }
            default:
                components = GenerationComponents(logitProcessorFactory: { SuppressTokenLogitProcessor(tokenID: 0) })
                    .appendingLogitProcessor { SuppressTokenLogitProcessor(tokenID: 0) }
            }
            let parameters = GenerateParameters(maxTokens: 20, temperature: 0)
            #expect(!(components.logitProcessor(parameters: parameters) is any MTPStatelessLogitProcessor))
            let drafter = ConstantDrafter()
            var iterator = try MTPSpeculativeTokenIterator(
                input: LMInput(tokens: MLXArray(Array(repeating: 1, count: 6))),
                mainModel: ConstantTarget(), drafter: drafter, parameters: parameters,
                blockSize: 8, components: components)
            #expect(iterator._nextAdaptiveDraftCountForTesting == nil)
            let clock = Clock()
            iterator._setAdaptiveDraftPolicyForTesting(.init(maximumBlockSize: 8), clock: clock.now)
            while iterator.next() != nil {}
            #expect(clock.calls == 0)
            #expect(iterator._nextAdaptiveDraftCountForTesting == 1)
            #expect(drafter.widths.contains(8))
        }
    }

    @Test("Built-in repetition penalties prevent stateless custom-processor adaptation")
    func penaltyProcessorGate() throws {
        try Device.withDefaultDevice(.cpu) {
            let components = GenerationComponents(logitProcessorFactory: { SuppressTokenLogitProcessor(tokenID: 0) })
            let parameters = GenerateParameters(maxTokens: 16, temperature: 0, repetitionPenalty: 1.1)
            #expect(!(components.logitProcessor(parameters: parameters) is any MTPStatelessLogitProcessor))
            let drafter = ConstantDrafter()
            var iterator = try MTPSpeculativeTokenIterator(
                input: LMInput(tokens: MLXArray(Array(repeating: 1, count: 6))),
                mainModel: ConstantTarget(), drafter: drafter, parameters: parameters,
                blockSize: 8, components: components)
            #expect(iterator._nextAdaptiveDraftCountForTesting == nil)
            let clock = Clock()
            iterator._setAdaptiveDraftPolicyForTesting(.init(maximumBlockSize: 8), clock: clock.now)
            while iterator.next() != nil {}
            #expect(clock.calls == 0)
            #expect(drafter.widths.contains(8))
        }
    }

    private func fixture() throws -> (Gemma4TextModel, Gemma4TextAssistantDraftModel) {
        let text: [String: Any] = [
            "model_type": "gemma4_text", "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 8, "global_head_dim": 8, "sliding_window": 4, "vocab_size": 32,
            "layer_types": ["sliding_attention", "full_attention"],
            "hidden_size_per_layer_input": 0, "num_kv_shared_layers": 0,
        ]
        let targetData = try JSONSerialization.data(withJSONObject: text)
        let draftData = try JSONSerialization.data(withJSONObject: [
            "text_config": text, "backbone_hidden_size": 16, "block_size": 8,
            "tie_word_embeddings": true, "use_ordered_embeddings": false,
        ])
        return (
            Gemma4TextModel(try JSONDecoder().decode(Gemma4TextConfiguration.self, from: targetData)),
            Gemma4TextAssistantDraftModel(
                try JSONDecoder().decode(Gemma4TextAssistantConfiguration.self, from: draftData))
        )
    }

    private final class Clock {
        private(set) var calls = 0
        func now() -> TimeInterval {
            defer { calls += 1 }
            return Double(calls)
        }
    }

    private struct IdentityProcessor: LogitProcessor {
        mutating func prompt(_ prompt: MLXArray) {}
        func process(logits: MLXArray) -> MLXArray { logits }
        mutating func didSample(token: MLXArray) {}
    }

    private final class ConstantDrafter: Module, MTPDrafterModel {
        private(set) var widths: [Int] = []
        let maximumBlockSize: Int?

        init(maximumBlockSize: Int? = nil) {
            self.maximumBlockSize = maximumBlockSize
            super.init()
        }

        func draftBlock(
            target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
            sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?,
            queryOffset: Int, blockSize: Int, sampler: any LogitSampler
        ) -> MLXArray {
            widths.append(blockSize)
            return MLXArray(Array(repeating: Int32(2), count: blockSize - 1)).reshaped([1, blockSize - 1])
        }
    }

    /// Writes real cache rows and accepts every draft by default. Its optional
    /// alternating nonfinite rows exercise rejection and per-position EOS fallback.
    private final class ConstantTarget: Module, LanguageModel {
        let nonfiniteEveryOtherRow: Bool

        init(nonfiniteEveryOtherRow: Bool = false) {
            self.nonfiniteEveryOtherRow = nonfiniteEveryOtherRow
            super.init()
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            [KVCacheSimple(), RotatingKVCache(maxSize: 8, keep: 0)]
        }

        func prepare(
            _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
        ) throws -> PrepareResult {
            .tokens(input.text)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            callAsFunction(LMInput(tokens: inputs).text, cache: cache, state: nil).logits
        }

        func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
            let length = input.tokens.dim(-1)
            let base = cache?.first?.offset ?? 0
            let logits: MLXArray
            if nonfiniteEveryOtherRow {
                let values = (base..<(base + length)).flatMap { position -> [Float] in
                    position.isMultiple(of: 2) ? [.nan, -.infinity, .infinity, .nan] : [200, 0, 100, 0]
                }
                logits = MLXArray(values).reshaped([1, length, 4])
            } else {
                logits = broadcast(MLXArray([Float(0), 0, 100, 0]).reshaped([1, 1, 4]), to: [1, length, 4])
            }
            let rows = MLXArray((base..<(base + length)).map { Float($0) }).reshaped([1, 1, length, 1])
            let presented = cache?.map { $0.update(keys: rows, values: rows) } ?? []
            guard state?[mtpEmitFlagKey] == true, presented.count == 2 else {
                return LMOutput(logits: logits)
            }
            var emitted = LMOutput.State()
            emitted[mtpLastHiddenStatesKey] = MLXArray.zeros([1, length, 2])
            emitted[mtpSharedKVStatesKey] = ["full_attention": presented[0], "sliding_attention": presented[1]]
            emitted[mtpSharedKVSourceIndicesKey] = ["full_attention": 0, "sliding_attention": 1]
            return LMOutput(logits: logits, state: emitted)
        }
    }
}
