import Foundation
import MLX
import Testing

@testable import ModelRunnerCore

@Suite("Laguna expert-conditioned activation statistics", .serialized)
struct LagunaRoutedActivationStatisticsTests {
    @Test("Selected experts receive conditional inputs, counts, and actual distinct down activations")
    func conditionalMoments() throws {
        let recorder = try LagunaRoutedActivationRecorder(minimumExpertPositions: 2)
        let indices = MLXArray([Int32(0), 1, 1, 2]).reshaped(1, 2, 2)
        let sharedInput = MLXArray([Float(1), 2, 3, 4]).reshaped(1, 2, 1, 1, 2)
        let downInput = MLXArray([Float(2), 3, 4, 5, 6, 7, 8, 9]).reshaped(1, 2, 2, 1, 2)
        recorder.observeRoutedProjection(path: "gate_up", input: sharedInput, indices: indices, expertCount: 4)
        recorder.observeRoutedProjection(path: "down", input: downInput, indices: indices, expertCount: 4)
        let results = try recorder.finalize()
        let gate = try #require(results.first { $0.path == "gate_up" })
        let down = try #require(results.first { $0.path == "down" })
        #expect(gate.expertPositionCounts == [1, 2, 1, 0])
        #expect(gate.eligibleExperts == [false, true, false, false])
        #expect(gate.secondMoments.asArray(Float.self) == [1, 4, 5, 10, 9, 16, 0, 0])
        #expect(down.secondMoments.asArray(Float.self) == [4, 9, 26, 37, 64, 81, 0, 0])
    }

    @Test("Sorted switch inputs accumulate across segments without averaging unrelated experts")
    func sortedInputs() throws {
        let recorder = try LagunaRoutedActivationRecorder(minimumExpertPositions: 1)
        let ids = MLXArray([Int32(0), 1, 1, 2])
        let input = MLXArray([Float(1), 2, 1, 2, 3, 4, 3, 4]).reshaped(4, 1, 2)
        for _ in 0..<2 {
            recorder.observeRoutedProjection(path: "sorted", input: input, indices: ids, expertCount: 4)
            try recorder.evaluatePending()
        }
        let result = try #require(recorder.finalize().first)
        #expect(result.expertPositionCounts == [2, 4, 2, 0])
        #expect(result.secondMoments.asArray(Float.self) == [1, 4, 5, 10, 9, 16, 0, 0])
    }

    @Test("Bad expert IDs fail before an indexed accumulation can execute")
    func invalidIDs() throws {
        let recorder = try LagunaRoutedActivationRecorder()
        recorder.observeRoutedProjection(
            path: "invalid", input: MLXArray.ones([1, 1, 2]),
            indices: MLXArray([Int32(5)]), expertCount: 4)
        #expect(throws: LagunaActivationStatisticsError.self) { try recorder.finalize() }
    }

    @Test("Observer follows both native sorting paths without changing model outputs")
    func modelParity() throws {
        let data = Data(
            #"{"model_type":"laguna","vocab_size":64,"hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":64,"max_position_embeddings":256,"sliding_window":8,"layer_types":["full_attention","sliding_attention"],"mlp_layer_types":["dense","sparse"],"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":128,"shared_expert_intermediate_size":128}"#
                .utf8)
        let model = LagunaModel(try JSONDecoder().decode(LagunaConfiguration.self, from: data))
        for length in [3, 40] {
            let tokens = MLXArray(Array(repeating: Int32(1), count: length)).reshaped(1, length)
            let baseline = model(tokens, cache: nil)
            MLX.eval(baseline)
            let recorder = try LagunaRoutedActivationRecorder(minimumExpertPositions: 1)
            try model.setRoutedActivationObserver(recorder)
            let observed = LagunaRuntimeTuning.$useCompiledBlockTail.withValue(false) {
                model(tokens, cache: nil)
            }
            MLX.eval(observed)
            let results = try recorder.finalize()
            #expect(results.count == 2)
            for result in results {
                #expect(result.expertPositionCounts.reduce(0, +) == length * 2)
                #expect(result.secondMoments.shape == [4, 128])
            }
            #expect(MLX.allClose(baseline, observed, rtol: 0, atol: 0).item(Bool.self))
            try model.setRoutedActivationObserver(nil)
        }
    }
}
