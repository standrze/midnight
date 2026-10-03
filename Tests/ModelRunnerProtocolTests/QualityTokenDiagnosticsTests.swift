import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import ModelRunnerCore

@Suite("Fixed-input quality token diagnostics")
struct QualityTokenDiagnosticsTests {
    @Test("Winner, near tie and target diagnostics refer to the same next-token positions")
    func nextTokenIdentity() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = QualityDiagnosticModel()
            let plain = try ModelQualityScoring.scoreNLL(tokens: [0, 1, 2, 3], model: model, prefillStepSize: 1)
            let scored = try ModelQualityScoring.scoreNLL(
                tokens: [0, 1, 2, 3], model: model, prefillStepSize: 1, tokenDiagnostics: true)
            let rows = try #require(scored.tokenDiagnostics)
            #expect(plain.tokenDiagnostics == nil)
            #expect(plain.nllSum == scored.nllSum)
            #expect(rows.map(\.position) == [1, 2, 3])
            #expect(rows.map(\.referenceTokenID) == [1, 2, 3])
            #expect(rows.map(\.winnerTokenID) == [0, 0, 3])
            #expect(rows.map(\.runnerUpTokenID) == [1, 1, 0])
            #expect(abs(rows[0].winnerMargin - 0.0001) < 0.000001)
            #expect(rows[1].winnerMargin == 0)
            #expect(rows[2].winnerMargin == 3)
            #expect(rows.reduce(0) { $0 + $1.referenceNLL } == scored.nllSum)
        }
    }

    @Test("Diagnostics require one-token model calls")
    func requireDecodeStep() throws {
        try Device.withDefaultDevice(.cpu) {
            for step in [0, 2, 512] {
                #expect(throws: ModelQualityScoringError.self) {
                    try ModelQualityScoring.scoreNLL(
                        tokens: [0, 1, 2, 3], model: QualityDiagnosticModel(),
                        prefillStepSize: step, tokenDiagnostics: true)
                }
            }
        }
    }
}

private final class QualityDiagnosticModel: Module, LLMModel {
    let vocabularySize = 4
    var loraLayers: [Module] { [] }
    let table = MLXArray(
        [Float(4), 3.9999, 0, -2, 1, 1, 0, -2, 0, 0, 0, 3, 0, 0, 0, 3], [4, 4])

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] { [] }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        table[inputs]
    }
}
