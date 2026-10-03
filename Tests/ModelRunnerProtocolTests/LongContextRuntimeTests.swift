import Foundation
import MLX
import MLXLMCommon
import ModelRunnerProtocol
import Testing

@testable import ModelRunnerCore

@Suite("Long-context runtime", .serialized)
struct LongContextRuntimeTests {
  @Test("Compression preserves native windows and runs through chunked prefill and decode")
  func hybridCompression() throws {
    try Device.withDefaultDevice(.gpu) {
      let data = Data(#"{"model_type":"laguna","vocab_size":64,"hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":64,"max_position_embeddings":256,"sliding_window":8,"layer_types":["full_attention","sliding_attention"],"mlp_layer_types":["dense","sparse"],"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":128,"shared_expert_intermediate_size":128}"#.utf8)
      let model = LagunaModel(try JSONDecoder().decode(LagunaConfiguration.self, from: data))
      for compression in LongContextOptions.Compression.allCases {
        let options = try LongContextOptions(prefillStepSize: 4, kvCompression: compression.rawValue)
        var parameters = GenerateParameters(maxTokens: 16, temperature: 0)
        try options.apply(to: &parameters)
        #expect(parameters.prefill.stepSize == 4)
        let status = try model.cacheStatus(parameters: parameters)
        #expect(status.attentionMaxSizes == [nil, 8])
        if compression != .none {
          #expect(status.pendingLayerCount + status.compressedLayerCount == 1)
          #expect(status.skippedLayerCount == 1)
        }
        var iterator = try TokenIterator(input: LMInput(tokens: MLXArray(Array(Int32(1)...24))),
          model: model, parameters: parameters)
        var tokens: [Int] = []
        while let token = iterator.next() { tokens.append(token) }
        #expect(tokens.count == 16)
        #expect(tokens.allSatisfy { (0..<64).contains($0) })
      }
    }
  }
}
