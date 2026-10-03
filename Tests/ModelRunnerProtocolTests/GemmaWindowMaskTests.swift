import Foundation
import MLX
import MLXLLM
import Testing

@Suite("Gemma retained-cache sliding attention")
struct GemmaWindowMaskTests {
    @Test("Chunked prefill and decode match full causal attention", arguments: [1, 2, 4, 8], [false, true])
    func chunkedAttention(_ chunk: Int, _ sliceWindow: Bool) throws {
        try Gemma4RuntimeTuning.$useWindowSlicing.withValue(sliceWindow) {
            try Device.withDefaultDevice(.cpu) {
                let config = try JSONDecoder().decode(
                    Gemma4TextConfiguration.self,
                    from: Data(
                        """
                        {"model_type":"gemma4_text","hidden_size":16,"intermediate_size":32,
                         "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
                         "head_dim":8,"global_head_dim":8,"sliding_window":4,"vocab_size":32,
                         "layer_types":["sliding_attention","full_attention"],
                         "hidden_size_per_layer_input":0,"num_kv_shared_layers":0}
                        """.utf8))
                let model = Gemma4TextModel(config)
                let tokens = MLXArray(Array(1...10)).reshaped(1, 10)
                let expected = model(tokens, cache: nil)
                eval(expected)
                let cache = try model.newCache(parameters: nil)
                var pieces: [MLXArray] = []
                for start in stride(from: 0, to: 10, by: chunk) {
                    let logits = model(tokens[0..., start..<min(start + chunk, 10)], cache: cache)
                    eval(logits)
                    pieces.append(logits)
                }
                let actual = concatenated(pieces, axis: 1)
                let error = MLX.max(abs(expected - actual)).item(Float.self)
                #expect(error < 0.0001)
            }
        }
    }
}
