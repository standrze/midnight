import Foundation
import ModelRunnerProtocol
import Testing

@Suite("Long-context admission")
struct LongContextOptionsTests {
    func profile(_ json: String, options: LongContextOptions = try! .init()) throws -> ModelMemoryProfile {
        try ModelMemoryProfile(configuration: Data(json.utf8), options: options)
    }

    @Test("Context includes requested output, and the caller cannot extend model limits")
    func contextBounds() throws {
        let json =
            #"{"max_position_embeddings":4096,"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2}"#
        let p = try profile(json)
        try p.validateContext(prompt: 3584, output: 512)
        #expect(throws: RequestAdmissionError.self) { try p.validateContext(prompt: 3585, output: 512) }
        #expect(throws: RequestAdmissionError.self) { try p.validateContext(prompt: Int.max, output: 512) }
        #expect(throws: RequestAdmissionError.self) { try profile(json, options: .init(contextLength: 8192)) }
        #expect(try profile(json, options: .init(contextLength: 2048)).contextLength == 2048)
    }

    @Test("Native sliding windows stop growing while full-attention layers continue")
    func hybridBudget() throws {
        let p = try profile(
            #"{"max_position_embeddings":32768,"hidden_size":64,"head_dim":16,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"sliding_window":512,"layer_types":["full_attention","sliding_attention"]}"#
        )
        #expect(p.hasKnownGeometry)
        #expect(p.kvBytes(tokens: 100) == 2 * 100 * 2 * 16 * 8)
        #expect(p.kvBytes(tokens: 4096) == (4096 + 512) * 2 * 16 * 8)
        #expect(p.kvBytes(tokens: 8192) - p.kvBytes(tokens: 4096) == 4096 * 2 * 16 * 8)
    }

    @Test("Nested text geometry and unknown topology remain conservatively bounded")
    func nestedAndFallback() throws {
        let p = try profile(
            #"{"text_config":{"max_position_embeddings":8192,"num_hidden_layers":4,"num_attention_heads":8,"hidden_size":256}}"#
        )
        #expect(p.contextLength == 8192)
        #expect(p.layerWindows.count == 4)
        let unknown = try profile("{}")
        #expect(unknown.contextLength == 4096)
        #expect(!unknown.hasKnownGeometry)
        #expect(unknown.kvBytes(tokens: 4096) > 0)
        #expect(try profile(#"{"max_position_embeddings":true}"#).contextLength == 4096)
    }

    @Test("Explicit bounded Gemma storage changes only sliding-layer admission")
    func boundedGemmaStorage() throws {
        let data = Data(
            #"{"model_type":"gemma4_text","num_hidden_layers":2,"num_attention_heads":8,"num_key_value_heads":2,"num_global_key_value_heads":1,"head_dim":256,"global_head_dim":512,"sliding_window":1024,"layer_types":["sliding_attention","full_attention"]}"#
                .utf8)
        let bounded = try ModelMemoryProfile(configuration: data, options: .init(), boundedGemmaSlidingCache: true)
        let retained = try ModelMemoryProfile(configuration: data, options: .init())
        #expect(bounded.kvBytes(tokens: 512) == retained.kvBytes(tokens: 512))
        #expect(bounded.kvBytes(tokens: 8192) == (1024 + 8192) * 512 * 8)
        #expect(retained.kvBytes(tokens: 8192) == 2 * 8192 * 512 * 8)
    }

    @Test("Gemma 4 budgets retained history and distinct global head dimensions")
    func gemmaRetainedHistory() throws {
        for type in ["gemma4", "gemma4_text", "gemma4_unified"] {
            let p = try profile(
                """
                {"model_type":"\(type)","text_config":{"num_hidden_layers":2,
                "num_attention_heads":32,"num_key_value_heads":16,"num_global_key_value_heads":4,
                "head_dim":256,"global_head_dim":512,"sliding_window":1024,
                "layer_types":["sliding_attention","full_attention"]}}
                """)
            #expect(p.layerWindows == [nil, nil])
            #expect(p.layerBytesPerToken == [16 * 256 * 8, 4 * 512 * 8])
            #expect(p.kvBytes(tokens: 8192) == 8192 * (16 * 256 + 4 * 512) * 8)
            #expect(p.kvBytes(tokens: 16384) == 2 * p.kvBytes(tokens: 8192))
        }
        let unknown = try profile(
            #"{"model_type":"gemma4_text","num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":256,"global_head_dim":512}"#
        )
        #expect(unknown.layerBytesPerToken == [2 * 512 * 8, 2 * 512 * 8])
    }

    @Test("Budgets saturate on overflow and include workspace plus resident weights")
    func arithmetic() throws {
        #expect(ModelMemoryProfile.add(Int.max, 1) == Int.max)
        #expect(ModelMemoryProfile.multiply(Int.max, 2) == Int.max)
        let p = try profile("{}")
        #expect(p.requestBytes(prompt: Int.max, output: 1, prefillStepSize: 512, residentBytes: 100) == Int.max)
        #expect(
            p.requestBytes(prompt: 1024, output: 256, prefillStepSize: 512, residentBytes: 100)
                > p.kvBytes(tokens: 1280))
        #expect(p.workspaceBytes(prefillStepSize: 2048) > p.workspaceBytes(prefillStepSize: 512))
    }

    @Test("Invalid settings fail before model loading")
    func options() throws {
        for size in [0, -1, 8193, Int.max] {
            #expect(throws: RequestAdmissionError.self) { try LongContextOptions(prefillStepSize: size) }
        }
        #expect(throws: RequestAdmissionError.self) { try LongContextOptions(contextLength: 0) }
        #expect(throws: RequestAdmissionError.self) { try LongContextOptions(kvCompression: "invalid") }
        #expect(try LongContextOptions().compression == .none)
        for value in LongContextOptions.Compression.allCases {
            #expect(try LongContextOptions(kvCompression: value.rawValue).compression == value)
        }
    }
}
