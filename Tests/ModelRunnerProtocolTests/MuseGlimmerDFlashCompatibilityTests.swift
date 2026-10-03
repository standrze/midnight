import Foundation
import Testing

@testable import ModelRunnerCore

@Suite("Muse Glimmer DFlash checkpoint contract")
struct MuseGlimmerDFlashCompatibilityTests {
    @Test("Official Muse target and assistant metadata agree")
    func validatesOfficialShape() throws {
        let descriptor = try MuseGlimmerDFlashCompatibility.validate(
            target: Data(target.utf8), assistant: Data(assistant.utf8), blockSize: nil)
        #expect(descriptor.hiddenSize == 6656)
        #expect(descriptor.vocabularySize == 202048)
        #expect(descriptor.targetLayerIDs == [1, 13, 25, 37, 49])
        #expect(descriptor.blockSize == 16)
    }

    @Test("Rejects an invalid requested block size before weight loading")
    func rejectsInvalidBlockSize() {
        #expect(throws: (any Error).self) {
            try MuseGlimmerDFlashCompatibility.validate(
                target: Data(target.utf8), assistant: Data(assistant.utf8), blockSize: 17)
        }
    }

    @Test("Assistant runtime decodes the published block architecture")
    func decodesAssistantRuntime() throws {
        let configuration = try JSONDecoder().decode(
            MuseGlimmerAssistantConfiguration.self, from: Data(runtimeAssistant.utf8))
        #expect(configuration.hiddenLayers == 5)
        #expect(configuration.attentionHeads == 32)
        #expect(configuration.keyValueHeads == 8)
        #expect(configuration.targetLayerIDs == [1, 13, 25, 37, 49])
    }

    private let target =
        #"{"model_type":"muse_glimmer","text_config":{"hidden_size":6656,"vocab_size":202048,"num_hidden_layers":52}}"#
    private let assistant =
        #"{"model_type":"muse_glimmer_assistant","architectures":["MuseGlimmerAssistantModel"],"hidden_size":6656,"num_hidden_layers":5,"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],"block_size":16,"mask_token_id":201818,"sliding_window":2048,"target_layer_ids":[1,13,25,37,49]}"#
    private let runtimeAssistant =
        #"{"model_type":"muse_glimmer_assistant","architectures":["MuseGlimmerAssistantModel"],"hidden_size":6656,"intermediate_size":19968,"num_hidden_layers":5,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"rms_norm_eps":0.00001,"rope_parameters":{"rope_theta":500000.0},"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],"block_size":16,"mask_token_id":201818,"sliding_window":2048,"target_layer_ids":[1,13,25,37,49]}"#
}
