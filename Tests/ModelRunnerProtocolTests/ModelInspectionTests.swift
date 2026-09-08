import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import ModelRunnerProtocol
import Testing

@testable import ModelRunnerCore

@Suite("Model inspection", .serialized)
struct ModelInspectionTests {
  @Test("Questions and output limits are bounded before inference")
  func boundsRequests() throws {
    #expect(try ModelInspection.maximumTokens(.init(question: "Hello")) == 32)
    for request in [InspectorTraceRequest(question: "  "), .init(question: "Hello", maxTokens: 0),
                    .init(question: "Hello", maxTokens: 65), .init(question: String(repeating: "x", count: 16_385))] {
      #expect(throws: ModelInspectionError.self) { try ModelInspection.maximumTokens(request) }
    }
  }

  @Test("Metadata preserves nested attention and expert configuration")
  func decodesMetadata() {
    let metadata = ModelInspectionMetadata.decode(Data(#"{"model_type":"mistral3","text_config":{"model_type":"ministral3","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"num_local_experts":8,"num_experts_per_tok":2,"layer_types":["full_attention","sliding_attention"]}}"#.utf8))
    #expect(metadata.modelType == "ministral3")
    #expect(metadata.hiddenSize == 64)
    #expect(metadata.attentionHeads == 4)
    #expect(metadata.kvHeads == 2)
    #expect(metadata.expertCount == 8)
    #expect(metadata.expertsPerToken == 2)
    #expect(metadata.layerTypes == ["full_attention", "sliding_attention"])
    #expect(InspectionLayerAddress.parse("language_model.model.layers.12.mlp.gate_proj") ==
      InspectionLayerAddress(index: 12, path: "language_model.model.layers.12"))
    #expect(InspectionLayerAddress.parse("lm_head") == nil)
  }

  @Test("Harmony semantic stops require confirmed tokens and the GPT-OSS runtime")
  func validatesHarmonyStops() {
    let tokenizer = InspectionTestTokenizer(controlTokens: ["<|ghissue|>": 11, "<|fim_suffix|>": 12,
      "<unk>": 0], unknownToken: "<unk>")
    #expect(ModelInspection.semanticStopTokenIDs(tokenizer: tokenizer, isGPTOSS: true) == [11, 12])
    #expect(ModelInspection.semanticStopTokenIDs(tokenizer: tokenizer, isGPTOSS: false).isEmpty)
    let renamed = InspectionTestTokenizer(controlTokens: ["<|call|>": 21, "<|return|>": 22])
    #expect(ModelInspection.semanticStopTokenIDs(tokenizer: renamed, isGPTOSS: true) == [21, 22])
  }

  @Test("Loaded topology describes packed tensors without logical parameter inflation")
  func describesQuantizedTopology() throws {
    try Device.withDefaultDevice(.cpu) {
      let model = makeModel()
      quantize(model: model, groupSize: 32, bits: 4)
      let descriptor = descriptor(model)
      #expect(descriptor.layerCount == 2)
      #expect(descriptor.traceSupported)
      #expect(descriptor.hiddenSize == 64)
      #expect(descriptor.storedElementCount == model.parameters().flattened().reduce(0) { $0 + $1.1.size })
      #expect(descriptor.weightBytes == model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes })
      #expect(descriptor.layers[0].modules.contains { $0.kind == "QuantizedLinear" })
      let quantized = descriptor.layers[0].modules.filter { $0.kind == "QuantizedLinear" }
      #expect(quantized.allSatisfy { $0.quantization == .init(format: "affine", bits: 4, groupSize: 32) })
      let data = try JSONEncoder().encode(descriptor)
      #expect(try JSONDecoder().decode(InspectorModel.self, from: data) == descriptor)
    }
  }

  @Test("Residual summaries measure known tensor values")
  func measuresResidualValues() throws {
    try Device.withDefaultDevice(.cpu) {
      let recorder = InspectionActivationRecorder()
      recorder.register(path: "model.layers.0.input_layernorm", layer: 0, site: "layer_input")
      recorder.observe(path: "model.layers.0.input_layernorm", input: MLXArray([Float(3), 4, 0, -2]).reshaped(1, 2, 2))
      try recorder.flush()
      let samples = try recorder.finalize(tokenCount: 2)[0].samples
      #expect(abs(samples[0].rms - sqrt(12.5)) < 0.00001)
      #expect(samples[0].maxAbs == 4)
      #expect(abs(samples[1].rms - sqrt(2)) < 0.00001)
      #expect(samples[1].maxAbs == 2)
      #expect(samples[0].channels.count == 16)
      #expect(samples[0].channels[0] == 3)
      #expect(samples[0].channels[15] == 4)
      #expect(throws: ModelInspectionError.self) { try recorder.finalize(tokenCount: 3) }
    }
  }

  @Test("Inspection restores exact norms and preserves quantized model logits")
  func restoresModel() throws {
    try Device.withDefaultDevice(.cpu) {
      let model = makeModel()
      quantize(model: model, groupSize: 32, bits: 4)
      let description = descriptor(model)
      let tokens = MLXArray([Int32(1), 2]).reshaped(1, 2)
      let baseline = model(tokens, cache: nil)
      eval(baseline)
      let originals = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
      let recorder = InspectionActivationRecorder()
      let installation = try InspectionNormInstallation(model: model, layers: description.layers, recorder: recorder)
      let observed = model(tokens, cache: nil)
      eval(observed)
      try recorder.flush()
      installation.restore()
      #expect(allClose(baseline, observed, rtol: 0, atol: 0).item(Bool.self))
      #expect(try recorder.finalize(tokenCount: 2).count == 4)
      let restored = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
      #expect(restored.count == originals.count)
      for (path, original) in originals { #expect(restored[path] === original) }
    }
  }

  @Test("Short decode traces every displayed token and leaves the graph unchanged")
  func tracesTokenSequence() throws {
    try Device.withDefaultDevice(.cpu) {
      let model = makeModel()
      quantize(model: model, groupSize: 32, bits: 4)
      let before = descriptor(model)
      let context = ModelContext(configuration: ModelConfiguration(id: "inspection-test"), model: model,
        processor: InspectionTestProcessor(), tokenizer: InspectionTestTokenizer())
      let trace = try ModelInspection.trace(context: context, descriptor: before,
        question: "test", promptTokens: [1, 2, 3], maximumTokens: 2)
      #expect(trace.promptTokenCount == 3)
      #expect(trace.generatedTokenCount == 2)
      #expect(trace.tokens.count == 5)
      #expect(trace.predictions?.map(\.afterTokenIndex) == [2, 3, 4])
      #expect(trace.predictions?.allSatisfy { p in
        p.candidates.count == 5 && p.candidates.allSatisfy { $0.probability.isFinite && $0.probability >= 0 && $0.probability <= 1 }
          && zip(p.candidates, p.candidates.dropFirst()).allSatisfy { $0.probability >= $1.probability }
      } == true)
      #expect(trace.layers.count == 4)
      #expect(trace.layers.allSatisfy { $0.samples.map(\.tokenIndex) == [0, 1, 2, 3, 4] })
      #expect(descriptor(model) == before)
      #expect(trace.stopReason == "length")
      #expect(trace.outputKind == "raw")
    }
  }

  @Test("A failed model evaluation also restores original normalization modules")
  func restoresAfterFailure() throws {
    try Device.withDefaultDevice(.cpu) {
      let model = makeModel()
      model.update(parameters: model.parameters().mapValues { MLXArray.full($0.shape, values: MLXArray(Float.nan)) })
      let original = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
      let context = ModelContext(configuration: ModelConfiguration(id: "inspection-test"), model: model,
        processor: InspectionTestProcessor(), tokenizer: InspectionTestTokenizer())
      #expect(throws: ModelInspectionError.self) {
        try ModelInspection.trace(context: context, descriptor: descriptor(model),
          question: "test", promptTokens: [1, 2], maximumTokens: 1)
      }
      let restored = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
      #expect(restored.count == original.count)
      for (path, module) in original { #expect(restored[path] === module) }
    }
  }

  private func makeModel() -> Mistral3TextModel {
    Mistral3TextModel(Mistral3TextConfiguration(hiddenSize: 64, hiddenLayers: 2,
      intermediateSize: 128, attentionHeads: 4, rmsNormEps: 0.00001,
      vocabularySize: 64, kvHeads: 2, layerTypes: ["full_attention", "sliding_attention"], slidingWindow: 32))
  }

  private func descriptor(_ model: Mistral3TextModel) -> InspectorModel {
    ModelInspection.descriptor(model: model, id: "test", metadata: ModelInspectionMetadata(
      modelType: "ministral3", layerTypes: ["full_attention", "sliding_attention"]), contextLength: 128)
  }
}

private struct InspectionTestTokenizer: MLXLMCommon.Tokenizer {
  var controlTokens: [String: Int] = [:]
  var bosToken: String? { nil }
  var eosToken: String? { nil }
  var unknownToken: String? = nil
  func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1, 2, 3] }
  func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { tokenIds.map { "t\($0)" }.joined() }
  func convertTokenToId(_ token: String) -> Int? { controlTokens[token] ?? unknownToken.flatMap { controlTokens[$0] } }
  func convertIdToToken(_ id: Int) -> String? { controlTokens.first { $0.value == id }?.key ?? "t\(id)" }
  func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                         additionalContext: [String: any Sendable]?) throws -> [Int] { [1, 2, 3] }
}

private struct InspectionTestProcessor: UserInputProcessor {
  func prepare(input: UserInput) async throws -> LMInput { LMInput(tokens: MLXArray([Int32(1), 2, 3])) }
}
