import Foundation
import MLX
import MLXNN
import Testing

@testable import ModelRunnerCore

@Suite("Bounded Laguna layerwise calibration", .serialized)
struct LagunaLayerwiseCalibrationTests {
  @Test("Native one-layer traversal with BF16 spool matches full-model logits and expert moments")
  func layerwiseMatchesFullModel() throws {
    let data = Data(#"{"model_type":"laguna","vocab_size":64,"hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":64,"max_position_embeddings":256,"sliding_window":8,"layer_types":["full_attention","sliding_attention"],"mlp_layer_types":["dense","sparse"],"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":128,"shared_expert_intermediate_size":128}"#.utf8)
    let configuration = try JSONDecoder().decode(LagunaConfiguration.self, from: data)
    let model = LagunaModel(configuration)
    let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map {
      ($0.0, $0.1.asType(.bfloat16))
    })
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    let fullObserver = try LagunaRoutedActivationRecorder(minimumExpertPositions: 1)
    try model.setRoutedActivationObserver(fullObserver)
    let segments = [MLXArray([Int32(1), 2, 3]).reshaped(1, 3),
      MLXArray((0..<40).map { Int32(($0 % 60) + 1) }).reshaped(1, 40)]
    let fullLogits = segments.map { model($0, cache: nil) }
    try MLX.checkedEval(fullLogits)
    let fullStatistics = try fullObserver.finalize()
    try model.setRoutedActivationObserver(nil)
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("laguna-layerwise-parity-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let embedding = try #require(weights["language_model.model.embed_tokens.weight"])
    var inputs = [URL]()
    for (index, tokens) in segments.enumerated() {
      let url = root.appendingPathComponent("embedding-\(index).safetensors")
      try MLX.save(arrays: ["hidden": embedding[tokens]], url: url)
      inputs.append(url)
    }
    let observer = try LagunaRoutedActivationRecorder(minimumExpertPositions: 1)
    for layer in 0..<configuration.calibrationLayerCount {
      let prefix = "language_model.model.layers.\(layer)."
      let block = try LagunaCalibrationBlock(configuration: configuration, layerIndex: layer,
        sourceWeights: weights.filter { $0.key.hasPrefix(prefix) })
      try block.setRoutedActivationObserver(observer)
      var nextInputs = [URL]()
      for (index, input) in inputs.enumerated() {
        let hidden = try #require(MLX.loadArrays(url: input)["hidden"])
        #expect(hidden.shape[1] == segments[index].shape[1])
        let output = block(hidden)
        let url = root.appendingPathComponent("layer-\(layer)-\(index).safetensors")
        try MLX.save(arrays: ["hidden": output], url: url)
        nextInputs.append(url)
      }
      inputs = nextInputs
    }
    let norm = try #require(weights["language_model.model.norm.weight"])
    let head = try #require(weights["language_model.lm_head.weight"])
    for (index, input) in inputs.enumerated() {
      let hidden = try #require(MLX.loadArrays(url: input)["hidden"])
      let normalized = MLXFast.rmsNorm(hidden, weight: norm, eps: configuration.calibrationRMSNormEpsilon)
      let logits = MLX.matmul(normalized, head.T)
      #expect(MLX.allClose(logits, fullLogits[index], rtol: 0, atol: 0).item(Bool.self))
    }
    let statistics = try observer.finalize()
    #expect(statistics.map(\.path) == fullStatistics.map(\.path))
    for (actual, expected) in zip(statistics, fullStatistics) {
      #expect(actual.expertPositionCounts == expected.expertPositionCounts)
      #expect(MLX.allClose(actual.secondMoments, expected.secondMoments, rtol: 1e-5, atol: 1e-6).item(Bool.self))
    }
  }

  @Test("Layerwise teacher rejects non-BF16 routed weights even when every dense Linear is BF16")
  func rejectsWrongExpertDType() throws {
    let data = Data(#"{"model_type":"laguna","vocab_size":64,"hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":64,"max_position_embeddings":256,"sliding_window":8,"layer_types":["full_attention","sliding_attention"],"mlp_layer_types":["dense","sparse"],"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":128,"shared_expert_intermediate_size":128}"#.utf8)
    let configuration = try JSONDecoder().decode(LagunaConfiguration.self, from: data)
    let model = LagunaModel(configuration)
    let prefix = "language_model.model.layers.1."
    let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened()
      .filter { $0.0.hasPrefix(prefix) }.map { ($0.0, $0.1.asType(.bfloat16)) })
    for invalidDType in [DType.float16, .float32] {
      var invalid = weights
      for key in invalid.keys where key.contains(".switch_mlp.") && key.hasSuffix(".weight") {
        invalid[key] = weights[key]!.asType(invalidDType)
      }
      #expect(throws: LagunaActivationStatisticsError.self) {
        try LagunaCalibrationBlock(configuration: configuration, layerIndex: 1, sourceWeights: invalid)
      }
    }
  }

  @Test("Selective reader accepts HF blob symlinks but rejects lexical index escapes")
  func symlinkAndTraversal() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("hf-reader-\(UUID().uuidString)")
    let snapshot = root.appendingPathComponent("snapshot")
    try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let original = MLXArray([Float(1), 2, 3]).asType(.bfloat16)
    let blob = root.appendingPathComponent("blob.safetensors")
    try MLX.save(arrays: ["weight": original], url: blob)
    try FileManager.default.createSymbolicLink(at: snapshot.appendingPathComponent("weights.safetensors"),
      withDestinationURL: blob)
    let index = snapshot.appendingPathComponent("model.safetensors.index.json")
    try JSONSerialization.data(withJSONObject: ["weight_map": ["weight": "weights.safetensors"]]).write(to: index)
    let reader = try SelectiveSafetensorsReader(directory: snapshot)
    #expect(MLX.arrayEqual(try reader.read("weight"), original).item(Bool.self))
    try JSONSerialization.data(withJSONObject: ["weight_map": ["weight": "../blob.safetensors"]]).write(to: index)
    let invalid = try SelectiveSafetensorsReader(directory: snapshot)
    #expect(throws: LagunaActivationStatisticsError.self) { try invalid.read("weight") }
  }

  @Test("Selective reads own their BF16 bytes after each tensor pool drains and the source is replaced")
  func selectiveReaderOwnsPayloadAfterPoolDrain() throws {
    try Device.withDefaultDevice(.cpu) {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent("reader-pool-lifetime-\(UUID().uuidString)")
      try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
      defer { try? FileManager.default.removeItem(at: root) }
      let shard = root.appendingPathComponent("weights.safetensors")
      try JSONSerialization.data(withJSONObject: ["weight_map": ["weight": "weights.safetensors"]])
        .write(to: root.appendingPathComponent("model.safetensors.index.json"))
      let reader = try SelectiveSafetensorsReader(directory: root)
      var loaded = [MLXArray]()
      var expected = [[Float]]()
      for index in 0..<8 {
        let values = (0..<16_384).map { Float(($0 + index * 17) % 128) / 8 }
        let original = MLXArray(values).reshaped([128, 128]).asType(.bfloat16)
        try MLX.save(arrays: ["weight": original], url: shard)
        // read() drains its own Foundation pool before returning. Delay all
        // assertions until later reads have reused temporary storage.
        loaded.append(try reader.read("weight"))
        expected.append(values) // Multiples of 1/8 below 16 are exact in BF16.
      }
      try FileManager.default.removeItem(at: shard)
      for (array, values) in zip(loaded, expected) {
        #expect(array.dtype == .bfloat16)
        #expect(array.shape == [128, 128])
        #expect(array.asType(.float32).asArray(Float.self) == values)
      }
    }
  }

  @Test("Selective reader loads only the indexed tensor's raw payload and preserves BF16")
  func selectiveReader() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("selective-tensors-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let original = MLXArray((0..<128).map { Float($0) / 13 }).reshaped(2, 64).asType(.bfloat16)
    try MLX.save(arrays: ["selected": original, "other": MLXArray.ones([4, 128])],
      url: root.appendingPathComponent("weights.safetensors"))
    let index = try JSONSerialization.data(withJSONObject: ["weight_map":
      ["selected": "weights.safetensors", "other": "weights.safetensors"]])
    try index.write(to: root.appendingPathComponent("model.safetensors.index.json"))
    let reader = try SelectiveSafetensorsReader(directory: root)
    let description = try reader.description(for: "selected")
    #expect(description.shape == [2, 64])
    #expect(description.dtype == .bfloat16)
    #expect(description.bytes == 256)
    let loaded = try reader.read("selected")
    #expect(loaded.dtype == .bfloat16)
    #expect(MLX.arrayEqual(loaded, original).item(Bool.self))
    #expect(throws: LagunaActivationStatisticsError.self) { try reader.read("missing") }
  }
}
