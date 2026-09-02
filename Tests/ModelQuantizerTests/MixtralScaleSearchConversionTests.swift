import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class MixtralScaleSearchConversionTests: XCTestCase {
  func testMixtralExpertsUseScaleSearchAndRoutersUseQ8() async throws {
    try await Device.withDefaultDevice(.cpu) {
      let fileManager = FileManager.default
      let root = fileManager.temporaryDirectory.appendingPathComponent(
        "model-runner-mixtral-conversion-\(UUID().uuidString)")
      let source = root.appendingPathComponent("source")
      let output = root.appendingPathComponent("output")
      let standardOutput = root.appendingPathComponent("standard-output")
      try fileManager.createDirectory(at: source, withIntermediateDirectories: true)
      defer { try? fileManager.removeItem(at: root) }

      let config = """
        {
          "model_type": "mixtral",
          "vocab_size": 128,
          "hidden_size": 64,
          "intermediate_size": 128,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "num_local_experts": 4,
          "num_experts_per_tok": 2,
          "rms_norm_eps": 0.00001,
          "rope_theta": 1000000.0,
          "tie_word_embeddings": false
        }
        """
      let configData = Data(config.utf8)
      try configData.write(to: source.appendingPathComponent("config.json"))

      let configuration = try JSONDecoder().decode(
        MixtralConfiguration.self, from: configData)
      let sourceModel = MixtralModel(configuration)
      let sourceArrays = Dictionary(
        uniqueKeysWithValues: sourceModel.parameters().flattened())
      try save(
        arrays: sourceArrays,
        metadata: ["format": "mlx"],
        url: source.appendingPathComponent("model.safetensors"),
        stream: .cpu
      )

      let routerQ8 = ModelConversionQuantization(
        bits: 8,
        groupSize: 64,
        mode: .affine,
        calibration: .standard
      )
      let result = try await LLMModelFactory.shared.convert(
        from: source,
        to: output,
        options: ModelConversionOptions(
          bits: 4,
          groupSize: 64,
          mode: .affine,
          calibration: .q4AffineScaleSearch,
          maxShardSize: 16 * 1_024 * 1_024,
          quantizationPredicate: { path, _ in
            if path.hasSuffix(".block_sparse_moe.gate") {
              return .quantize(routerQ8)
            }
            return .quantize()
          }
        )
      )
      let standardResult = try await LLMModelFactory.shared.convert(
        from: source,
        to: standardOutput,
        options: ModelConversionOptions(
          bits: 4,
          groupSize: 64,
          mode: .affine,
          calibration: .standard,
          maxShardSize: 16 * 1_024 * 1_024,
          quantizationPredicate: { path, _ in
            if path.hasSuffix(".block_sparse_moe.gate") {
              return .quantize(routerQ8)
            }
            return .quantize()
          }
        )
      )

      XCTAssertFalse(result.weightsURLs.isEmpty)
      let outputConfig = try JSONDecoder.json5().decode(
        BaseConfiguration.self,
        from: Data(contentsOf: output.appendingPathComponent("config.json"))
      )
      let quantization = try XCTUnwrap(outputConfig.perLayerQuantization)
      XCTAssertEqual(quantization.quantization?.bits, 4)
      XCTAssertEqual(quantization.quantization?.groupSize, 64)
      for layer in 0 ..< 2 {
        let router = "model.layers.\(layer).block_sparse_moe.gate"
        guard case .quantize(let routerQuantization)? =
          quantization.perLayerQuantization[router]
        else {
          return XCTFail("missing Q8 override for \(router)")
        }
        XCTAssertEqual(routerQuantization.bits, 8)
        XCTAssertEqual(routerQuantization.groupSize, 64)
      }

      var outputArrays = [String: MLXArray]()
      for weightsURL in result.weightsURLs {
        outputArrays.merge(
          try loadArrays(url: weightsURL, stream: .cpu),
          uniquingKeysWith: { _, replacement in replacement }
        )
      }
      XCTAssertNotNil(
        outputArrays["model.layers.0.block_sparse_moe.gate.scales"])
      XCTAssertNotNil(
        outputArrays["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.scales"])
      XCTAssertNotNil(
        outputArrays["model.layers.0.block_sparse_moe.switch_mlp.up_proj.biases"])
      XCTAssertNotNil(
        outputArrays["model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"])

      var standardArrays = [String: MLXArray]()
      for weightsURL in standardResult.weightsURLs {
        standardArrays.merge(
          try loadArrays(url: weightsURL, stream: .cpu),
          uniquingKeysWith: { _, replacement in replacement }
        )
      }
      XCTAssertEqual(Set(outputArrays.keys), Set(standardArrays.keys))
      for (key, scaleSearchArray) in outputArrays {
        let standardArray = try XCTUnwrap(
          standardArrays[key], "missing standard control array \(key)")
        XCTAssertEqual(scaleSearchArray.shape, standardArray.shape, key)
        XCTAssertEqual(scaleSearchArray.dtype, standardArray.dtype, key)
      }
    }
  }
}
