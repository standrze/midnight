import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@_spi(Benchmark) @testable import ModelRunnerCore

@Suite("Experimental Laguna gate/up SiLU eligibility", .serialized)
struct LagunaFusedGatherSiluEligibilityTests {
  @Test("Only the opted-in inference decode trace can select the candidate")
  func executionPolicy() {
    func allowed(enabled: Bool = true, compiled: Bool = true, router: Bool = true,
      training: Bool = false, observer: Bool = false) -> Bool
    {
      LagunaFusedGateUpSiluEligibility.allows(runtimeEnabled: enabled,
        useCompiledTail: compiled, useFusedRouter: router, training: training,
        hasCalibrationObserver: observer)
    }
    #if os(macOS)
    #expect(allowed())
    #else
    #expect(!allowed())
    #endif
    #expect(!allowed(enabled: false))
    #expect(!allowed(compiled: false))
    #expect(!allowed(router: false))
    #expect(!allowed(training: true))
    #expect(!allowed(observer: true))
    #expect(LagunaRuntimeTuning.useFusedGateUpSilu == nil)
  }

  @Test("Bindings reject unsupported formats, learned biases, and tensor layouts")
  func parameterPolicy() {
    Device.withDefaultDevice(.cpu) {
      #expect(LagunaFusedGateUpSiluBinding(metadataLayer()) != nil)
      for bits in [3, 8] {
        #expect(LagunaFusedGateUpSiluBinding(metadataLayer(bits: bits)) == nil)
      }
      #expect(LagunaFusedGateUpSiluBinding(metadataLayer(groupSize: 32)) == nil)
      #expect(LagunaFusedGateUpSiluBinding(metadataLayer(mode: .mxfp4)) == nil)
      #expect(LagunaFusedGateUpSiluBinding(metadataLayer(learnedBias: true)) == nil)
      #expect(LagunaFusedGateUpSiluBinding(metadataLayer(affineBias: false)) == nil)
      #expect(LagunaFusedGateUpSiluBinding(metadataLayer(scaleDType: .float16)) == nil)
      #expect(LagunaFusedGateUpSiluBinding(metadataLayer(experts: 128)) == nil)
      #expect(LagunaFusedGateUpSiluBinding(metadataLayer(packedWidth: 128)) == nil)
      #expect(LagunaFusedGateUpSiluBinding(FusedGateUpSwitchGLU(
        inputDims: 64, hiddenDims: 64, numExperts: 8)) == nil)
    }
  }

  @Test("Unsupported inputs and CPU execution return stock-fallback signals without evaluation")
  func inputPolicy() throws {
    try Device.withDefaultDevice(.cpu) {
      let weight = MLXArray.zeros([256, 1024, 256], dtype: .uint32)
      let scales = MLXArray.zeros([256, 1024, 32], dtype: .bfloat16)
      let x = MLXArray.zeros([1, 1, 2048], dtype: .bfloat16)
      let ids = MLXArray.zeros([1, 1, 8], dtype: .uint32)
      func supports(_ input: MLXArray, _ indices: MLXArray) -> Bool {
        LagunaFusedGatherSiluKernel.supports(input: input, weight: weight,
          scales: scales, biases: scales, indices: indices)
      }
      #expect(supports(x, ids))
      #expect(!supports(x.asType(.float16), ids))
      #expect(!supports(x.asType(.float32), ids))
      #expect(!supports(MLXArray.zeros([1, 2, 2048], dtype: .bfloat16), ids))
      #expect(!supports(x, MLXArray.zeros([1, 1, 4], dtype: .uint32)))
      #expect(!supports(x, ids.asType(.int32)))
      #expect(LagunaFusedGatherSiluKernel.callAsFunction(input: x, weight: weight,
        scales: scales, biases: scales, indices: ids) == nil)
      let binding = try #require(LagunaFusedGateUpSiluBinding(metadataLayer()))
      #expect(binding(x, ids) == nil)
    }
  }

  // These are unevaluated metadata fixtures. Even the large shapes allocate no
  // model buffers and never execute a Metal kernel in the default test suite.
  private func metadataLayer(bits: Int = 4, groupSize: Int = 64,
    mode: QuantizationMode = .affine, learnedBias: Bool = false,
    affineBias: Bool = true, scaleDType: DType = .bfloat16,
    experts: Int = 256, packedWidth: Int = 256) -> FusedGateUpSwitchGLU
  {
    let layer = FusedGateUpSwitchGLU(inputDims: 2048, hiddenDims: 512, numExperts: experts)
    let base = SwitchLinear(inputDims: 2048, outputDims: 1024, numExperts: experts,
      weight: MLXArray.zeros([experts, 1024, 2048], dtype: .bfloat16),
      bias: learnedBias ? MLXArray.zeros([experts, 1024], dtype: .bfloat16) : nil)
    let gate = QuantizedSwitchLinear(base,
      weight: MLXArray.zeros([experts, 1024, packedWidth], dtype: .uint32),
      scales: MLXArray.zeros([experts, 1024, 32], dtype: scaleDType),
      biases: affineBias ? MLXArray.zeros([experts, 1024, 32], dtype: scaleDType) : nil,
      groupSize: groupSize, bits: bits, mode: mode)
    layer.update(modules: ModuleChildren.unflattened(["gate_up_proj": gate]))
    return layer
  }
}

#if os(macOS)
@Suite("Experimental Laguna gate/up SiLU Metal parity", .serialized,
  .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_RUN_FUSED_GATHER_REGRESSION"] == "1"))
struct LagunaFusedGatherSiluParityTests {
  @Test("Bound packed weights and original down projection exactly match stock with strided inputs")
  func boundProjectionParity() throws {
    try Device.withDefaultDevice(.gpu) {
      // Quantize one seeded expert then broadcast its packed tensors. This also
      // tests noncontiguous expert-axis views without allocating BF16 E256 weights.
      let layer = FusedGateUpSwitchGLU(inputDims: 2048, hiddenDims: 512, numExperts: 256)
      let gate = quantizedProjection(input: 2048, output: 1024, seed: 901)
      let down = quantizedProjection(input: 512, output: 2048, seed: 902)
      layer.update(modules: ModuleChildren.unflattened([
        "gate_up_proj": gate, "down_proj": down]))
      layer.train(false)
      let binding = try #require(LagunaFusedGateUpSiluBinding(layer))
      let stock = compile { inputs in [layer(inputs[0], inputs[1])] }
      let candidate = compile { inputs in [binding(inputs[0], inputs[1])!] }
      for (caseIndex, selected) in [
        [255, 0, 3, 71, 128, 14, 219, 42],
        [7, 7, 0, 255, 7, 128, 42, 42],
      ].enumerated() {
        let storage = MLXRandom.normal([2048, 2], dtype: .bfloat16,
          key: MLXRandom.key(UInt64(903 + caseIndex)))
        let input = storage[0..., 0].reshaped([1, 1, 2048])
        let indexStorage = MLXArray(selected.flatMap { [UInt32($0), 0] }).reshaped([8, 2])
        let indices = indexStorage[0..., 0].reshaped([1, 1, 8])
        let expected = stock([input, indices])[0]
        let actual = candidate([input, indices])[0]
        eval(expected, actual)
        #expect(expected.shape == [1, 1, 8, 2048])
        #expect(abs(expected).max().item(Float.self) > 0.01)
        #expect(arrayEqual(expected, actual).item(Bool.self))
        // Switching back uses the original compiled graph on the same modules.
        #expect(arrayEqual(stock([input, indices])[0], expected).item(Bool.self))
      }
    }
  }

  private func quantizedProjection(input: Int, output: Int, seed: UInt64) -> QuantizedSwitchLinear {
    let values = MLXRandom.normal([1, output, input], dtype: .bfloat16,
      scale: 0.02, key: MLXRandom.key(seed))
    let q = quantized(values, groupSize: 64, bits: 4)
    let base = SwitchLinear(inputDims: input, outputDims: output, numExperts: 256,
      weight: broadcast(values, to: [256, output, input]))
    // Distinct expert scales make wrong expert-index selection observable.
    let factors = (MLXArray(0..<256).asType(.float32) / 1024 + 0.875)
      .asType(.bfloat16).reshaped([256, 1, 1])
    return QuantizedSwitchLinear(base,
      weight: broadcast(q.wq, to: [256, output, input / 8]),
      scales: broadcast(q.scales, to: [256, output, input / 64]) * factors,
      biases: broadcast(q.biases!, to: [256, output, input / 64]) * factors,
      groupSize: 64, bits: 4)
  }
}
#endif
