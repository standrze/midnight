import Cmlx
import Foundation
import MLX
import MLXLMCommon
import ModelRunnerProtocol

/// Owns weight arrays until their driver advice has been reset on unload. Serving
/// never mutates these weights; adapters and speculative assistants are excluded.
final class CUDAWeightPlacement: @unchecked Sendable {
    let plan: CUDALayerOffloadPlan
    private var bindings: [(array: MLXArray, pointer: UnsafeMutableRawPointer, bytes: Int)] = []

    init(model: any LanguageModel, mode: CUDALayerOffloadMode, gpuBudget: Int, hostBudget: Int) throws {
        let parameters = model.parameters().flattened().sorted { $0.0 < $1.0 }
        try MLX.checkedEval(parameters.map(\.1))
        var layers: [String: Int] = [:]
        var stationary = 0
        for (path, array) in parameters {
            if let layer = CUDALayerOffloadPlan.layerPath(for: path), array.nbytes >= 16_384 {
                layers[layer, default: 0] = ModelMemoryProfile.add(layers[layer, default: 0], array.nbytes)
            } else {
                stationary = ModelMemoryProfile.add(stationary, array.nbytes)
            }
        }
        plan = try CUDALayerOffloadPlan(
            mode: mode, layerBytes: layers, stationaryBytes: stationary, gpuBudget: gpuBudget, hostBudget: hostBudget)
        // Avoid advising the allocator's shared scalar pool and tiny normalization
        // weights. Larger tensors must be contiguous and have independent ranges;
        // aliases are rejected so a host layer cannot also migrate through a GPU alias.
        var ranges: [Range<UInt>] = []
        do {
            for (path, array) in parameters where array.nbytes >= 16_384 {
                let strides = mlx_array_strides(array.ctx)!
                var contiguousStride = 1
                for index in array.shape.indices.reversed() {
                    guard array.shape[index] == 1 || Int(strides[index]) == contiguousStride else {
                        throw RequestAdmissionError.configuration(
                            "CUDA RAM offload requires contiguous weights: \(path)")
                    }
                    contiguousStride *= array.shape[index]
                }
                guard let data = mlx_array_data_uint8(array.ctx) else {
                    throw RequestAdmissionError.configuration("Cannot access managed CUDA weight: \(path)")
                }
                let pointer = UnsafeMutableRawPointer(mutating: data)
                let start = UInt(bitPattern: pointer)
                guard start <= UInt.max - UInt(array.nbytes) else {
                    throw RequestAdmissionError.configuration("CUDA weight address overflow")
                }
                let range = start..<(start + UInt(array.nbytes))
                guard !ranges.contains(where: { $0.overlaps(range) }) else {
                    throw RequestAdmissionError.configuration(
                        "CUDA RAM offload does not support aliased weights: \(path)")
                }
                ranges.append(range)
                // Record before changing advice so a partial driver failure rolls back.
                bindings.append((array, pointer, array.nbytes))
                let host = CUDALayerOffloadPlan.layerPath(for: path).map { plan.layerPaths.contains($0) } ?? false
                try CUDAMemoryRuntime.place(pointer, bytes: array.nbytes, host: host)
            }
        } catch {
            reset()
            throw error
        }
        print(
            "CUDA weight placement: RAM layers=\(plan.layerPaths.sorted()) "
                + "host=\(plan.hostWeightBytes) GPU=\(plan.deviceWeightBytes) "
                + "workspace reserve=\(plan.workspaceReserveBytes) bytes")
    }

    deinit { reset() }

    private func reset() {
        for binding in bindings {
            do {
                try CUDAMemoryRuntime.reset(binding.pointer, bytes: binding.bytes)
            } catch {
                // An unusable CUDA context must not recycle advised buffers into
                // another model. Disable recycling before releasing the arrays.
                Memory.cacheLimit = 0
                Memory.clearCache()
                print("CUDA weight-placement cleanup failed; allocator recycling disabled: \(error)")
            }
            withExtendedLifetime(binding.array) {}
        }
        bindings.removeAll()
    }
}
