import CUDAMemory
import Foundation
import ModelRunnerProtocol

/// Narrow native CUDA driver boundary. The C bridge restores thread-local contexts.
enum CUDAMemoryRuntime {
    static func snapshot() throws -> CUDAMemorySnapshot {
        var free: UInt64 = 0
        var total: UInt64 = 0
        var managed: Int32 = 0
        try call { error, count in
            midnight_cuda_memory_snapshot(&free, &total, &managed, error, count)
        }
        guard total <= UInt64(Int.max), free <= total, total > 0 else {
            throw MLXResourceLimitError.cudaMemoryUnavailable
        }
        return CUDAMemorySnapshot(totalBytes: Int(total), freeBytes: Int(free), supportsLayerOffload: managed != 0)
    }

    static func place(_ pointer: UnsafeMutableRawPointer, bytes: Int, host: Bool) throws {
        try call { error, count in
            midnight_cuda_place_weights(pointer, bytes, host ? 1 : 0, error, count)
        }
    }

    static func reset(_ pointer: UnsafeMutableRawPointer, bytes: Int) throws {
        try call { error, count in midnight_cuda_reset_weights(pointer, bytes, error, count) }
    }

    private static func call(_ operation: (UnsafeMutablePointer<CChar>, Int) -> Int32) throws {
        var message = [CChar](repeating: 0, count: 512)
        let status = message.withUnsafeMutableBufferPointer { operation($0.baseAddress!, $0.count) }
        guard status == 0 else {
            throw RequestAdmissionError.configuration(
                message.withUnsafeBufferPointer { String(cString: $0.baseAddress!) })
        }
    }
}
