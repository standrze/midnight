import MLX
import XCTest

#if !MLX_CPU_BACKEND
    final class MLXCUDAKernelTests: XCTestCase {
        func testUnavailableCUDAIsRecoverable() throws {
            guard !MLXFast.isCUDAKernelAvailable else {
                throw XCTSkip("This assertion covers runtimes without an available CUDA backend.")
            }
            XCTAssertThrowsError(try makeCopyKernel()) { error in
                XCTAssertEqual(error as? MLXFast.CUDAKernelError, .unavailable)
            }
            #if os(macOS)
                // A CUDA-built Linux runtime can require a visible NVIDIA device even
                // for CPU allocation/default-stream initialization. Metal does not.
                try Device.withDefaultDevice(.cpu) {
                    let before = (MLXArray([1, 2, 3]) + 1).asArray(Int32.self)
                    XCTAssertThrowsError(try makeCopyKernel()) { error in
                        XCTAssertEqual(error as? MLXFast.CUDAKernelError, .unavailable)
                    }
                    XCTAssertEqual((MLXArray([1, 2, 3]) + 1).asArray(Int32.self), before)
                    XCTAssertEqual(Device.defaultDevice().deviceType, .cpu)
                }
            #endif
        }

        func testFactoryRejectsInvalidMetadataWithoutTouchingCUDA() {
            func rejected(
                inputs: [String] = ["input"], outputs: [String] = ["output"],
                source: String = "", sharedMemory: Int = 0
            ) {
                XCTAssertThrowsError(
                    try MLXFast.cudaKernel(
                        name: "invalid_metadata", inputNames: inputs, outputNames: outputs,
                        source: source, sharedMemory: sharedMemory)
                ) { error in
                    guard case .invalidConfiguration = error as? MLXFast.CUDAKernelError else {
                        return XCTFail("Expected configuration error, received \(error)")
                    }
                }
            }
            rejected(outputs: [])
            rejected(inputs: [""])
            rejected(outputs: ["input"])
            rejected(source: "before\0after")
            rejected(sharedMemory: -1)
            rejected(sharedMemory: Int(Int32.max) + 1)
        }

        #if os(macOS)
            func testExistingMetalKernelStillWorksAfterCUDARejection() throws {
                XCTAssertFalse(MLXFast.isCUDAKernelAvailable)
                XCTAssertThrowsError(try makeCopyKernel())
                let kernel = MLXFast.metalKernel(
                    name: "cuda_binding_metal_control", inputNames: ["input"], outputNames: ["output"],
                    source: """
                        uint i = thread_position_in_grid.x;
                        if (i < 7) output[i] = input[i] + 1;
                        """)
                let input = MLXArray(0..<7).asType(.float32)
                let output = kernel(
                    [input], grid: (7, 1, 1), threadGroup: (4, 1, 1),
                    outputShapes: [[7]], outputDTypes: [.float32], stream: .gpu)[0]
                try withError { eval(output) }
                XCTAssertEqual(output.asArray(Float.self), (1...7).map(Float.init))
            }
        #endif

        func testCUDATemplatesMultipleOutputsAndPaddedGrid() throws {
            try requireCUDA()
            let kernel = try MLXFast.cudaKernel(
                name: "cuda_binding_templates", inputNames: ["input", "bias"],
                outputNames: ["values", "indices"],
                source: """
                    const int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i < N - 1) {
                        T value = T(input[i]);
                        values[i] = enabled ? value + T(bias[0]) : value;
                        indices[i] = i + offset;
                    }
                    """, sharedMemory: 32)
            // A strided input exercises the binding's row-contiguous option. A partial
            // final block and an unwritten final element cover grid and initialization.
            let input = MLXArray(0..<262).asType(.float32).reshaped([131, 2])[0..., 0]
            let result = try kernel(
                [input, Float(2)], template: [("T", DType.float32), ("N", 131), ("enabled", true), ("offset", 3)],
                grid: (131, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [[131], [131]], outputDTypes: [.float32, .int32],
                initValue: -99, stream: .gpu)
            try withError { eval(result) }
            XCTAssertEqual(result[0].asArray(Float.self), (0..<130).map { Float($0 * 2 + 2) } + [-99])
            XCTAssertEqual(result[1].asArray(Int32.self), (0..<130).map { Int32($0 + 3) } + [-99])
        }

        func testCUDARejectsInvalidLaunchesAndCPUStream() throws {
            try requireCUDA()
            let kernel = try makeCopyKernel()
            let input = MLXArray([Float(1), 2, 3])
            XCTAssertEqual(input.dtype, .float32)
            func rejected(
                inputs: [MLXArray] = [], grid: (Int, Int, Int) = (3, 1, 1),
                group: (Int, Int, Int) = (32, 1, 1), shapes: [[Int]] = [[3]],
                dtypes: [DType] = [.float32], template: [(String, any KernelTemplateArg)]? = nil,
                stream: StreamOrDevice = .gpu
            ) {
                XCTAssertThrowsError(
                    try kernel(
                        inputs, template: template, grid: grid, threadGroup: group,
                        outputShapes: shapes, outputDTypes: dtypes, stream: stream)
                ) { error in
                    guard case .invalidConfiguration = error as? MLXFast.CUDAKernelError else {
                        return XCTFail("Expected configuration error, received \(error)")
                    }
                }
            }
            rejected()
            rejected(inputs: [input], shapes: [])
            rejected(inputs: [input], dtypes: [])
            rejected(inputs: [input], grid: (0, 1, 1))
            rejected(inputs: [input], group: (0, 1, 1))
            rejected(inputs: [input], group: (64, 64, 1))
            rejected(inputs: [input], grid: (Int(Int32.max), 1, 1))
            rejected(inputs: [input], shapes: [[-1]])
            rejected(inputs: [input], template: [("N", Int(Int32.max) + 1)])
            rejected(inputs: [input], template: [("N", UnsupportedTemplateArgument())])
            rejected(inputs: [input], stream: .cpu)
            // Failed configuration must leave the same prepared kernel reusable.
            let result = try kernel(
                [input], grid: (3, 1, 1), threadGroup: (32, 1, 1),
                outputShapes: [[3]], outputDTypes: [.float32], stream: .gpu)
            try withError { eval(result) }
            XCTAssertEqual(result[0].asArray(Float.self), [1, 2, 3])
            XCTAssertEqual(input.asArray(Float.self), [1, 2, 3])
        }

        private func requireCUDA() throws {
            try XCTSkipUnless(MLXFast.isCUDAKernelAvailable, "CUDA execution requires an available NVIDIA MLX runtime.")
        }

        private func makeCopyKernel() throws -> MLXFast.CUDAKernel {
            try MLXFast.cudaKernel(
                name: "cuda_binding_copy", inputNames: ["input"], outputNames: ["output"],
                source: """
                    const int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i < input_shape[0]) output[i] = input[i];
                    """)
        }

        private struct UnsupportedTemplateArgument: KernelTemplateArg {}
    }
#endif
