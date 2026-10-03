import ModelRunnerProtocol
import Testing

@Suite("MLX resource limits")
struct MLXResourceLimitsTests {
    private let gibibyte = 1_073_741_824
    private let mebibyte = 1_048_576

    @Test("CUDA budgets scale across cards and account for occupied VRAM")
    func cudaCapacity() throws {
        for capacity in [8, 24, 48, 80] {
            let snapshot = CUDAMemorySnapshot(
                totalBytes: capacity * gibibyte, freeBytes: (capacity - 3) * gibibyte,
                supportsLayerOffload: true)
            let limits = try MLXResourceLimits.resolve(
                for: .cuda, physicalMemoryBytes: 128 * UInt64(gibibyte), cudaMemory: snapshot, environment: [:])
            #expect(limits.memoryLimitBytes == snapshot.freeBytes - snapshot.reserveBytes)
            #expect(limits.maximumMemoryLimitBytes == limits.memoryLimitBytes)
        }
        #expect(throws: MLXResourceLimitError.cudaMemoryUnavailable) {
            try MLXResourceLimits.resolve(for: .cuda, physicalMemoryBytes: 64 * UInt64(gibibyte), environment: [:])
        }
        for snapshot in [
            CUDAMemorySnapshot(totalBytes: 0, freeBytes: 0, supportsLayerOffload: false),
            CUDAMemorySnapshot(totalBytes: 8 * gibibyte, freeBytes: 9 * gibibyte, supportsLayerOffload: false),
            CUDAMemorySnapshot(totalBytes: 8 * gibibyte, freeBytes: 512 * mebibyte, supportsLayerOffload: false),
        ] {
            #expect(throws: MLXResourceLimitError.self) {
                try MLXResourceLimits.resolve(
                    for: .cuda, physicalMemoryBytes: 64 * UInt64(gibibyte), cudaMemory: snapshot, environment: [:])
            }
        }
    }

    private var snapshot: CUDAMemorySnapshot {
        .init(totalBytes: 24 * gibibyte, freeBytes: 24 * gibibyte, supportsLayerOffload: true)
    }

    @Test("CUDA defaults use free device memory and preserve headroom")
    func cudaDefaults() throws {
        let limits = try MLXResourceLimits.resolve(
            for: .cuda,
            physicalMemoryBytes: 64 * UInt64(gibibyte),
            cudaMemory: snapshot,
            environment: [:]
        )

        #expect(limits.memoryLimitBytes == snapshot.usableBytes)
        #expect(limits.maximumMemoryLimitBytes == snapshot.usableBytes)
        #expect(limits.cacheLimitBytes == 128 * mebibyte)
        #expect(limits.maximumCacheLimitBytes == 1_024 * mebibyte)
    }

    @Test("CUDA overrides may not exceed the absolute bounds")
    func cudaBounds() throws {
        let accepted = try MLXResourceLimits.resolve(
            for: .cuda,
            physicalMemoryBytes: 64 * UInt64(gibibyte),
            cudaMemory: snapshot,
            environment: [
                MLXResourceLimits.memoryLimitEnvironmentKey: "20",
                MLXResourceLimits.cacheLimitEnvironmentKey: "1024",
            ]
        )
        #expect(accepted.memoryLimitBytes == 20 * gibibyte)
        #expect(accepted.cacheLimitBytes == 1_024 * mebibyte)

        #expect(throws: MLXResourceLimitError.self) {
            try MLXResourceLimits.resolve(
                for: .cuda,
                physicalMemoryBytes: 64 * UInt64(gibibyte),
                cudaMemory: snapshot,
                environment: [MLXResourceLimits.memoryLimitEnvironmentKey: "23"]
            )
        }
        #expect(throws: MLXResourceLimitError.self) {
            try MLXResourceLimits.resolve(
                for: .cuda,
                physicalMemoryBytes: 64 * UInt64(gibibyte),
                cudaMemory: snapshot,
                environment: [MLXResourceLimits.cacheLimitEnvironmentKey: "1025"]
            )
        }
    }

    @Test("Metal scales with device capacity and preserves host headroom")
    func metalBounds() throws {
        let large = try MLXResourceLimits.resolve(
            for: .metal,
            physicalMemoryBytes: 128 * UInt64(gibibyte), recommendedWorkingSetBytes: 96 * gibibyte,
            environment: [:])
        #expect(large.memoryLimitBytes == 96 * gibibyte)
        #expect(large.maximumMemoryLimitBytes == 96 * gibibyte)
        #expect(large.maximumCacheLimitBytes == 2 * gibibyte)
        let small = try MLXResourceLimits.resolve(
            for: .metal,
            physicalMemoryBytes: 8 * UInt64(gibibyte), recommendedWorkingSetBytes: 6 * gibibyte,
            environment: [:])
        #expect(small.memoryLimitBytes == 6 * gibibyte)
        #expect(throws: MLXResourceLimitError.self) {
            try MLXResourceLimits.resolve(
                for: .metal, physicalMemoryBytes: 8 * UInt64(gibibyte),
                recommendedWorkingSetBytes: 6 * gibibyte,
                environment: [MLXResourceLimits.memoryLimitEnvironmentKey: "7"])
        }
    }

    @Test("Host reserve is configurable but cannot consume the entire machine")
    func hostReserve() throws {
        let limits = try MLXResourceLimits.resolve(
            for: .metal,
            physicalMemoryBytes: 64 * UInt64(gibibyte),
            recommendedWorkingSetBytes: 48 * gibibyte, cudaMemory: snapshot,
            environment: [MLXResourceLimits.reserveEnvironmentKey: "24"])
        #expect(limits.memoryLimitBytes == 40 * gibibyte)
        for reserve in ["0", "1", "64", "nan"] {
            #expect(throws: MLXResourceLimitError.self) {
                try MLXResourceLimits.resolve(
                    for: .metal, physicalMemoryBytes: 64 * UInt64(gibibyte),
                    cudaMemory: snapshot,
                    environment: [MLXResourceLimits.reserveEnvironmentKey: reserve])
            }
        }
    }

    @Test("Malformed, non-finite, signed, and empty values fail closed")
    func malformedOverrides() {
        for value in ["", " ", "nan", "inf", "-1", "+1", "1e1", ".5", "5."] {
            #expect(throws: MLXResourceLimitError.self) {
                try MLXResourceLimits.resolve(
                    for: .cuda,
                    physicalMemoryBytes: 64 * UInt64(gibibyte),
                    cudaMemory: snapshot,
                    environment: [MLXResourceLimits.memoryLimitEnvironmentKey: value]
                )
            }
        }
    }

    @Test("A zero cache disables recycling but a zero memory limit is rejected")
    func zeroLimits() throws {
        let limits = try MLXResourceLimits.resolve(
            for: .cuda,
            physicalMemoryBytes: 64 * UInt64(gibibyte),
            cudaMemory: snapshot,
            environment: [MLXResourceLimits.cacheLimitEnvironmentKey: "0"]
        )
        #expect(limits.cacheLimitBytes == 0)
        #expect(throws: MLXResourceLimitError.self) {
            try MLXResourceLimits.resolve(
                for: .cuda,
                physicalMemoryBytes: 64 * UInt64(gibibyte),
                cudaMemory: snapshot,
                environment: [MLXResourceLimits.memoryLimitEnvironmentKey: "0"]
            )
        }
    }

    @Test("The cache may not exceed a lowered memory ceiling")
    func cacheBelowMemory() {
        #expect(throws: MLXResourceLimitError.self) {
            try MLXResourceLimits.resolve(
                for: .cuda,
                physicalMemoryBytes: 64 * UInt64(gibibyte),
                cudaMemory: snapshot,
                environment: [
                    MLXResourceLimits.memoryLimitEnvironmentKey: "0.0625",
                    MLXResourceLimits.cacheLimitEnvironmentKey: "128",
                ]
            )
        }
    }

    @Test("Auto must be resolved before resource policy selection")
    func rejectsAuto() {
        #expect(throws: MLXResourceLimitError.unresolvedEngine(.auto)) {
            try MLXResourceLimits.resolve(
                for: .auto,
                physicalMemoryBytes: 64 * UInt64(gibibyte),
                cudaMemory: snapshot,
                environment: [:]
            )
        }
    }
}
