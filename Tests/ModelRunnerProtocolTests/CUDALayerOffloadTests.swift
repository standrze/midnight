import ModelRunnerProtocol
import Testing

@testable import ModelRunnerCore

@Suite("CUDA RAM layer planning")
struct CUDALayerOffloadTests {
    private let gib = 1_073_741_824

    @Test("Auto offloads only enough whole layers and reserves workspace")
    func automaticSelection() throws {
        let layers = Dictionary(uniqueKeysWithValues: (0..<12).map { ("model.layers.\($0)", gib) })
        let plan = try CUDALayerOffloadPlan(
            mode: .automatic, layerBytes: layers, stationaryBytes: gib, gpuBudget: 10 * gib, hostBudget: 8 * gib)
        #expect(
            plan.layerPaths == [
                "model.layers.7", "model.layers.8", "model.layers.9", "model.layers.10", "model.layers.11",
            ])
        #expect(plan.hostWeightBytes == 5 * gib)
        #expect(plan.deviceWeightBytes == 8 * gib)
        #expect(plan.deviceWeightBytes + plan.workspaceReserveBytes <= 10 * gib)
        let fitting = try CUDALayerOffloadPlan(
            mode: .automatic, layerBytes: ["model.layers.0": gib], stationaryBytes: gib,
            gpuBudget: 10 * gib, hostBudget: 0)
        #expect(fitting.layerPaths.isEmpty)
    }

    @Test("Explicit counts are honored and infeasible plans fail before placement")
    func explicitCounts() throws {
        let layers = ["model.layers.0": gib, "model.layers.1": 2 * gib]
        let plan = try CUDALayerOffloadPlan(
            mode: .layers(1), layerBytes: layers, stationaryBytes: gib, gpuBudget: 8 * gib, hostBudget: 4 * gib)
        #expect(plan.layerPaths == ["model.layers.1"])
        #expect(plan.hostWeightBytes == 2 * gib)
        for (mode, gpu, host) in [
            (CUDALayerOffloadMode.layers(3), 8 * gib, 8 * gib), (.layers(1), 2 * gib, 8 * gib),
            (.automatic, 3 * gib, gib),
        ] {
            #expect(throws: RequestAdmissionError.self) {
                try CUDALayerOffloadPlan(
                    mode: mode, layerBytes: layers, stationaryBytes: gib, gpuBudget: gpu, hostBudget: host)
            }
        }
        #expect(throws: RequestAdmissionError.self) {
            try CUDALayerOffloadPlan(
                mode: .automatic, layerBytes: [:], stationaryBytes: 4 * gib, gpuBudget: 3 * gib, hostBudget: 8 * gib)
        }
    }

    @Test("Layer identification recognizes native and upstream decoder layouts")
    func paths() {
        #expect(CUDALayerOffloadPlan.layerPath(for: "model.layers.12.mlp.scales") == "model.layers.12")
        #expect(CUDALayerOffloadPlan.layerPath(for: "transformer.h.3.attn.weight") == "transformer.h.3")
        #expect(CUDALayerOffloadPlan.layerPath(for: "blocks.0.mlp.weight") == "blocks.0")
        #expect(CUDALayerOffloadPlan.layerPath(for: "model.embed_tokens.weight") == nil)
        #expect(CUDALayerOffloadPlan.layerPath(for: "layers.bad.weight") == nil)
    }

    @Test("Offload settings reject malformed values and other engines")
    func settings() throws {
        #expect(try CUDALayerOffloadMode(environment: [:], engine: .cuda) == .disabled)
        #expect(
            try CUDALayerOffloadMode(environment: [CUDALayerOffloadMode.environmentKey: "auto"], engine: .cuda)
                == .automatic)
        #expect(
            try CUDALayerOffloadMode(environment: [CUDALayerOffloadMode.environmentKey: "4"], engine: .cuda)
                == .layers(4))
        for raw in ["", "-1", "+1", "1.5", " 1", "AUTO", "99999999999999999999"] {
            #expect(throws: RequestAdmissionError.self) {
                try CUDALayerOffloadMode(environment: [CUDALayerOffloadMode.environmentKey: raw], engine: .cuda)
            }
        }
        #expect(throws: RequestAdmissionError.self) {
            try CUDALayerOffloadMode(environment: [CUDALayerOffloadMode.environmentKey: "auto"], engine: .metal)
        }
    }

    @Test("Available host RAM respects occupied memory, cgroup usage, and reserve")
    func hostMemory() throws {
        let meminfo = "MemTotal: 67108864 kB\nMemAvailable: 33554432 kB\n"
        #expect(try CUDAHostMemory.availableBytes(meminfo: meminfo) == 32 * gib)
        let available = try CUDAHostMemory.availableBytes(
            meminfo: meminfo, cgroupLimit: String(24 * gib), cgroupCurrent: String(8 * gib))
        #expect(available == 16 * gib)
        let budget = try CUDAHostMemory.budget(physical: UInt64(64 * gib), available: available, environment: [:])
        #expect(budget == available - (64 * gib / 5))
        #expect(
            try CUDAHostMemory.budget(
                physical: UInt64(64 * gib), available: available, environment: [CUDAHostMemory.limitKey: "2"]) == 2
                * gib)
        #expect(
            try CUDAHostMemory.availableBytes(meminfo: meminfo, cgroupLimit: "max\n", cgroupCurrent: "12") == 32 * gib)
        for raw in ["", "-1", "nan", "1e1", "5.", ".5", "4"] {
            #expect(throws: RequestAdmissionError.self) {
                try CUDAHostMemory.budget(
                    physical: UInt64(64 * gib), available: available, environment: [CUDAHostMemory.limitKey: raw])
            }
        }
        #expect(throws: RequestAdmissionError.self) { try CUDAHostMemory.availableBytes(meminfo: "MemTotal: 100 kB") }
        #expect(throws: RequestAdmissionError.self) {
            try CUDAHostMemory.availableBytes(meminfo: meminfo, cgroupLimit: "broken", cgroupCurrent: "1")
        }
    }
}
