import ModelRunnerProtocol
import Testing

@Suite("Model engine selection")
struct ModelEngineTests {
    @Test("Auto resolves to the compiled backend")
    func autoResolution() throws {
        #expect(try ModelEngine.auto.resolve(for: .metal) == .metal)
        #expect(try ModelEngine.auto.resolve(for: .cuda) == .cuda)
        #expect(try ModelEngine.auto.resolve(for: .cpu) == .cpu)
    }

    @Test("CPU is available in every build")
    func cpuResolution() throws {
        #expect(try ModelEngine.cpu.resolve(for: .metal) == .cpu)
        #expect(try ModelEngine.cpu.resolve(for: .cuda) == .cpu)
        #expect(try ModelEngine.cpu.resolve(for: .cpu) == .cpu)
    }

    @Test("GPU engines require their matching backend")
    func gpuValidation() throws {
        #expect(try ModelEngine.metal.resolve(for: .metal) == .metal)
        #expect(try ModelEngine.cuda.resolve(for: .cuda) == .cuda)
        #expect(throws: ModelEngineError.self) {
            try ModelEngine.cuda.resolve(for: .metal)
        }
        #expect(throws: ModelEngineError.self) {
            try ModelEngine.metal.resolve(for: .cuda)
        }
    }

    @Test("Arguments are normalized and validated")
    func argumentParsing() throws {
        #expect(try ModelEngine(argument: " CUDA ") == .cuda)
        #expect(throws: ModelEngineError.self) {
            try ModelEngine(argument: "vulkan")
        }
    }
}
