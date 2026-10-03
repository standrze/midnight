import Foundation
import Metal

private enum ProbeError: Error {
    case failure(String)
}

private enum Precision: String, CaseIterable {
    case float16
    case bfloat16

    func encode(_ value: Float) -> UInt16 {
        switch self {
        case .float16:
            Float16(value).bitPattern
        case .bfloat16:
            UInt16(truncatingIfNeeded: (value.bitPattern &+ 0x7FFF &+ ((value.bitPattern >> 16) & 1)) >> 16)
        }
    }

    func decode(_ value: UInt16) -> Float {
        switch self {
        case .float16:
            Float(Float16(bitPattern: value))
        case .bfloat16:
            Float(bitPattern: UInt32(value) << 16)
        }
    }

    var relativeTolerance: Double {
        self == .float16 ? 0.0011 : 0.008
    }
}

private struct Shape {
    var rows: UInt32
    var columns: UInt32
    var reduction: UInt32
    var groupSize: UInt32

    init(_ rows: Int, _ columns: Int, _ reduction: Int, _ groupSize: Int) {
        self.rows = UInt32(rows)
        self.columns = UInt32(columns)
        self.reduction = UInt32(reduction)
        self.groupSize = UInt32(groupSize)
    }

    var label: String {
        "M\(rows)-N\(columns)-K\(reduction)-G\(groupSize)"
    }
}

private struct Fixture {
    let shape: Shape
    let precision: Precision
    let input: [UInt16]
    let packed: [UInt32]
    let scales: [UInt16]
    let biases: [UInt16]

    init(shape: Shape, precision: Precision) {
        self.shape = shape
        self.precision = precision
        let m = Int(shape.rows)
        let n = Int(shape.columns)
        let k = Int(shape.reduction)
        let g = Int(shape.groupSize)
        input = (0..<m * k).map { index in
            precision.encode(Float((index * 17 + index / k * 13) % 101 - 50) / 53)
        }
        packed = (0..<n * k / 8).map { index in
            var word: UInt32 = 0
            for nibble in 0..<8 {
                let code = UInt32((index * 7 + nibble * 11 + index / (k / 8)) % 16)
                word |= code << (nibble * 4)
            }
            return word
        }
        scales = (0..<n * k / g).map { index in
            precision.encode(index % 19 == 0 ? 0 : Float(index % 13 - 6) / 37)
        }
        biases = (0..<n * k / g).map { index in
            precision.encode(Float(index % 17 - 8) / 29)
        }
    }

    // Independent scalar reference: stored metadata, decoded grid, operand rounding,
    // then a double-precision dot product. It never uses GPU-produced weights.
    func reference(at index: Int) -> (value: Double, absoluteSum: Double) {
        let k = Int(shape.reduction)
        let n = Int(shape.columns)
        let g = Int(shape.groupSize)
        let row = index / n
        let column = index % n
        var sum = 0.0
        var absoluteSum = 0.0
        for reduction in 0..<k {
            let word = packed[column * (k / 8) + reduction / 8]
            let code = Float((word >> (4 * (reduction % 8))) & 15)
            let group = column * (k / g) + reduction / g
            let weight = precision.decode(
                precision.encode(
                    precision.decode(scales[group]) * code + precision.decode(biases[group])
                ))
            let product = Double(precision.decode(input[row * k + reduction])) * Double(weight)
            sum += product
            absoluteSum += abs(product)
        }
        return (sum, absoluteSum)
    }
}

private struct Execution {
    let output: [UInt16]
    let gpuSeconds: Double
    let wallSeconds: Double
    let guardsIntact: Bool
}

private final class Runner {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipelines: [Precision: [String: MTLComputePipelineState]]

    init(source: String) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw ProbeError.failure("Metal GPU unavailable. This probe needs native GPU access.")
        }
        self.device = device
        self.queue = queue
        var pipelines: [Precision: [String: MTLComputePipelineState]] = [:]
        for precision in Precision.allCases {
            let options = MTLCompileOptions()
            options.languageVersion = .version4_0
            options.mathMode = .safe
            options.mathFloatingPointFunctions = .precise
            options.preprocessorMacros = ["USE_BFLOAT": NSNumber(value: precision == .bfloat16)]
            let library = try device.makeLibrary(source: source, options: options)
            var kernels: [String: MTLComputePipelineState] = [:]
            for name in ["affine_q4_staged", "affine_q4_cooperative"] {
                guard let function = library.makeFunction(name: name) else {
                    throw ProbeError.failure("Missing kernel: \(name)")
                }
                kernels[name] = try device.makeComputePipelineState(function: function)
            }
            pipelines[precision] = kernels
        }
        self.pipelines = pipelines
    }

    func buffer<T>(_ array: [T]) throws -> MTLBuffer {
        let buffer = array.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }
        guard let buffer else { throw ProbeError.failure("Buffer allocation failed") }
        return buffer
    }

    func prepare(_ fixture: Fixture) throws -> [MTLBuffer] {
        try [buffer(fixture.input), buffer(fixture.packed), buffer(fixture.scales), buffer(fixture.biases)]
    }

    func execute(
        _ fixture: Fixture, buffers: [MTLBuffer], kernel: String, repetitions: Int = 1
    ) throws -> Execution {
        guard let pipeline = pipelines[fixture.precision]?[kernel] else {
            throw ProbeError.failure("Missing pipeline")
        }
        let count = Int(fixture.shape.rows * fixture.shape.columns)
        let guardCount = 128
        let sentinel: UInt16 = 0x55AA
        var initial = [UInt16](repeating: sentinel, count: count + guardCount * 2)
        for index in guardCount..<guardCount + count { initial[index] = 0x7FFF }
        let output = try buffer(initial)
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw ProbeError.failure("Command allocation failed")
        }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        encoder.setBuffer(output, offset: guardCount * 2, index: 4)
        var shape = fixture.shape
        encoder.setBytes(&shape, length: MemoryLayout<Shape>.size, index: 5)
        let grid = MTLSize(width: (Int(shape.columns) + 31) / 32, height: (Int(shape.rows) + 15) / 16, depth: 1)
        for _ in 0..<repetitions {
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        encoder.endEncoding()
        let start = ProcessInfo.processInfo.systemUptime
        command.commit()
        command.waitUntilCompleted()
        let elapsed = ProcessInfo.processInfo.systemUptime - start
        guard command.status == .completed else {
            throw ProbeError.failure("GPU command failed: \(String(describing: command.error))")
        }
        let pointer = output.contents().bindMemory(to: UInt16.self, capacity: initial.count)
        let values = Array(UnsafeBufferPointer(start: pointer + guardCount, count: count))
        let guards = (0..<guardCount).allSatisfy {
            pointer[$0] == sentinel && pointer[guardCount + count + $0] == sentinel
        }
        return Execution(
            output: values,
            gpuSeconds: (command.gpuEndTime - command.gpuStartTime) / Double(repetitions),
            wallSeconds: elapsed / Double(repetitions), guardsIntact: guards
        )
    }
}

private func validate(_ fixture: Fixture, _ staged: Execution, _ cooperative: Execution, sample: Bool) throws
    -> [String: Any]
{
    guard staged.guardsIntact, cooperative.guardsIntact else {
        throw ProbeError.failure("Output guard corrupted: \(fixture.shape.label)")
    }
    let mismatches = zip(staged.output, cooperative.output).filter { $0 != $1 }.count
    guard mismatches == 0 else {
        throw ProbeError.failure("Staged/cooperative outputs differ in \(mismatches) entries: \(fixture.shape.label)")
    }
    guard cooperative.output.allSatisfy({ fixture.precision.decode($0).isFinite }) else {
        throw ProbeError.failure("Nonfinite or unwritten output: \(fixture.shape.label)")
    }
    let count = cooperative.output.count
    let indices =
        sample
        ? Array(Set((0..<min(256, count)).map { $0 * (count - 1) / (min(256, count) - 1) })).sorted() : Array(0..<count)
    var maximumError = 0.0
    var maximumToleranceRatio = 0.0
    for index in indices {
        let reference = fixture.reference(at: index)
        let actual = Double(fixture.precision.decode(cooperative.output[index]))
        let error = abs(actual - reference.value)
        let tolerance =
            0.00001 + abs(reference.value) * fixture.precision.relativeTolerance + reference.absoluteSum * 0.000008
        maximumError = max(maximumError, error)
        maximumToleranceRatio = max(maximumToleranceRatio, error / tolerance)
        guard error <= tolerance else {
            throw ProbeError.failure(
                "Reference mismatch \(fixture.shape.label) \(fixture.precision.rawValue) index \(index): \(actual) vs \(reference.value), tolerance \(tolerance)"
            )
        }
    }
    return [
        "shape": fixture.shape.label, "precision": fixture.precision.rawValue,
        "reference_entries": indices.count, "output_entries": count,
        "bitwise_control_mismatches": mismatches, "guards_intact": true,
        "maximum_absolute_error": maximumError, "maximum_tolerance_ratio": maximumToleranceRatio,
    ]
}

private func percentile(_ values: [Double], _ fraction: Double) -> Double {
    let sorted = values.sorted()
    let position = Double(sorted.count - 1) * fraction
    let lower = Int(position)
    let upper = min(lower + 1, sorted.count - 1)
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - Double(lower))
}

@main
private enum TensorOpsQuantizedMatmulProbe {
    static func main() {
        do {
            try run()
        } catch {
            fputs("Probe failed: \(error)\n", stderr)
            exit(1)
        }
    }

    static func run() throws {
        let arguments = CommandLine.arguments
        guard arguments.count == 3 || (arguments.count == 4 && arguments[3] == "--benchmark") else {
            throw ProbeError.failure("Usage: probe FIXTURE_DIRECTORY OUTPUT_DIRECTORY [--benchmark]")
        }
        let fixtureDirectory = URL(fileURLWithPath: arguments[1])
        let outputDirectory = URL(fileURLWithPath: arguments[2])
        let source = try String(contentsOf: fixtureDirectory.appendingPathComponent("affine-q4.metal"), encoding: .utf8)
        let runner = try Runner(source: source)
        print("GPU: \(runner.device.name); OS: \(ProcessInfo.processInfo.operatingSystemVersionString)")
        var checks: [[String: Any]] = []
        let shapes = [
            Shape(1, 31, 64, 32), Shape(4, 35, 128, 64), Shape(16, 32, 128, 128),
            Shape(17, 65, 192, 64), Shape(33, 35, 512, 32), Shape(17, 33, 2816, 128),
        ]
        for precision in Precision.allCases {
            for shape in shapes {
                let fixture = Fixture(shape: shape, precision: precision)
                let buffers = try runner.prepare(fixture)
                let staged = try runner.execute(fixture, buffers: buffers, kernel: "affine_q4_staged")
                let cooperative = try runner.execute(fixture, buffers: buffers, kernel: "affine_q4_cooperative")
                checks.append(try validate(fixture, staged, cooperative, sample: false))
                print("PASS \(precision.rawValue) \(shape.label)")
            }
        }
        var benchmarks: [[String: Any]] = []
        if arguments.count == 4 {
            let benchmarkShapes =
                [1, 4, 16, 64, 512].map { Shape($0, 1024, 1024, 64) }
                + [Shape(64, 2048, 2816, 64), Shape(512, 2048, 2816, 64)]
            for precision in Precision.allCases {
                for shape in benchmarkShapes {
                    let fixture = Fixture(shape: shape, precision: precision)
                    let buffers = try runner.prepare(fixture)
                    let staged = try runner.execute(fixture, buffers: buffers, kernel: "affine_q4_staged")
                    let cooperative = try runner.execute(fixture, buffers: buffers, kernel: "affine_q4_cooperative")
                    let check = try validate(fixture, staged, cooperative, sample: true)
                    for _ in 0..<3 {
                        _ = try runner.execute(fixture, buffers: buffers, kernel: "affine_q4_staged")
                        _ = try runner.execute(fixture, buffers: buffers, kernel: "affine_q4_cooperative")
                    }
                    var trials: [[String: Any]] = []
                    var stagedTimes: [Double] = []
                    var cooperativeTimes: [Double] = []
                    for trial in 0..<20 {
                        let order =
                            trial.isMultiple(of: 2)
                            ? ["affine_q4_staged", "affine_q4_cooperative"]
                            : ["affine_q4_cooperative", "affine_q4_staged"]
                        var timing: [String: Any] = ["trial": trial, "order": order]
                        for kernel in order {
                            let execution = try runner.execute(
                                fixture, buffers: buffers, kernel: kernel, repetitions: 5)
                            guard execution.gpuSeconds > 0, execution.gpuSeconds.isFinite else {
                                throw ProbeError.failure("GPU timestamps unavailable")
                            }
                            guard execution.guardsIntact, execution.output == cooperative.output else {
                                throw ProbeError.failure("Benchmark output changed: \(fixture.shape.label)")
                            }
                            timing[kernel] = [
                                "gpu_seconds": execution.gpuSeconds, "wall_seconds": execution.wallSeconds,
                            ]
                            if kernel == "affine_q4_staged" {
                                stagedTimes.append(execution.gpuSeconds)
                            } else {
                                cooperativeTimes.append(execution.gpuSeconds)
                            }
                        }
                        trials.append(timing)
                    }
                    let speedup = percentile(stagedTimes, 0.5) / percentile(cooperativeTimes, 0.5)
                    let forwardRatios = stagedTimes.indices.filter { $0.isMultiple(of: 2) }
                        .map { stagedTimes[$0] / cooperativeTimes[$0] }
                    let reverseRatios = stagedTimes.indices.filter { !$0.isMultiple(of: 2) }
                        .map { stagedTimes[$0] / cooperativeTimes[$0] }
                    benchmarks.append([
                        "validation": check, "trials": trials, "repetitions_per_command": 5,
                        "staged_median_seconds": percentile(stagedTimes, 0.5),
                        "cooperative_median_seconds": percentile(cooperativeTimes, 0.5),
                        "staged_p95_seconds": percentile(stagedTimes, 0.95),
                        "cooperative_p95_seconds": percentile(cooperativeTimes, 0.95),
                        "median_speedup_vs_matched_control": speedup,
                        "staged_first_paired_median_speedup": percentile(forwardRatios, 0.5),
                        "cooperative_first_paired_median_speedup": percentile(reverseRatios, 0.5),
                    ])
                    print(
                        String(
                            format: "%@ %@: %.3fx vs matched staging control", precision.rawValue, fixture.shape.label,
                            speedup))
                }
            }
        }
        // Native scale-plane FP4 support is an OS 27 API. Never turn an uncompiled
        // future shader into a claim of tested native FP4 execution.
        let mxStatus =
            ProcessInfo.processInfo.operatingSystemVersion.majorVersion < 27
            ? "unsupported_os_requires_27" : "scaffold_not_validated"
        let report: [String: Any] = [
            "schema_version": 1,
            "device": runner.device.name,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "control": "Matched single-simdgroup staging kernel, not stock MLX",
            "quantization": "Existing affine Q4 grid; no re-quantization",
            "correctness": checks, "benchmarks": benchmarks,
            "direct_mxfp4": ["status": mxStatus, "minimum_os_major": 27],
            "scope": "Primitive probe only; no MLX dispatch or model integration",
            "compiler_options": ["metal_language": "4.0", "math_mode": "safe", "floating_point_functions": "precise"],
            "timing_scope":
                "GPU command duration per dispatch; wall time commit-to-completion. Allocation, compilation and model execution excluded.",
            "reference_tolerance":
                "1e-5 + abs(dot)*dtypeTolerance + sum(abs(products))*8e-6; dtypeTolerance FP16=0.0011, BF16=0.008",
        ]
        let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try json.write(to: outputDirectory.appendingPathComponent("report.json"), options: .atomic)
        print("Report: \(outputDirectory.appendingPathComponent("report.json").path)")
    }
}
