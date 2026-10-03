import Foundation
import ModelRunnerProtocol

/// Opt-in managed weight placement. Computation and KV caches remain on CUDA.
enum CUDALayerOffloadMode: Equatable, Sendable {
    static let environmentKey = "MODEL_RUNNER_CUDA_OFFLOAD_LAYERS"
    case disabled
    case automatic
    case layers(Int)

    init(environment: [String: String], engine: ModelEngine) throws {
        let raw = environment[Self.environmentKey] ?? "0"
        if raw == "0" {
            self = .disabled
            return
        }
        guard engine == .cuda else {
            throw RequestAdmissionError.configuration("\(Self.environmentKey) requires the CUDA engine")
        }
        if raw == "auto" {
            self = .automatic
        } else if !raw.isEmpty, raw.allSatisfy({ $0.isASCII && $0.isNumber }), let count = Int(raw), count > 0 {
            self = .layers(count)
        } else {
            throw RequestAdmissionError.configuration(
                "\(Self.environmentKey) must be 0, auto, or a positive layer count")
        }
    }
}

/// Pure selection policy, based on actual parameter payloads rather than shard sizes.
struct CUDALayerOffloadPlan: Equatable, Sendable {
    let layerPaths: Set<String>
    let hostWeightBytes: Int
    let deviceWeightBytes: Int
    let workspaceReserveBytes: Int

    static func layerPath(for parameter: String) -> String? {
        let parts = parameter.split(separator: ".").map(String.init)
        for index in parts.indices.dropLast() {
            if ["layers", "h", "blocks"].contains(parts[index]), Int(parts[index + 1]) != nil {
                return parts[...(index + 1)].joined(separator: ".")
            }
        }
        return nil
    }

    init(mode: CUDALayerOffloadMode, layerBytes: [String: Int], stationaryBytes: Int, gpuBudget: Int, hostBudget: Int)
        throws
    {
        guard stationaryBytes >= 0, layerBytes.values.allSatisfy({ $0 > 0 }), gpuBudget > 0, hostBudget >= 0 else {
            throw RequestAdmissionError.configuration("Invalid CUDA weight-placement sizes")
        }
        let total = layerBytes.values.reduce(stationaryBytes, ModelMemoryProfile.add)
        let reserve = max(512 * 1_048_576, gpuBudget / 8)
        let deviceWeightBudget = max(0, gpuBudget - reserve)
        let ordered = layerBytes.keys.sorted { $0.compare($1, options: .numeric) == .orderedDescending }
        var selected = Set<String>()
        var hostBytes = 0
        let requested: Int?
        if case .layers(let count) = mode { requested = count } else { requested = nil }
        if let requested, requested > ordered.count {
            throw RequestAdmissionError.configuration(
                "Requested \(requested) CUDA RAM layers, but only \(ordered.count) eligible layers exist")
        }
        for path in ordered {
            let needed: Bool
            switch mode {
            case .disabled: needed = false
            case .automatic: needed = total - hostBytes > deviceWeightBudget
            case .layers(let count): needed = selected.count < count
            }
            if !needed { break }
            hostBytes = ModelMemoryProfile.add(hostBytes, layerBytes[path]!)
            selected.insert(path)
        }
        guard hostBytes <= hostBudget else {
            throw RequestAdmissionError.memoryExceeded(required: hostBytes, available: hostBudget)
        }
        guard total - hostBytes <= deviceWeightBudget else {
            throw RequestAdmissionError.configuration(
                "CUDA weights still exceed the GPU budget after layer placement; increase the RAM layer count, use auto, or choose smaller weights"
            )
        }
        self.layerPaths = selected
        self.hostWeightBytes = hostBytes
        self.deviceWeightBytes = total - hostBytes
        self.workspaceReserveBytes = reserve
    }
}

enum CUDAHostMemory {
    static let limitKey = "MODEL_RUNNER_CUDA_HOST_LIMIT_GIB"

    /// MemAvailable includes reclaimable pages. A cgroup ceiling additionally bounds it.
    static func availableBytes(meminfo: String, cgroupLimit: String? = nil, cgroupCurrent: String? = nil) throws -> Int
    {
        let line = meminfo.split(separator: "\n").first { $0.hasPrefix("MemAvailable:") }
        let parts = line?.split(whereSeparator: \.isWhitespace) ?? []
        guard parts.count == 3, parts[2] == "kB", let kib = Int(parts[1]), kib >= 0, kib <= Int.max / 1024 else {
            throw RequestAdmissionError.configuration("Linux MemAvailable could not be determined for CUDA RAM offload")
        }
        var available = kib * 1024
        if let cgroupLimit, cgroupLimit.trimmingCharacters(in: .whitespacesAndNewlines) != "max" {
            guard let limit = Int(cgroupLimit.trimmingCharacters(in: .whitespacesAndNewlines)), limit >= 0,
                let current = cgroupCurrent.flatMap({ Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }),
                current >= 0
            else { throw RequestAdmissionError.configuration("Invalid cgroup memory ceiling for CUDA RAM offload") }
            available = min(available, max(0, limit - current))
        }
        return available
    }

    static func budget(physical: UInt64, available: Int, environment: [String: String]) throws -> Int {
        let physical = Int(min(physical, UInt64(Int.max)))
        let reserve = max(2 * 1_073_741_824, physical / 5)
        let maximum = max(0, min(available, physical) - reserve)
        guard let raw = environment[limitKey] else { return maximum }
        let pieces = raw.split(separator: ".", omittingEmptySubsequences: false)
        guard pieces.count <= 2, pieces.allSatisfy({ !$0.isEmpty && $0.allSatisfy { $0.isASCII && $0.isNumber } }),
            let value = Double(raw), value.isFinite, value > 0,
            value * 1_073_741_824 <= Double(maximum)
        else {
            throw RequestAdmissionError.configuration(
                "\(limitKey) must be a positive plain decimal within available host RAM after reserve")
        }
        return Int(value * 1_073_741_824)
    }

    static func resolve(environment: [String: String]) throws -> Int {
        let meminfo = try String(contentsOfFile: "/proc/meminfo", encoding: .utf8)
        let membership = try String(contentsOfFile: "/proc/self/cgroup", encoding: .utf8)
        let unified = membership.split(separator: "\n").first { $0.hasPrefix("0::") }.map { String($0.dropFirst(3)) }
        let cgroupURL = unified.map { URL(fileURLWithPath: "/sys/fs/cgroup").appendingPathComponent($0) }
        let limit = cgroupURL.flatMap {
            try? String(contentsOf: $0.appendingPathComponent("memory.max"), encoding: .utf8)
        }
        let current = cgroupURL.flatMap {
            try? String(contentsOf: $0.appendingPathComponent("memory.current"), encoding: .utf8)
        }
        return try budget(
            physical: ProcessInfo.processInfo.physicalMemory,
            available: availableBytes(meminfo: meminfo, cgroupLimit: limit, cgroupCurrent: current),
            environment: environment)
    }
}
