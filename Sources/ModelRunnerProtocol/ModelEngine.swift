import Foundation

/// Selects the device runtime requested for a model load.
///
/// `auto` uses the backend compiled into this build; an explicit Metal or CUDA
/// request fails if that backend is unavailable.
public enum ModelEngine: String, Codable, CaseIterable, Sendable {
    case auto
    case metal
    case cuda
    case cpu

    /// Parses a case-insensitive command-line engine name.
    public init(argument: String) throws {
        let normalized = argument.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let engine = Self(rawValue: normalized) else {
            throw ModelEngineError.unknownEngine(argument)
        }
        self = engine
    }

    /// Resolves `auto` and validates an explicit request against the compiled backend.
    public func resolve(for backend: CompiledMLXBackend = .current) throws -> ModelEngine {
        switch self {
        case .auto:
            return backend.engine
        case .cpu:
            return .cpu
        case .metal where backend == .metal:
            return .metal
        case .cuda where backend == .cuda:
            return .cuda
        case .metal, .cuda:
            throw ModelEngineError.unavailable(requested: self, compiled: backend)
        }
    }
}

/// The MLX backend selected when Midnight was built.
public enum CompiledMLXBackend: String, Codable, Sendable {
    case metal
    case cuda
    case cpu

    /// The backend indicated by this build's Swift compilation flags.
    public static var current: Self {
        #if MLX_METAL_BACKEND
            .metal
        #elseif MLX_CUDA_BACKEND
            .cuda
        #else
            .cpu
        #endif
    }

    /// The corresponding user-facing engine selection.
    public var engine: ModelEngine {
        switch self {
        case .metal: .metal
        case .cuda: .cuda
        case .cpu: .cpu
        }
    }
}

/// A command-line engine name is unknown or unavailable in this build.
public enum ModelEngineError: LocalizedError, Equatable {
    case unknownEngine(String)
    case unavailable(requested: ModelEngine, compiled: CompiledMLXBackend)

    /// User-facing explanation for this error.
    public var errorDescription: String? {
        switch self {
        case .unknownEngine(let value):
            let choices = ModelEngine.allCases.map(\.rawValue).joined(separator: ", ")
            return "Unknown engine '\(value)'. Choose one of: \(choices)."
        case .unavailable(let requested, let compiled):
            return "Engine '\(requested.rawValue)' is unavailable in this \(compiled.rawValue) build."
        }
    }
}
