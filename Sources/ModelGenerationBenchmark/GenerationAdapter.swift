import Foundation

/// File provenance does not establish tensor compatibility or adapter quality.
struct GenerationAdapterReport: Encodable {
    var path: String
    var requestedScale: Float?
    var configuredScale: Float?
    var effectiveScale: Float?
    var configSha256: String?
    var weightsSha256: String?
    var configBytes: Int?
    var weightsBytes: Int?
    var validation = "pending"
    var loaded = false
    var provenanceVerification = "observed_preload"
    var filesUnchangedAfterLoad: Bool?
    var filesUnchangedAfterRun: Bool?
}

// Match LoRAContainer.from(directory:)'s native schema and defaults. Do not
// introduce a second adapter loader or instantiate MLX during --validate-only.
private struct NativeGenerationAdapterConfiguration: Decodable {
    enum FineTuneType: String, Decodable {
        case lora
        case dora
    }

    struct Parameters: Decodable {
        let rank: Int
        let scale: Float
        let dropout: Float
        let keys: [String]?

        enum CodingKeys: String, CodingKey {
            case rank, scale, dropout, keys
        }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            rank = try container.decode(Int.self, forKey: .rank)
            scale = try container.decode(Float.self, forKey: .scale)
            let dropout = try container.decodeIfPresent(Float.self, forKey: .dropout) ?? 0
            guard (0.0..<1.0).contains(dropout) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .dropout, in: container,
                    debugDescription: "LoRA dropout must be in the range [0, 1)")
            }
            self.dropout = dropout
            keys = try container.decodeIfPresent([String].self, forKey: .keys)
        }
    }

    let numLayers: Int
    let fineTuneType: FineTuneType
    let loraParameters: Parameters

    enum CodingKeys: String, CodingKey {
        case numLayers = "num_layers"
        case fineTuneType = "fine_tune_type"
        case loraParameters = "lora_parameters"
    }
}

func preflightGenerationAdapter(_ adapter: inout GenerationAdapterReport) throws {
    let directory = URL(fileURLWithPath: adapter.path, isDirectory: true)
    let config = try Data(contentsOf: directory.appendingPathComponent("adapter_config.json"))
    adapter.configBytes = config.count
    adapter.configSha256 = generationSHA256(config)
    let configuration = try JSONDecoder().decode(NativeGenerationAdapterConfiguration.self, from: config)
    adapter.configuredScale = configuration.loraParameters.scale
    adapter.effectiveScale = adapter.requestedScale ?? configuration.loraParameters.scale
    let weights = try Data(contentsOf: directory.appendingPathComponent("adapters.safetensors"))
    adapter.weightsBytes = weights.count
    adapter.weightsSha256 = generationSHA256(weights)
    adapter.validation = "files_and_config_only"
    if adapter.configSha256 == nil || adapter.weightsSha256 == nil {
        adapter.provenanceVerification = "observed_preload_only_sha256_unavailable"
    }
}

enum GenerationAdapterVerificationPhase: Equatable {
    case load
    case run
}

func verifyGenerationAdapterFiles(
    _ adapter: inout GenerationAdapterReport, after phase: GenerationAdapterVerificationPhase
) throws {
    guard let configHash = adapter.configSha256, let weightsHash = adapter.weightsSha256 else {
        adapter.provenanceVerification = "observed_preload_only_sha256_unavailable"
        return
    }
    if phase == .load {
        adapter.filesUnchangedAfterLoad = false
    } else {
        adapter.filesUnchangedAfterRun = false
    }
    adapter.provenanceVerification = "verification_failed"
    let directory = URL(fileURLWithPath: adapter.path, isDirectory: true)
    let config = try Data(contentsOf: directory.appendingPathComponent("adapter_config.json"))
    let weights = try Data(contentsOf: directory.appendingPathComponent("adapters.safetensors"))
    guard config.count == adapter.configBytes, weights.count == adapter.weightsBytes,
        generationSHA256(config) == configHash, generationSHA256(weights) == weightsHash
    else {
        throw GenerationBenchmarkError.invalidResult("adapter files changed after preflight")
    }
    if phase == .load {
        adapter.filesUnchangedAfterLoad = true
        adapter.provenanceVerification = "checked_after_load"
    } else {
        adapter.filesUnchangedAfterRun = true
        adapter.provenanceVerification = "checked_after_run"
    }
}
