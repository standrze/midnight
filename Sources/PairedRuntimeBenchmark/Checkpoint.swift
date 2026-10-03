import ArgumentParser
import Crypto
import Foundation
import ModelRunnerProtocol

struct CheckpointRecord: Encodable {
    let arm: String
    let path: String
    let configurationSHA256: String
    let quantizationJSON: String?
    let storedWeightBytes: Int
    let shards: [ShardRecord]
}

struct ShardRecord: Encodable {
    let name: String
    let bytes: Int
    let modificationDate: Date?
}

struct CheckedCheckpoint {
    let record: CheckpointRecord
    let profile: ModelMemoryProfile

    init(arm: String, path: String, options: LongContextOptions) throws {
        let directory = canonicalURL(path)
        let configuration = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let object = try JSONSerialization.jsonObject(with: configuration) as? [String: Any],
            object["model_type"] as? String == "gemma3_text"
        else {
            throw ValidationError("Arm \(arm) must be a native Gemma 3 text checkpoint.")
        }
        let expected = [
            "hidden_size": 640, "intermediate_size": 2048, "num_hidden_layers": 18,
            "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256, "vocab_size": 262_144,
        ]
        for (key, value) in expected {
            guard let actual = object[key] as? NSNumber, actual.stringValue == String(value) else {
                throw ValidationError("Arm \(arm) must have Gemma 3 270M geometry: \(key)=\(value).")
            }
        }
        try Self.checkQuantization(object["quantization"], arm: arm)
        try Self.checkQuantization(object["quantization_config"], arm: arm)
        let quantizationJSON = try object["quantization"].map {
            String(decoding: try JSONSerialization.data(withJSONObject: $0, options: [.sortedKeys]), as: UTF8.self)
        }
        let shardURLs = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey, .contentModificationDateKey]
        ).filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !shardURLs.isEmpty else {
            throw ValidationError("Arm \(arm) has no safetensors shards.")
        }
        let shards = try shardURLs.map { url -> ShardRecord in
            let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey, .contentModificationDateKey])
            guard values.isRegularFile == true, let bytes = values.fileSize, bytes > 0 else {
                throw ValidationError("Invalid safetensors shard: \(url.path)")
            }
            return ShardRecord(
                name: url.lastPathComponent, bytes: bytes, modificationDate: values.contentModificationDate)
        }
        record = CheckpointRecord(
            arm: arm, path: directory.path, configurationSHA256: sha256(configuration),
            quantizationJSON: quantizationJSON,
            storedWeightBytes: shards.reduce(0) { ModelMemoryProfile.add($0, $1.bytes) }, shards: shards)
        profile = try ModelMemoryProfile(configuration: configuration, options: options)
    }

    private static func checkQuantization(_ value: Any?, arm: String) throws {
        guard let object = value as? [String: Any] else {
            return
        }
        if let bits = object["bits"] as? NSNumber,
            !(4...16).contains(bits.intValue) || bits.doubleValue != Double(bits.intValue)
        {
            throw ValidationError("Arm \(arm) contains an unsupported bit width; this experiment requires >=4 bits.")
        }
        for nested in object.values where nested is [String: Any] {
            try checkQuantization(nested, arm: arm)
        }
    }
}

func canonicalURL(_ path: String) -> URL {
    URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
        .standardizedFileURL.resolvingSymlinksInPath()
}

func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func runtimeEnvironment() -> [String: String] {
    ProcessInfo.processInfo.environment.filter { key, _ in
        let runtime =
            key.hasPrefix("MODEL_RUNNER_") || key.hasPrefix("MIDNIGHT_GEMMA")
            || key.hasPrefix("MIDNIGHT_METAL_") || key.hasPrefix("MIDNIGHT_MTP_") || key.hasPrefix("MLX_")
        let upper = key.uppercased()
        let secret =
            upper.hasSuffix("_TOKEN")
            || ["SECRET", "PASSWORD", "API_KEY", "ACCESS_KEY"].contains { upper.contains($0) }
        return runtime && !secret
    }
}
