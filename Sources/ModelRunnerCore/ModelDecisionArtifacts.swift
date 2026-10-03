import CryptoKit
import Foundation
import ModelRunnerProtocol

/// Verify that a decision contract belongs to the selected local weights.
enum ModelDecisionArtifacts {
    static func validate(contract: DecisionModelContract, base: URL, adapter: URL?) throws {
        if let expected = contract.baseFingerprint {
            guard try modelHash(base) == expected else {
                throw DecisionError.invalidContract("Base fingerprint differs from the decision export.")
            }
        } else {
            let metadata = base.appendingPathComponent("midnight-download.json")
            guard let bytes = try? Data(contentsOf: metadata),
                let identity = try JSONSerialization.jsonObject(with: bytes) as? [String: String],
                identity["repository"] == contract.model, identity["revision"] == contract.revision
            else {
                throw DecisionError.invalidContract(
                    "Decision scoring requires the pinned publisher base or an export base fingerprint.")
            }
        }
        if let expected = contract.adapterFingerprint {
            guard let adapter, try hash(adapter.appendingPathComponent("adapters.safetensors")) == expected else {
                throw DecisionError.invalidContract("Adapter fingerprint differs from the decision export.")
            }
        }
    }

    private static func hash(_ file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var digest = SHA256()
        while let bytes = try handle.read(upToCount: 1_048_576), !bytes.isEmpty { digest.update(data: bytes) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func modelHash(_ directory: URL) throws -> String {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter {
                $0.pathExtension == "safetensors"
                    || ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"].contains(
                        $0.lastPathComponent)
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard files.contains(where: { $0.pathExtension == "safetensors" }) else {
            throw DecisionError.invalidContract("No base weights.")
        }
        var digest = SHA256()
        for file in files { digest.update(data: Data((file.lastPathComponent + ":" + (try hash(file)) + "\n").utf8)) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
