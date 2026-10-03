import ArgumentParser
import Foundation

/// Publisher-owned download choices; unavailable entries never substitute another source.
enum ModelDownloadCatalog {
    struct Entry: Sendable {
        let name: String
        let repository: String
        let assistant: String?
        let note: String
        let unavailableReason: String?
    }

    static let entries: [Entry] =
        [
            Entry(
                name: "nimble-9b", repository: "bespokelabs/Bespoke-Nimble-9B", assistant: nil,
                note: "Decision model · official BF16 Qwen3.5-9B base + Bespoke LoRA; prepare in Afterglow.",
                unavailableReason: nil),
            Entry(
                name: "gemma-26b", repository: "google/gemma-4-26B-A4B-it", assistant: nil,
                note: "Tier 1", unavailableReason: "No verified publisher-owned 4-bit native MLX release."),
            Entry(
                name: "gemma-31b", repository: "google/gemma-4-31B-it", assistant: nil,
                note: "Tier 1",
                unavailableReason: "Official GGUF/ compressed-tensors QAT needs a different loading path."),
            Entry(
                name: "gpt-oss-20b", repository: "openai/gpt-oss-20b", assistant: nil,
                note: "Tier 2 · official MXFP4; MLX can expand weights in memory; no official assistant verified.",
                unavailableReason: nil),
            Entry(
                name: "laguna-xs-2.1", repository: "poolside/Laguna-XS-2.1-NVFP4-mlx",
                assistant: "poolside/Laguna-XS-2.1-DFlash-NVFP4",
                note: "Tier 2 · official NVFP4 MLX + BF16 NVFP4-target assistant; pairing requires runtime evaluation.",
                unavailableReason: nil),
            Entry(
                name: "qwen-28b", repository: "Qwen/Qwen3.8-27B", assistant: nil,
                note: "Tier 3", unavailableReason: "No verified publisher-owned 4-bit native MLX release."),
            Entry(
                name: "laguna-s-2.1", repository: "poolside/Laguna-S-2.1-NVFP4-mlx", assistant: nil,
                note: "Official NVFP4 MLX; exceeds the default 30 GB limit; assistant bundle not selected.",
                unavailableReason: nil),
            Entry(
                name: "gpt-oss-120b", repository: "openai/gpt-oss-120b", assistant: nil,
                note: "Official MXFP4; exceeds the default 30 GB limit; MLX can expand weights in memory.",
                unavailableReason: nil),
            Entry(
                name: "devstral-small-2", repository: "mistralai/Devstral-Small-2-24B-Instruct-2512", assistant: nil,
                note: "Support coverage", unavailableReason: "No verified publisher-owned 4-bit native MLX release."),
        ] + pendingSupport

    private static let pendingSupport: [Entry] = [
        ("gemma-e2b", "google/gemma-4-E2B-it"),
        ("gemma-e4b", "google/gemma-4-E4B-it"),
        ("gemma-12b", "google/gemma-4-12B-it"),
        ("gemma3-270m", "google/gemma-3-270m-it"),
        ("gemma3-1b", "google/gemma-3-1b-it"),
        ("gemma3-4b", "google/gemma-3-4b-it"),
        ("gemma3-12b", "google/gemma-3-12b-it"),
        ("gemma3-27b", "google/gemma-3-27b-it"),
        ("voxtral-mini", "mistralai/Voxtral-Mini-3B-2507"),
        ("voxtral-small", "mistralai/Voxtral-Small-24B-2507"),
        ("voxtral-realtime", "mistralai/Voxtral-Mini-4B-Realtime-2602"),
        ("voxtral-tts", "mistralai/Voxtral-4B-TTS-2603"),
    ].map { name, repository in
        Entry(
            name: name, repository: repository, assistant: nil, note: "Support coverage",
            unavailableReason: "No verified publisher-owned approximately 4-bit native MLX source is registered.")
    }

    static func selection(repository: String) throws -> Entry {
        guard let entry = entries.first(where: { $0.repository == repository }) else {
            throw ValidationError(
                "No approved publisher-owned approximately 4-bit source is registered for \(repository).")
        }
        if let reason = entry.unavailableReason {
            throw ValidationError(reason)
        }
        return entry
    }

    /// Metadata must confirm the quantization rather than trusting a repository's name.
    static func validateQuantization(_ data: Data) throws {
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let quantization = (object?["quantization"] ?? object?["quantization_config"]) as? [String: Any]
        let method = quantization?["quant_method"] as? String
        guard quantization?["bits"] as? Int == 4 || method == "mxfp4" else {
            throw ValidationError("The publisher configuration does not confirm supported approximately 4-bit weights.")
        }
    }
}
