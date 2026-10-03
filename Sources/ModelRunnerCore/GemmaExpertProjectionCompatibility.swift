import Foundation
import ModelRunnerProtocol

/// The experimental packed-row conversion has only been validated for this layout.
public enum GemmaExpertProjectionCompatibility {
    /// Rejects unsupported expert projection settings for the selected backend.
    public static func validate(configuration: Data, engine: ModelEngine, hasAdapter: Bool) throws {
        guard engine == .metal, !hasAdapter else {
            throw RequestAdmissionError.configuration(
                "Combined Gemma expert projections require Metal without an adapter")
        }
        guard let root = try JSONSerialization.jsonObject(with: configuration) as? [String: Any],
            let type = root["model_type"] as? String,
            ["gemma4", "gemma4_text"].contains(type)
        else {
            throw RequestAdmissionError.configuration("Combined expert projections require Gemma 4 26B-A4B")
        }
        let text = root["text_config"] as? [String: Any] ?? root
        for (key, expected) in [
            "hidden_size": 2816, "num_hidden_layers": 30,
            "num_experts": 128, "top_k_experts": 8, "moe_intermediate_size": 704,
        ] {
            guard text[key] as? Int == expected else {
                throw RequestAdmissionError.configuration("Unsupported Gemma expert layout: \(key)")
            }
        }
        guard text["enable_moe_block"] as? Bool == true,
            let quantization = root["quantization"] as? [String: Any],
            matches(quantization)
        else {
            throw RequestAdmissionError.configuration(
                "Combined A4B projections require affine Q4 weights with group size 64")
        }
        for (key, value) in quantization
        where key.contains("experts") && (key.contains("gate_proj") || key.contains("up_proj")) {
            guard let setting = value as? [String: Any], matches(setting) else {
                throw RequestAdmissionError.configuration("Unsupported expert quantization override: \(key)")
            }
        }
    }

    private static func matches(_ value: [String: Any]) -> Bool {
        let mode = value["mode"] == nil ? "affine" : value["mode"] as? String
        return value["bits"] as? Int == 4 && value["group_size"] as? Int == 64 && mode == "affine"
    }
}
