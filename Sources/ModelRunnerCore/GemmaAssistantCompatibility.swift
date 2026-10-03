import Foundation
import ModelRunnerProtocol

/// Reject incompatible shared-KV checkpoints before constructing a generation graph.
public enum GemmaAssistantCompatibility {
    /// Validates a Gemma target and assistant checkpoint before loading weights.
    public static func validate(target: Data, assistant: Data, blockSize: Int?, quantizationBits: Int? = nil) throws
        -> Int
    {
        guard let root = try JSONSerialization.jsonObject(with: target) as? [String: Any],
            let type = root["model_type"] as? String,
            ["gemma4", "gemma4_text"].contains(type)
        else {
            throw RequestAdmissionError.configuration("Gemma assistant requires a Gemma 4 text target")
        }
        let text = root["text_config"] as? [String: Any] ?? root
        guard let assistantRoot = try JSONSerialization.jsonObject(with: assistant) as? [String: Any],
            let assistantText = assistantRoot["text_config"] as? [String: Any]
        else {
            throw RequestAdmissionError.configuration("Gemma assistant has no text configuration")
        }
        guard assistantRoot["model_type"] as? String == "gemma4_assistant" else {
            throw RequestAdmissionError.configuration("Checkpoint is not gemma4_assistant")
        }
        if let bits = quantizationBits {
            guard [4, 8].contains(bits), assistantRoot["quantization"] == nil,
                assistantRoot["quantization_config"] == nil
            else {
                throw RequestAdmissionError.configuration(
                    "Assistant quantization requires an unquantized source and 4 or 8 bits")
            }
        }
        guard (text["attention_k_eq_v"] as? Bool ?? false) == (assistantText["attention_k_eq_v"] as? Bool ?? false),
            NSDictionary(dictionary: text["rope_parameters"] as? [String: Any] ?? [:]).isEqual(
                to: assistantText["rope_parameters"] as? [String: Any] ?? [:])
        else {
            throw RequestAdmissionError.configuration(
                "Gemma assistant rotary attention configuration does not match target")
        }
        for key in [
            "hidden_size", "vocab_size", "num_key_value_heads", "head_dim", "global_head_dim", "sliding_window",
        ] {
            let expected = key == "hidden_size" ? assistantRoot["backbone_hidden_size"] : assistantText[key]
            guard let actual = text[key] as? Int, let expected = expected as? Int,
                actual > 0, actual == expected
            else {
                throw RequestAdmissionError.configuration("Gemma assistant target mismatch: \(key)")
            }
        }
        let globalHeads = text["num_global_key_value_heads"] as? Int ?? text["num_key_value_heads"] as? Int
        let draftGlobalHeads =
            assistantText["num_global_key_value_heads"] as? Int ?? assistantText["num_key_value_heads"] as? Int
        guard globalHeads == draftGlobalHeads,
            let layers = text["layer_types"] as? [String],
            let draftLayers = assistantText["layer_types"] as? [String],
            !draftLayers.isEmpty, Set(draftLayers).isSubset(of: Set(layers)),
            let layerCount = assistantText["num_hidden_layers"] as? Int,
            layerCount == draftLayers.count,
            assistantText["num_kv_shared_layers"] as? Int == layerCount
        else {
            throw RequestAdmissionError.configuration("Gemma assistant shared attention layout does not match target")
        }
        let maximum = assistantRoot["block_size"] as? Int ?? 4
        let effective = blockSize ?? maximum
        guard effective >= 2, effective <= maximum else {
            throw RequestAdmissionError.configuration("Gemma assistant block size must be 2...\(maximum)")
        }
        return effective
    }
}
