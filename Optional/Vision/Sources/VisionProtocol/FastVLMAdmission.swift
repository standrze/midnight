import Foundation

/// Bounds derived from the pinned native FastVLM implementation, before model allocation/prefill.
/// Its two stride-2 stem convolutions and four stride-2 patch merges turn 1024² pixels
/// into 16² image positions. Other architectures need their own verified admission rule.
public struct FastVLMAdmission: Sendable {
    public static let imagePositions = 256
    public let contextLimit: Int
    /// Validates the supported FastVLM image layout and context limit.
    public init(configuration: Data, processor: Data) throws {
        guard let config = try JSONSerialization.jsonObject(with: configuration) as? [String: Any],
            let preprocessor = try JSONSerialization.jsonObject(with: processor) as? [String: Any],
            let modelType = config["model_type"] as? String, ["fastvlm", "llava_qwen2"].contains(modelType),
            let vision = config["vision_config"] as? [String: Any],
            (vision["layers"] as? [Int])?.count == 5,
            vision["downsamples"] as? [Bool] == [true, true, true, true, true],
            vision["down_stride"] as? Int == 2, vision["down_patch_size"] as? Int == 7,
            vision["image_size"] as? Int == 1024,
            config["image_token_index"] == nil || config["image_token_index"] as? Int == -200,
            preprocessor["processor_class"] as? String == "FastVLMProcessor",
            let crop = preprocessor["crop_size"] as? [String: Int], crop["width"] == 1024, crop["height"] == 1024,
            let configuredContext = config["tokenizer_model_max_length"] as? Int, configuredContext > 0
        else {
            throw VisionError(
                "unsupported_vision_model",
                "This initial native worker supports FastVLM with its verified 1024-pixel, 256-image-position layout. Other vision architectures are not admitted yet."
            )
        }
        contextLimit = min(8192, configuredContext)
    }
    /// Returns the expanded prompt positions after image admission and token limits.
    public func promptPositions(tokenIDs: [Int], maximumTokens: Int) throws -> Int {
        guard tokenIDs.filter({ $0 == -200 }).count == 1 else {
            throw VisionError(
                "invalid_image_placeholders",
                "The rendered vision prompt must contain exactly one image placeholder. Do not include literal image control tokens in message text."
            )
        }
        let expanded = tokenIDs.count - 1 + Self.imagePositions
        guard expanded <= contextLimit - maximumTokens else {
            throw VisionError(
                "context_too_large",
                "Vision prompt including all 256 image positions plus requested output exceeds the \(contextLimit)-token context budget."
            )
        }
        return expanded
    }
}
