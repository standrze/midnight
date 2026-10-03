import Foundation
import MLXLMCommon

/// Local checkpoints do not inherit the model-ID presets in MLX's registry.
/// Select protocols from checkpoint metadata so renaming a folder is harmless.
enum LocalModelChatConfiguration {
    static func make(
        directory: URL,
        modelType: String?,
        additionalEOSTokens: Set<String> = []
    ) -> ModelConfiguration {
        var configuration = ModelConfiguration(
            directory: directory,
            extraEOSTokens: Set(["<end_of_turn>", "<turn|>"]).union(additionalEOSTokens)
        )
        if modelType == "muse_glimmer" {
            // ATEM's token-aware Onyx decoder separates recipient headers, private
            // reasoning and function calls. The JSON fallback emits them as text.
            configuration.toolCallFormat = .atem
            configuration.extraEOSTokens.formUnion(["<|eot|>", "<|end_of_text|>"])
            // <|eom|> ends a frame, not a turn: reasoning may precede a tool call.
        }
        if modelType == "laguna" {
            configuration.toolCallFormat = .glm4
            configuration.reasoningConfig = .thinkTagsWithEnableThinking
            configuration.extraEOSTokens.insert("</assistant>")
        }
        return configuration
    }
}
