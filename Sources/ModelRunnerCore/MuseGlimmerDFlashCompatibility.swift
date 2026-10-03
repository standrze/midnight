import Foundation
import ModelRunnerProtocol

/// The checkpoint-level contract between Muse Glimmer and its DFlash drafter.
///
/// This deliberately validates only metadata: the target and drafter have
/// different numbers of KV heads, and the drafter borrows the target embedding
/// and LM head instead of carrying either tensor itself. Loading either set of
/// weights before this check makes an otherwise actionable configuration error
/// look like an allocator or tensor-name failure.
public enum MuseGlimmerDFlashCompatibility {
    /// Validated target and drafter geometry before any weights are loaded.
    public struct Descriptor: Equatable, Sendable {
        public let hiddenSize: Int
        public let vocabularySize: Int
        public let targetLayerCount: Int
        public let targetLayerIDs: [Int]
        /// Maximum draft token IDs proposed per verification block.
        public let blockSize: Int
        public let maskTokenID: Int
        /// Target attention window measured in token positions.
        public let slidingWindow: Int
    }

    /// Checks target and drafter JSON metadata before loading either checkpoint.
    public static func validate(target: Data, assistant: Data, blockSize requested: Int?) throws -> Descriptor {
        guard let targetRoot = try JSONSerialization.jsonObject(with: target) as? [String: Any],
            targetRoot["model_type"] as? String == "muse_glimmer",
            let text = targetRoot["text_config"] as? [String: Any]
        else {
            throw RequestAdmissionError.configuration("Muse DFlash requires a muse_glimmer target")
        }
        guard let draft = try JSONSerialization.jsonObject(with: assistant) as? [String: Any],
            draft["model_type"] as? String == "muse_glimmer_assistant",
            (draft["architectures"] as? [String])?.contains("MuseGlimmerAssistantModel") == true
        else {
            throw RequestAdmissionError.configuration("Checkpoint is not a Muse Glimmer assistant")
        }
        func positive(_ value: Any?) -> Int? {
            guard let number = value as? NSNumber, number.intValue > 0,
                number.doubleValue == Double(number.intValue)
            else {
                return nil
            }
            return number.intValue
        }
        guard let hiddenSize = positive(text["hidden_size"]),
            let vocabularySize = positive(text["vocab_size"]),
            let targetLayerCount = positive(text["num_hidden_layers"]),
            let draftHiddenSize = positive(draft["hidden_size"]), draftHiddenSize == hiddenSize,
            let blockMaximum = positive(draft["block_size"]),
            let maskTokenID = draft["mask_token_id"] as? Int,
            let slidingWindow = positive(draft["sliding_window"]),
            let targetLayerIDs = draft["target_layer_ids"] as? [Int], !targetLayerIDs.isEmpty,
            targetLayerIDs == targetLayerIDs.sorted(), Set(targetLayerIDs).count == targetLayerIDs.count,
            targetLayerIDs.allSatisfy({ (0..<targetLayerCount).contains($0) })
        else {
            throw RequestAdmissionError.configuration("Muse assistant target dimensions or layer taps are incompatible")
        }
        guard maskTokenID >= 0, maskTokenID < vocabularySize else {
            throw RequestAdmissionError.configuration("Muse assistant mask_token_id is outside the target vocabulary")
        }
        guard let types = draft["layer_types"] as? [String],
            types.count == positive(draft["num_hidden_layers"]),
            types.allSatisfy({ $0 == "sliding_attention" })
        else {
            throw RequestAdmissionError.configuration("Muse assistant requires sliding_attention draft layers")
        }
        let effectiveBlockSize = requested ?? blockMaximum
        guard (2...blockMaximum).contains(effectiveBlockSize) else {
            throw RequestAdmissionError.configuration("Muse assistant block size must be 2...\(blockMaximum)")
        }
        return Descriptor(
            hiddenSize: hiddenSize, vocabularySize: vocabularySize,
            targetLayerCount: targetLayerCount, targetLayerIDs: targetLayerIDs,
            blockSize: effectiveBlockSize, maskTokenID: maskTokenID, slidingWindow: slidingWindow)
    }
}
