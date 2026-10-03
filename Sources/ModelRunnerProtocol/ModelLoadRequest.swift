import Foundation

/// Per-load overrides. Omitted values are resolved from the selected checkpoint's
/// settings instead of inheriting options from the previously loaded model.
public struct ModelLoadRequest: Codable, Equatable, Sendable {
    public let model: String
    public let name: String?
    public let adapter: String?
    public let adapterScale: Float?
    public let dflashModel: String?
    /// Maximum draft token IDs proposed per DFlash verification block.
    public let dflashBlockSize: Int?
    public let gemmaAssistantModel: String?
    /// Maximum draft token IDs proposed per Gemma assistant verification block.
    public let gemmaAssistantBlockSize: Int?
    public let gemmaAssistantQuantizationBits: Int?
    public let autoAssistant: Bool?
    /// Maximum emitted token IDs in one response.
    public let maxTokens: Int?
    /// Combined prompt and reserved output capacity, in token positions.
    public let contextLength: Int?
    /// Prompt token positions evaluated per prefill chunk.
    public let prefillStepSize: Int?
    public let kvCompression: String?
    public let engine: String?
    /// Optional compare-and-swap guards for clients restoring a prior selection.
    /// The generation is the managed model lifecycle generation, not a token count.
    public let expectedGeneration: UInt64?
    public let expectedInstanceID: String?

    /// Creates per-load model overrides and optional compare-and-swap guards.
    public init(
        model: String,
        name: String? = nil,
        adapter: String? = nil,
        adapterScale: Float? = nil,
        dflashModel: String? = nil,
        dflashBlockSize: Int? = nil,
        gemmaAssistantModel: String? = nil,
        gemmaAssistantBlockSize: Int? = nil,
        gemmaAssistantQuantizationBits: Int? = nil,
        autoAssistant: Bool? = nil,
        maxTokens: Int? = nil,
        contextLength: Int? = nil,
        prefillStepSize: Int? = nil,
        kvCompression: String? = nil,
        engine: String? = nil,
        expectedGeneration: UInt64? = nil,
        expectedInstanceID: String? = nil
    ) {
        self.model = model
        self.name = name
        self.adapter = adapter
        self.adapterScale = adapterScale
        self.dflashModel = dflashModel
        self.dflashBlockSize = dflashBlockSize
        self.gemmaAssistantModel = gemmaAssistantModel
        self.gemmaAssistantBlockSize = gemmaAssistantBlockSize
        self.gemmaAssistantQuantizationBits = gemmaAssistantQuantizationBits
        self.autoAssistant = autoAssistant
        self.maxTokens = maxTokens
        self.contextLength = contextLength
        self.prefillStepSize = prefillStepSize
        self.kvCompression = kvCompression
        self.engine = engine
        self.expectedGeneration = expectedGeneration
        self.expectedInstanceID = expectedInstanceID
    }
}

/// Request body for explicitly unloading the currently managed model.
public struct ModelUnloadRequest: Decodable, Sendable {
    /// Managed model lifecycle generation required for conditional unload.
    public let expectedGeneration: UInt64?
    /// Control-plane instance ID required for conditional unload.
    public let expectedInstanceID: String?
}
