import Foundation

/// Per-load overrides. Omitted values are resolved from the selected checkpoint's
/// settings instead of inheriting options from the previously loaded model.
public struct ModelLoadRequest: Codable, Equatable, Sendable {
    public let model: String
    public let name: String?
    public let adapter: String?
    public let adapterScale: Float?
    public let dflashModel: String?
    public let dflashBlockSize: Int?
    public let maxTokens: Int?
    public let contextLength: Int?
    public let prefillStepSize: Int?
    public let kvCompression: String?
    public let engine: String?

    public init(
        model: String,
        name: String? = nil,
        adapter: String? = nil,
        adapterScale: Float? = nil,
        dflashModel: String? = nil,
        dflashBlockSize: Int? = nil,
        maxTokens: Int? = nil,
        contextLength: Int? = nil,
        prefillStepSize: Int? = nil,
        kvCompression: String? = nil,
        engine: String? = nil
    ) {
        self.model = model
        self.name = name
        self.adapter = adapter
        self.adapterScale = adapterScale
        self.dflashModel = dflashModel
        self.dflashBlockSize = dflashBlockSize
        self.maxTokens = maxTokens
        self.contextLength = contextLength
        self.prefillStepSize = prefillStepSize
        self.kvCompression = kvCompression
        self.engine = engine
    }
}
