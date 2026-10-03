import Foundation

/// Arms bounded summary capture for the next ordinary text generation on a loaded runtime.
public struct InspectorRecordingRequest: Codable, Equatable, Sendable {
    public var model: String?
    public var runtimeGeneration: UInt64?
    public var layers: [Int]
    public var sites: [InspectorObservationSite]
    public var maxTokens: Int
    public var maxCaptureBytes: Int

    /// Captures the final evaluated prompt token and at most `maxTokens` emitted tokens.
    public init(
        model: String? = nil, runtimeGeneration: UInt64? = nil, layers: [Int],
        sites: [InspectorObservationSite] = InspectorObservationSite.allCases,
        maxTokens: Int = 32, maxCaptureBytes: Int = 4 * 1_048_576
    ) {
        self.model = model
        self.runtimeGeneration = runtimeGeneration
        self.layers = layers
        self.sites = sites
        self.maxTokens = maxTokens
        self.maxCaptureBytes = maxCaptureBytes
    }

    private enum CodingKeys: String, CodingKey {
        case model, runtimeGeneration, layers, sites, maxTokens, maxCaptureBytes
    }

    /// Omitted limits use the same defaults as the Swift initializer.
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            model: try values.decodeIfPresent(String.self, forKey: .model),
            runtimeGeneration: try values.decodeIfPresent(UInt64.self, forKey: .runtimeGeneration),
            layers: try values.decode([Int].self, forKey: .layers),
            sites: try values.decodeIfPresent([InspectorObservationSite].self, forKey: .sites)
                ?? InspectorObservationSite.allCases,
            maxTokens: try values.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 32,
            maxCaptureBytes: try values.decodeIfPresent(Int.self, forKey: .maxCaptureBytes) ?? 4 * 1_048_576)
    }
}

/// In-memory recording status. Trace tokens contain only the bounded observed window.
public struct InspectorRecordingSession: Codable, Equatable, Sendable {
    public var id: String
    public var status: String
    public var createdAt: Double
    public var message: String?
    public var model: InspectorModel?
    public var trace: InspectorTrace?
    public var cachedPromptTokenCount: Int?

    /// Creates a recording status independent of the client's conversation transport.
    public init(
        id: String, status: String, createdAt: Double, message: String? = nil,
        model: InspectorModel? = nil, trace: InspectorTrace? = nil, cachedPromptTokenCount: Int? = nil
    ) {
        self.id = id
        self.status = status
        self.createdAt = createdAt
        self.message = message
        self.model = model
        self.trace = trace
        self.cachedPromptTokenCount = cachedPromptTokenCount
    }
}
