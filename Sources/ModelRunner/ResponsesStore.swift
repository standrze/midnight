import Foundation
import ModelRunnerProtocol

/// Completed Responses API results retained for retrieval and conversation continuation.
/// Limits apply to the JSON representation of every retained ID and complete entry;
/// the entry count also bounds the store's bookkeeping overhead.
actor ResponsesStore {
    struct Entry: Codable, Equatable, Sendable {
        let response: OpenAIJSONValue
        /// Conversation history without the request's top-level `instructions`.
        let messages: [OpenAIMessage]
        let inputItems: [OpenAIJSONValue]
        let model: String
    }

    private struct StoredEntry: Sendable {
        let entry: Entry
        let byteCount: Int
        let expiresAt: Date
    }

    private struct EncodedEntry: Encodable {
        let id: String
        let entry: Entry
    }

    private let maximumEntries: Int
    private let maximumBytes: Int
    private let timeToLive: TimeInterval
    private var entries: [String: StoredEntry] = [:]
    private var storageOrder: [String] = []
    private var storedBytes = 0

    init(
        maximumEntries: Int = 128,
        maximumBytes: Int = 64 * 1024 * 1024,
        timeToLive: TimeInterval = 3600
    ) {
        self.maximumEntries = max(0, maximumEntries)
        self.maximumBytes = max(0, maximumBytes)
        self.timeToLive = timeToLive.isFinite ? max(0, timeToLive) : 0
    }

    /// Returns false if this complete entry cannot fit. Failed insertion never
    /// evicts a live result, including an existing result under the same ID.
    @discardableResult
    func store(id: String, entry: Entry, now: Date = Date()) -> Bool {
        purgeExpired(now: now)
        guard maximumEntries > 0, maximumBytes > 0, timeToLive > 0,
            now.timeIntervalSinceReferenceDate.isFinite,
            let data = try? JSONEncoder().encode(EncodedEntry(id: id, entry: entry)),
            data.count <= maximumBytes
        else { return false }

        let expiresAt = now.addingTimeInterval(timeToLive)
        guard expiresAt.timeIntervalSinceReferenceDate.isFinite else { return false }

        discard(id: id)
        while entries.count >= maximumEntries || storedBytes > maximumBytes - data.count {
            guard let oldest = storageOrder.first else { return false }
            discard(id: oldest)
        }
        entries[id] = StoredEntry(entry: entry, byteCount: data.count, expiresAt: expiresAt)
        storageOrder.append(id)
        storedBytes += data.count
        return true
    }

    func get(id: String, now: Date = Date()) -> Entry? {
        purgeExpired(now: now)
        return entries[id]?.entry
    }

    @discardableResult
    func remove(id: String, now: Date = Date()) -> Bool {
        purgeExpired(now: now)
        return discard(id: id)
    }

    private func purgeExpired(now: Date) {
        let expired = entries.compactMap { id, value in
            now >= value.expiresAt ? id : nil
        }
        for id in expired { discard(id: id) }
    }

    @discardableResult
    private func discard(id: String) -> Bool {
        guard let removed = entries.removeValue(forKey: id) else { return false }
        storedBytes -= removed.byteCount
        storageOrder.removeAll { $0 == id }
        return true
    }
}

enum ResponsesAPIRoute: Equatable, Sendable {
    case create
    case response(id: String)
    case inputItems(id: String)

    static func parse(uri: String) -> Self? {
        guard let components = components(uri: uri),
            let segments = decodedPathSegments(components.percentEncodedPath)
        else { return nil }

        if segments == ["v1", "responses"] { return .create }
        guard segments.count == 3 || segments.count == 4,
            segments[0] == "v1", segments[1] == "responses",
            validResponseID(segments[2])
        else { return nil }

        if segments.count == 3 { return .response(id: segments[2]) }
        if segments[3] == "input_items" { return .inputItems(id: segments[2]) }
        return nil
    }

    var allowedMethods: Set<String> {
        switch self {
        case .create: ["POST"]
        case .response: ["GET", "DELETE"]
        case .inputItems: ["GET"]
        }
    }

    func allows(method: String) -> Bool {
        allowedMethods.contains(method.uppercased())
    }

    static func queryItems(uri: String) -> [URLQueryItem] {
        components(uri: uri)?.queryItems ?? []
    }

    private static func components(uri: String) -> URLComponents? {
        // Accept the origin-form request target only. Fragments are not valid in
        // HTTP request targets, and URLComponents otherwise silently accepts them.
        guard uri.hasPrefix("/"), !uri.hasPrefix("//"), !uri.contains("#"),
            uri.utf8.allSatisfy({ $0 > 0x20 && $0 != 0x7f }),
            hasValidPercentEscapes(uri),
            let result = URLComponents(string: "http://midnight.local\(uri)")
        else { return nil }
        return result
    }

    private static func decodedPathSegments(_ encodedPath: String) -> [String]? {
        let segments = encodedPath.split(separator: "/", omittingEmptySubsequences: false)
        guard segments.first == "", segments.dropFirst().allSatisfy({ !$0.isEmpty })
        else { return nil }
        var decoded: [String] = []
        for segment in segments.dropFirst() {
            guard let value = String(segment).removingPercentEncoding,
                !value.contains("/"), !value.contains("\\"), !value.contains("\0"),
                value != ".", value != ".."
            else { return nil }
            decoded.append(value)
        }
        return decoded
    }

    private static func validResponseID(_ id: String) -> Bool {
        let bytes = Array(id.utf8)
        guard id.hasPrefix("resp_"), bytes.count > 5, bytes.count <= 256 else { return false }
        return bytes.allSatisfy { byte in
            (byte >= 65 && byte <= 90) || (byte >= 97 && byte <= 122)
                || (byte >= 48 && byte <= 57) || byte == 95 || byte == 45
        }
    }

    private static func hasValidPercentEscapes(_ value: String) -> Bool {
        let bytes = Array(value.utf8)
        var index = 0
        while index < bytes.count {
            if bytes[index] == 37 {
                guard index + 2 < bytes.count,
                    isHex(bytes[index + 1]), isHex(bytes[index + 2])
                else { return false }
                index += 3
            } else {
                index += 1
            }
        }
        return true
    }

    private static func isHex(_ byte: UInt8) -> Bool {
        (byte >= 48 && byte <= 57) || (byte >= 65 && byte <= 70)
            || (byte >= 97 && byte <= 102)
    }
}
