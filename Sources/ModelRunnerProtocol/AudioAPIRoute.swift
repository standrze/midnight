import Foundation

public enum AudioAPIRoute: Equatable, Sendable {
    case speech
    case voices
    case voice(id: String)
    case voiceSample(id: String)

    public static func parse(uri: String) -> AudioAPIRoute? {
        guard let components = URLComponents(string: "http://midnight.local\(uri)"),
            let segments = decodedPathSegments(components.percentEncodedPath)
        else { return nil }

        if segments == ["v1", "audio", "speech"] { return .speech }
        if segments == ["v1", "audio", "voices"] { return .voices }
        if segments.count == 4, Array(segments.prefix(3)) == ["v1", "audio", "voices"] {
            return .voice(id: segments[3])
        }
        if segments.count == 5,
            Array(segments.prefix(3)) == ["v1", "audio", "voices"],
            segments[4] == "sample"
        {
            return .voiceSample(id: segments[3])
        }
        return nil
    }

    public var allowedMethods: Set<String> {
        switch self {
        case .speech:
            ["POST"]
        case .voices:
            ["GET", "POST"]
        case .voice:
            ["GET", "PATCH", "DELETE"]
        case .voiceSample:
            ["GET"]
        }
    }

    public func allows(method: String) -> Bool {
        allowedMethods.contains(method.uppercased())
    }

    public static func queryItems(uri: String) -> [URLQueryItem] {
        URLComponents(string: "http://midnight.local\(uri)")?.queryItems ?? []
    }

    private static func decodedPathSegments(_ encodedPath: String) -> [String]? {
        let encodedSegments = encodedPath.split(separator: "/", omittingEmptySubsequences: false)
        guard encodedSegments.first == "", encodedSegments.dropFirst().allSatisfy({ !$0.isEmpty })
        else { return nil }

        var result: [String] = []
        for encoded in encodedSegments.dropFirst() {
            guard let value = String(encoded).removingPercentEncoding,
                !value.isEmpty,
                value != ".",
                value != "..",
                !value.contains("/"),
                !value.contains("\0")
            else { return nil }
            result.append(value)
        }
        return result
    }
}
