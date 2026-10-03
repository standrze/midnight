import CoreFoundation
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// Reports an invalid corpus input, source, or output condition.
public struct CorpusError: Error, CustomStringConvertible {
    public let description: String
    /// Creates a corpus error with a caller-facing explanation.
    public init(_ description: String) { self.description = description }
}

// Python's original tools count Unicode scalars and trim these exact whitespace
// characters. Swift Character/grapheme counts would change corpus boundaries.
private let whitespace = CharacterSet(
    charactersIn:
        "\u{9}\u{a}\u{b}\u{c}\u{d}\u{1c}\u{1d}\u{1e}\u{1f} \u{85}\u{a0}\u{1680}\u{2000}\u{2001}\u{2002}\u{2003}\u{2004}\u{2005}\u{2006}\u{2007}\u{2008}\u{2009}\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}"
)
private func trim(_ text: String) -> String {
    let scalars = text.unicodeScalars
    var start = scalars.startIndex
    var end = scalars.endIndex
    while start != end && whitespace.contains(scalars[start]) {
        scalars.formIndex(after: &start)
    }
    while start != end {
        let previous = scalars.index(before: end)
        guard whitespace.contains(scalars[previous]) else {
            break
        }
        end = previous
    }
    return String(scalars[start..<end])
}
private func normalizedNewlines(_ text: String) -> String {
    guard text.utf8.contains(13) else {
        return text
    }
    var result = [UInt8]()
    result.reserveCapacity(text.utf8.count)
    var previousWasCR = false
    for byte in text.utf8 {
        if byte != 10 || !previousWasCR {
            result.append(byte == 13 ? 10 : byte)
        }
        previousWasCR = byte == 13
    }
    return String(decoding: result, as: UTF8.self)
}
private func paragraphs(_ text: String) -> [String] {
    // ASCII delimiter matching on UTF-8 avoids Foundation's Unicode search and
    // bridging cost without changing any scalar or grapheme in the payload.
    let bytes = Array(text.utf8)
    var result = [String]()
    var start = 0
    var position = 0
    while position + 1 < bytes.count {
        if bytes[position] == 10 && bytes[position + 1] == 10 {
            let part = trim(String(decoding: bytes[start..<position], as: UTF8.self))
            if !part.isEmpty {
                result.append(part)
            }
            position += 2
            start = position
        } else {
            position += 1
        }
    }
    let last = trim(String(decoding: bytes[start...], as: UTF8.self))
    if !last.isEmpty {
        result.append(last)
    }
    return result
}
private func utf8(_ data: Data) throws -> String {
    guard let text = String(data: data, encoding: .utf8) else {
        throw CorpusError("Input is not valid UTF-8")
    }
    return text
}
// Stable JSON string encoding: UTF-8, literal slashes, and Python-compatible
// escapes. Key ordering/spacing below are part of the existing dataset hashes.
private func quoted(_ text: String) -> String {
    var result = "\""
    for scalar in text.unicodeScalars {
        switch scalar.value {
        case 34: result += "\\\""
        case 92: result += "\\\\"
        case 8: result += "\\b"
        case 9: result += "\\t"
        case 10: result += "\\n"
        case 12: result += "\\f"
        case 13: result += "\\r"
        case 0..<32: result += String(format: "\\u%04x", scalar.value)
        default: result.unicodeScalars.append(scalar)
        }
    }
    return result + "\""
}
private func record(_ id: String, _ category: String, _ text: String, spaced: Bool) -> String {
    let s = spaced ? " " : ""
    return "{\"id\":\(s)\(quoted(id)),\(s)\"category\":\(s)\(quoted(category)),\(s)\"text\":\(s)\(quoted(text))}\n"
}
/// Counts and SHA-256 digests for a prepared text corpus.
public struct TextResult: Sendable {
    public let samples: Int
    public let characters: Int
    public let sourceSHA256: String
    public let outputSHA256: String
}
/// Builds deterministic held-out text samples from UTF-8 or JSON input.
public enum TextCorpus {
    /// Returns JSONL bytes and a Unicode-scalar count for selected samples.
    ///
    /// `targetCharacters` and the returned count use Unicode scalars. Invalid
    /// input or too few paragraphs throws `CorpusError`.
    public static func payload(
        data: Data, json: Bool, samples: Int = 64, targetCharacters: Int = 2500,
        idPrefix: String = "heldout", category: String = "language-modeling"
    ) throws -> (Data, Int) {
        guard samples > 0, targetCharacters >= 128 else {
            throw CorpusError("--samples must be positive and --target-characters must be at least 128")
        }
        var text = normalizedNewlines(try utf8(data))
        if json {
            let value = try JSONSerialization.jsonObject(with: Data(text.utf8))
            guard let rows = (value as? [String: Any])?["instances"] as? [Any] ?? value as? [Any] else {
                throw CorpusError("JSON input must be a list or contain an 'instances' list")
            }
            text = try rows.enumerated().compactMap { index, row in
                guard let content = (row as? [String: Any])?["text"] as? String else {
                    throw CorpusError("JSON text row \(index) does not contain a string 'text'")
                }
                let value = trim(content)
                return value.isEmpty ? nil : value
            }.joined(separator: "\n\n")
        }
        var chunks = [String]()
        var pending = [String]()
        var pendingSize = 0
        for paragraph in paragraphs(text) {
            let length = paragraph.unicodeScalars.count
            let addition = length + (pending.isEmpty ? 0 : 2)
            if !pending.isEmpty && pendingSize + addition > targetCharacters {
                chunks.append(pending.joined(separator: "\n\n"))
                pending = []
                pendingSize = 0
            }
            if length > targetCharacters && pending.isEmpty {
                let scalars = Array(paragraph.unicodeScalars)
                for start in stride(from: 0, to: length, by: targetCharacters) {
                    let piece = trim(
                        String(String.UnicodeScalarView(scalars[start..<min(start + targetCharacters, length)])))
                    if !piece.isEmpty {
                        chunks.append(piece)
                    }
                }
                continue
            }
            pending.append(paragraph)
            // Preserve the original pre-flush addition for byte-identical selection.
            pendingSize += addition
        }
        if !pending.isEmpty {
            chunks.append(pending.joined(separator: "\n\n"))
        }
        guard chunks.count >= samples else {
            throw CorpusError("requested \(samples) samples but only built \(chunks.count) chunks")
        }
        var output = ""
        var characters = 0
        for i in 0..<samples {
            let index = samples == 1 ? 0 : i * (chunks.count - 1) / (samples - 1)
            characters += chunks[index].unicodeScalars.count
            output += record(String(format: "%@-%03d", idPrefix, i + 1), category, chunks[index], spaced: false)
        }
        return (Data(output.utf8), characters)
    }
    /// Reads a source file and writes its sampled JSONL corpus to a distinct output URL.
    ///
    /// Files with a `.json` extension use the JSON input path. The result
    /// includes SHA-256 digests for the input and written output.
    public static func prepare(
        source: URL, output: URL, samples: Int = 64, targetCharacters: Int = 2500,
        idPrefix: String = "heldout", category: String = "language-modeling"
    ) throws -> TextResult {
        guard
            source.standardizedFileURL.resolvingSymlinksInPath() != output.standardizedFileURL.resolvingSymlinksInPath()
        else {
            throw CorpusError("source and output must differ")
        }
        let data = try Data(contentsOf: source)
        let (result, characters) = try payload(
            data: data, json: source.pathExtension.lowercased() == "json", samples: samples,
            targetCharacters: targetCharacters, idPrefix: idPrefix, category: category)
        try FileManager.default.createDirectory(
            at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        try result.write(to: output)
        return TextResult(
            samples: samples, characters: characters, sourceSHA256: corpusSHA256(data),
            outputSHA256: corpusSHA256(result))
    }
}

/// Repository location and expected hashes for one pinned reference source.
public struct PinnedSource: Codable, Sendable {
    public var name: String
    public var repo: String
    public var revision: String
    public var path: String
    // These public names also serve as pinned-manifest JSON keys.
    // swift-format-ignore: AlwaysUseLowerCamelCase
    public var source_sha256: String
    // swift-format-ignore: AlwaysUseLowerCamelCase
    public var corpus_sha256: String
    /// Raw GitHub URL formed from the pinned repository, revision, and path.
    public var url: URL { URL(string: "https://raw.githubusercontent.com/\(repo)/\(revision)/\(path)")! }
}
/// Pinned inputs and combined digest for the reference corpora.
public struct PinnedManifest: Codable, Sendable {
    public var sources: [PinnedSource]
    // Keep the published property and manifest JSON key stable.
    // swift-format-ignore: AlwaysUseLowerCamelCase
    public var combined_sha256: String
    public var purpose: String
    /// Loads the manifest bundled with `CorpusPreparationCore`.
    public static func bundled() throws -> Self {
        try JSONDecoder().decode(
            Self.self, from: Data(contentsOf: Bundle.module.url(forResource: "pinned-sources", withExtension: "json")!))
    }
}
/// Encodable summary of the verified reference files written to disk.
public struct ReferenceResult: Encodable, Sendable {
    public let corpus: String
    // Keep the encoded output field stable.
    // swift-format-ignore: AlwaysUseLowerCamelCase
    public let sample_count: Int
    public let sha256: String
    public let purpose: String
    public let files: [String: String]
}
/// Validates and prepares the pinned MBPP and GSM8K reference corpora.
public enum ReferenceCorpus {
    /// Returns data only when its SHA-256 digest matches `expected`.
    public static func verify(_ data: Data, expected: String, description: String) throws -> Data {
        let actual = corpusSHA256(data)
        guard actual == expected else {
            throw CorpusError("SHA256 mismatch for \(description): expected \(expected), got \(actual)")
        }
        return data
    }
    /// Converts an ordered source JSONL dataset into its deterministic reference JSONL payload.
    ///
    /// Supported names are `mbpp` and `gsm8k`; malformed or reordered source
    /// records throw `CorpusError`.
    public static func payload(name: String, data: Data) throws -> Data {
        let rows = try utf8(data).components(separatedBy: .newlines).filter { !trim($0).isEmpty }.map {
            guard let row = try JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any] else {
                throw CorpusError("Expected JSON objects")
            }
            return row
        }
        func field(_ row: [String: Any], _ key: String) throws -> String {
            guard let value = row[key] as? String else {
                throw CorpusError("Missing or invalid string: \(key)")
            }
            return value
        }
        var output = ""
        switch name {
        case "mbpp":
            let selected = try rows.filter { row in
                guard let number = row["task_id"] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                    number.doubleValue == Double(number.intValue)
                else {
                    throw CorpusError("Invalid MBPP task_id")
                }
                return (11...74).contains(number.intValue)
            }
            guard selected.compactMap({ ($0["task_id"] as? NSNumber)?.intValue }) == Array(11...74) else {
                throw CorpusError("MBPP source must contain ordered tasks 11...74 exactly once")
            }
            for (index, row) in selected.enumerated() {
                output += record(
                    "mbpp-test-\(index + 11)", "code-reference",
                    "Task: \(try field(row,"text"))\nSolution:\n\(try field(row,"code"))", spaced: true)
            }
        case "gsm8k":
            guard rows.count >= 64 else {
                throw CorpusError("Expected exactly 64 records for gsm8k, got \(rows.count)")
            }
            for (index, row) in rows.prefix(64).enumerated() {
                output += record(
                    "gsm8k-test-\(index)", "math-reference",
                    "Question: \(try field(row,"question"))\nAnswer: \(try field(row,"answer"))", spaced: true)
            }
        default: throw CorpusError("Unknown source: \(name)")
        }
        return Data(output.utf8)
    }
    /// Downloads a source using a 60-second request timeout and requires an HTTP success status.
    public static func download(_ url: URL) async throws -> Data {
        var request = URLRequest(url: url, timeoutInterval: 60)
        request.setValue("midnight-pinned-corpus/1", forHTTPHeaderField: "User-Agent")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let response = response as? HTTPURLResponse, (200..<300).contains(response.statusCode) else {
            throw CorpusError("Download failed: \(url)")
        }
        return data
    }
    /// Verifies pinned inputs, writes each source and derived corpus, and returns their file summary.
    ///
    /// `offline` requires an existing verified source in `output` or `cache`.
    /// The optional `fetch` closure allows a caller to supply a transport.
    public static func prepare(
        output: URL, cache: URL? = nil, offline: Bool = false,
        manifest: PinnedManifest? = nil,
        fetch: @Sendable (URL) async throws -> Data = download
    ) async throws -> ReferenceResult {
        let manifest = try manifest ?? .bundled()
        let fm = FileManager.default
        for directory in [output, cache].compactMap({ $0 }) {
            var isDirectory: ObjCBool = false
            if fm.fileExists(atPath: directory.path, isDirectory: &isDirectory) {
                guard isDirectory.boolValue else {
                    throw CorpusError("Not a directory: \(directory.path)")
                }
            } else if directory == cache {
                throw CorpusError("Cache is not a directory: \(directory.path)")
            }
        }
        var files = [(String, Data)]()
        var combined = Data()
        var entries = [String]()
        for source in manifest.sources {
            let name = "\(source.name)-source.jsonl"
            let candidates = [output.appendingPathComponent(name), cache?.appendingPathComponent(name)].compactMap {
                $0
            }
            let data: Data
            if let existing = candidates.first(where: { fm.fileExists(atPath: $0.path) }) {
                data = try verify(
                    Data(contentsOf: existing), expected: source.source_sha256, description: existing.path)
            } else {
                guard !offline else {
                    throw CorpusError("Offline source missing: \(name); provide --cache-dir with verified source files")
                }
                data = try await verify(
                    fetch(source.url), expected: source.source_sha256, description: source.url.absoluteString)
            }
            let result = try verify(
                payload(name: source.name, data: data), expected: source.corpus_sha256,
                description: "\(source.name) corpus")
            files.append((name, data))
            files.append(("\(source.name)-64.jsonl", result))
            combined.append(result)
            let pairs = [
                ("name", source.name), ("repo", source.repo), ("revision", source.revision), ("path", source.path),
                ("url", source.url.absoluteString), ("source_sha256", source.source_sha256),
                ("corpus_sha256", source.corpus_sha256),
            ]
            entries.append(
                "  {\n" + pairs.map { "    \(quoted($0.0)): \(quoted($0.1)),\n" }.joined()
                    + "    \"sample_count\": 64,\n    \"purpose\": \(quoted(manifest.purpose))\n  }")
        }
        _ = try verify(combined, expected: manifest.combined_sha256, description: "code-math-128.jsonl")
        files.append(("code-math-128.jsonl", combined))
        files.append(("provenance.json", Data(("[\n" + entries.joined(separator: ",\n") + "\n]\n").utf8)))
        for (name, data) in files {
            let path = output.appendingPathComponent(name)
            if fm.fileExists(atPath: path.path) {
                guard (try? Data(contentsOf: path)) == data else {
                    throw CorpusError("Refusing to overwrite differing output: \(path.path)")
                }
            }
        }
        try fm.createDirectory(at: output, withIntermediateDirectories: true)
        for (name, data) in files {
            let path = output.appendingPathComponent(name)
            if !fm.fileExists(atPath: path.path) {
                try data.write(to: path, options: .withoutOverwriting)
            }
        }
        return ReferenceResult(
            corpus: output.appendingPathComponent("code-math-128.jsonl").standardizedFileURL.path,
            sample_count: 128, sha256: manifest.combined_sha256, purpose: manifest.purpose,
            files: Dictionary(uniqueKeysWithValues: files.map { ($0.0, corpusSHA256($0.1)) }))
    }
}
