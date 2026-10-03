import Foundation
import Testing
import Tokenizers

@Suite("Added-token regex segmentation")
struct AddedTokenRegexTests {
    @Test("Ordinary text and empty input preserve their token IDs")
    func ordinaryText() async throws {
        for addedTokens in [[], [AddedToken(id: 1000, content: "<x>")]] {
            let tokenizer = try await makeTokenizer(addedTokens: addedTokens)
            expectTokens(
                tokenizer, text: "a b\tc\n", pieces: ["a", " ", "b", "\t", "c", "\n"],
                ids: [97, 32, 98, 9, 99, 10])
            expectTokens(tokenizer, text: "", pieces: [], ids: [])
        }
    }

    @Test("Longer added tokens win over earlier shorter alternatives")
    func overlappingTokens() async throws {
        let tokenizer = try await makeTokenizer(addedTokens: [
            AddedToken(id: 1000, content: "a"),
            AddedToken(id: 1001, content: "ab"),
            AddedToken(id: 1002, content: "abc"),
        ])
        expectTokens(
            tokenizer, text: "abcabac", pieces: ["abc", "ab", "a", "c"],
            ids: [1002, 1001, 1000, 99])
    }

    @Test("Adjacent tokens at either edge remain distinct, including nonspecial added tokens")
    func adjacentTokens() async throws {
        let tokenizer = try await makeTokenizer(addedTokens: [
            AddedToken(id: 1000, content: "<x>"),
            AddedToken(id: 1001, content: "<y>", special: false),
        ])
        expectTokens(
            tokenizer, text: "<x><y><x>", pieces: ["<x>", "<y>", "<x>"], ids: [1000, 1001, 1000])
        expectTokens(
            tokenizer, text: "a<x><y>b", pieces: ["a", "<x>", "<y>", "b"], ids: [97, 1000, 1001, 98])
    }

    @Test("UTF-16 offsets preserve combining marks and multiscalar graphemes")
    func unicodeBoundaries() async throws {
        let tokenizer = try await makeTokenizer(addedTokens: [
            AddedToken(id: 1000, content: "<x>"), AddedToken(id: 1001, content: "<y>"),
        ])
        expectTokens(
            tokenizer, text: "é<x>e\u{0301}<y>中🙂<x>👩\u{200D}💻",
            pieces: ["é", "<x>", "e", "\u{0301}", "<y>", "中", "🙂", "<x>", "👩", "\u{200D}", "💻"],
            ids: [233, 1000, 101, 769, 1001, 20013, 128578, 1000, 128105, 8205, 128187])

        let graphemeTokenizer = try await makeTokenizer(addedTokens: [
            AddedToken(id: 1000, content: "👩\u{200D}💻"), AddedToken(id: 1001, content: "🙂"),
        ])
        expectTokens(
            graphemeTokenizer, text: "a👩\u{200D}💻b🙂👩\u{200D}💻",
            pieces: ["a", "👩\u{200D}💻", "b", "🙂", "👩\u{200D}💻"],
            ids: [97, 1000, 98, 1001, 1000])
    }

    @Test("Regex metacharacters in added tokens are matched literally")
    func escapedMetacharacters() async throws {
        let tokenizer = try await makeTokenizer(addedTokens: [
            AddedToken(id: 1000, content: "["), AddedToken(id: 1001, content: ".*"),
            AddedToken(id: 1002, content: "("), AddedToken(id: 1003, content: "\\"),
            AddedToken(id: 1004, content: "a|b"), AddedToken(id: 1005, content: "$^"),
        ])
        expectTokens(
            tokenizer, text: #"c[.*(\a|b$^c"#,
            pieces: ["c", "[", ".*", "(", "\\", "a|b", "$^", "c"],
            ids: [99, 1000, 1001, 1002, 1003, 1004, 1005, 99])
    }

    @Test("Equal grapheme-length prefix ties preserve capture-regex ordering")
    func equalGraphemeLengthPrefixes() async throws {
        let short = "👩"
        let long = "👩\u{200D}💻"
        #expect(short.count == long.count)
        for alternatives in [[short, long], [long, short]] {
            let tokenizer = try await makeTokenizer(
                addedTokens: alternatives.map {
                    AddedToken(id: $0 == short ? 1000 : 1001, content: $0)
                })
            let text = "a\(long)b\(short)"
            let sections = try legacyCapturedSections(text: text, alternatives: alternatives)
            let pieces = sections.flatMap {
                alternatives.contains($0) ? [$0] : $0.unicodeScalars.map { String($0) }
            }
            let ids = try pieces.map { piece in
                if piece == short {
                    return 1000
                }
                if piece == long {
                    return 1001
                }
                return Int(try #require(piece.unicodeScalars.first).value)
            }
            expectTokens(tokenizer, text: text, pieces: pieces, ids: ids)
        }
    }

    @Test("Missing IDs remain ordinary text and duplicate contents retain the last ID")
    func malformedAndDuplicateEntries() async throws {
        let tokenizer = try await makeTokenizer(addedTokens: [
            AddedToken(id: nil, content: "<missing>"),
            AddedToken(id: 1000, content: "<x>"), AddedToken(id: 1001, content: "<x>"),
        ])
        expectTokens(
            tokenizer, text: "<missing><x>",
            pieces: ["<", "m", "i", "s", "s", "i", "n", "g", ">", "<x>"],
            ids: [60, 109, 105, 115, 115, 105, 110, 103, 62, 1001])
    }

    @Test("An explicitly empty added token keeps its zero-width boundaries")
    func emptyAddedToken() async throws {
        // A capture of an empty literal is valid ICU syntax. An empty regex
        // pattern is not; silently falling back to ordinary BPE loses these IDs.
        let tokenizer = try await makeTokenizer(addedTokens: [AddedToken(id: 1000, content: "")])
        expectTokens(tokenizer, text: "", pieces: [""], ids: [1000])
        expectTokens(
            tokenizer, text: "ab", pieces: ["", "a", "", "b", ""], ids: [1000, 97, 1000, 98, 1000])

        let mixed = try await makeTokenizer(addedTokens: [
            AddedToken(id: 1000, content: ""), AddedToken(id: 1001, content: "<x>"),
        ])
        let text = "a<x>b"
        let pieces = try legacyCapturedSections(text: text, alternatives: ["", "<x>"]).flatMap {
            $0.isEmpty || $0 == "<x>" ? [$0] : $0.unicodeScalars.map { String($0) }
        }
        let vocabulary = ["": 1000, "<x>": 1001, "a": 97, "b": 98]
        let ids = try pieces.map { try #require(vocabulary[$0]) }
        expectTokens(mixed, text: text, pieces: pieces, ids: ids)
    }

    @Test("Left and right strip flags remove only the requested surrounding whitespace")
    func whitespaceStripping() async throws {
        let cases: [(Bool, Bool, [String], [Int])] = [
            (false, false, ["a", " ", "\t", "<x>", "\n", " ", "b"], [97, 32, 9, 1000, 10, 32, 98]),
            (true, false, ["a", "<x>", "\n", " ", "b"], [97, 1000, 10, 32, 98]),
            (false, true, ["a", " ", "\t", "<x>", "b"], [97, 32, 9, 1000, 98]),
            (true, true, ["a", "<x>", "b"], [97, 1000, 98]),
        ]
        for (left, right, pieces, ids) in cases {
            let tokenizer = try await makeTokenizer(addedTokens: [
                AddedToken(id: 1000, content: "<x>", leftStrip: left, rightStrip: right)
            ])
            expectTokens(tokenizer, text: "a \t<x>\n b", pieces: pieces, ids: ids)
        }
    }

    @Test("An unused stripping token preserves every other token and its whitespace")
    func unusedStrippingToken() async throws {
        // Any stripping token selects the capture-aware fallback for the complete
        // alternation, even when it is nonspecial and does not occur in the input.
        for (left, right) in [(true, false), (false, true), (true, true)] {
            let tokenizer = try await makeTokenizer(addedTokens: [
                AddedToken(id: 1000, content: "<x>"),
                AddedToken(id: 1001, content: "<y>", special: false),
                AddedToken(id: 1002, content: "<unused>", leftStrip: left, rightStrip: right, special: false),
            ])
            expectTokens(
                tokenizer, text: "a \t<x>\n b<y><x>",
                pieces: ["a", " ", "\t", "<x>", "\n", " ", "b", "<y>", "<x>"],
                ids: [97, 32, 9, 1000, 10, 32, 98, 1001, 1000])
        }
    }

    private struct AddedToken {
        let id: Int?
        let content: String
        var leftStrip = false
        var rightStrip = false
        var special = true

        var json: [String: Any] {
            var value: [String: Any] = [
                "content": content, "lstrip": leftStrip, "rstrip": rightStrip,
                "special": special, "normalized": false, "single_word": false,
            ]
            if let id {
                value["id"] = id
            }
            return value
        }
    }

    private func makeTokenizer(addedTokens: [AddedToken]) async throws -> any Tokenizer {
        // No merges or preprocessing: ordinary text exposes individual Unicode
        // scalars, while each recognized added token contributes exactly one ID.
        let scalars =
            (32...126).map { UInt32($0) }
            + "\t\n\ré中🙂👩\u{200D}💻\u{0301}".unicodeScalars.map(\.value)
            + addedTokens.flatMap { $0.content.unicodeScalars.map(\.value) }
        var vocabulary = ["<unk>": 0]
        for value in Set(scalars) {
            if let scalar = UnicodeScalar(value) {
                vocabulary[String(scalar)] = Int(value)
            }
        }
        let configuration: [String: Any] = [
            "tokenizer_class": "PreTrainedTokenizer", "unk_token": "<unk>",
            "clean_up_tokenization_spaces": false,
        ]
        let data: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocabulary, "merges": [[String]](), "byte_fallback": false],
            "added_tokens": addedTokens.map(\.json),
        ]
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("midnight-added-token-regex-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: configuration)
            .write(to: directory.appendingPathComponent("tokenizer_config.json"))
        try JSONSerialization.data(withJSONObject: data)
            .write(to: directory.appendingPathComponent("tokenizer.json"))
        return try await AutoTokenizer.from(modelFolder: directory)
    }

    private func legacyCapturedSections(text: String, alternatives: [String]) throws -> [String] {
        // The old capture-aware splitter is an oracle only for partial-grapheme
        // boundaries, whose NSRange conversion must remain Foundation-defined.
        let pattern = alternatives.sorted { $0.count > $1.count }
            .map { "(\(NSRegularExpression.escapedPattern(for: $0)))" }.joined(separator: "|")
        let regex = try NSRegularExpression(pattern: pattern)
        let matches = regex.matches(in: text, range: NSRange(text.startIndex..<text.endIndex, in: text))
        if matches.isEmpty {
            return [text]
        }
        var sections: [String] = []
        var start = text.startIndex
        for match in matches {
            guard let range = Range(match.range, in: text) else {
                continue
            }
            if start < range.lowerBound {
                sections.append(String(text[start..<range.lowerBound]))
            }
            start = range.upperBound
            for index in (0..<match.numberOfRanges).reversed() {
                if let capture = Range(match.range(at: index), in: text) {
                    sections.append(String(text[capture]))
                    break
                }
            }
        }
        if start < text.endIndex {
            sections.append(String(text[start...]))
        }
        return sections
    }

    private func expectTokens(
        _ tokenizer: any Tokenizer, text: String, pieces: [String], ids: [Int],
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(tokenizer.tokenize(text: text) == pieces, sourceLocation: sourceLocation)
        #expect(tokenizer.encode(text: text, addSpecialTokens: false) == ids, sourceLocation: sourceLocation)
    }
}
