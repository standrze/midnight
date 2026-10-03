import Foundation

// A Foundation-only measurement of the pinned tokenizer's added-token split.
// It deliberately excludes Jinja, BPE, model loading, and GPU execution.
struct AddedToken: Decodable {
    let content: String
    let lstrip: Bool
    let rstrip: Bool
}

struct TokenizerData: Decodable {
    let addedTokens: [AddedToken]

    enum CodingKeys: String, CodingKey {
        case addedTokens = "added_tokens"
    }
}

struct Trial: Codable {
    let pair: Int
    let order: String
    let variant: String
    let millisecondsPerSplit: Double
    let checksum: Int
}

struct Report: Codable {
    let scope: String
    let tokenizerPath: String
    let textPath: String
    let textUTF8Bytes: Int
    let addedTokenCount: Int
    let strippingTokenCount: Int
    let legacyCaptureGroups: Int
    let candidateCaptureGroups: Int
    let exactSegments: Bool
    let segmentCount: Int
    let iterations: Int
    let warmupsPerVariant: Int
    let trials: [Trial]
}

func makeRegex(_ tokens: [AddedToken], optimized: Bool) throws -> NSRegularExpression {
    let needsCaptures = !optimized || tokens.contains { $0.lstrip || $0.rstrip || $0.content.isEmpty }
    let pattern = tokens.map { token in
        let literal = NSRegularExpression.escapedPattern(for: token.content)
        if !needsCaptures {
            return literal
        }
        let prefix = token.lstrip ? #"\s*"# : ""
        let suffix = token.rstrip ? #"\s*"# : ""
        return "\(prefix)(\(literal))\(suffix)"
    }.joined(separator: "|")
    return try NSRegularExpression(pattern: pattern)
}

// Same full-match and reversed-capture selection as the pinned
// String+PreTokenization.swift split(by: NSRegularExpression) implementation.
func split(_ text: String, by regex: NSRegularExpression) -> [String] {
    let matches = regex.matches(in: text, range: NSRange(text.startIndex..<text.endIndex, in: text))
    if matches.isEmpty {
        return [text]
    }
    var result: [String] = []
    var start = text.startIndex
    for match in matches {
        guard let matchRange = Range(match.range, in: text) else {
            continue
        }
        if start < matchRange.lowerBound {
            result.append(String(text[start..<matchRange.lowerBound]))
        }
        start = matchRange.upperBound
        for index in (0..<match.numberOfRanges).reversed() {
            if let range = Range(match.range(at: index), in: text) {
                result.append(String(text[range]))
                break
            }
        }
    }
    if start < text.endIndex {
        result.append(String(text[start...]))
    }
    return result
}

func run() throws {
    let arguments = Array(CommandLine.arguments.dropFirst())
    guard arguments.count == 2 else {
        throw NSError(
            domain: "AddedTokenRegexBenchmark", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Usage: benchmark-added-token-regex TOKENIZER_JSON RENDERED_TEXT"])
    }
    let data = try Data(contentsOf: URL(fileURLWithPath: arguments[0]))
    let tokens = try JSONDecoder().decode(TokenizerData.self, from: data).addedTokens.sorted {
        $0.content.count > $1.content.count
    }
    let text = try String(contentsOfFile: arguments[1], encoding: .utf8)
    let legacy = try makeRegex(tokens, optimized: false)
    let candidate = try makeRegex(tokens, optimized: true)
    let expected = split(text, by: legacy)
    guard expected == split(text, by: candidate) else {
        throw NSError(
            domain: "AddedTokenRegexBenchmark", code: 2,
            userInfo: [NSLocalizedDescriptionKey: "Candidate segmentation differs from original"])
    }
    let warmups = 3
    let iterations = 10
    for _ in 0..<warmups {
        precondition(split(text, by: legacy) == expected)
        precondition(split(text, by: candidate) == expected)
    }
    var trials: [Trial] = []
    for pair in 0..<8 {
        let order = pair.isMultiple(of: 2) ? ["legacy", "candidate"] : ["candidate", "legacy"]
        for variant in order {
            let regex = variant == "legacy" ? legacy : candidate
            var checksum = 0
            let start = ContinuousClock.now
            for _ in 0..<iterations {
                let segments = split(text, by: regex)
                checksum += segments.reduce(0) { $0 + $1.utf8.count }
            }
            let duration = start.duration(to: .now).components
            let seconds = Double(duration.seconds) + Double(duration.attoseconds) / 1e18
            precondition(checksum == iterations * expected.reduce(0) { $0 + $1.utf8.count })
            trials.append(
                Trial(
                    pair: pair + 1, order: order.joined(separator: "/"), variant: variant,
                    millisecondsPerSplit: seconds * 1_000 / Double(iterations), checksum: checksum))
        }
    }
    let report = Report(
        scope: "Warm added-token regex splitting only; excludes Jinja, BPE and model execution",
        tokenizerPath: arguments[0], textPath: arguments[1], textUTF8Bytes: text.utf8.count,
        addedTokenCount: tokens.count, strippingTokenCount: tokens.filter { $0.lstrip || $0.rstrip }.count,
        legacyCaptureGroups: legacy.numberOfCaptureGroups, candidateCaptureGroups: candidate.numberOfCaptureGroups,
        exactSegments: true, segmentCount: expected.count, iterations: iterations, warmupsPerVariant: warmups,
        trials: trials)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    print(String(decoding: try encoder.encode(report), as: UTF8.self))
}

try run()
