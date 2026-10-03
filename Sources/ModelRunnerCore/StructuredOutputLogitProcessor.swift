import Foundation
import MLX
import MLXLMCommon
import ModelRunnerProtocol

/// Masks the actual model vocabulary before sampling. The state belongs to one
/// generation; read failure/completion only after its producer task has joined.
final class StructuredOutputLogitProcessor: LogitProcessor, @unchecked Sendable {
    private let grammar: StructuredOutputGrammar
    private let vocabulary: StructuredOutputVocabulary
    private let eosTokenIDs: Set<Int>
    private var state: StructuredOutputGrammar.State
    private var failure: StructuredOutputDecodingError?
    private var ended = false
    private struct MaskKey: Hashable {
        let state: StructuredOutputGrammar.State
        let vocabularyCount: Int
    }
    // Repeated string/number grammar states are common. Retain only host masks,
    // so copies can safely use a different MLX device or stream.
    private var maskCache: [MaskKey: [Bool]] = [:]
    private static let maximumCachedMasks = 64

    init(
        grammar: StructuredOutputGrammar,
        modelDirectory: URL,
        tokenizer: any MLXLMCommon.Tokenizer,
        eosTokenIDs: Set<Int>
    ) throws {
        let url = modelDirectory.appendingPathComponent("tokenizer.json")
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw StructuredOutputDecodingError.unsupportedTokenizer(
                "Structured output requires a readable tokenizer.json: \(error.localizedDescription)")
        }
        let vocabulary = try StructuredOutputVocabulary(tokenizerData: data, tokenizer: tokenizer)
        guard !eosTokenIDs.isEmpty else {
            throw StructuredOutputDecodingError.unsupportedTokenizer(
                "Structured output requires a configured end-of-sequence token.")
        }
        self.grammar = grammar
        self.vocabulary = vocabulary
        self.eosTokenIDs = eosTokenIDs
        self.state = grammar.initialState
    }

    /// Also allows tiny, real-byte vocabularies in CPU-only decoder tests.
    init(grammar: StructuredOutputGrammar, tokenBytes: [Int: [UInt8]], eosTokenIDs: Set<Int>) throws {
        guard !eosTokenIDs.isEmpty else {
            throw StructuredOutputDecodingError.unsupportedTokenizer("An EOS token is required.")
        }
        self.grammar = grammar
        self.vocabulary = try StructuredOutputVocabulary(tokenBytes: tokenBytes)
        self.eosTokenIDs = eosTokenIDs
        self.state = grammar.initialState
    }

    private init(
        grammar: StructuredOutputGrammar,
        vocabulary: StructuredOutputVocabulary,
        eosTokenIDs: Set<Int>
    ) {
        self.grammar = grammar
        self.vocabulary = vocabulary
        self.eosTokenIDs = eosTokenIDs
        self.state = grammar.initialState
    }

    func copy() -> StructuredOutputLogitProcessor {
        let copy = StructuredOutputLogitProcessor(
            grammar: grammar, vocabulary: vocabulary, eosTokenIDs: eosTokenIDs)
        copy.state = state
        copy.failure = failure
        copy.ended = ended
        copy.maskCache = maskCache
        return copy
    }

    /// Immutable token lookup is safe while the producer mutates grammar state.
    /// Consumers assemble these bytes and emit only complete UTF-8 prefixes;
    /// per-token String decoding would corrupt scalars split across token IDs.
    func bytes(for tokenID: Int) -> [UInt8]? {
        guard !eosTokenIDs.contains(tokenID) else {
            return nil
        }
        return vocabulary.tokenBytes[tokenID]
    }

    var isComplete: Bool { grammar.isComplete(state) }

    func throwIfFailed() throws {
        if let failure {
            throw failure
        }
    }

    func prompt(_ prompt: MLXArray) {
        // Prompt/template tokens do not form part of the requested JSON document.
        state = grammar.initialState
        failure = nil
        ended = false
        maskCache.removeAll(keepingCapacity: true)
    }

    func process(logits: MLXArray) -> MLXArray {
        let count = logits.dim(-1)
        let terminalIDs = eosTokenIDs.filter { (0..<count).contains($0) }
        guard let fallbackID = terminalIDs.min() else {
            failure = .generation("The model logits contain no configured EOS token.")
            return terminalLogits(logits, tokenID: 0)
        }
        if failure != nil || ended || grammar.isComplete(state) {
            return terminalLogits(logits, tokenID: fallbackID)
        }

        guard let allowed = allowedTokenMask(vocabularyCount: count) else {
            failure = .generation("No vocabulary token can continue the requested JSON structure.")
            return terminalLogits(logits, tokenID: fallbackID)
        }

        let suppressed = MLXArray(-Float.infinity).asType(logits.dtype)
        let mask = broadcast(MLXArray(allowed), to: logits.shape)
        let result = which(mask .&& isFinite(logits), logits, suppressed)
        // Avoid passing an all-infinite row into softmax/argmax. Such a row could
        // otherwise sample a prohibited token without reporting the model failure.
        guard any(isFinite(result)).item(Bool.self) else {
            failure = .generation("The model produced no finite logit for a permitted JSON token.")
            return terminalLogits(logits, tokenID: fallbackID)
        }
        return result
    }

    private func allowedTokenMask(vocabularyCount count: Int) -> [Bool]? {
        let key = MaskKey(state: state, vocabularyCount: count)
        if let cached = maskCache[key] {
            return cached
        }

        var allowed = Array(repeating: false, count: count)
        var frontier: [(Int, StructuredOutputGrammar.State)] = [(0, state)]
        var hasAllowedToken = false
        while let (nodeID, prefixState) = frontier.popLast() {
            let node = vocabulary.nodes[nodeID]
            for tokenID in node.tokenIDs where (0..<count).contains(tokenID) && !eosTokenIDs.contains(tokenID) {
                allowed[tokenID] = true
                hasAllowedToken = true
            }
            for edge in node.edges {
                if let next = grammar.advancing(prefixState, byte: edge.byte) {
                    frontier.append((edge.node, next))
                }
            }
        }
        guard hasAllowedToken else {
            return nil
        }
        if maskCache.count >= Self.maximumCachedMasks {
            maskCache.removeAll(keepingCapacity: true)
        }
        maskCache[key] = allowed
        return allowed
    }

    func didSample(token: MLXArray) {
        guard failure == nil else {
            return
        }
        let id = token.item(Int.self)
        if eosTokenIDs.contains(id) {
            if !grammar.isComplete(state) {
                failure = .generation("Generation ended before completing the requested JSON structure.")
            }
            ended = true
            return
        }
        guard !ended, let bytes = vocabulary.tokenBytes[id],
            let next = grammar.advancing(state, bytes: bytes)
        else {
            failure = .generation("A sampled token violated the requested JSON structure.")
            return
        }
        state = next
    }

    private func terminalLogits(_ logits: MLXArray, tokenID: Int) -> MLXArray {
        let selected = arange(logits.dim(-1), dtype: .int32) .== Int32(tokenID)
        // Preserve a dependency on model evaluation even when EOS was non-finite;
        // this makes the token iterator finish its pending prompt/cache graph.
        let finite = which(isFinite(logits), logits, MLXArray(0 as Float).asType(logits.dtype))
        return which(selected, finite, MLXArray(-Float.infinity).asType(logits.dtype))
    }
}

enum StructuredOutputDecodingError: LocalizedError, Equatable {
    case unsupportedTokenizer(String)
    case generation(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedTokenizer(let message), .generation(let message): message
        }
    }
}

/// A prefix trie shares grammar transitions among tokens with the same bytes.
/// Decoder cleanup is deliberately not applied: the structured-output stream
/// emits this exact byte mapping, preserving JSON punctuation and whitespace.
struct StructuredOutputVocabulary: Sendable {
    struct Edge: Sendable {
        let byte: UInt8
        let node: Int
    }
    struct Node: Sendable {
        var edges: [Edge] = []
        var tokenIDs: [Int] = []
    }

    let tokenBytes: [Int: [UInt8]]
    let nodes: [Node]

    init(tokenBytes: [Int: [UInt8]]) throws {
        guard tokenBytes.keys.allSatisfy({ (0...1_000_000).contains($0) }),
            tokenBytes.values.allSatisfy({ $0.count <= 4096 }),
            tokenBytes.values.reduce(0, { $0 + $1.count }) <= 32 * 1024 * 1024
        else {
            throw StructuredOutputDecodingError.unsupportedTokenizer("Tokenizer vocabulary exceeds supported bounds.")
        }
        // Zero-byte tokens can make no progress and must never be sampled.
        self.tokenBytes = tokenBytes.filter { !$0.value.isEmpty }
        var nodes = [Node()]
        for (id, bytes) in self.tokenBytes.sorted(by: { $0.key < $1.key }) {
            var index = 0
            for byte in bytes {
                if let edge = nodes[index].edges.first(where: { $0.byte == byte }) {
                    index = edge.node
                } else {
                    let next = nodes.count
                    nodes.append(Node())
                    nodes[index].edges.append(Edge(byte: byte, node: next))
                    index = next
                }
            }
            nodes[index].tokenIDs.append(id)
        }
        self.nodes = nodes
    }

    init(tokenizerData: Data, tokenizer: (any MLXLMCommon.Tokenizer)? = nil) throws {
        guard let root = try JSONSerialization.jsonObject(with: tokenizerData) as? [String: Any],
            let model = root["model"] as? [String: Any],
            let type = model["type"] as? String, ["BPE", "Unigram"].contains(type),
            let decoder = root["decoder"] as? [String: Any]
        else {
            throw StructuredOutputDecodingError.unsupportedTokenizer(
                "Structured output currently supports tokenizer.json BPE or Unigram vocabularies with ByteLevel or SentencePiece decoders."
            )
        }
        let decoding = try ByteDecoding(decoder: decoder)
        var pieces: [Int: String] = [:]
        if let vocab = model["vocab"] as? [String: Int] {
            for (piece, id) in vocab {
                pieces[id] = piece
            }
        } else if let vocab = model["vocab"] as? [[Any]] {
            for (id, entry) in vocab.enumerated() {
                guard let piece = entry.first as? String else {
                    throw StructuredOutputDecodingError.unsupportedTokenizer("Invalid Unigram vocabulary entry.")
                }
                pieces[id] = piece
            }
        } else {
            throw StructuredOutputDecodingError.unsupportedTokenizer(
                "Unrecognized tokenizer vocabulary representation.")
        }

        var added: Set<Int> = []
        var special: Set<Int> = []
        for entry in root["added_tokens"] as? [[String: Any]] ?? [] {
            guard let id = entry["id"] as? Int, let piece = entry["content"] as? String else {
                continue
            }
            pieces[id] = piece
            added.insert(id)
            if entry["special"] as? Bool == true {
                special.insert(id)
            }
        }
        if let unknown = tokenizer?.unknownTokenId {
            special.insert(unknown)
        }
        if let bos = tokenizer?.bosToken, let id = tokenizer?.convertTokenToId(bos) {
            special.insert(id)
        }

        var bytes: [Int: [UInt8]] = [:]
        for (id, piece) in pieces where !special.contains(id) {
            if let tokenizer, tokenizer.convertIdToToken(id) != piece {
                throw StructuredOutputDecodingError.unsupportedTokenizer(
                    "The loaded tokenizer vocabulary does not match tokenizer.json at token \(id).")
            }
            bytes[id] = try decoding.bytes(piece, added: added.contains(id))
        }
        try self.init(tokenBytes: bytes)
    }

    private struct ByteDecoding {
        enum Kind {
            case byteLevel
            case sentencePiece(replacement: String, fallback: Bool)
        }
        let kind: Kind

        init(decoder: [String: Any]) throws {
            switch decoder["type"] as? String {
            case "ByteLevel":
                kind = .byteLevel
            case "Metaspace":
                guard let replacement = decoder["replacement"] as? String, !replacement.isEmpty else {
                    throw Self.unsupported()
                }
                kind = .sentencePiece(replacement: replacement, fallback: false)
            case "Sequence":
                guard let decoders = decoder["decoders"] as? [[String: Any]] else {
                    throw Self.unsupported()
                }
                if decoders.count == 1, decoders[0]["type"] as? String == "ByteLevel" {
                    kind = .byteLevel
                    return
                }
                // This is the standard Llama/Mistral/Gemma SentencePiece byte-fallback
                // decoder. Reordering transformations or arbitrary regex replacements
                // requires different byte semantics and is rejected explicitly.
                let types = decoders.compactMap { $0["type"] as? String }
                guard
                    types == ["Replace", "ByteFallback", "Fuse", "Strip"]
                        || types == ["Replace", "ByteFallback", "Fuse"]
                else {
                    throw Self.unsupported()
                }
                guard let pattern = decoders[0]["pattern"] as? [String: Any],
                    let replacement = pattern["String"] as? String, !replacement.isEmpty,
                    decoders[0]["content"] as? String == " "
                else {
                    throw Self.unsupported()
                }
                if types.last == "Strip" {
                    let strip = decoders[3]
                    guard strip["content"] as? String == " ",
                        strip["start"] as? Int == 1, strip["stop"] as? Int == 0
                    else {
                        throw Self.unsupported()
                    }
                }
                // Preserving SentencePiece's optional initial space is harmless JSON
                // whitespace and makes the emitted bytes strictly append-only.
                kind = .sentencePiece(replacement: replacement, fallback: true)
            default:
                throw Self.unsupported()
            }
        }

        func bytes(_ piece: String, added: Bool) throws -> [UInt8] {
            switch kind {
            case .byteLevel:
                if added {
                    return Array(piece.utf8)
                }
                return try piece.unicodeScalars.map { scalar in
                    guard let byte = Self.byteAlphabet[scalar.value] else {
                        throw Self.unsupported()
                    }
                    return byte
                }
            case .sentencePiece(let replacement, let fallback):
                if fallback, piece.utf8.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">"),
                    let byte = UInt8(piece.dropFirst(3).dropLast(), radix: 16)
                {
                    return [byte]
                }
                return Array(piece.replacingOccurrences(of: replacement, with: " ").utf8)
            }
        }

        private static func unsupported() -> StructuredOutputDecodingError {
            .unsupportedTokenizer(
                "This tokenizer decoder is not supported for constrained structured output. Supported decoders are ByteLevel, Metaspace, and the standard SentencePiece Replace/ByteFallback/Fuse/Strip sequence."
            )
        }

        private static let byteAlphabet: [UInt32: UInt8] = {
            let visible = Array(33...126) + Array(161...172) + Array(174...255)
            let visibleSet = Set(visible)
            var mapping = Dictionary(uniqueKeysWithValues: visible.map { (UInt32($0), UInt8($0)) })
            var extra: UInt32 = 256
            for byte in 0...255 where !visibleSet.contains(byte) {
                mapping[extra] = UInt8(byte)
                extra += 1
            }
            return mapping
        }()
    }
}
