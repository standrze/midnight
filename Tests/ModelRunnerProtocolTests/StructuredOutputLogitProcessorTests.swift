import Foundation
import MLX
import ModelRunnerProtocol
import Testing

@testable import ModelRunnerCore

@Suite("Structured output token masking", .serialized)
struct StructuredOutputLogitProcessorTests {
    @Test("The mask rejects prose and premature EOS, retains batch shape, then forces EOS")
    func masksBeforeSampling() throws {
        try Device.withDefaultDevice(.cpu) {
            let grammar = try StructuredOutputGrammar(responseFormat: .jsonObject)
            let processor = try StructuredOutputLogitProcessor(
                grammar: grammar,
                tokenBytes: [
                    1: Array("{".utf8), 2: Array("}".utf8), 3: Array("prose".utf8),
                    4: Array(" ".utf8), 5: Array("{}".utf8),
                ], eosTokenIDs: [0])
            processor.prompt(MLXArray([3, 3, 3]))
            let logits = MLXArray([Float(100), 2, 3, 99, 4, 5], [1, 6])
            let first = processor.process(logits: logits)
            #expect(first.shape == [1, 6])
            let expectedFirst: [Float] = [-.infinity, 2, -.infinity, -.infinity, 4, 5]
            #expect(first.asArray(Float.self) == expectedFirst)
            processor.didSample(token: MLXArray(1))
            processor.didSample(token: MLXArray(2))
            #expect(processor.isComplete)
            let completed = processor.process(logits: logits).asArray(Float.self)
            let expectedCompleted: [Float] = [100, -.infinity, -.infinity, -.infinity, -.infinity, -.infinity]
            #expect(completed == expectedCompleted)
            processor.didSample(token: MLXArray(0))
            try processor.throwIfFailed()
            #expect(processor.bytes(for: 0) == nil)
        }
    }

    @Test("UTF-8 scalars can span vocabulary tokens without lossy replacement")
    func splitUnicode() throws {
        try Device.withDefaultDevice(.cpu) {
            let processor = try StructuredOutputLogitProcessor(
                grammar: StructuredOutputGrammar(responseFormat: .jsonObject),
                tokenBytes: [
                    1: Array("{\"name\":\"".utf8), 2: [0xC3], 3: [0xA9],
                    4: [0xFF], 5: Array("\"}".utf8),
                ], eosTokenIDs: [0])
            let logits = MLXArray(Array(repeating: Float(1), count: 6), [1, 6])
            processor.didSample(token: MLXArray(1))
            processor.didSample(token: MLXArray(2))
            let partial = processor.process(logits: logits).asArray(Float.self)
            let expectedPartial: [Float] = [-.infinity, -.infinity, -.infinity, 1, -.infinity, -.infinity]
            #expect(partial == expectedPartial)
            processor.didSample(token: MLXArray(3))
            processor.didSample(token: MLXArray(5))
            #expect(processor.isComplete)
            let raw = [1, 2, 3, 5].flatMap { processor.bytes(for: $0) ?? [] }
            #expect(String(bytes: raw, encoding: .utf8) == "{\"name\":\"é\"}")
            try processor.throwIfFailed()
        }
    }

    @Test("Schema constraints exclude tokens of the wrong value type")
    func schemaTypes() throws {
        try Device.withDefaultDevice(.cpu) {
            let format = try JSONDecoder().decode(
                OpenAIResponseFormat.self,
                from: Data(
                    #"{"type":"json_schema","json_schema":{"name":"result","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}}}"#
                        .utf8))
            let processor = try StructuredOutputLogitProcessor(
                grammar: StructuredOutputGrammar(responseFormat: format),
                tokenBytes: [
                    1: Array("{\"answer\":".utf8), 2: Array("42".utf8),
                    3: Array("\"yes\"}".utf8), 4: Array("null".utf8),
                ], eosTokenIDs: [0])
            processor.didSample(token: MLXArray(1))
            let logits = MLXArray(Array(repeating: Float(1), count: 5))
            let expected: [Float] = [-.infinity, -.infinity, -.infinity, 1, -.infinity]
            #expect(processor.process(logits: logits).asArray(Float.self) == expected)
            processor.didSample(token: MLXArray(3))
            #expect(processor.isComplete)
        }
    }

    @Test("Copies own independent grammar state and prompt starts a fresh document")
    func copyAndPrompt() throws {
        let processor = try StructuredOutputLogitProcessor(
            grammar: StructuredOutputGrammar(responseFormat: .jsonObject),
            tokenBytes: [1: Array("{}".utf8)], eosTokenIDs: [0])
        let copy = processor.copy()
        processor.didSample(token: MLXArray(1))
        #expect(processor.isComplete)
        #expect(!copy.isComplete)
        processor.prompt(MLXArray([1]))
        #expect(!processor.isComplete)
        try processor.throwIfFailed()
    }

    @Test("Repeated string states reuse masks without reusing logits or vocabulary shape")
    func repeatedStringMasks() throws {
        try Device.withDefaultDevice(.cpu) {
            let processor = try StructuredOutputLogitProcessor(
                grammar: StructuredOutputGrammar(responseFormat: .jsonObject),
                tokenBytes: [
                    1: Array("{\"answer\":\"".utf8), 2: Array("a".utf8),
                    3: Array("b".utf8), 4: Array("\"}".utf8), 5: [0xFF], 6: Array("c".utf8),
                ],
                eosTokenIDs: [0])
            processor.didSample(token: MLXArray(1))
            let initial = MLXArray(Array(repeating: Float(1), count: 6), [1, 6])
            let expected: [Float] = [-.infinity, -.infinity, 1, 1, 1, -.infinity]
            #expect(processor.process(logits: initial).asArray(Float.self) == expected)
            #expect(processor.process(logits: initial).asArray(Float.self) == expected)
            processor.didSample(token: MLXArray(2))
            let changed = MLXArray(Array(repeating: Float(2), count: 6), [1, 6])
            let expectedChanged: [Float] = [-.infinity, -.infinity, 2, 2, 2, -.infinity]
            #expect(processor.process(logits: changed).asArray(Float.self) == expectedChanged)
            let expanded = MLXArray(Array(repeating: Float(3), count: 7), [1, 7])
            let expectedExpanded: [Float] = [-.infinity, -.infinity, 3, 3, 3, -.infinity, 3]
            #expect(processor.process(logits: expanded).asArray(Float.self) == expectedExpanded)

            let copy = processor.copy()
            processor.prompt(MLXArray([1]))
            let expectedReset: [Float] = [-.infinity, 1, -.infinity, -.infinity, -.infinity, -.infinity]
            #expect(processor.process(logits: initial).asArray(Float.self) == expectedReset)
            #expect(copy.process(logits: initial).asArray(Float.self) == expected)
            try processor.throwIfFailed()
            try copy.throwIfFailed()
        }
    }

    @Test("Impossible continuations terminate with a reported error")
    func impossibleContinuation() throws {
        try Device.withDefaultDevice(.cpu) {
            let processor = try StructuredOutputLogitProcessor(
                grammar: StructuredOutputGrammar(responseFormat: .jsonObject),
                tokenBytes: [1: Array("hello".utf8)], eosTokenIDs: [0])
            let result = processor.process(logits: MLXArray([Float(2), 10])).asArray(Float.self)
            let expected: [Float] = [2, -.infinity]
            #expect(result == expected)
            #expect(throws: StructuredOutputDecodingError.self) { try processor.throwIfFailed() }
            #expect(!processor.isComplete)
        }
    }

    @Test("All-nonfinite legal logits cannot silently produce invalid JSON")
    func nonfiniteModelOutput() throws {
        try Device.withDefaultDevice(.cpu) {
            let processor = try StructuredOutputLogitProcessor(
                grammar: StructuredOutputGrammar(responseFormat: .jsonObject),
                tokenBytes: [1: Array("{}".utf8)], eosTokenIDs: [0])
            let result = processor.process(logits: MLXArray([Float.nan, -.infinity])).asArray(Float.self)
            let expected: [Float] = [0, -.infinity]
            #expect(result == expected)
            #expect(throws: StructuredOutputDecodingError.self) { try processor.throwIfFailed() }
        }
    }

    @Test("ByteLevel vocabulary keeps split bytes and treats added tokens literally")
    func byteLevelVocabulary() throws {
        let data = Data(
            #"{"model":{"type":"BPE","vocab":{"Ġ":0,"Ã":1,"©":2,"{}":3}},"decoder":{"type":"ByteLevel"},"added_tokens":[{"id":4,"content":"Ġliteral","special":false},{"id":5,"content":"<eos>","special":true}]}"#
                .utf8)
        let vocabulary = try StructuredOutputVocabulary(tokenizerData: data)
        #expect(vocabulary.tokenBytes[0] == [32])
        #expect(vocabulary.tokenBytes[1] == [0xC3])
        #expect(vocabulary.tokenBytes[2] == [0xA9])
        #expect(vocabulary.tokenBytes[4] == Array("Ġliteral".utf8))
        #expect(vocabulary.tokenBytes[5] == nil)

        let sequence = Data(
            #"{"model":{"type":"BPE","vocab":{"Ġ{":0}},"decoder":{"type":"Sequence","decoders":[{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}]}}"#
                .utf8)
        #expect(try StructuredOutputVocabulary(tokenizerData: sequence).tokenBytes[0] == Array(" {".utf8))
    }

    @Test("SentencePiece byte fallback and metaspace retain exact output bytes")
    func sentencePieceVocabulary() throws {
        let data = Data(
            #"{"model":{"type":"BPE","vocab":{"▁{":0,"<0xC3>":1,"<0xA9>":2,"}":3}},"decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]}}"#
                .utf8)
        let vocabulary = try StructuredOutputVocabulary(tokenizerData: data)
        #expect(vocabulary.tokenBytes[0] == Array(" {".utf8))
        #expect(vocabulary.tokenBytes[1] == [0xC3])
        #expect(vocabulary.tokenBytes[2] == [0xA9])

        let unigram = Data(
            #"{"model":{"type":"Unigram","vocab":[["▁hello",0.0],["world",-1.0]]},"decoder":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always"}}"#
                .utf8)
        let unigramVocabulary = try StructuredOutputVocabulary(tokenizerData: unigram)
        #expect(unigramVocabulary.tokenBytes[0] == Array(" hello".utf8))
        #expect(unigramVocabulary.tokenBytes[1] == Array("world".utf8))
    }

    @Test("Unknown decoder transformations fail before model generation")
    func unsupportedDecoder() {
        let data = Data(#"{"model":{"type":"BPE","vocab":{"a":0}},"decoder":{"type":"WordPiece","cleanup":true}}"#.utf8)
        #expect(throws: StructuredOutputDecodingError.self) {
            try StructuredOutputVocabulary(tokenizerData: data)
        }
    }
}
