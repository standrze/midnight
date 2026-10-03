import MLXLMCommon
import Testing

@Suite("Gemma 4 reasoning delimiters")
struct GemmaReasoningDelimiterTests {
    private func emitter() -> ReasoningEventEmitter {
        ReasoningEventEmitter(
            config: .init(
                startDelimiter: "<|channel>thought", endDelimiter: "<channel|>",
                promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: false)),
            primedInside: false)
    }

    @Test("Every chunk boundary preserves separate reasoning and answer")
    func splitBoundaries() {
        let text = "<|channel>thoughtCalculate 17 × 19<channel|>323"
        for offset in 0...text.count {
            var decoder = emitter()
            let split = text.index(text.startIndex, offsetBy: offset)
            let segments =
                decoder.process(String(text[..<split]))
                + decoder.process(String(text[split...])) + decoder.finalize()
            var reasoning = ""
            var answer = ""
            for segment in segments {
                switch segment {
                case .reasoning(let text): reasoning += text
                case .response(let text): answer += text
                }
            }
            #expect(reasoning == "Calculate 17 × 19")
            #expect(answer == "323")
            #expect(!decoder.isInsideReasoning)
        }
    }

    @Test("Truncated reasoning remains reasoning at end of generation")
    func unfinishedThought() {
        var decoder = emitter()
        let segments = decoder.process("<|channel>thoughtStill considering<chan") + decoder.finalize()
        var reasoning = ""
        for segment in segments {
            switch segment {
            case .reasoning(let text): reasoning += text
            case .response: Issue.record("Unfinished reasoning leaked into answer")
            }
        }
        #expect(reasoning == "Still considering<chan")
    }
}
