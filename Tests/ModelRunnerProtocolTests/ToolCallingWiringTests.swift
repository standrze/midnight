import Foundation
import Testing

@Suite("MLX tool-calling wiring")
struct ToolCallingWiringTests {
    @Test("Core supplies tools to both prompt rendering and output parsing")
    func coreWiresMLXToolAPIs() throws {
        let root = packageRoot()
        let source = try String(
            contentsOf: root.appendingPathComponent(
                "Sources/ModelRunnerCore/LocalModelRunner.swift"
            ),
            encoding: .utf8
        )

        #expect(source.contains("UserInput(chat: chatMessages, tools: toolSpecs)"))
        #expect(source.contains("tools: toolSpecs"))
        #expect(source.contains("case .toolCall(let call):"))
        #expect(source.contains("case .rejectedToolCall(let rejection):"))
        #expect(source.contains("return .tool(content, id: callID"))
        #expect(source.contains(".assistant(message.content ?? \"\", toolCalls: toolCalls)"))
    }

    @Test("HTTP stream exposes tool-call deltas and the OpenAI finish reason")
    func serverWiresOpenAIStream() throws {
        let source = try String(
            contentsOf: packageRoot().appendingPathComponent(
                "Sources/ModelRunner/ModelHTTPServer.swift"
            ),
            encoding: .utf8
        )

        #expect(source.contains("tools: completion.tools"))
        #expect(source.contains("case .toolCall(let call):"))
        #expect(source.contains("OpenAIToolCallDelta("))
        #expect(source.contains("finishReason = \"tool_calls\""))
    }

    private func packageRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
