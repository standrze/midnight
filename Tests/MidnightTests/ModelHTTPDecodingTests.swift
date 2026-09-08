import Foundation
import Testing

@testable import Midnight

@Suite("Chat request decoding diagnostics")
struct ModelHTTPDecodingTests {
    @Test("Missing nested fields include the absent key and array indices")
    func missingNestedField() throws {
        let issue = try diagnose(#"{"messages":[{"content":[{"type":"text","text":"a"},{"text":"b"}]}]}"#)

        #expect(issue.param == "messages[0].content[1].type")
        #expect(issue.message.hasPrefix("messages[0].content[1].type: "))
        #expect(issue.message.contains("No value associated with key"))
    }

    @Test("Missing top-level fields identify the required key")
    func missingTopLevelField() throws {
        let issue = try diagnose("{}")

        #expect(issue.param == "messages")
        #expect(issue.message.hasPrefix("messages: "))
    }

    @Test("Type mismatches retain the complete field path and decoder detail")
    func nestedTypeMismatch() throws {
        let issue = try diagnose(#"{"messages":[{"content":[{"type":"text","text":42}]}]}"#)

        #expect(issue.param == "messages[0].content[0].text")
        #expect(issue.message.hasPrefix("messages[0].content[0].text: "))
        #expect(issue.message.contains("String"))
    }

    @Test("Required null values identify their nested field")
    func nestedNullValue() throws {
        let issue = try diagnose(#"{"messages":[{"content":[{"type":"text","text":null}]}]}"#)

        #expect(issue.param == "messages[0].content[0].text")
        #expect(issue.message.hasPrefix("messages[0].content[0].text: "))
        #expect(issue.message.contains("null"))
    }

    @Test("Invalid enum values retain the nested decoder path")
    func corruptedNestedValue() throws {
        let issue = try diagnose(#"{"messages":[{"content":[{"type":"image","text":"a"}]}]}"#)

        #expect(issue.param == "messages[0].content[0].type")
        #expect(issue.message.hasPrefix("messages[0].content[0].type: "))
        #expect(issue.message.contains("Cannot initialize"))
    }

    @Test("Malformed JSON has decoder detail without inventing a field")
    func malformedJSON() throws {
        let issue = try diagnose("{")

        #expect(issue.param == nil)
        #expect(issue.message.contains("JSON"))
    }

    private func diagnose(_ json: String) throws -> (message: String, param: String?) {
        do {
            _ = try JSONDecoder().decode(Request.self, from: Data(json.utf8))
            throw UnexpectedSuccess.decoded
        } catch let error as DecodingError {
            return ModelHTTPServer.chatDecodingIssue(error)
        }
    }

    private enum UnexpectedSuccess: Error {
        case decoded
    }

    private struct Request: Decodable {
        let messages: [Message]
    }

    private struct Message: Decodable {
        let content: [Part]
    }

    private struct Part: Decodable {
        enum Kind: String, Decodable {
            case text
        }

        let type: Kind
        let text: String
    }
}
