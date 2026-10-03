import Foundation
import ModelRunnerProtocol
import Testing

@testable import Midnight

@Suite("Chat request decoding diagnostics")
struct ModelHTTPDecodingTests {
    @Test("Chat validation runs without a channel and retains parsed limits")
    func chatValidationWithoutChannel() throws {
        let server = ModelHTTPServer(
            servedModelName: "local", tokenLimit: try GenerationTokenLimit(configuredMaximum: 64))
        let body = Data(
            #"{"model":"local","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":12,"stop":["END"]}"#
                .utf8)

        let request = try server.decodeAndValidateChatRequest(body)

        #expect(request.completion.model == "local")
        #expect(request.requestedMaximumTokens == 12)
        #expect(request.stop == ["END"])
    }

    @Test("Chat validation identifies conflicting token-limit fields before I/O")
    func conflictingTokenLimitsWithoutChannel() throws {
        let server = ModelHTTPServer(
            servedModelName: "local", tokenLimit: try GenerationTokenLimit(configuredMaximum: 64))
        let body = Data(
            #"{"model":"local","messages":[{"role":"user","content":"Hi"}],"max_tokens":8,"max_completion_tokens":12}"#
                .utf8)

        do {
            _ = try server.decodeAndValidateChatRequest(body)
            Issue.record("Expected a conflicting-token-limit error")
        } catch let error as ModelHTTPError {
            #expect(error.param == "max_completion_tokens")
            #expect(error.code == "invalid_parameter")
        }
    }

    @Test("Unsupported top-level Chat Completions fields are rejected")
    func unsupportedTopLevelFields() throws {
        let server = ModelHTTPServer(
            servedModelName: "local", tokenLimit: try GenerationTokenLimit(configuredMaximum: 64))

        for field in ["seed", "presence_penalty", "topP"] {
            let body = Data(
                """
                {"model":"local","messages":[{"role":"user","content":"Hi"}],"\(field)":1}
                """.utf8)

            do {
                _ = try server.decodeAndValidateChatRequest(body)
                Issue.record("Expected an unsupported-field error for \(field)")
            } catch let error as ModelHTTPError {
                #expect(error.param == field)
                #expect(error.code == "invalid_json")
                #expect(error.message.contains("Unsupported Chat Completions parameter '\(field)'"))
            }
        }
    }

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
