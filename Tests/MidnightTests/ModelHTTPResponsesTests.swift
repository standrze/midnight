import Foundation
import ModelRunnerProtocol
import NIOCore
import NIOEmbedded
import NIOHTTP1
import Testing

@testable import Midnight

@Suite("Responses HTTP endpoints", .timeLimit(.minutes(1)))
struct ModelHTTPResponsesTests {
    @Test("Stored responses are retrievable, pageable, and deletable without a model")
    func storedEndpointsWithoutModel() async throws {
        for managed in [false, true] {
            let store = ResponsesStore()
            let stored = entry(id: "resp_saved", items: inputItems(3))
            #expect(await store.store(id: "resp_saved", entry: stored))
            let server = try makeServer(store: store, managed: managed)
            let result = try await request(server, .GET, "/v1/responses/resp_saved?stream=false")
            #expect(result.status == .ok)
            #expect(try result.json() == stored.response)
            let page = try await request(server, .GET, "/v1/responses/resp_saved/input_items?order=asc&limit=2")
            #expect(page.status == .ok)
            let pageBody = try object(page.json())
            #expect(pageBody["data"] == .array(Array(stored.inputItems.prefix(2))))
            #expect(pageBody["first_id"] == .string("msg_0"))
            #expect(pageBody["last_id"] == .string("msg_1"))
            #expect(pageBody["has_more"] == .bool(true))
            let deletion = try await request(server, .DELETE, "/v1/responses/resp_saved")
            #expect(deletion.status == .ok)
            #expect(
                try deletion.json()
                    == .object([
                        "id": .string("resp_saved"), "object": .string("response.deleted"), "deleted": .bool(true),
                    ]))
            #expect(await store.get(id: "resp_saved") == nil)
            let deleted = try await request(server, .GET, "/v1/responses/resp_saved")
            #expect(deleted.status == .notFound)
            #expect(try deleted.error().code == "response_not_found")
        }
    }

    @Test("Missing and expired history returns an OpenAI error for every stored endpoint")
    func missingAndExpired() async throws {
        let store = ResponsesStore(timeToLive: 10)
        #expect(
            await store.store(id: "resp_expired", entry: entry(id: "resp_expired"), now: Date().addingTimeInterval(-20))
        )
        let server = try makeServer(store: store, managed: true)
        for id in ["resp_missing", "resp_expired"] {
            for (method, suffix) in [(HTTPMethod.GET, ""), (.DELETE, ""), (.GET, "/input_items")] {
                let response = try await request(server, method, "/v1/responses/\(id)\(suffix)")
                #expect(response.status == .notFound)
                #expect(try response.error().code == "response_not_found")
                #expect(try response.error().param == "response_id")
                #expect(response.contentType?.hasPrefix("application/json") == true)
            }
        }
    }

    @Test("Unsupported methods return 405 before model admission")
    func unsupportedMethods() async throws {
        for managed in [false, true] {
            let server = try makeServer(managed: managed)
            for (method, path) in [
                (HTTPMethod.GET, "/v1/responses"), (.DELETE, "/v1/responses"),
                (.POST, "/v1/responses/resp_missing"), (.PATCH, "/v1/responses/resp_missing"),
                (.POST, "/v1/responses/resp_missing/input_items"), (.DELETE, "/v1/responses/resp_missing/input_items"),
            ] {
                let response = try await request(server, method, path)
                #expect(response.status == .methodNotAllowed)
                #expect(try response.error().code == "method_not_allowed")
            }
        }
    }

    @Test("Stored endpoint query parameters are validated before returning results")
    func invalidQueries() async throws {
        let store = ResponsesStore()
        #expect(await store.store(id: "resp_saved", entry: entry(id: "resp_saved", items: inputItems(2))))
        let server = try makeServer(store: store, managed: true)
        for (method, suffix, param) in [
            (HTTPMethod.GET, "?stream=true", "stream"), (.GET, "?stream=", "stream"),
            (.GET, "?stream=false&stream=false", "stream"), (.GET, "?extra=1", "extra"),
            (.DELETE, "?stream=false", "stream"),
            (.GET, "/input_items?limit=0", "limit"), (.GET, "/input_items?limit=101", "limit"),
            (.GET, "/input_items?limit=no", "limit"), (.GET, "/input_items?order=invalid", "order"),
            (.GET, "/input_items?after=missing", "after"), (.GET, "/input_items?extra=1", "extra"),
            (.GET, "/input_items?limit=1&limit=2", "limit"), (.GET, "/input_items?order", "order"),
        ] {
            let response = try await request(server, method, "/v1/responses/resp_saved\(suffix)")
            #expect(response.status == .badRequest)
            #expect(try response.error().param == param)
            #expect(try response.error().code == "invalid_parameter")
        }
        #expect(await store.get(id: "resp_saved") != nil)
    }

    @Test("Creating a response for an unknown model on an empty managed listener returns model not found")
    func unknownModelOnEmptyManagedListener() async throws {
        let response = try await request(
            makeServer(managed: true), .POST, "/v1/responses",
            body: #"{"model":"local","input":"Hello","stream":true}"#)
        // Named requests select from the installed catalog, even while idle.
        // An unknown ID is a model lookup error, not temporary unavailability.
        #expect(response.status == .notFound)
        #expect(try response.error().code == "model_not_found")
        #expect(response.contentType?.hasPrefix("application/json") == true)
    }

    @Test("Invalid and unsupported creation options fail before SSE or model execution")
    func invalidCreationPreflight() async throws {
        let server = try makeServer()
        let malformed = try await request(server, .POST, "/v1/responses", body: "{")
        #expect(malformed.status == .badRequest)
        #expect(try malformed.error().code == "invalid_json")
        let cases: [(String, String)] = [
            (
                #"{"model":"local","input":[{"role":"user","content":[{"type":"input_image","image_url":"https://example.com/image.png"}]}],"stream":true}"#,
                "input[0].content[0].type"
            ),
            (#"{"model":"local","input":"Hello","tools":[{"type":"web_search"}],"stream":true}"#, "tools[0].type"),
            (
                #"{"model":"local","input":"Hello","tools":[{"type":"function","name":"test","strict":true}],"stream":true}"#,
                "tools[0].strict"
            ),
            (
                #"{"model":"local","input":"Hello","text":{"format":{"type":"unknown"}},"stream":true}"#,
                "text.format.type"
            ),
            (
                #"{"model":"local","input":"Hello","text":{"format":{"type":"json_schema","name":"answer","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"string","pattern":"x"}},"required":["answer"],"additionalProperties":false}}},"stream":true}"#,
                "text.format"
            ),
            (
                #"{"model":"local","input":"Hello","tools":[{"type":"function","name":"test","strict":false}],"text":{"format":{"type":"json_object"}},"stream":true}"#,
                "text.format"
            ),
            (#"{"model":"local","input":"Hello","background":true,"stream":true}"#, "background"),
            (#"{"model":"local","input":"Hello","max_output_tokens":0,"stream":true}"#, "max_output_tokens"),
        ]
        for (body, param) in cases {
            let response = try await request(server, .POST, "/v1/responses", body: body)
            #expect(response.status == .badRequest)
            #expect(try response.error().param == param)
            #expect(response.contentType?.hasPrefix("application/json") == true)
            #expect(String(decoding: response.body, as: UTF8.self).contains("event:") == false)
        }
        let query = try await request(
            server, .POST, "/v1/responses?unknown=1", body: #"{"model":"local","input":"Hello"}"#)
        #expect(query.status == .badRequest)
        #expect(try query.error().param == "unknown")
        let valid = try await request(server, .POST, "/v1/responses", body: #"{"model":"local","input":"Hello"}"#)
        #expect(valid.status == .notImplemented)
        #expect(try valid.error().code == "unsupported_model_feature")
    }

    @Test("Conversation continuation rejects missing history, model mismatches, and unanswered calls")
    func continuationPreflight() async throws {
        let store = ResponsesStore()
        #expect(await store.store(id: "resp_other", entry: entry(id: "resp_other", model: "different")))
        let pending = ResponsesStore.Entry(
            response: .object(["output": .array([])]),
            messages: [.init(role: "assistant", content: nil, toolCalls: [call("call_pending")])], inputItems: [],
            model: "local")
        #expect(await store.store(id: "resp_pending", entry: pending))
        #expect(await store.store(id: "resp_existing", entry: entry(id: "resp_existing", items: inputItems(1))))
        let server = try makeServer(store: store)
        let missing = try await request(
            server, .POST, "/v1/responses",
            body: #"{"model":"local","input":"Continue","previous_response_id":"resp_absent","stream":true}"#)
        #expect(missing.status == .notFound)
        #expect(try missing.error().code == "response_not_found")
        #expect(try missing.error().param == "previous_response_id")
        let other = try await request(
            server, .POST, "/v1/responses",
            body: #"{"model":"local","input":"Continue","previous_response_id":"resp_other","stream":true}"#)
        #expect(other.status == .badRequest)
        #expect(try other.error().param == "previous_response_id")
        let unanswered = try await request(
            server, .POST, "/v1/responses",
            body: #"{"model":"local","input":"Continue","previous_response_id":"resp_pending","stream":true}"#)
        #expect(unanswered.status == .badRequest)
        #expect(try unanswered.error().param == "input")
        let answered = try await request(
            server, .POST, "/v1/responses",
            body:
                #"{"model":"local","input":[{"type":"function_call_output","call_id":"call_pending","output":"result"}],"previous_response_id":"resp_pending"}"#
        )
        #expect(answered.status == .notImplemented)
        #expect(try answered.error().code == "unsupported_model_feature")
        let duplicate = try await request(
            server, .POST, "/v1/responses",
            body:
                #"{"model":"local","input":[{"id":"msg_0","role":"user","content":"Repeated ID"}],"previous_response_id":"resp_existing","stream":true}"#
        )
        #expect(duplicate.status == .badRequest)
        #expect(try duplicate.error().param == "input")
    }

    @Test("Input pagination follows order and cursor with a default limit of twenty")
    func inputPagination() throws {
        let items = inputItems(25)
        let defaultPage = try object(ModelHTTPServer.responsesInputPage(items, query: []))
        #expect(defaultPage["object"] == .string("list"))
        #expect(defaultPage["data"] == .array(Array(items.reversed().prefix(20))))
        #expect(defaultPage["first_id"] == .string("msg_24"))
        #expect(defaultPage["last_id"] == .string("msg_5"))
        #expect(defaultPage["has_more"] == .bool(true))
        let ascending = try object(
            ModelHTTPServer.responsesInputPage(
                items,
                query: [
                    .init(name: "order", value: "asc"), .init(name: "limit", value: "2"),
                    .init(name: "after", value: "msg_21"),
                ]))
        #expect(ascending["data"] == .array(Array(items[22...23])))
        #expect(ascending["has_more"] == .bool(true))
        let descending = try object(
            ModelHTTPServer.responsesInputPage(
                items,
                query: [
                    .init(name: "order", value: "desc"), .init(name: "after", value: "msg_2"),
                ]))
        #expect(descending["data"] == .array([items[1], items[0]]))
        #expect(descending["has_more"] == .bool(false))
        let exhausted = try object(
            ModelHTTPServer.responsesInputPage(items, query: [.init(name: "after", value: "msg_0")]))
        #expect(exhausted["data"] == .array([]))
        #expect(exhausted["first_id"] == .null)
        #expect(exhausted["last_id"] == .null)
        #expect(exhausted["has_more"] == .bool(false))
        #expect(
            try ModelHTTPServer.responsesInputPage([], query: [])
                == .object([
                    "object": .string("list"), "data": .array([]), "first_id": .null, "last_id": .null,
                    "has_more": .bool(false),
                ]))
    }

    @Test("Input pagination rejects invalid, repeated, empty, or unsupported options")
    func paginationErrors() {
        let cases: [([URLQueryItem], String)] = [
            ([.init(name: "limit", value: "0")], "limit"), ([.init(name: "limit", value: "101")], "limit"),
            ([.init(name: "limit", value: "1.5")], "limit"), ([.init(name: "order", value: "up")], "order"),
            ([.init(name: "after", value: "missing")], "after"), ([.init(name: "limit", value: nil)], "limit"),
            ([.init(name: "after", value: "")], "after"), ([.init(name: "before", value: "msg_1")], "before"),
            ([.init(name: "limit", value: "1"), .init(name: "limit", value: "2")], "limit"),
        ]
        for (query, param) in cases {
            do {
                _ = try ModelHTTPServer.responsesInputPage(inputItems(2), query: query)
                Issue.record("Expected an invalid pagination parameter")
            } catch let error as ModelHTTPError {
                #expect(error.status == .badRequest)
                #expect(error.param == param)
            } catch {
                Issue.record("Unexpected pagination error: \(error)")
            }
        }
    }

    @Test("Parallel tool results may arrive out of order and assistant text may accompany calls")
    func validToolHistories() throws {
        let calls = OpenAIMessage(role: "assistant", content: nil, toolCalls: [call("call_a"), call("call_b")])
        try ModelHTTPServer.validateResponsesToolHistory([
            .init(role: "user", content: "Use tools"), calls,
            .init(role: "tool", content: "B", toolCallID: "call_b"),
            .init(role: "tool", content: "A", toolCallID: "call_a"), .init(role: "user", content: "Continue"),
        ])
        try ModelHTTPServer.validateResponsesToolHistory([
            .init(role: "assistant", content: "Checking."),
            .init(role: "assistant", content: nil, toolCalls: [call("call_a")]),
            .init(role: "assistant", content: "One moment."),
            .init(role: "tool", content: "A", toolCallID: "call_a"),
        ])
    }

    @Test("Tool histories reject missing, duplicate, unknown results and reused call IDs")
    func invalidToolHistories() {
        let assistant = OpenAIMessage(role: "assistant", content: nil, toolCalls: [call("call_a")])
        let answer = OpenAIMessage(role: "tool", content: "A", toolCallID: "call_a")
        let histories: [[OpenAIMessage]] = [
            [assistant], [assistant, .init(role: "user", content: "Continue"), answer],
            [assistant, answer, answer], [.init(role: "tool", content: "orphan", toolCallID: "call_unknown")],
            [assistant, .init(role: "tool", content: "wrong", toolCallID: "call_b")],
            [assistant, .init(role: "tool", content: "missing id")],
            [.init(role: "assistant", content: nil, toolCalls: [call("call_a"), call("call_a")]), answer],
            [assistant, answer, assistant, answer],
            [.init(role: "assistant", content: nil, toolCalls: [call("")])],
        ]
        for history in histories {
            do {
                try ModelHTTPServer.validateResponsesToolHistory(history)
                Issue.record("Expected invalid tool history")
            } catch let error as ModelHTTPError {
                #expect(error.status == .badRequest)
                #expect(error.param == "input")
            } catch {
                Issue.record("Unexpected tool history error: \(error)")
            }
        }
    }

    private func makeServer(store: ResponsesStore = ResponsesStore(), managed: Bool = false) throws -> ModelHTTPServer {
        if managed {
            return ModelHTTPServer(
                manager: ModelLifecycleManager(validate: { _ in
                    throw ResponsesHTTPFailure.unexpectedLoad
                }), responsesStore: store)
        }
        return ModelHTTPServer(
            servedModelName: "local",
            tokenLimit: try GenerationTokenLimit(configuredMaximum: 128), responsesStore: store)
    }

    private func entry(id: String, items: [OpenAIJSONValue] = [], model: String = "local") -> ResponsesStore.Entry {
        .init(
            response: .object([
                "id": .string(id), "object": .string("response"), "status": .string("completed"), "output": .array([]),
            ]),
            messages: [.init(role: "user", content: "Hello"), .init(role: "assistant", content: "Hi")],
            inputItems: items, model: model)
    }

    private func inputItems(_ count: Int) -> [OpenAIJSONValue] {
        (0..<count).map {
            .object([
                "id": .string("msg_\($0)"), "type": .string("message"), "role": .string("user"),
                "content": .string("Input \($0)"),
            ])
        }
    }

    private func call(_ id: String) -> OpenAIToolCall {
        .init(id: id, function: .init(name: "lookup", arguments: "{}"))
    }

    private func object(_ value: OpenAIJSONValue) throws -> [String: OpenAIJSONValue] {
        guard case .object(let fields) = value else {
            throw ResponsesHTTPFailure.unexpectedBody
        }
        return fields
    }

    private func request(
        _ server: ModelHTTPServer, _ method: HTTPMethod, _ uri: String,
        body: String = ""
    ) async throws -> Response {
        let channel = await NIOAsyncTestingChannel(handler: ModelHTTPRequestHandler(server: server))
        try await channel.connect(to: SocketAddress(ipAddress: "127.0.0.1", port: 9999)).get()
        do {
            let headers = HTTPHeaders([("content-type", "application/json")])
            _ = try await channel.writeInbound(
                HTTPServerRequestPart.head(
                    HTTPRequestHead(version: .http1_1, method: method, uri: uri, headers: headers)))
            if !body.isEmpty {
                var buffer = channel.allocator.buffer(capacity: body.utf8.count)
                buffer.writeString(body)
                _ = try await channel.writeInbound(HTTPServerRequestPart.body(buffer))
            }
            _ = try await channel.writeInbound(HTTPServerRequestPart.end(nil))
            var head: HTTPResponseHead?
            var data = Data()
            let deadline = ContinuousClock.now.advanced(by: .seconds(3))
            while ContinuousClock.now < deadline {
                guard let part = try await channel.readOutbound(as: HTTPServerResponsePart.self) else {
                    try await Task.sleep(for: .milliseconds(5))
                    continue
                }
                switch part {
                case .head(let value): head = value
                case .body(.byteBuffer(var buffer)):
                    if let bytes = buffer.readBytes(length: buffer.readableBytes) {
                        data.append(contentsOf: bytes)
                    }
                case .body(.fileRegion): throw ResponsesHTTPFailure.unexpectedBody
                case .end:
                    let head = try #require(head)
                    _ = try await channel.finish(acceptAlreadyClosed: true)
                    return Response(
                        status: head.status, body: data, contentType: head.headers.first(name: "content-type"))
                }
            }
            throw ResponsesHTTPFailure.timedOut
        } catch {
            _ = try? await channel.finish(acceptAlreadyClosed: true)
            throw error
        }
    }

    private struct Response {
        let status: HTTPResponseStatus
        let body: Data
        let contentType: String?
        func json() throws -> OpenAIJSONValue { try JSONDecoder().decode(OpenAIJSONValue.self, from: body) }
        func error() throws -> ErrorEnvelope.Detail { try JSONDecoder().decode(ErrorEnvelope.self, from: body).error }
    }

    private struct ErrorEnvelope: Decodable {
        struct Detail: Decodable {
            let code: String?
            let param: String?
        }
        let error: Detail
    }
}

private enum ResponsesHTTPFailure: Error { case timedOut, unexpectedBody, unexpectedLoad }
