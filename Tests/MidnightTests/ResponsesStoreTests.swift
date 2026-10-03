import Foundation
import ModelRunnerProtocol
import Testing

@testable import Midnight

@Suite("Responses API history")
struct ResponsesStoreTests {
    private let epoch = Date(timeIntervalSince1970: 1_000)

    @Test("Missing and deleted responses are absent")
    func deletion() async {
        let store = ResponsesStore()
        #expect(await store.get(id: "resp_missing", now: epoch) == nil)
        #expect(await store.remove(id: "resp_missing", now: epoch) == false)
        let original = entry("hello")
        #expect(await store.store(id: "resp_a", entry: original, now: epoch))
        #expect(await store.get(id: "resp_a", now: epoch) == original)
        #expect(await store.remove(id: "resp_a", now: epoch))
        #expect(await store.remove(id: "resp_a", now: epoch) == false)
        #expect(await store.get(id: "resp_a", now: epoch) == nil)
    }

    @Test("Capacity evicts by storage order without promoting reads")
    func oldestEviction() async {
        let store = ResponsesStore(maximumEntries: 2)
        #expect(await store.store(id: "resp_a", entry: entry("a"), now: epoch))
        #expect(await store.store(id: "resp_b", entry: entry("b"), now: epoch))
        #expect(await store.get(id: "resp_a", now: epoch) != nil)
        #expect(await store.store(id: "resp_c", entry: entry("c"), now: epoch))
        #expect(await store.get(id: "resp_a", now: epoch) == nil)
        #expect(await store.get(id: "resp_b", now: epoch) != nil)
        #expect(await store.get(id: "resp_c", now: epoch) != nil)
    }

    @Test("TTL expires at the absolute deadline and reads do not extend it")
    func absoluteExpiration() async {
        let store = ResponsesStore(timeToLive: 10)
        #expect(await store.store(id: "resp_a", entry: entry("a"), now: epoch))
        #expect(await store.get(id: "resp_a", now: epoch.addingTimeInterval(9)) != nil)
        #expect(await store.get(id: "resp_a", now: epoch.addingTimeInterval(10)) == nil)
        #expect(await store.remove(id: "resp_a", now: epoch.addingTimeInterval(10)) == false)
    }

    @Test("Expired entries release capacity before a new result is stored")
    func expirationReclaimsBytes() async throws {
        let value = entry("same")
        let budget = try encodedSize(id: "resp_a", entry: value)
        let store = ResponsesStore(maximumBytes: budget, timeToLive: 10)
        #expect(await store.store(id: "resp_a", entry: value, now: epoch))
        #expect(await store.store(id: "resp_b", entry: value, now: epoch.addingTimeInterval(10)))
        #expect(await store.get(id: "resp_b", now: epoch.addingTimeInterval(10)) == value)
    }

    @Test("Byte accounting includes transcript, canonical inputs, model, and response")
    func completeByteAccounting() async throws {
        let value = entry("日本語 \"escaped\" \\ text")
        let budget = try encodedSize(id: "resp_a", entry: value)
        let exact = ResponsesStore(maximumBytes: budget)
        #expect(await exact.store(id: "resp_a", entry: value, now: epoch))
        let tooSmall = ResponsesStore(maximumBytes: budget - 1)
        #expect(await tooSmall.store(id: "resp_a", entry: value, now: epoch) == false)

        let variants = [
            ResponsesStore.Entry(
                response: value.response, messages: value.messages + [.init(role: "user", content: "extra")],
                inputItems: value.inputItems, model: value.model),
            ResponsesStore.Entry(
                response: value.response, messages: value.messages, inputItems: value.inputItems + [.string("extra")],
                model: value.model),
            ResponsesStore.Entry(
                response: value.response, messages: value.messages, inputItems: value.inputItems,
                model: value.model + "extra"),
            ResponsesStore.Entry(
                response: .string(String(repeating: "x", count: budget)), messages: value.messages,
                inputItems: value.inputItems, model: value.model),
        ]
        for larger in variants {
            #expect(await exact.store(id: "resp_a", entry: larger, now: epoch) == false)
            #expect(await exact.get(id: "resp_a", now: epoch) == value)
        }
    }

    @Test("Byte pressure evicts the oldest entry and deletion releases bytes")
    func byteEvictionAndDeletion() async throws {
        let value = entry("value")
        let size = try encodedSize(id: "resp_a", entry: value)
        let store = ResponsesStore(maximumBytes: size * 2)
        for id in ["resp_a", "resp_b", "resp_c"] {
            #expect(await store.store(id: id, entry: value, now: epoch))
        }
        #expect(await store.get(id: "resp_a", now: epoch) == nil)
        #expect(await store.get(id: "resp_b", now: epoch) == value)
        #expect(await store.remove(id: "resp_b", now: epoch))
        #expect(await store.store(id: "resp_d", entry: value, now: epoch))
        #expect(await store.get(id: "resp_c", now: epoch) == value)
        #expect(await store.get(id: "resp_d", now: epoch) == value)
    }

    @Test("Oversized and unencodable entries do not evict live history")
    func failedStorePreservesExistingEntries() async throws {
        let value = entry("value")
        let store = ResponsesStore(maximumEntries: 1, maximumBytes: try encodedSize(id: "resp_a", entry: value))
        #expect(await store.store(id: "resp_a", entry: value, now: epoch))
        #expect(
            await store.store(id: "resp_b", entry: entry(String(repeating: "x", count: 1_000)), now: epoch) == false)
        let invalid = ResponsesStore.Entry(response: .number(.infinity), messages: [], inputItems: [], model: "test")
        #expect(await store.store(id: "resp_b", entry: invalid, now: epoch) == false)
        #expect(await store.get(id: "resp_a", now: epoch) == value)
    }

    @Test("Replacing a result accounts for its new size and storage order")
    func replacement() async {
        let store = ResponsesStore(maximumEntries: 2)
        #expect(await store.store(id: "resp_a", entry: entry("a"), now: epoch))
        #expect(await store.store(id: "resp_b", entry: entry("b"), now: epoch))
        #expect(await store.store(id: "resp_a", entry: entry("replacement"), now: epoch))
        #expect(await store.store(id: "resp_c", entry: entry("c"), now: epoch))
        #expect(await store.get(id: "resp_b", now: epoch) == nil)
        #expect(await store.get(id: "resp_a", now: epoch) == entry("replacement"))
    }

    @Test("Continuation branches retain independent value histories")
    func independentBranches() async throws {
        let store = ResponsesStore()
        let original = entry("parent")
        #expect(await store.store(id: "resp_parent", entry: original, now: epoch))
        let parent = try #require(await store.get(id: "resp_parent", now: epoch))
        let first = ResponsesStore.Entry(
            response: .string("first"), messages: parent.messages + [.init(role: "user", content: "branch one")],
            inputItems: parent.inputItems + [.string("one")], model: parent.model)
        let second = ResponsesStore.Entry(
            response: .string("second"), messages: parent.messages + [.init(role: "user", content: "branch two")],
            inputItems: parent.inputItems + [.string("two")], model: parent.model)
        #expect(await store.store(id: "resp_first", entry: first, now: epoch))
        #expect(await store.store(id: "resp_second", entry: second, now: epoch))
        #expect(await store.remove(id: "resp_parent", now: epoch))
        #expect(await store.get(id: "resp_first", now: epoch) == first)
        #expect(await store.get(id: "resp_second", now: epoch) == second)
        #expect(first.messages.last?.content == "branch one")
        #expect(second.messages.last?.content == "branch two")
    }

    @Test("Nonpositive and nonfinite retention settings disable storage")
    func disabledStorage() async {
        let disabled = [
            ResponsesStore(maximumEntries: 0), ResponsesStore(maximumEntries: -1), ResponsesStore(maximumBytes: 0),
            ResponsesStore(maximumBytes: -1), ResponsesStore(timeToLive: 0), ResponsesStore(timeToLive: -1),
            ResponsesStore(timeToLive: .infinity), ResponsesStore(timeToLive: .nan),
        ]
        for store in disabled {
            #expect(await store.store(id: "resp_a", entry: entry("value"), now: epoch) == false)
        }
    }

    private func entry(_ text: String) -> ResponsesStore.Entry {
        .init(
            response: .object(["output": .string(text)]),
            messages: [.init(role: "user", content: text)],
            inputItems: [.object(["id": .string("msg_a"), "text": .string(text)])],
            model: "test-model"
        )
    }

    private func encodedSize(id: String, entry: ResponsesStore.Entry) throws -> Int {
        struct Payload: Encodable {
            let id: String
            let entry: ResponsesStore.Entry
        }
        return try JSONEncoder().encode(Payload(id: id, entry: entry)).count
    }
}

@Suite("Responses API routing")
struct ResponsesAPIRouteTests {
    @Test("Exact endpoints accept query strings and decoded IDs")
    func routes() {
        #expect(ResponsesAPIRoute.parse(uri: "/v1/responses") == .create)
        #expect(ResponsesAPIRoute.parse(uri: "/v1/responses?trace=1") == .create)
        #expect(ResponsesAPIRoute.parse(uri: "/v1/responses/resp_A-123_b") == .response(id: "resp_A-123_b"))
        #expect(
            ResponsesAPIRoute.parse(uri: "/v1/responses/resp_%61/input_items?limit=20&order=desc")
                == .inputItems(id: "resp_a"))
        #expect(
            ResponsesAPIRoute.queryItems(uri: "/v1/responses/resp_a/input_items?after=msg_a&limit=20") == [
                URLQueryItem(name: "after", value: "msg_a"), URLQueryItem(name: "limit", value: "20"),
            ])
    }

    @Test("Malformed paths, invalid IDs, and extra segments do not match")
    func rejectsInvalidRoutes() {
        let paths = [
            "v1/responses", "//v1/responses", "http://example.com/v1/responses",
            "/v1/responses/", "/v1//responses", "/v1/responses#fragment", "/v1/responses?x=%",
            "/v1/responses/resp_", "/v1/responses/not_response", "/v1/responses/resp_😀",
            "/v1/responses/resp_a%2Fb", "/v1/responses/resp_a%5Cb", "/v1/responses/resp_a%00",
            "/v1/responses/resp_a%20b", "/v1/responses/resp_a%FF", "/v1/responses/resp_a%2",
            "/v1/responses/resp_a%GG", "/v1/responses/resp_a/input_items/extra",
            "/v1/responses/resp_a/input_items/", "/v1/responses/resp_a/cancel",
            "/v1/responses/resp_a?x=hello world", "/v1/responses/resp_a\n",
            "/v1/responses/resp_" + String(repeating: "a", count: 252),
        ]
        for path in paths {
            #expect(ResponsesAPIRoute.parse(uri: path) == nil, "Unexpected route for \(path)")
        }
    }

    @Test("Each route exposes the supported HTTP methods")
    func methods() {
        #expect(ResponsesAPIRoute.create.allowedMethods == ["POST"])
        #expect(ResponsesAPIRoute.response(id: "resp_a").allowedMethods == ["GET", "DELETE"])
        #expect(ResponsesAPIRoute.inputItems(id: "resp_a").allowedMethods == ["GET"])
        #expect(ResponsesAPIRoute.create.allows(method: "post"))
        #expect(ResponsesAPIRoute.create.allows(method: "GET") == false)
    }
}
