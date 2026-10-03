#if os(macOS)
    import Foundation
    import Testing
    import NIOHTTP1
    import VisionProtocol
    import VisionHTTP

    private actor StubBackend: VisionGenerating {
        var calls = 0
        var drains = 0
        let error: VisionError?
        init(error: VisionError? = nil) { self.error = error }
        func answer(_ request: VisionRequest) async throws -> VisionAnswer {
            calls += 1
            if let error {
                throw error
            }
            return VisionAnswer(text: "ORBIT 731", promptTokens: 47, completionTokens: 5, finishReason: "stop")
        }
        func waitUntilIdle() { drains += 1 }
        func memorySnapshot() -> [String: Int] { ["active_bytes": 123, "peak_bytes": 456] }
    }

    @Suite("Optional vision HTTP contract")
    struct VisionHTTPTests {
        @Test func managedControlRoutesRequireOwningToken() async throws {
            let backend = StubBackend()
            let server = VisionHTTPServer(backend: backend, model: "test", controlToken: "owner-token")
            let refused = await server.response(method: .POST, uri: "/v1/vision/drain", data: Data())
            #expect(refused.0 == .unauthorized)
            #expect(await backend.drains == 0)
            let drained = await server.response(
                method: .POST, uri: "/v1/vision/drain", data: Data(), authorization: "Bearer owner-token")
            #expect(drained.0 == .ok)
            #expect(await backend.drains == 1)
            #expect(await backend.calls == 0)
            let status = await server.response(
                method: .GET, uri: "/v1/vision/status", data: Data(), authorization: "Bearer owner-token")
            let json = try #require(JSONSerialization.jsonObject(with: status.1) as? [String: Any])
            #expect(json["memory_scope"] as? String == "vision_worker")
            #expect((json["memory"] as? [String: Int])?["active_bytes"] == 123)
            #expect((json["memory"] as? [String: Int])?["peak_bytes"] == 456)
        }
        @Test func modelAndStatusDoNotGenerate() async throws {
            let backend = StubBackend()
            let server = VisionHTTPServer(backend: backend, model: "test")
            let catalog = await server.response(method: .GET, uri: "/v1/models", data: Data())
            #expect(catalog.0 == .ok)
            let json = try #require(JSONSerialization.jsonObject(with: catalog.1) as? [String: Any])
            #expect((json["data"] as? [[String: Any]])?.first?["id"] as? String == "test")
            #expect(await backend.calls == 0)
            let status = await server.response(method: .GET, uri: "/v1/vision/status", data: Data())
            #expect(status.0 == .ok)
        }
        @Test func chatResponseHasOpenAIEnvelope() async throws {
            let backend = StubBackend()
            let server = VisionHTTPServer(backend: backend, model: "vision-test")
            let result = await server.response(
                method: .POST, uri: "/v1/chat/completions", data: try VisionRequestTests().body())
            #expect(result.0 == .ok)
            let json = try #require(JSONSerialization.jsonObject(with: result.1) as? [String: Any])
            #expect(json["object"] as? String == "chat.completion")
            #expect((json["usage"] as? [String: Int])?["total_tokens"] == 52)
            let choice = try #require((json["choices"] as? [[String: Any]])?.first)
            #expect((choice["message"] as? [String: String])?["content"] == "ORBIT 731")
            #expect(await backend.calls == 1)
        }
        @Test func rejectsBeforeBackend() async throws {
            let backend = StubBackend()
            let server = VisionHTTPServer(backend: backend, model: "vision-test")
            for body in [Data("{".utf8), try VisionRequestTests().body { $0["model"] = "text" }] {
                let result = await server.response(method: .POST, uri: "/v1/chat/completions", data: body)
                #expect(result.0 == .badRequest)
            }
            #expect(await backend.calls == 0)
        }
        @Test func busyIsExplicitAndUnsupportedRoutesDoNotGenerate() async throws {
            let backend = StubBackend(error: VisionError("vision_busy", "Busy"))
            let server = VisionHTTPServer(backend: backend, model: "vision-test")
            let result = await server.response(
                method: .POST, uri: "/v1/chat/completions", data: try VisionRequestTests().body())
            #expect(result.0 == .tooManyRequests)
            let unsupported = await server.response(method: .POST, uri: "/v1/responses", data: Data())
            #expect(unsupported.0 == .notFound)
            #expect(await backend.calls == 1)
        }
    }

#endif
