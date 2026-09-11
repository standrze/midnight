import Foundation
import ModelRunnerProtocol
import NIOCore
import NIOEmbedded
import NIOHTTP1
import Testing

@testable import Midnight

@Suite("Persistent model HTTP lifecycle", .timeLimit(.minutes(1)))
struct ModelHTTPLifecycleTests {
    @Test("An empty listener reports status and rejects inference without a model")
    func emptyListener() async throws {
        let manager = makeManager()
        let server = ModelHTTPServer(manager: manager)
        let status = try await request(server, .GET, "/v1/runtime")
        #expect(status.status == .ok)
        let runtime = try status.decode(Runtime.self)
        #expect(runtime.phase == "empty")
        #expect(runtime.processID > 0)
        #expect(runtime.loadedModel == nil)
        #expect(runtime.modelGeneration == 0)

        let unloaded = try await request(server, .POST, "/v1/runtime/unload", body: "{}")
        #expect(unloaded.status == .accepted)
        #expect(try unloaded.decode(Runtime.self).operationID != nil)
        #expect(try unloaded.decode(Runtime.self).modelGeneration == 0)
        let unknown = try await request(server, .GET, "/unknown")
        #expect(unknown.status == .notFound)

        let inspector = try await request(server, .GET, "/v1/inspector/runtime")
        #expect(try inspector.decode(Runtime.self).processID == runtime.processID)
        let models = try await request(server, .GET, "/v1/models")
        #expect(models.status == .ok)
        #expect(try models.decode(Models.self).data.isEmpty)
        let missing = try await request(server, .GET, "/v1/models/absent")
        #expect(missing.status == .notFound)
        #expect(try missing.errorCode() == "model_not_found")

        for route in ["/v1/chat/completions", "/v1/audio/speech", "/v1/inspector/trace"] {
            let unavailable = try await request(server, .POST, route, body: "{}")
            #expect(unavailable.status == .serviceUnavailable)
            #expect(try unavailable.errorCode() == "model_unavailable")
        }
    }

    @Test("Load is accepted asynchronously and model metadata changes on replacement")
    func asynchronousLoadAndReplacement() async throws {
        let releaseLoad = LifecycleHTTPGate()
        let manager = makeManager(delayedLoad: releaseLoad)
        let server = ModelHTTPServer(manager: manager)

        let accepted = try await request(server, .POST, "/v1/runtime/load",
            body: #"{"model":"delayed","name":"alpha one"}"#)
        #expect(accepted.status == .accepted)
        let operation = try accepted.decode(Runtime.self)
        #expect(operation.phase == "loading")
        #expect(operation.operationID != nil)
        #expect(operation.targetModel == "alpha one")

        let loading = try await request(server, .GET, "/v1/runtime")
        #expect(try loading.decode(Runtime.self).operationID == operation.operationID)
        #expect(try loading.decode(Runtime.self).phase == "loading")
        let emptyModels = try await request(server, .GET, "/v1/models")
        #expect(try emptyModels.decode(Models.self).data.isEmpty)
        let unavailable = try await request(server, .POST, "/v1/chat/completions", body: "{}")
        #expect(unavailable.status == .serviceUnavailable)
        for (route, body) in [
            ("/v1/runtime/load", #"{"model":"second"}"#),
            ("/v1/runtime/unload", "{}"),
        ] {
            let busy = try await request(server, .POST, route, body: body)
            #expect(busy.status == .conflict)
            #expect(try busy.errorCode() == "model_transition_in_progress")
        }

        await releaseLoad.open()
        try await waitForPhase(.ready, manager: manager)
        let firstModels = try await request(server, .GET, "/v1/models")
        let first = try #require(firstModels.decode(Models.self).data.first)
        #expect(first.id == "alpha one")
        #expect(first.object == "model")
        #expect(first.ownedBy == "midnight")
        let single = try await request(server, .GET, "/v1/models/alpha%20one")
        #expect(single.status == .ok)
        #expect(try single.decode(Model.self).created == first.created)
        let firstState = await manager.state()

        let replacement = try await request(server, .POST, "/v1/runtime/load",
            body: #"{"model":"second","name":"beta"}"#)
        #expect(replacement.status == .accepted)
        try await waitForPhase(.ready, manager: manager)
        let secondModels = try await request(server, .GET, "/v1/models")
        #expect(try secondModels.decode(Models.self).data.map(\.id) == ["beta"])
        let secondState = await manager.state()
        #expect(secondState.modelGeneration > firstState.modelGeneration)
        #expect(secondState.operationID != firstState.operationID)
        let oldModel = try await request(server, .GET, "/v1/models/alpha%20one")
        #expect(oldModel.status == .notFound)

        let unloaded = try await request(server, .POST, "/v1/runtime/unload", body: "{}")
        #expect(unloaded.status == .accepted)
        try await waitForPhase(.empty, manager: manager)
        let finalModels = try await request(server, .GET, "/v1/models")
        #expect(try finalModels.decode(Models.self).data.isEmpty)
        let finalStatus = try await request(server, .GET, "/v1/runtime")
        #expect(try finalStatus.decode(Runtime.self).processID == operation.processID)
    }

    @Test("Draining closes inference admission and waits for an existing request lease")
    func drainingExistingRequest() async throws {
        let manager = makeManager()
        let server = ModelHTTPServer(manager: manager)
        _ = try await request(server, .POST, "/v1/runtime/load", body: #"{"model":"first"}"#)
        try await waitForPhase(.ready, manager: manager)
        let entered = LifecycleHTTPGate()
        let release = LifecycleHTTPGate()
        let activeRequest = Task {
            try await manager.withModel { model in
                #expect(model.servedModelName == "first")
                await entered.open()
                await release.wait()
            }
        }
        await entered.wait()

        let unload = try await request(server, .POST, "/v1/runtime/unload", body: "{}")
        #expect(unload.status == .accepted)
        #expect(try unload.decode(Runtime.self).phase == "draining")
        #expect(await manager.state().phase == .draining)
        let unavailable = try await request(server, .POST, "/v1/chat/completions", body: "{}")
        #expect(unavailable.status == .serviceUnavailable)
        let busy = try await request(server, .POST, "/v1/runtime/load", body: #"{"model":"second"}"#)
        #expect(busy.status == .conflict)

        await release.open()
        try await activeRequest.value
        try await waitForPhase(.empty, manager: manager)
    }

    @Test("Invalid requests preserve the current model and failed loads allow recovery")
    func invalidSelectionAndRecovery() async throws {
        let manager = makeManager()
        let server = ModelHTTPServer(manager: manager)
        _ = try await request(server, .POST, "/v1/runtime/load", body: #"{"model":"first"}"#)
        try await waitForPhase(.ready, manager: manager)
        let generation = await manager.state().modelGeneration

        for invalidJSON in ["{", "{}", #"{"model":42}"#, #"{"model":"invalid"}"#] {
            let response = try await request(server, .POST, "/v1/runtime/load", body: invalidJSON)
            #expect(response.status == .badRequest)
            #expect(try response.errorCode() == "invalid_model_selection")
            #expect(await manager.state().loadedModel?.id == "first")
            #expect(await manager.state().modelGeneration == generation)
        }
        for invalidJSON in ["{", "[]", #"{"extra":true}"#] {
            let response = try await request(server, .POST, "/v1/runtime/unload", body: invalidJSON)
            #expect(response.status == .badRequest)
            #expect(await manager.state().loadedModel?.id == "first")
        }
        let oversized = try await request(server, .POST, "/v1/runtime/load",
            body: String(repeating: " ", count: 32 * 1024 + 1))
        #expect(oversized.status == .payloadTooLarge)
        #expect(try oversized.errorCode() == "request_too_large")

        let failure = try await request(server, .POST, "/v1/runtime/load", body: #"{"model":"broken"}"#)
        #expect(failure.status == .accepted)
        try await waitForPhase(.empty, manager: manager)
        let failedStatus = try await request(server, .GET, "/v1/runtime")
        #expect(try failedStatus.decode(Runtime.self).lastError?.contains("Deliberate load failure") == true)
        let failedModels = try await request(server, .GET, "/v1/models")
        #expect(try failedModels.decode(Models.self).data.isEmpty)

        let retry = try await request(server, .POST, "/v1/runtime/load", body: #"{"model":"recovered"}"#)
        #expect(retry.status == .accepted)
        try await waitForPhase(.ready, manager: manager)
        let recovered = try await request(server, .GET, "/v1/runtime")
        let state = try recovered.decode(Runtime.self)
        #expect(state.loadedModel?.id == "recovered")
        #expect(state.lastError == nil)
    }

    @Test("Model control accepts local JSON clients and rejects other request contexts")
    func controlRequestRestrictions() throws {
        let local = try SocketAddress(ipAddress: "127.0.0.1", port: 9999)
        func head(_ headers: HTTPHeaders) -> HTTPRequestHead {
            HTTPRequestHead(version: .http1_1, method: .POST, uri: "/v1/runtime/load", headers: headers)
        }
        let json = HTTPHeaders([("content-type", "application/json")])
        for address in ["127.0.0.1", "::1", "::ffff:127.0.0.1"] {
            try ModelHTTPServer.validateControlRequest(head: head(json),
                remoteAddress: SocketAddress(ipAddress: address, port: 9999))
        }
        try ModelHTTPServer.validateControlRequest(
            head: head(HTTPHeaders([("Content-Type", "Application/JSON; charset=utf-8")])),
            remoteAddress: local)
        for address in [nil, try SocketAddress(ipAddress: "192.0.2.1", port: 9999)] {
            #expect(throws: (any Error).self) {
                try ModelHTTPServer.validateControlRequest(head: head(json), remoteAddress: address)
            }
        }
        for headers in [
            HTTPHeaders(),
            HTTPHeaders([("content-type", "text/plain")]),
            HTTPHeaders([("content-type", "application/json"), ("content-type", "application/json")]),
            HTTPHeaders([("content-type", "application/json"), ("origin", "http://localhost")]),
            HTTPHeaders([("content-type", "application/json"), ("sec-fetch-site", "cross-site")]),
        ] {
            #expect(throws: (any Error).self) {
                try ModelHTTPServer.validateControlRequest(head: head(headers), remoteAddress: local)
            }
        }
    }

    private func makeManager(delayedLoad: LifecycleHTTPGate? = nil) -> ModelLifecycleManager {
        ModelLifecycleManager(validate: { request in
            guard request.model != "invalid" else { throw LifecycleHTTPFailure.invalidSelection }
            let name = request.name ?? request.model
            return ModelLoadOperation(name: name) {
                if request.model == "delayed" { await delayedLoad?.wait() }
                if request.model == "broken" { throw LifecycleHTTPFailure.loadFailed }
                return LoadedModel(servedModelName: name,
                    tokenLimit: try GenerationTokenLimit(configuredMaximum: request.maxTokens ?? 128))
            }
        })
    }

    private func request(_ server: ModelHTTPServer, _ method: HTTPMethod, _ uri: String,
                         body: String = "") async throws -> Response {
        let channel = await NIOAsyncTestingChannel(handler: ModelHTTPRequestHandler(server: server))
        try await channel.connect(to: SocketAddress(ipAddress: "127.0.0.1", port: 9999)).get()
        do {
            let headers = HTTPHeaders([("content-type", "application/json")])
            _ = try await channel.writeInbound(HTTPServerRequestPart.head(
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
                    if let bytes = buffer.readBytes(length: buffer.readableBytes) { data.append(contentsOf: bytes) }
                case .body(.fileRegion): throw LifecycleHTTPFailure.unexpectedBody
                case .end:
                    let head = try #require(head)
                    _ = try await channel.finish(acceptAlreadyClosed: true)
                    return Response(status: head.status, body: data)
                }
            }
            throw LifecycleHTTPFailure.timedOut
        } catch {
            _ = try? await channel.finish(acceptAlreadyClosed: true)
            throw error
        }
    }

    private func waitForPhase(_ phase: ModelLifecycleState.Phase,
                              manager: ModelLifecycleManager) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while ContinuousClock.now < deadline {
            if await manager.state().phase == phase { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        throw LifecycleHTTPFailure.timedOut
    }

    private struct Response {
        let status: HTTPResponseStatus
        let body: Data
        func decode<Value: Decodable>(_ type: Value.Type) throws -> Value {
            try JSONDecoder().decode(type, from: body)
        }
        func errorCode() throws -> String? { try decode(ErrorEnvelope.self).error.code }
    }

    private struct Runtime: Decodable {
        let processID: Int32
        let phase: String
        let operationID: String?
        let modelGeneration: UInt64
        let loadedModel: Model?
        let targetModel: String?
        let lastError: String?
    }

    private struct Model: Decodable {
        let id: String
        let created: Int
        let object: String
        let ownedBy: String
        enum CodingKeys: String, CodingKey {
            case id, created, object
            case ownedBy = "owned_by"
        }
    }

    private struct Models: Decodable { let data: [Model] }
    private struct ErrorEnvelope: Decodable {
        struct Detail: Decodable { let code: String? }
        let error: Detail
    }
}

private enum LifecycleHTTPFailure: LocalizedError {
    case invalidSelection, loadFailed, timedOut, unexpectedBody
    var errorDescription: String? {
        switch self {
        case .invalidSelection: "Deliberate invalid selection"
        case .loadFailed: "Deliberate load failure"
        case .timedOut: "HTTP lifecycle test timed out"
        case .unexpectedBody: "Unexpected file body in JSON response"
        }
    }
}

private actor LifecycleHTTPGate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        guard !isOpen else { return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func open() {
        isOpen = true
        let waiters = waiters
        self.waiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }
}
