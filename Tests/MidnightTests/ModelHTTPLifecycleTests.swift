import Foundation
import ModelRunnerProtocol
import NIOCore
import NIOEmbedded
import NIOHTTP1
import Testing

@testable import Midnight
@testable import ModelRunnerCore

@Suite("Persistent model HTTP lifecycle", .timeLimit(.minutes(1)))
struct ModelHTTPLifecycleTests {
    @Test("Runtime generation metrics encode without enabling per-token monitoring")
    func runtimeGenerationMetrics() async throws {
        let manager = makeManager()
        var state = await manager.state()
        #expect(state.generationMetrics == nil)
        state.generationMetrics = LocalModelRunnerMetrics(
            promptTokenCount: 512, cachedPromptTokenCount: 128, generationTokenCount: 32,
            promptTokensPerSecond: 1200, tokensPerSecond: 42.5, stopReason: "stop")
        let object = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(state)) as? [String: Any])
        let metrics = try #require(object["generationMetrics"] as? [String: Any])
        #expect(metrics["tokensPerSecond"] as? Double == 42.5)
        #expect(metrics["cachedPromptTokenCount"] as? Int == 128)
        let empty = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(await manager.state())) as? [String: Any])
        #expect(empty["generationMetrics"] == nil)
    }

    @Test("Without a configured key, discovery accepts clients without authorization")
    func requestsWithoutConfiguredKey() async throws {
        let manager = makeManager()
        let server = ModelHTTPServer(manager: manager)
        let state = try await request(server, .GET, "/v1/runtime")
        #expect(state.status == .ok)
        let models = try await request(server, .GET, "/v1/models")
        #expect(models.status == .ok)
        let placeholder = try await request(
            server, .GET, "/v1/runtime", extraHeaders: [("authorization", "Bearer unused")])
        #expect(placeholder.status == .ok)
        #expect(await manager.state().phase == .empty)
    }

    @Test("A configured API key protects routes before they reach the manager")
    func apiKeyRequiredForEveryRoute() async throws {
        let manager = makeManager()
        let key = "0123456789abcdef0123456789abcdef"
        let server = ModelHTTPServer(
            manager: manager, authentication: try APIKeyAuthentication(key: key))

        for path in ["/v1/runtime", "/v1/models", "/v1/unknown", "/v1/inspector/recordings/\(UUID().uuidString)"] {
            let missing = try await request(server, .GET, path)
            #expect(missing.status == .unauthorized)
            #expect(try missing.errorCode() == "invalid_api_key")
            #expect(missing.headers.first(name: "www-authenticate") == "Bearer realm=\"Midnight\"")
            let wrong = try await request(
                server, .GET, path, extraHeaders: [("authorization", "Bearer wrong")])
            #expect(wrong.status == .unauthorized)
        }

        let duplicate = try await request(
            server, .GET, "/v1/runtime",
            extraHeaders: [("authorization", "Bearer \(key)"), ("authorization", "Bearer \(key)")])
        #expect(duplicate.status == .unauthorized)

        let accepted = try await request(
            server, .GET, "/v1/runtime", extraHeaders: [("authorization", "Bearer \(key)")])
        #expect(accepted.status == .ok)
        #expect(await manager.state().phase == .empty)
    }

    @Test("Recording retrieval and cancellation bypass a busy model lease")
    func recordingDuringGeneration() async throws {
        let manager = makeManager()
        _ = try await manager.load(ModelLoadRequest(model: "fixture"))
        try await waitForPhase(.ready, manager: manager)
        let store = InspectorRecordingStore()
        let descriptor = InspectorModel(
            id: "fixture", modelType: "test", runtimeType: "test", layerCount: 0,
            hiddenSize: 8, storedElementCount: 0, weightBytes: 0,
            traceSupported: true, traceReason: nil, layers: [])
        let armed = try store.arm(model: descriptor)
        #expect(store.claim(id: armed.id))
        let server = ModelHTTPServer(manager: manager, recordingStore: store)
        let admitted = LifecycleHTTPGate()
        let release = LifecycleHTTPGate()
        let generation = Task {
            try await manager.withModel { _ in
                await admitted.open()
                await release.wait()
            }
        }
        await admitted.wait()
        let path = "/v1/inspector/recordings/\(armed.id)"
        do {
            let status = try await request(server, .GET, path)
            #expect(status.status == .ok)
            #expect(try status.decode(InspectorRecordingSession.self).status == "recording")
            let cancelled = try await request(server, .DELETE, path)
            #expect(cancelled.status == .ok)
            #expect(try cancelled.decode(InspectorRecordingSession.self).status == "cancelled")
        } catch {
            await release.open()
            _ = try? await generation.value
            throw error
        }
        await release.open()
        try await generation.value
        _ = try await manager.unload()
        try await waitForPhase(.empty, manager: manager)
        let retained = try await request(server, .GET, path)
        #expect(retained.status == .ok)
        #expect(try retained.decode(InspectorRecordingSession.self).model?.id == "fixture")
    }

    @Test("Recording routes validate methods and unknown IDs without a loaded model")
    func recordingRouteValidation() async throws {
        let direct = ModelHTTPServer(
            servedModelName: "fixture", tokenLimit: try GenerationTokenLimit(configuredMaximum: 64))
        for server in [direct, ModelHTTPServer(manager: makeManager())] {
            let path = "/v1/inspector/recordings/\(UUID().uuidString)"
            for method in [HTTPMethod.GET, .DELETE] {
                let missing = try await request(server, method, path)
                #expect(missing.status == .notFound)
                #expect(try missing.errorCode() == "recording_not_found")
            }
            #expect(try await request(server, .POST, path, body: "{}").status == .methodNotAllowed)
            #expect(try await request(server, .GET, "/v1/inspector/recordings").status == .methodNotAllowed)
            #expect(try await request(server, .GET, path + "/extra").status == .notFound)
        }
        let unsupported = try await request(
            direct, .POST, "/v1/inspector/recordings", body: #"{"layers":[0]}"#)
        #expect(unsupported.status == .notImplemented)
        let malformed = try await request(direct, .POST, "/v1/inspector/recordings", body: "{")
        #expect(malformed.status == .badRequest)
        let oversized = try await request(
            direct, .POST, "/v1/inspector/recordings", body: String(repeating: " ", count: 32 * 1024 + 1))
        #expect(oversized.status == .payloadTooLarge)
    }

    @Test("Cache reset validates local JSON requests and requires a loaded text model")
    func cacheResetRoute() async throws {
        let manager = makeManager()
        let server = ModelHTTPServer(manager: manager)
        let path = "/v1/runtime/reset-cache"
        let empty = try await request(server, .POST, path, body: "{}")
        #expect(empty.status == .serviceUnavailable)
        for body in ["", "[]", "null", "{", #"{"extra":true}"#] {
            let response = try await request(server, .POST, path, body: body)
            #expect(response.status == .badRequest)
        }
        let remote = try await request(server, .POST, path, body: "{}", remoteIP: "192.0.2.1")
        #expect(remote.status == .forbidden)
        let origin = try await request(
            server, .POST, path, body: "{}",
            extraHeaders: [("origin", "http://localhost")])
        #expect(origin.status == .forbidden)
        _ = try await manager.load(ModelLoadRequest(model: "fixture"))
        try await waitForPhase(.ready, manager: manager)
        let unsupported = try await request(server, .POST, path, body: "{}")
        #expect(unsupported.status == .badRequest)
        #expect(await manager.state().phase == .ready)
    }
    @Test("Installed model discovery and speech requests automatically select their model")
    func automaticSpeechSelection() async throws {
        let voices = try LifecycleHTTPVoice().voiceCatalog
        let manager = ModelLifecycleManager(
            validate: { request in
                ModelLoadOperation(name: request.model) {
                    LoadedModel(
                        servedModelName: request.model,
                        tokenLimit: try GenerationTokenLimit(configuredMaximum: 64),
                        speechSynthesizer: CatalogHTTPVoice(servedModelName: request.model, voiceCatalog: voices),
                        loadRequest: request)
                }
            },
            catalog: {
                ["voice-a", "voice-b"].map { id in
                    let selection = ModelLoadRequest(model: id)
                    return InstalledModelEntry(
                        request: selection,
                        descriptor: ModelLifecycleDescriptor(
                            id: id, created: 0, contextLength: nil, prefillStepSize: nil,
                            kvCompression: nil, memoryLimitBytes: nil, maximumOutputTokens: nil,
                            defaultOutputTokens: nil, modality: "voice", loadRequest: selection))
                }
            })
        let server = ModelHTTPServer(manager: manager)
        let discovery = try await request(server, .GET, "/v1/models")
        #expect(try discovery.decode(Models.self).data.map(\.id) == ["voice-a", "voice-b"])
        #expect(await manager.state().phase == .empty)
        let voice = try #require(voices.voices.first).apiID
        for id in ["voice-a", "voice-b", "voice-a"] {
            let result = try await request(
                server, .POST, "/v1/audio/speech",
                body: "{\"model\":\"\(id)\",\"input\":\"Hello\",\"voice\":\"\(voice)\",\"response_format\":\"wav\"}")
            #expect(result.status == .ok)
            #expect(await manager.state().loadedModel?.id == id)
        }
        let unknown = try await request(
            server, .POST, "/v1/audio/speech",
            body: "{\"model\":\"/tmp/not-a-catalog-id\",\"input\":\"Hello\",\"voice\":\"\(voice)\"}")
        #expect(unknown.status == .notFound)
        #expect(await manager.state().loadedModel?.id == "voice-a")
        _ = try await manager.unload()
        try await waitForPhase(.empty, manager: manager)
        let retained = try await request(server, .GET, "/v1/models/voice-b")
        #expect(retained.status == .ok)
        _ = try await manager.setModelAvailable(false, id: "voice-b")
        let filtered = try await request(server, .GET, "/v1/models")
        #expect(try filtered.decode(Models.self).data.map(\.id) == ["voice-a"])
        #expect(try await request(server, .GET, "/v1/models/voice-b").status == .notFound)
        for (method, path, body) in [
            (HTTPMethod.POST, "/v1/chat/completions", #"{"model":"voice-b","messages":[]}"#),
            (.POST, "/v1/responses", #"{"model":"voice-b","input":"Hello"}"#),
            (.POST, "/v1/audio/speech", #"{"model":"voice-b","input":"Hello"}"#),
            (.GET, "/v1/audio/voices?model=voice-b", ""),
            (.POST, "/v1/runtime/load", #"{"model":"voice-b","name":"another-alias"}"#),
        ] {
            let blocked = try await request(server, method, path, body: body)
            #expect(blocked.status == .serviceUnavailable)
            #expect(String(decoding: blocked.body, as: UTF8.self).contains("model_unavailable"))
        }
        _ = try await manager.setModelAvailable(true, id: "voice-b")
        #expect(await manager.state().phase == .empty)
        #expect(try await request(server, .GET, "/v1/models/voice-b").status == .ok)
    }

    @Test("Mistral speech rejects input longer than 4096 characters")
    func mistralSpeechInputLimit() async throws {
        let speech = try LifecycleHTTPVoice()
        let server = ModelHTTPServer(
            servedModelName: speech.servedModelName,
            tokenLimit: try GenerationTokenLimit(configuredMaximum: 64),
            speechSynthesizer: speech)
        let input = String(repeating: "a", count: 4_097)
        let body = try JSONSerialization.data(withJSONObject: ["input": input])
        let response = try await request(
            server,
            .POST,
            "/v1/audio/speech",
            body: String(decoding: body, as: UTF8.self))

        #expect(response.status == .unprocessableEntity)
        let detail = try #require(response.decode(ValidationEnvelope.self).detail.first)
        #expect(detail.loc == ["body", "input"])
        #expect(detail.msg == "input must contain between 1 and 4096 characters.")
    }

    @Test("Existing model routes expose a display card without changing routing IDs")
    func modelCardMetadata() async throws {
        let card = ModelCard(name: "Friendly Polish name", description: "Editable metadata")
        let expected = ModelCard(name: card.name, description: card.description, capabilities: .init())
        struct CardResponse: Decodable {
            let id: String
            // The API response uses this exact JSON field name.
            // swift-format-ignore: AlwaysUseLowerCamelCase
            let model_card: ModelCard
        }
        struct CardList: Decodable { let data: [CardResponse] }
        let direct = ModelHTTPServer(
            servedModelName: "stable-id",
            tokenLimit: try GenerationTokenLimit(configuredMaximum: 64), modelCard: card)
        let manager = ModelLifecycleManager(validate: { request in
            ModelLoadOperation(name: request.model) {
                LoadedModel(
                    servedModelName: request.model,
                    tokenLimit: try GenerationTokenLimit(configuredMaximum: 64), modelCard: card)
            }
        })
        _ = try await manager.load(ModelLoadRequest(model: "stable-id"))
        try await waitForPhase(.ready, manager: manager)
        for server in [direct, ModelHTTPServer(manager: manager)] {
            let single = try await request(server, .GET, "/v1/models/stable-id")
            #expect(single.status == .ok)
            #expect(try single.decode(CardResponse.self).id == "stable-id")
            #expect(try single.decode(CardResponse.self).model_card == expected)
            let list = try await request(server, .GET, "/v1/models")
            #expect(try list.decode(CardList.self).data.first?.model_card == expected)
            let wrongID = try await request(server, .GET, "/v1/models/Friendly%20Polish%20name")
            #expect(wrongID.status == .notFound)
        }
        _ = try await manager.unload()
        try await waitForPhase(.empty, manager: manager)
    }

    @Test(
        "Model cards expose backend capabilities across direct serving and managed replacement",
        arguments: [false, true])
    func modelCardCapabilities(hasCard: Bool) async throws {
        let speech = try LifecycleHTTPVoice()
        let tokenLimit = try GenerationTokenLimit(configuredMaximum: 64)
        let manager = ModelLifecycleManager(validate: { request in
            let name = request.model
            let card = hasCard ? Self.cardWithFalseClaims(model: name) : nil
            return ModelLoadOperation(name: name) {
                LoadedModel(
                    servedModelName: name, tokenLimit: tokenLimit,
                    speechSynthesizer: name == "speech" ? speech : nil,
                    vision: name == "vision" ? LifecycleHTTPVision() : nil, modelCard: card)
            }
        })
        let managed = ModelHTTPServer(manager: manager)
        let cases: [(String, ModelCard.Capabilities)] = [
            ("text", .init()), ("vision", .init(vision: true)), ("speech", .init(audioOutput: true)),
        ]
        var firstSnapshot: ModelCard?
        for (name, capabilities) in cases {
            let card = hasCard ? Self.cardWithFalseClaims(model: name) : nil
            let expected = ModelCard(
                name: hasCard ? "Display name" : name,
                description: hasCard ? "Publisher metadata" : nil, capabilities: capabilities,
                voices: name == "speech" ? speech.voiceCatalog.modelCardVoices : nil)
            let direct = ModelHTTPServer(
                servedModelName: name, tokenLimit: tokenLimit,
                speechSynthesizer: name == "speech" ? speech : nil,
                vision: name == "vision" ? LifecycleHTTPVision() : nil, modelCard: card)
            _ = try await manager.load(ModelLoadRequest(model: name))
            try await waitForPhase(.ready, manager: manager)
            for server in [direct, managed] {
                let single = try await request(server, .GET, "/v1/models/\(name)")
                #expect(single.status == .ok)
                #expect(try single.decode(Model.self).id == name)
                #expect(try single.decode(Model.self).modelCard == expected)
                let list = try await request(server, .GET, "/v1/models")
                #expect(list.status == .ok)
                #expect(try list.decode(Models.self).data.map(\.id) == [name])
                #expect(try list.decode(Models.self).data.first?.modelCard == expected)
            }
            let runtime = try await request(managed, .GET, "/v1/runtime")
            #expect(try runtime.decode(Runtime.self).loadedModel?.modelCard == expected)
            #expect(try await manager.withModel { $0.modelCard } == expected)
            if name == "text" {
                firstSnapshot = try runtime.decode(Runtime.self).loadedModel?.modelCard
            }
        }
        #expect(firstSnapshot?.capabilities == ModelCard.Capabilities())
        _ = try await manager.unload()
        try await waitForPhase(.empty, manager: manager)
    }

    private static func cardWithFalseClaims(model: String) -> ModelCard {
        ModelCard(
            name: "Display name", description: "Publisher metadata",
            capabilities:
                .init(vision: model != "vision", audioInput: true, audioOutput: model != "speech"))
    }

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

        let accepted = try await request(
            server, .POST, "/v1/runtime/load",
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

        let replacement = try await request(
            server, .POST, "/v1/runtime/load",
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
        let oversized = try await request(
            server, .POST, "/v1/runtime/load",
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
            try ModelHTTPServer.validateControlRequest(
                head: head(json),
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

    @Test("Download removal is explicit, protects active files, and works after unload")
    func managedDownloadRemoval() async throws {
        let root = FileManager.default.temporaryDirectory.resolvingSymlinksInPath()
            .appendingPathComponent("midnight-http-removal-\(UUID().uuidString)")
        let model = root.appendingPathComponent("downloaded")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try JSONEncoder().encode(["repository": "owner/model", "revision": String(repeating: "a", count: 40)])
            .write(to: model.appendingPathComponent("midnight-download.json"))
        let manager = ModelLifecycleManager(
            validate: { _ in
                ModelLoadOperation(name: "alias") {
                    LoadedModel(
                        servedModelName: "alias", tokenLimit: try GenerationTokenLimit(configuredMaximum: 64),
                        protectedDirectories: [model])
                }
            }, downloads: ManagedModelDownloads(root: root))
        let server = ModelHTTPServer(manager: manager)
        _ = try await manager.load(ModelLoadRequest(model: "downloaded"))
        try await waitForPhase(.ready, manager: manager)
        let list = try await request(server, .GET, "/v1/runtime/downloads")
        #expect(list.status == .ok)
        #expect(try list.decode(ManagedModelDownloadList.self).data.first?.inUse == true)
        let active = try await request(server, .POST, "/v1/runtime/remove", body: #"{"model":"downloaded"}"#)
        #expect(active.status == .conflict)
        #expect(try active.errorCode() == "model_in_use")
        #expect(FileManager.default.fileExists(atPath: model.path))
        _ = try await manager.unload()
        try await waitForPhase(.empty, manager: manager)
        let removed = try await request(server, .POST, "/v1/runtime/remove", body: #"{"model":"downloaded"}"#)
        #expect(removed.status == .ok)
        #expect(try removed.decode(ManagedModelRemoval.self).deleted)
        #expect(!FileManager.default.fileExists(atPath: model.path))
        let missing = try await request(server, .POST, "/v1/runtime/remove", body: #"{"model":"downloaded"}"#)
        #expect(missing.status == .notFound)
        #expect(try missing.errorCode() == "model_not_found")
    }

    @Test("Removal rejects malformed and cross-origin requests without touching files")
    func removalRequestValidation() async throws {
        let root = FileManager.default.temporaryDirectory.resolvingSymlinksInPath()
            .appendingPathComponent("midnight-http-removal-\(UUID().uuidString)")
        let model = root.appendingPathComponent("downloaded")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try JSONEncoder().encode(["repository": "owner/model", "revision": String(repeating: "a", count: 40)])
            .write(to: model.appendingPathComponent("midnight-download.json"))
        let manager = ModelLifecycleManager(
            validate: { _ in throw LifecycleHTTPFailure.invalidSelection },
            downloads: ManagedModelDownloads(root: root))
        let server = ModelHTTPServer(manager: manager)
        for invalid in [
            "{", "{}", "[]", #"{"model":42}"#, #"{"model":"../downloaded"}"#,
            #"{"model":"downloaded","force":true}"#,
        ] {
            let response = try await request(server, .POST, "/v1/runtime/remove", body: invalid)
            #expect(response.status == .badRequest)
        }
        let oversized = try await request(
            server, .POST, "/v1/runtime/remove", body: String(repeating: " ", count: 32 * 1024 + 1))
        #expect(oversized.status == .payloadTooLarge)
        let crossOrigin = try await request(
            server, .POST, "/v1/runtime/remove", body: #"{"model":"downloaded"}"#,
            extraHeaders: [("origin", "http://localhost")])
        #expect(crossOrigin.status == .forbidden)
        let remote = try await request(
            server, .POST, "/v1/runtime/remove", body: #"{"model":"downloaded"}"#,
            remoteIP: "192.0.2.5")
        #expect(remote.status == .forbidden)
        #expect(FileManager.default.fileExists(atPath: model.path))
    }

    @Test("Guarded selections reject a stale restore and inspection before changing the model")
    func guardedSelections() async throws {
        let manager = makeManager()
        let server = ModelHTTPServer(manager: manager)
        _ = try await manager.load(ModelLoadRequest(model: "original"))
        try await waitForPhase(.ready, manager: manager)
        let original = await manager.state()
        let instance = try #require(original.instanceID)
        let load = ModelLoadRequest(
            model: "vision", expectedGeneration: original.modelGeneration,
            expectedInstanceID: instance)
        let accepted = try await request(
            server, .POST, "/v1/runtime/load",
            body: String(decoding: JSONEncoder().encode(load), as: UTF8.self))
        #expect(accepted.status == .accepted)
        try await waitForPhase(.ready, manager: manager)

        let stale = try await request(
            server, .POST, "/v1/runtime/load",
            body: String(decoding: JSONEncoder().encode(load), as: UTF8.self))
        #expect(stale.status == .conflict)
        #expect(try stale.errorCode() == "stale_model_selection")
        let wrongInstance = try await request(
            server, .POST, "/v1/runtime/unload",
            body: #"{"expectedInstanceID":"a-different-server"}"#)
        #expect(wrongInstance.status == .conflict)
        #expect(try wrongInstance.errorCode() == "stale_model_selection")

        let trace = InspectorTraceRequest(
            question: "hello", maxTokens: 0, layers: [0],
            tokenPositions: .init(prefill: [-1], decode: []), runtimeGeneration: original.modelGeneration)
        let staleTrace = try await request(
            server, .POST, "/v1/inspector/trace",
            body: String(decoding: JSONEncoder().encode(trace), as: UTF8.self))
        #expect(staleTrace.status == .conflict)
        #expect(try staleTrace.errorCode() == "stale_model_selection")
        #expect(await manager.state().loadedModel?.id == "vision")
        for body in [
            #"{"expectedGeneration":-1}"#, #"{"expectedGeneration":"1"}"#,
            #"{"expectedInstanceID":42}"#,
        ] {
            let rejected = try await request(server, .POST, "/v1/runtime/unload", body: body)
            #expect(rejected.status == .badRequest)
            #expect(await manager.state().loadedModel?.id == "vision")
        }
        let generation = await manager.state().modelGeneration
        let payload = try JSONSerialization.data(withJSONObject: [
            "expectedGeneration": generation, "expectedInstanceID": instance,
        ])
        let unload = try await request(
            server, .POST, "/v1/runtime/unload",
            body: String(decoding: payload, as: UTF8.self))
        #expect(unload.status == .accepted)
        try await waitForPhase(.empty, manager: manager)
    }

    @Test("Managed vision forwards image bodies and worker errors, and rejects unsupported APIs")
    func managedVisionRouting() async throws {
        let vision = LifecycleHTTPVision()
        let manager = ModelLifecycleManager(validate: { _ in
            ModelLoadOperation(name: "vision") {
                LoadedModel(
                    servedModelName: "vision",
                    tokenLimit: try GenerationTokenLimit(configuredMaximum: 64), vision: vision)
            }
        })
        let server = ModelHTTPServer(manager: manager)
        _ = try await manager.load(ModelLoadRequest(model: "vision"))
        try await waitForPhase(.ready, manager: manager)
        let body =
            #"{"model":"vision","messages":[{"role":"user","content":[{"type":"text","text":"Read it"},{"type":"image_url","image_url":{"url":"data:image/png;base64,YQ=="}}]}]}"#
        let response = try await request(server, .POST, "/v1/chat/completions", body: body)
        #expect(response.status == .unprocessableEntity)
        #expect(try response.errorCode() == "test_image_error")
        #expect(await vision.bodies == [Data(body.utf8)])
        let status = try await request(server, .GET, "/v1/vision/status")
        #expect(status.status == .ok)
        #expect(String(decoding: status.body, as: UTF8.self).contains("vision-ready"))
        for route in ["/v1/responses", "/v1/audio/speech", "/v1/inspector/trace"] {
            let rejected = try await request(server, .POST, route, body: "{}")
            #expect(rejected.status == .notImplemented)
            #expect(try rejected.errorCode() == "unsupported_model_feature")
        }
        let models = try await request(server, .GET, "/v1/models")
        #expect(try models.decode(Models.self).data.map(\.id) == ["vision"])
        _ = try await manager.unload()
        try await waitForPhase(.empty, manager: manager)
        #expect(await vision.didShutdown)
    }

    private func makeManager(delayedLoad: LifecycleHTTPGate? = nil) -> ModelLifecycleManager {
        ModelLifecycleManager(validate: { request in
            guard request.model != "invalid" else {
                throw LifecycleHTTPFailure.invalidSelection
            }
            let name = request.name ?? request.model
            return ModelLoadOperation(name: name) {
                if request.model == "delayed" {
                    await delayedLoad?.wait()
                }
                if request.model == "broken" {
                    throw LifecycleHTTPFailure.loadFailed
                }
                return LoadedModel(
                    servedModelName: name,
                    tokenLimit: try GenerationTokenLimit(configuredMaximum: request.maxTokens ?? 128))
            }
        })
    }

    private func request(
        _ server: ModelHTTPServer, _ method: HTTPMethod, _ uri: String,
        body: String = "", extraHeaders: [(String, String)] = [],
        remoteIP: String = "127.0.0.1"
    ) async throws -> Response {
        let channel = await NIOAsyncTestingChannel(handler: ModelHTTPRequestHandler(server: server))
        try await channel.connect(to: SocketAddress(ipAddress: remoteIP, port: 9999)).get()
        do {
            let headers = HTTPHeaders([("content-type", "application/json")] + extraHeaders)
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
                case .body(.fileRegion): throw LifecycleHTTPFailure.unexpectedBody
                case .end:
                    let head = try #require(head)
                    _ = try await channel.finish(acceptAlreadyClosed: true)
                    return Response(status: head.status, headers: head.headers, body: data)
                }
            }
            throw LifecycleHTTPFailure.timedOut
        } catch {
            _ = try? await channel.finish(acceptAlreadyClosed: true)
            throw error
        }
    }

    private func waitForPhase(
        _ phase: ModelLifecycleState.Phase,
        manager: ModelLifecycleManager
    ) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while ContinuousClock.now < deadline {
            if await manager.state().phase == phase {
                return
            }
            try await Task.sleep(for: .milliseconds(5))
        }
        throw LifecycleHTTPFailure.timedOut
    }

    private struct Response {
        let status: HTTPResponseStatus
        let headers: HTTPHeaders
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
        let modelCard: ModelCard?
        enum CodingKeys: String, CodingKey {
            case id, created, object
            case ownedBy = "owned_by"
            case modelCard = "model_card"
        }
    }

    private struct Models: Decodable { let data: [Model] }
    private struct ErrorEnvelope: Decodable {
        struct Detail: Decodable { let code: String? }
        let error: Detail
    }

    private struct ValidationEnvelope: Decodable {
        struct Detail: Decodable {
            let loc: [String]
            let msg: String
        }

        let detail: [Detail]
    }
}

private struct CatalogHTTPVoice: LocalSpeechSynthesizing {
    let servedModelName: String
    let voiceCatalog: VoxtralVoiceCatalog
    func stream(request: LocalSpeechSynthesisRequest) async -> AsyncThrowingStream<LocalSpeechSynthesisEvent, Error> {
        AsyncThrowingStream {
            $0.yield(.audio(Data([1, 2, 3, 4])))
            $0.yield(.completed(.init(promptTokens: 1, completionTokens: 1)))
            $0.finish()
        }
    }
    func waitUntilIdle() async {}
}

private struct LifecycleHTTPVoice: LocalSpeechSynthesizing {
    let servedModelName = "speech"
    let voiceCatalog: VoxtralVoiceCatalog

    init() throws {
        let fixture = FileManager.default.temporaryDirectory
            .appendingPathComponent("midnight-http-capabilities-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixture, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixture) }
        try Data(#"{"model_type":"voxtral_tts","multimodal":{"audio_tokenizer_args":{"voice":{"en_female":0}}}}"#.utf8)
            .write(to: fixture.appendingPathComponent("config.json"))
        voiceCatalog = try #require(try VoxtralVoiceCatalog(modelDirectory: fixture.path))
    }

    func stream(request: LocalSpeechSynthesisRequest) async -> AsyncThrowingStream<LocalSpeechSynthesisEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }

    func waitUntilIdle() async {}
}

private actor LifecycleHTTPVision: VisionModelServing {
    private(set) var bodies: [Data] = []
    private(set) var didShutdown = false
    func complete(_ body: Data) async throws -> VisionHTTPResponse {
        bodies.append(body)
        return VisionHTTPResponse(status: 422, body: Data(#"{"error":{"code":"test_image_error"}}"#.utf8))
    }
    func status() async throws -> VisionHTTPResponse {
        VisionHTTPResponse(status: 200, body: Data(#"{"phase":"vision-ready"}"#.utf8))
    }
    func waitUntilIdle() async {}
    func shutdown() async { didShutdown = true }
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
        guard !isOpen else {
            return
        }
        await withCheckedContinuation { waiters.append($0) }
    }
    func open() {
        isOpen = true
        let waiters = waiters
        self.waiters.removeAll()
        for waiter in waiters {
            waiter.resume()
        }
    }
}
