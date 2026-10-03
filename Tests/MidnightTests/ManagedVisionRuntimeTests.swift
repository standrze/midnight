import Foundation
import ModelRunnerProtocol
import Testing

@testable import Midnight

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

@Suite("Explicit managed vision runtime", .serialized)
struct ManagedVisionRuntimeTests {
    @Test("Text preflight never consults the optional worker environment")
    func textDoesNotConsultVision() throws {
        let fixture = try VisionWorkerFixture()
        defer { fixture.remove() }
        try Data(#"{"model_type":"llama","max_position_embeddings":32768}"#.utf8)
            .write(to: fixture.directory.appendingPathComponent("config.json"))
        let loader = ModelLoader(
            defaultEngine: "cpu",
            visionEnvironment: {
                Issue.record("Text preflight consulted the vision environment")
                return [:]
            })
        let value = try loader.validate(ModelLoadRequest(model: fixture.directory.path))
        #expect(value.modality == "text")
        #expect(value.modelCard?.capabilities == ModelCard.Capabilities())
        #expect(value.loadRequest?.model == fixture.directory.path)
        #expect(value.loadRequest?.maxTokens == nil)
        let restored = try loader.validate(#require(value.loadRequest))
        #expect(restored.tokenLimit == value.tokenLimit)
        #expect(restored.longContext.contextLength == 32768)
        let capped = try loader.validate(ModelLoadRequest(model: fixture.directory.path, maxTokens: 96))
        #expect(capped.loadRequest?.maxTokens == 96)
        #expect(try loader.validate(#require(capped.loadRequest)).tokenLimit == capped.tokenLimit)
    }

    @Test("Adapter bundle restore retains the bundle's automatic output policy")
    func bundleRestorePolicy() throws {
        let fixture = try VisionWorkerFixture()
        defer { fixture.remove() }
        let base = fixture.directory.appendingPathComponent("base-model")
        let adapter = fixture.directory.appendingPathComponent("adapter")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: false)
        try FileManager.default.createDirectory(at: adapter, withIntermediateDirectories: false)
        try Data(#"{"model_type":"llama","max_position_embeddings":32768}"#.utf8).write(
            to: base.appendingPathComponent("config.json"))
        try Data().write(to: base.appendingPathComponent("model.safetensors"))
        try Data(#"{"maximumTokens":96}"#.utf8).write(to: base.appendingPathComponent("midnight.json"))
        try Data("{}".utf8).write(to: adapter.appendingPathComponent("adapter_config.json"))
        try Data().write(to: adapter.appendingPathComponent("adapters.safetensors"))
        let loader = ModelLoader(defaultEngine: "cpu")
        let original = try loader.validate(ModelLoadRequest(model: fixture.directory.path))
        #expect(original.loadRequest?.model == fixture.directory.path)
        #expect(original.loadRequest?.maxTokens == nil)
        #expect(original.tokenLimit.configuredMaximum == 32767)
        let restored = try loader.validate(#require(original.loadRequest))
        #expect(restored.tokenLimit == original.tokenLimit)
        #expect(restored.selection.adapterPath == original.selection.adapterPath)
    }

    #if os(macOS)
        @Test("Vision validates worker and incompatible controls before allocating weights")
        func visionPreflight() throws {
            let fixture = try VisionWorkerFixture()
            defer { fixture.remove() }
            #expect(throws: (any Error).self) {
                try ModelLoader(visionEnvironment: { [:] }).validate(ModelLoadRequest(model: fixture.directory.path))
            }
            let path = fixture.executable.path
            let loader = ModelLoader(visionEnvironment: { ["MIDNIGHT_VISION_WORKER": path] })
            let value = try loader.validate(ModelLoadRequest(model: fixture.directory.path))
            #expect(value.modality == "vision")
            #expect(value.modelCard?.capabilities == ModelCard.Capabilities(vision: true))
            #expect(value.tokenLimit.configuredMaximum == 1024)
            #expect(value.tokenLimit.defaultTokens == 512)
            #expect(value.loadRequest?.contextLength == nil)
            #expect(
                !FileManager.default.fileExists(atPath: fixture.directory.appendingPathComponent("worker-pid").path))
            for request in [
                ModelLoadRequest(model: fixture.directory.path, contextLength: 4096),
                ModelLoadRequest(model: fixture.directory.path, engine: "cpu"),
                ModelLoadRequest(model: fixture.directory.path, maxTokens: 1025),
            ] {
                #expect(throws: (any Error).self) { try loader.validate(request) }
            }
        }

        @Test("Explicit worker launch, status, output limit and shutdown own the child process")
        func workerLifecycle() async throws {
            let fixture = try VisionWorkerFixture()
            defer { fixture.remove() }
            let worker = try await fixture.launch()
            let status = try await worker.status()
            #expect(status.status == 200)
            let answer = try await worker.complete(Data(#"{"model":"test-vision"}"#.utf8))
            #expect(answer.status == 200)
            #expect(try JSONSerialization.jsonObject(with: answer.body) as? [String: Int] == ["max_tokens": 64])
            #expect(try await worker.complete(Data(#"{"max_tokens":65}"#.utf8)).status == 400)
            await worker.shutdown()
            await #expect(throws: (any Error).self) { try await worker.status() }
            #expect(!fixture.workerIsAlive)
        }

        @Test("Client cancellation waits for remote drain before the request returns")
        func cancellationDrains() async throws {
            let fixture = try VisionWorkerFixture()
            defer { fixture.remove() }
            let worker = try await fixture.launch()
            let request = Task { try await worker.complete(Data(#"{"hold":true}"#.utf8)) }
            for _ in 0..<200 {
                if FileManager.default.fileExists(atPath: fixture.directory.appendingPathComponent("entered").path) {
                    break
                }
                try await Task.sleep(for: .milliseconds(10))
            }
            #expect(FileManager.default.fileExists(atPath: fixture.directory.appendingPathComponent("entered").path))
            request.cancel()
            do {
                _ = try await request.value
                Issue.record("Cancelled request unexpectedly succeeded")
            } catch {}
            #expect(FileManager.default.fileExists(atPath: fixture.directory.appendingPathComponent("drained").path))
            #expect(try await worker.status().status == 200)
            await worker.shutdown()
            #expect(!fixture.workerIsAlive)
        }

        @Test("Missing readiness stops the failed child and preserves the launch error")
        func startupFailureStopsChild() async throws {
            let fixture = try VisionWorkerFixture(mode: "no-ready")
            defer { fixture.remove() }
            await #expect(throws: (any Error).self) { try await fixture.launch(startupSeconds: 0.2) }
            #expect(!fixture.workerIsAlive)
        }
    #endif
}

private struct VisionWorkerFixture: Sendable {
    let directory: URL
    let executable: URL
    init(mode: String = "normal") throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "vision-worker-test-\(UUID().uuidString)")
        executable = directory.appendingPathComponent("fake-worker")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(
            #"{"model_type":"fastvlm","tokenizer_model_max_length":8192,"vision_config":{"layers":[2,12,24,4,2],"downsamples":[true,true,true,true,true],"down_stride":2,"down_patch_size":7,"image_size":1024}}"#
                .utf8
        ).write(to: directory.appendingPathComponent("config.json"))
        try Data(#"{"processor_class":"FastVLMProcessor","crop_size":{"width":1024,"height":1024}}"#.utf8).write(
            to: directory.appendingPathComponent("preprocessor_config.json"))
        try Data().write(to: directory.appendingPathComponent("model.safetensors"))
        let script = #"""
            #!/usr/bin/env python3
            import http.server,json,os,pathlib,sys,threading,time
            a=dict(zip(sys.argv[1::2],sys.argv[2::2])); root=pathlib.Path(a['--model'])
            def watch_parent():
              while os.getppid()==int(a['--parent-pid']):time.sleep(.1)
              os._exit(0)
            threading.Thread(target=watch_parent,daemon=True).start()
            (root/'worker-pid').write_text(str(os.getpid()))
            if MODE == 'no-ready': time.sleep(30); sys.exit()
            busy=threading.Event()
            class Handler(http.server.BaseHTTPRequestHandler):
              def log_message(self,*args): pass
              def send(self,value):
                data=json.dumps(value).encode(); self.send_response(200); self.send_header('Content-Length',str(len(data))); self.end_headers()
                try:self.wfile.write(data)
                except OSError:pass
              def do_GET(self):self.send({'phase':'ready','model':a['--served-model-name']})
              def do_POST(self):
                data=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))) or b'{}')
                if self.path.endswith('/drain'):
                  while busy.is_set():time.sleep(.01)
                  (root/'drained').write_text('yes'); self.send({'idle':True}); return
                if data.get('hold'):
                  busy.set(); (root/'entered').write_text('yes'); time.sleep(.3); busy.clear()
                self.send({'max_tokens':data.get('max_tokens')})
            server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
            pathlib.Path(a['--ready-file']).write_text(json.dumps({'port':server.server_port,'pid':os.getpid(),'model':a['--served-model-name'],'token':a['--control-token']}))
            server.serve_forever()
            """#.replacingOccurrences(of: "MODE", with: "'\(mode)'")
        try Data(script.utf8).write(to: executable)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)
    }
    func launch(startupSeconds: Double = 5) async throws -> ManagedVisionRuntime {
        try await ManagedVisionRuntime.launch(
            configuration: ManagedVisionConfiguration(
                executable: executable, memoryGiB: 4, contextLength: 8192), model: directory,
            name: "test-vision", managedRoot: directory.appendingPathComponent("absent-root"),
            tokenLimit: GenerationTokenLimit(configuredMaximum: 64), startupSeconds: startupSeconds)
    }
    var workerIsAlive: Bool {
        guard let text = try? String(contentsOf: directory.appendingPathComponent("worker-pid"), encoding: .utf8),
            let pid = Int32(text)
        else {
            return false
        }
        return kill(pid, 0) == 0
    }
    func remove() { try? FileManager.default.removeItem(at: directory) }
}
