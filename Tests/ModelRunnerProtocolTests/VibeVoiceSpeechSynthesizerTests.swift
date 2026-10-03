import Foundation
import ModelRunnerProtocol
import Testing

@testable import ModelRunnerCore

#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

@Suite("VibeVoice worker lifecycle", .serialized, .timeLimit(.minutes(1)))
struct VibeVoiceSpeechSynthesizerTests {
    @Test("Cancelling a stream interrupts worker I/O and drains before the response timeout")
    func cancellationDrainsPromptly() async throws {
        let fixture = try VibeVoiceWorkerFixture()
        defer { fixture.remove() }
        let synthesizer = try await VibeVoiceSpeechSynthesizer(
            servedModelName: "vibevoice-test",
            workerExecutableURL: fixture.executable,
            workerArguments: [fixture.entered.path, fixture.firstRequest.path, fixture.firstWorkerPID.path],
            workerEnvironment: [:],
            responseTimeout: 3,
            shutdownGracePeriod: 0.1
        )
        let request = LocalSpeechSynthesisRequest(input: "hello", voiceID: "default", format: .wav)
        let stream = await synthesizer.stream(request: request)
        let consumer = Task {
            do {
                for try await _ in stream {}
            } catch {}
        }
        try await fixture.waitUntilEntered()

        let startedCancellation = ContinuousClock.now
        consumer.cancel()
        await consumer.value
        await synthesizer.waitUntilIdle()
        let cancellationDuration = startedCancellation.duration(to: .now)

        #expect(cancellationDuration < .seconds(1))
        #expect(try !fixture.firstWorkerIsAlive())

        var recoveredEvents: [LocalSpeechSynthesisEvent] = []
        for try await event in await synthesizer.stream(request: request) {
            recoveredEvents.append(event)
        }
        await synthesizer.waitUntilIdle()
        #expect(
            recoveredEvents == [
                .audio(Data([1, 2])),
                .completed(LocalSpeechUsage(promptTokens: 1, completionTokens: 2)),
            ])
    }
}

private struct VibeVoiceWorkerFixture: Sendable {
    let directory: URL
    let executable: URL
    let entered: URL
    let firstRequest: URL
    let firstWorkerPID: URL

    init() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "vibevoice-worker-test-\(UUID().uuidString)", isDirectory: true)
        executable = directory.appendingPathComponent("fake-worker")
        entered = directory.appendingPathComponent("entered")
        firstRequest = directory.appendingPathComponent("first-request")
        firstWorkerPID = directory.appendingPathComponent("first-worker-pid")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let script = #"""
            #!/bin/sh
            printf '{"ready":true,"voices":["default"]}\n'
            IFS= read -r request
            if [ ! -e "$2" ]; then
                : > "$2"
                printf '%s' "$$" > "$3"
                : > "$1"
                trap '' TERM
                exec /bin/sleep 30
            fi
            printf '{"audio":"AQI=","promptTokens":1,"completionTokens":2}\n'
            """#
        try Data(script.utf8).write(to: executable)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)
    }

    func waitUntilEntered() async throws {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(2))
        while clock.now < deadline {
            if FileManager.default.fileExists(atPath: entered.path) {
                return
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        throw VibeVoiceWorkerFixtureError.workerDidNotReceiveRequest
    }

    func remove() {
        try? FileManager.default.removeItem(at: directory)
    }

    func firstWorkerIsAlive() throws -> Bool {
        let value = try String(contentsOf: firstWorkerPID, encoding: .utf8)
        guard let processIdentifier = Int32(value) else {
            throw VibeVoiceWorkerFixtureError.invalidProcessIdentifier
        }
        #if canImport(Darwin) || canImport(Glibc)
            return kill(processIdentifier, 0) == 0 || errno == EPERM
        #else
            return false
        #endif
    }
}

private enum VibeVoiceWorkerFixtureError: Error {
    case invalidProcessIdentifier
    case workerDidNotReceiveRequest
}
