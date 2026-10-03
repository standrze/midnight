import ArgumentParser
import Foundation
import ModelRunnerProtocol
import Testing

@testable import Midnight

@Suite("Server CLI compatibility")
struct ServerCommandTests {
    @Test func allOptionsReachLoadRequest() throws {
        let command = try MidnightCommand.parse([
            "-m", "target", "--name", "alias", "--adapter", "adapter", "--adapter-scale", "0.5",
            "--host", "localhost", "-p", "9000", "-c", "settings.json", "--verbose", "--engine", "metal",
            "--max-tokens", "100", "--context-length", "4096", "--prefill-step-size", "256",
            "--kv-compression", "affine8", "--dflash-model", "draft", "--dflash-block-size", "3",
            "--gemma-assistant-model", "assistant", "--gemma-assistant-block-size", "4",
            "--gemma-assistant-quantization-bits", "8", "--no-auto-assistant",
        ])
        #expect(command.listener.config == "settings.json")
        let resolved = try ServerConfiguration(command: command, settings: nil)
        #expect(resolved.host == "localhost")
        #expect(resolved.port == 9000)
        #expect(resolved.verbose)
        #expect(resolved.engine == "metal")
        let load = try #require(resolved.initialLoad)
        #expect(load.model == "target")
        #expect(load.name == "alias")
        #expect(load.adapter == "adapter")
        #expect(load.adapterScale == 0.5)
        #expect(load.maxTokens == 100)
        #expect(load.contextLength == 4096)
        #expect(load.prefillStepSize == 256)
        #expect(load.kvCompression == "affine8")
        #expect(load.engine == "metal")
        #expect(load.dflashModel == "draft")
        #expect(load.dflashBlockSize == 3)
        #expect(load.gemmaAssistantModel == "assistant")
        #expect(load.gemmaAssistantBlockSize == 4)
        #expect(load.gemmaAssistantQuantizationBits == 8)
        #expect(load.autoAssistant == false)
    }

    @Test func precedenceAndIdle() throws {
        let settings = try JSONDecoder().decode(
            ModelStackSettings.self,
            from: Data(
                #"{"mlxRunner":{"modelPath":"configured","host":"localhost","port":9001,"maximumTokens":80}}"#.utf8)
        ).mlxRunner
        let configured = try ServerConfiguration(command: MidnightCommand.parse([]), settings: settings)
        #expect(configured.initialLoad?.model == "configured")
        #expect(configured.initialLoad?.maxTokens == nil)  // ModelLoader applies checkpoint-aware settings.
        #expect(configured.host == "localhost")
        #expect(configured.port == 9001)
        let overridden = try ServerConfiguration(
            command: MidnightCommand.parse(["--model", "other", "--host", "127.0.0.1", "--port", "9002"]),
            settings: settings)
        #expect(overridden.initialLoad?.model == "other")
        #expect(overridden.host == "127.0.0.1")
        #expect(overridden.port == 9002)
        let idle = try ServerConfiguration(command: MidnightCommand.parse(["--idle"]), settings: settings)
        #expect(idle.initialLoad == nil)
        let defaults = try ServerConfiguration(command: MidnightCommand.parse(["--idle"]), settings: nil)
        #expect(defaults.host == "127.0.0.1")
        #expect(defaults.port == 8080)
        #expect(throws: (any Error).self) { try ServerConfiguration(command: MidnightCommand.parse([]), settings: nil) }
        #expect(throws: (any Error).self) {
            try ServerConfiguration(command: MidnightCommand.parse(["--idle", "--model", "x"]), settings: nil)
        }
    }

    @Test func authAndHelp() throws {
        #expect(try MidnightCommand.parseAsRoot(["auth", "login"]) is AuthCommand.Login)
        #expect(try MidnightCommand.parseAsRoot(["auth", "logout"]) is AuthCommand.Logout)
        #expect(try MidnightCommand.parseAsRoot(["auth", "status"]) is AuthCommand.Status)
        for arguments in [["--help"], ["--version"], ["auth", "--help"], ["download", "--help"]] {
            do {
                var help = try MidnightCommand.parseAsRoot(arguments)
                try help.run()
            } catch {
                #expect(MidnightCommand.exitCode(for: error) == .success)
            }
        }
        #expect(MidnightCommand.helpMessage().contains("GENERATION & CACHE"))
    }

    @Test func listingSkipsConfig() async throws {
        var command = try MidnightCommand.parse([
            "--list", "--config", "/missing/midnight-test-settings.json", "--idle", "--model", "ignored",
        ])
        try await command.run()
    }
}
