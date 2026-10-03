import ArgumentParser
import Foundation
import ModelRunnerProtocol
import Testing
import loom

@testable import Midnight
@testable import ModelRunnerCore

@Suite("Terminal server console")
struct ServerConsoleTests {
    @Test @MainActor func downloadCatalogRendersProgressAndUnavailableSources() {
        var frame = Frame(width: 120, height: 28)
        ServerConsole.render(
            frame: &frame, endpoint: "local", listening: true, state: nil,
            models: [], selectedIndex: 0, message: "Ready", logs: [],
            actionMessage: "Delete saved-model from disk? y confirm · n/Esc cancel",
            downloadEntries: ModelDownloadCatalog.entries, downloadIndex: 2,
            downloadStatus: "Downloading publisher/model · 42%")
        let text = frame.buffer.plainText
        #expect(text.contains("Publisher downloads"))
        #expect(text.contains("unavailable"))
        #expect(text.contains("42%"))
        #expect(text.contains("y confirm"))
        #expect(text.contains("c cancel"))
        #expect(text.contains("t token status"))
    }

    @Test func automaticConsoleRequiresAnInteractiveTerminal() {
        #expect(
            ServerConsole.shouldPresent(
                disabled: false, inputIsTerminal: true, outputIsTerminal: true, terminalType: "xterm-256color"))
        #expect(
            !ServerConsole.shouldPresent(
                disabled: true, inputIsTerminal: true, outputIsTerminal: true, terminalType: "xterm"))
        #expect(
            !ServerConsole.shouldPresent(
                disabled: false, inputIsTerminal: false, outputIsTerminal: true, terminalType: "xterm"))
        #expect(
            !ServerConsole.shouldPresent(
                disabled: false, inputIsTerminal: true, outputIsTerminal: false, terminalType: "xterm"))
        #expect(
            !ServerConsole.shouldPresent(
                disabled: false, inputIsTerminal: true, outputIsTerminal: true, terminalType: "dumb"))
    }

    @Test func interactiveLaunchCanStartWithoutAModel() throws {
        let command = try MidnightCommand.parse([])
        let interactive = try ServerConfiguration(command: command, settings: nil, consoleEnabled: true)
        #expect(interactive.initialLoad == nil)
        #expect(interactive.consoleEnabled)
        #expect(throws: (any Error).self) {
            try ServerConfiguration(command: command, settings: nil, consoleEnabled: false)
        }
        let plain = try ServerConfiguration(command: MidnightCommand.parse(["--idle", "--no-ui"]), settings: nil)
        #expect(!plain.consoleEnabled)
        #expect(MidnightCommand.helpMessage().contains("--no-ui"))
    }

    @Test @MainActor func renderingTracksModelSelectionAndRuntimeErrors() {
        let models = (0..<20).map { index in
            ModelLifecycleDescriptor(
                id: "model-\(index)", created: 0, contextLength: nil, prefillStepSize: nil,
                kvCompression: nil, memoryLimitBytes: nil, maximumOutputTokens: nil,
                defaultOutputTokens: nil, modality: "text", loadRequest: ModelLoadRequest(model: "model-\(index)"))
        }
        let state = ModelLifecycleState(
            instanceID: "test", processID: 1, phase: .ready, operationID: nil, modelGeneration: 1,
            loadedModel: models[19], targetModel: nil, lastError: "Load failed: fixture",
            memory: LocalModelMemorySnapshot(activeBytes: 104_857_600, cachedBytes: 0, peakBytes: 104_857_600))
        var frame = Frame(width: 80, height: 24)
        ServerConsole.render(
            frame: &frame, endpoint: "http://localhost:8080/v1", listening: true, state: state,
            models: models, selectedIndex: 19, message: "Ready", logs: ["Request completed"],
            unavailableModels: ["model-18"])
        let text = frame.buffer.plainText
        #expect(text.contains("READY  ·  model-19"))
        #expect(text.contains("100 MiB active"))
        #expect(text.contains("Load failed: fixture"))
        #expect(text.contains("Request completed"))
        #expect(frame.buffer[1, 17].style.reversed)
        #expect(text.contains("Enter load"))
        #expect(text.contains("unavailable"))
        #expect(text.contains("a availability"))
        #expect(text.contains("− model-18"))
        #expect(text.contains("+ model-19"))
        for (width, height) in [(0, 0), (1, 1), (20, 5), (40, 12)] {
            var small = Frame(width: width, height: height)
            ServerConsole.render(
                frame: &small, endpoint: "local", listening: false, state: nil,
                models: [], selectedIndex: 0, message: "Starting", logs: [])
            #expect(small.buffer.width == width)
            #expect(small.buffer.height == height)
        }
    }
}
