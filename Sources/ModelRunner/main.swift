import ArgumentParser
import Foundation
import ModelRunnerCore
import ModelRunnerProtocol

@main
struct MidnightCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "midnight",
        abstract: "Midnight Runner — serve a local MLX model through OpenAI-compatible chat and local audio APIs.",
        version: "0.2.0-beta.5",
        subcommands: [
            DownloadCommand.self, RemoveModelCommand.self, UnloadModelCommand.self,
            AuthCommand.self, APIKeyCommand.self,
        ]
    )

    @OptionGroup(title: "Listener & Configuration") var listener: ListenerOptions
    @OptionGroup(title: "Model & Adapters") var selection: ModelSelectionOptions
    @OptionGroup(title: "Generation & Cache") var generation: GenerationOptions
    @OptionGroup(title: "Speculative Decoding") var speculation: SpeculativeDecodingOptions

    @OptionGroup var listing: ModelListOptions

    mutating func run() async throws {
        defer { clearModelRunnerMLXStreams() }

        if listing.list {
            let directory = ModelCatalog.defaultDirectory()
            let models = ModelLoader().installedModels(roots: [directory]).map(\.descriptor.id)
            print("Available models in \(directory.path):")
            if models.isEmpty {
                print("  (none)")
            } else {
                for model in models {
                    print("  \(model)")
                }
            }
            return
        }

        let settings = try ModelStackSettings.load(explicitPath: listener.config)
        let configuration = try ServerConfiguration(command: self, settings: settings?.mlxRunner)
        try await configuration.run()
    }
}
