import ArgumentParser
import CorpusPreparationCore
import Foundation

@main
struct PrepareCorpus: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "model-runner-prepare-corpus",
        abstract: "Prepare reproducible text and pinned reference corpora without Python or a model runtime.",
        subcommands: [Text.self, Reference.self])
}
struct Text: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Build a deterministic JSONL benchmark from text or JSON text rows.")
    @Argument var source: String
    @Argument var output: String
    @Option var samples = 64
    @Option var targetCharacters = 2500
    @Option var idPrefix = "heldout"
    @Option var category = "language-modeling"
    mutating func validate() throws {
        guard samples > 0, targetCharacters >= 128 else {
            throw ValidationError("--samples must be positive and --target-characters must be at least 128")
        }
    }
    mutating func run() throws {
        let result = try TextCorpus.prepare(
            source: URL(fileURLWithPath: source), output: URL(fileURLWithPath: output),
            samples: samples, targetCharacters: targetCharacters, idPrefix: idPrefix, category: category)
        print("samples=\(result.samples) characters=\(result.characters)")
        print("source_sha256=\(result.sourceSHA256)")
        print("output_sha256=\(result.outputSHA256)")
    }
}
struct Reference: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Reproduce the pinned 128-record code/math reference corpus.")
    @Option var outputDir: String
    @Option var cacheDir: String?
    @Flag var offline = false
    mutating func run() async throws {
        let result = try await ReferenceCorpus.prepare(
            output: URL(fileURLWithPath: outputDir),
            cache: cacheDir.map { URL(fileURLWithPath: $0) }, offline: offline)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        print(String(decoding: try encoder.encode(result), as: UTF8.self))
    }
}
