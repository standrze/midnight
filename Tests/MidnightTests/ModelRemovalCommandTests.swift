import ArgumentParser
import Testing

@testable import Midnight

@Suite("Model-management root command parsing")
struct ModelRemovalCommandTests {
    @Test(
        "The remove subcommand owns its list flag",
        arguments: [
            ["remove", "--list", "--endpoint", "http://127.0.0.1:19941"],
            ["remove", "--endpoint", "http://127.0.0.1:19941", "--list"],
            ["--model", "active-model", "remove", "--list", "--endpoint", "http://127.0.0.1:19941"],
            ["remove", "--list-models", "--endpoint", "http://127.0.0.1:19941"],
            ["--list-models", "remove", "--endpoint", "http://127.0.0.1:19941"],
        ])
    func listsDownloads(arguments: [String]) throws {
        let command = try #require(try MidnightCommand.parseAsRoot(arguments) as? RemoveModelCommand)
        #expect(command.listing.list)
        #expect(command.model == nil)
        #expect(command.connection.endpoint == "http://127.0.0.1:19941")
    }

    @Test(
        "Removal keeps the explicitly selected download",
        arguments: [
            ["remove", "selected-download"],
            ["--model", "active-model", "remove", "selected-download"],
        ])
    func removalTarget(arguments: [String]) throws {
        let command = try #require(try MidnightCommand.parseAsRoot(arguments) as? RemoveModelCommand)
        #expect(!command.listing.list)
        #expect(command.model == "selected-download")
    }

    @Test("Global listing remains a root command")
    func globalListing() throws {
        let command = try #require(try MidnightCommand.parseAsRoot(["--list"]) as? MidnightCommand)
        #expect(command.listing.list)
        #expect(command.selection.model == nil)
        let alias = try #require(try MidnightCommand.parseAsRoot(["--list-models"]) as? MidnightCommand)
        #expect(alias.listing.list)
    }
    @Test("Download commands retain their own listing flag and repository")
    func downloadParsing() throws {
        let list = try #require(try MidnightCommand.parseAsRoot(["download", "--list"]) as? DownloadCommand)
        #expect(list.listing.list)
        #expect(list.model == nil)
        let selected = try #require(
            try MidnightCommand.parseAsRoot(["download", "owner/model", "--dry-run"]) as? DownloadCommand)
        #expect(selected.model == "owner/model")
        #expect(selected.dryRun)
        #expect(!selected.listing.list)
    }

    @Test("Unload retains its endpoint without changing global model selection")
    func unloadParsing() throws {
        let unload = try #require(
            try MidnightCommand.parseAsRoot(["unload", "--endpoint", "http://127.0.0.1:19941"]) as? UnloadModelCommand)
        #expect(unload.connection.endpoint == "http://127.0.0.1:19941")
    }

    @Test(
        "Removal requires exactly one explicit operation",
        arguments: [
            ["remove"], ["--model", "active-model", "remove"],
            ["remove", "selected-download", "--list"], ["--list", "remove", "selected-download"],
            ["remove", "selected-download", "--list-models"], ["--list-models", "remove", "selected-download"],
        ])
    func invalidRemoval(arguments: [String]) {
        #expect(throws: (any Error).self) { try MidnightCommand.parseAsRoot(arguments) }
    }
}
