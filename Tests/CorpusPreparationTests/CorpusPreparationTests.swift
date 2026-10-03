import Foundation
import Testing

@testable import CorpusPreparationCore

private let fixtures = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!
private struct TextGolden: Decodable {
    let name: String
    let input: String
    let json: Bool
    let samples: Int
    let target: Int
    let expected: String
    let stdout: String
}
private struct Sandbox {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("midnight-corpus-\(UUID().uuidString)")
    var output: URL { root.appendingPathComponent("output") }
    var cache: URL { root.appendingPathComponent("cache") }
    init() throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: fixtures.appendingPathComponent("reference-cache"), to: cache)
    }
    func cleanup() { try? FileManager.default.removeItem(at: root) }
}
private func manifest() throws -> PinnedManifest {
    try JSONDecoder().decode(
        PinnedManifest.self, from: Data(contentsOf: fixtures.appendingPathComponent("reference-manifest.json")))
}
private func forbiddenDownload(_ url: URL) async throws -> Data {
    Issue.record("Unexpected network request: \(url)")
    throw CorpusError("Network forbidden")
}
private actor Requests {
    var urls = [URL]()
    func record(_ url: URL) { urls.append(url) }
}

@Suite("Native corpus preparation")
struct CorpusPreparationTests {
    @Test func textMatchesFrozenPythonOutputs() throws {
        let cases = try JSONDecoder().decode(
            [TextGolden].self, from: Data(contentsOf: fixtures.appendingPathComponent("text-goldens.json")))
        for item in cases {
            let (data, characters) = try TextCorpus.payload(
                data: Data(item.input.utf8), json: item.json,
                samples: item.samples, targetCharacters: item.target)
            #expect(data == Data(item.expected.utf8), "Golden mismatch: \(item.name)")
            #expect(
                item.stdout
                    == "samples=\(item.samples) characters=\(characters)\nsource_sha256=\(corpusSHA256(Data(item.input.utf8)))\noutput_sha256=\(corpusSHA256(data))\n"
            )
        }
    }
    @Test func textErrorsAndOutputProtection() throws {
        for data in [Data("{}".utf8), Data("[{\"text\":null}]".utf8), Data("[\"text\"]".utf8), Data([0xff])] {
            #expect(throws: (any Error).self) { try TextCorpus.payload(data: data, json: true, samples: 1) }
        }
        #expect(throws: CorpusError.self) { try TextCorpus.payload(data: Data("tiny".utf8), json: false, samples: 2) }
        #expect(throws: CorpusError.self) { try TextCorpus.payload(data: Data(), json: false, samples: 0) }
        #expect(throws: CorpusError.self) { try TextCorpus.payload(data: Data(), json: false, targetCharacters: 127) }
        let box = try Sandbox()
        defer { box.cleanup() }
        let source = box.root.appendingPathComponent("source.txt")
        try Data("preserve me".utf8).write(to: source)
        let alias = box.root.appendingPathComponent("alias.txt")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: source)
        #expect(throws: CorpusError.self) { try TextCorpus.prepare(source: source, output: alias, samples: 1) }
        #expect(try String(contentsOf: source, encoding: .utf8) == "preserve me")
        try Data("existing".utf8).write(to: box.output)
        #expect(throws: CorpusError.self) { try TextCorpus.prepare(source: source, output: box.output, samples: 5) }
        #expect(try String(contentsOf: box.output, encoding: .utf8) == "existing")
        _ = try TextCorpus.prepare(source: source, output: box.output, samples: 1)
        #expect(try String(contentsOf: box.output, encoding: .utf8).contains("preserve me"))
    }
    @Test func exactReferenceFilesAndReuse() async throws {
        let box = try Sandbox()
        defer { box.cleanup() }
        let result = try await ReferenceCorpus.prepare(
            output: box.output, cache: box.cache, offline: true, manifest: manifest(), fetch: forbiddenDownload)
        let expected = fixtures.appendingPathComponent("reference-expected")
        let paths = try FileManager.default.contentsOfDirectory(at: expected, includingPropertiesForKeys: nil)
        #expect(paths.count == 6)
        var dates = [String: Date]()
        for path in paths {
            let actual = box.output.appendingPathComponent(path.lastPathComponent)
            #expect(try Data(contentsOf: actual) == Data(contentsOf: path), "Mismatch: \(path.lastPathComponent)")
            #expect(result.files[path.lastPathComponent] == corpusSHA256(try Data(contentsOf: path)))
            dates[path.lastPathComponent] = try actual.resourceValues(forKeys: [.contentModificationDateKey])
                .contentModificationDate
        }
        let again = try await ReferenceCorpus.prepare(
            output: box.output, offline: true, manifest: manifest(), fetch: forbiddenDownload)
        #expect(again.sha256 == result.sha256)
        for (name, date) in dates {
            #expect(
                try box.output.appendingPathComponent(name).resourceValues(forKeys: [.contentModificationDateKey])
                    .contentModificationDate == date)
        }
    }
    @Test func corruptCacheNeverFallsBackToDownload() async throws {
        let box = try Sandbox()
        defer { box.cleanup() }
        try Data("tampered".utf8).write(to: box.cache.appendingPathComponent("gsm8k-source.jsonl"))
        await #expect(throws: CorpusError.self) {
            try await ReferenceCorpus.prepare(
                output: box.output, cache: box.cache, manifest: manifest(), fetch: forbiddenDownload)
        }
        #expect(!FileManager.default.fileExists(atPath: box.output.path))
    }
    @Test func missingOfflineInputsNeverDownload() async throws {
        let box = try Sandbox()
        defer { box.cleanup() }
        await #expect(throws: CorpusError.self) {
            try await ReferenceCorpus.prepare(
                output: box.output, offline: true, manifest: manifest(), fetch: forbiddenDownload)
        }
        #expect(!FileManager.default.fileExists(atPath: box.output.path))
    }
    @Test func conflictsArePreflightedBeforeAnyWrites() async throws {
        let box = try Sandbox()
        defer { box.cleanup() }
        try FileManager.default.createDirectory(at: box.output, withIntermediateDirectories: true)
        let conflict = box.output.appendingPathComponent("provenance.json")
        try Data("existing work".utf8).write(to: conflict)
        await #expect(throws: CorpusError.self) {
            try await ReferenceCorpus.prepare(
                output: box.output, cache: box.cache, offline: true, manifest: manifest(), fetch: forbiddenDownload)
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: box.output.path) == ["provenance.json"])
        #expect(try String(contentsOf: conflict, encoding: .utf8) == "existing work")
    }
    @Test func derivedAndCombinedHashesMustMatch() async throws {
        let box = try Sandbox()
        defer { box.cleanup() }
        var invalid = try manifest()
        invalid.sources[0].corpus_sha256 = String(repeating: "0", count: 64)
        await #expect(throws: CorpusError.self) {
            try await ReferenceCorpus.prepare(
                output: box.output, cache: box.cache, offline: true, manifest: invalid, fetch: forbiddenDownload)
        }
        invalid = try manifest()
        invalid.combined_sha256 = String(repeating: "0", count: 64)
        await #expect(throws: CorpusError.self) {
            try await ReferenceCorpus.prepare(
                output: box.output, cache: box.cache, offline: true, manifest: invalid, fetch: forbiddenDownload)
        }
        #expect(!FileManager.default.fileExists(atPath: box.output.path))
    }
    @Test func downloadsUsePinnedURLsAndVerifyBeforeWriting() async throws {
        let box = try Sandbox()
        defer { box.cleanup() }
        let pins = try manifest()
        let requests = Requests()
        let inputs = try Dictionary(
            uniqueKeysWithValues: pins.sources.map {
                ($0.url, try Data(contentsOf: box.cache.appendingPathComponent("\($0.name)-source.jsonl")))
            })
        let result = try await ReferenceCorpus.prepare(output: box.output, manifest: pins) { url in
            await requests.record(url)
            return try #require(inputs[url])
        }
        #expect(await requests.urls == pins.sources.map(\.url))
        #expect(result.sample_count == 128)
        let badOutput = box.root.appendingPathComponent("bad")
        await #expect(throws: CorpusError.self) {
            try await ReferenceCorpus.prepare(output: badOutput, manifest: pins) { _ in Data("bad download".utf8) }
        }
        #expect(!FileManager.default.fileExists(atPath: badOutput.path))
    }
    @Test func rejectsReorderedMissingOrMalformedRecords() throws {
        let original = try String(
            contentsOf: fixtures.appendingPathComponent("reference-cache/mbpp-source.jsonl"), encoding: .utf8)
        let lines = original.split(separator: "\n")
        #expect(throws: CorpusError.self) {
            try ReferenceCorpus.payload(name: "mbpp", data: Data(lines.reversed().joined(separator: "\n").utf8))
        }
        #expect(throws: CorpusError.self) {
            try ReferenceCorpus.payload(name: "mbpp", data: Data("{\"task_id\":true}".utf8))
        }
        #expect(throws: CorpusError.self) {
            try ReferenceCorpus.payload(name: "gsm8k", data: Data("{\"question\":\"a\",\"answer\":\"b\"}".utf8))
        }
        #expect(throws: CorpusError.self) { try ReferenceCorpus.payload(name: "other", data: Data()) }
    }
    @Test func bundledPinsRemainOriginal() throws {
        let pins = try PinnedManifest.bundled()
        #expect(pins.sources.map(\.name) == ["mbpp", "gsm8k"])
        #expect(pins.combined_sha256 == "5fc84a9794338a138e7284f415018323aa6bfec54227d09db8a95ad562e532f6")
    }
    @Test func portableHashMatchesKnownAnswersAndPlatformHash() {
        let vectors = [
            ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            ("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            (
                String(repeating: "a", count: 1_000_000),
                "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
            ),
        ]
        for (input, hash) in vectors {
            #expect(portableCorpusSHA256(Data(input.utf8)) == hash)
            #expect(corpusSHA256(Data(input.utf8)) == hash)
        }
        for count in [55, 56, 63, 64, 65, 119, 120, 127, 128, 1025] {
            let data = Data((0..<count).map { UInt8(truncatingIfNeeded: $0) })
            #expect(portableCorpusSHA256(data) == corpusSHA256(data))
        }
    }
}
