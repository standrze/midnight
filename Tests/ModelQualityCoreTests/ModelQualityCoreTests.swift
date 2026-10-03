import Foundation
import Testing

@testable import ModelQualityCore

@Suite("Generic teacher-forced quality benchmark support")
struct ModelQualityCoreTests {
  @Test("JSONL corpus parsing preserves authored order and metadata")
  func parsesCorpus() throws {
    let url = try temporaryCorpus(
      """
      {"id":"first","category":"prose","text":"Alpha beta."}

      {"id":"second","text":"Gamma delta."}
      """
    )
    defer { try? FileManager.default.removeItem(at: url) }

    let samples = try ModelQualityCore.loadCorpus(from: url)

    #expect(samples == [
      ModelQualityCorpusSample(id: "first", category: "prose", text: "Alpha beta."),
      ModelQualityCorpusSample(id: "second", text: "Gamma delta."),
    ])
    #expect(ModelQualityCore.corpusFingerprint(samples).hasPrefix("fnv1a64:"))
  }

  @Test("Malformed, duplicate, and blank corpus records fail closed")
  func rejectsInvalidCorpus() throws {
    let malformed = try temporaryCorpus("{not-json}\n")
    defer { try? FileManager.default.removeItem(at: malformed) }
    #expect(throws: ModelQualityCoreError.self) {
      try ModelQualityCore.loadCorpus(from: malformed)
    }

    let duplicate = try temporaryCorpus(
      """
      {"id":"same","text":"One."}
      {"id":"same","text":"Two."}
      """
    )
    defer { try? FileManager.default.removeItem(at: duplicate) }
    do {
      _ = try ModelQualityCore.loadCorpus(from: duplicate)
      Issue.record("Expected the duplicate id to be rejected")
    } catch let error as ModelQualityCoreError {
      #expect(error == .duplicateID(id: "same", line: 2))
    }

    let blank = try temporaryCorpus("{\"id\":\"blank\",\"text\":\"  \\n\"}\n")
    defer { try? FileManager.default.removeItem(at: blank) }
    #expect(throws: ModelQualityCoreError.self) {
      try ModelQualityCore.loadCorpus(from: blank)
    }
  }

  @Test("Token bounding and fingerprints are deterministic and order-sensitive")
  func boundsAndFingerprintsTokens() throws {
    let bounded = try ModelQualityCore.boundedTokens([11, 22, 33, 44], maximumCount: 3)

    #expect(bounded == [11, 22, 33])
    #expect(throws: ModelQualityCoreError.self) {
      try ModelQualityCore.boundedTokens([11, 22], maximumCount: 1)
    }
    #expect(
      ModelQualityCore.tokenIDFingerprint(bounded)
        == ModelQualityCore.tokenIDFingerprint([11, 22, 33])
    )
    #expect(
      ModelQualityCore.tokenIDFingerprint(bounded)
        != ModelQualityCore.tokenIDFingerprint([33, 22, 11])
    )
    #expect(
      ModelQualityCore.combinedTokenIDFingerprint([
        ModelQualityTokenSequence(sampleID: "a", tokenIDs: [1, 2]),
        ModelQualityTokenSequence(sampleID: "b", tokenIDs: [3]),
      ])
        != ModelQualityCore.combinedTokenIDFingerprint([
          ModelQualityTokenSequence(sampleID: "a", tokenIDs: [1]),
          ModelQualityTokenSequence(sampleID: "b", tokenIDs: [2, 3]),
        ])
    )
  }

  @Test("Aggregate NLL is weighted by scored tokens")
  func computesTokenWeightedSummary() throws {
    let short = try ModelQualitySampleResult(
      id: "short",
      originalTokenCount: 3,
      evaluatedTokenCount: 3,
      tokenIDFingerprint: "short-fingerprint",
      nllSum: 2
    )
    let long = try ModelQualitySampleResult(
      id: "long",
      originalTokenCount: 9,
      evaluatedTokenCount: 9,
      tokenIDFingerprint: "long-fingerprint",
      nllSum: 16
    )

    let summary = try ModelQualityCore.summarize([short, long])

    #expect(summary.sampleCount == 2)
    #expect(summary.scoredTokenCount == 10)
    #expect(abs(summary.tokenWeightedNLL - 1.8) < 1e-12)
    #expect(abs(summary.perplexity - exp(1.8)) < 1e-12)
    #expect(summary.tokenWeightedNLL != (short.nll + long.nll) / 2)
  }

  @Test("The checked-in smoke corpus is valid and nontrivial")
  func validatesAuthoredCorpus() throws {
    let repositoryRoot = URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
    let corpusURL = repositoryRoot
      .appendingPathComponent("Benchmarks/Quality/general-smoke.jsonl")

    let samples = try ModelQualityCore.loadCorpus(from: corpusURL)

    #expect(samples.count == 10)
    #expect(Set(samples.map(\.id)).count == samples.count)
    #expect(Set(samples.compactMap(\.category)).count >= 6)
  }

  private func temporaryCorpus(_ contents: String) throws -> URL {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("model-quality-\(UUID().uuidString).jsonl")
    try Data(contents.utf8).write(to: url, options: .atomic)
    return url
  }
}
