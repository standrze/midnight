import Foundation
import ModelQualityCore
#if canImport(CryptoKit)
import CryptoKit
#endif

struct GenerationInput: Decodable, Sendable {
  let id: String
  let category: String
  let prompt: String

  private struct Field: CodingKey {
    let stringValue: String
    let intValue: Int? = nil
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
  }

  init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: Field.self)
    let allowed: Set<String> = ["id", "category", "prompt", "metadata"]
    let unknown = Set(container.allKeys.map(\.stringValue)).subtracting(allowed)
    guard unknown.isEmpty else {
      throw GenerationBenchmarkError.invalidInput(
        "public records permit only id, category, prompt and optional metadata; unexpected fields: \(unknown.sorted().joined(separator: ", "))")
    }
    id = try container.decode(String.self, forKey: Field(stringValue: "id")!)
    category = try container.decode(String.self, forKey: Field(stringValue: "category")!)
    prompt = try container.decode(String.self, forKey: Field(stringValue: "prompt")!)
    guard !id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      !category.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    else { throw GenerationBenchmarkError.invalidInput("id, category and prompt must be nonblank strings") }
    // Public metadata is intentionally not forwarded to the model or copied
    // into its result. A scorer can join it from the fingerprinted public corpus.
  }
}

enum GenerationBenchmarkError: Error, LocalizedError {
  case invalidInput(String)
  case invalidResult(String)
  case sampleFailed(String, String)

  var errorDescription: String? {
    switch self {
    case .invalidInput(let message): "Invalid generation benchmark input: \(message)"
    case .invalidResult(let message): "Invalid generation result: \(message)"
    case .sampleFailed(let id, let message): "Generation sample '\(id)' failed: \(message)"
    }
  }
}

func readGenerationInputs(_ data: Data) throws -> [GenerationInput] {
  guard let text = String(data: data, encoding: .utf8) else {
    throw GenerationBenchmarkError.invalidInput("corpus must be UTF-8 JSONL")
  }
  var inputs = [GenerationInput]()
  var ids = Set<String>()
  for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
    if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
    let record: GenerationInput
    do { record = try JSONDecoder().decode(GenerationInput.self, from: Data(line.utf8)) }
    catch {
      throw GenerationBenchmarkError.invalidInput("JSONL line \(index + 1): \(error.localizedDescription)")
    }
    guard ids.insert(record.id).inserted else {
      throw GenerationBenchmarkError.invalidInput("duplicate sample id '\(record.id)' at JSONL line \(index + 1)")
    }
    inputs.append(record)
  }
  guard !inputs.isEmpty else { throw GenerationBenchmarkError.invalidInput("corpus contains no records") }
  return inputs
}

func generationCorpusFingerprint(_ records: [GenerationInput]) -> String {
  ModelQualityCore.corpusFingerprint(records.map {
    ModelQualityCorpusSample(id: $0.id, category: $0.category, text: $0.prompt)
  })
}

func generationSHA256(_ data: Data) -> String? {
  #if canImport(CryptoKit)
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  #else
    // Corpus and prepared-token FNV fingerprints remain available on builds
    // without CryptoKit. A missing SHA256 is explicit, never relabeled FNV.
    nil
  #endif
}

func generationURL(_ path: String) -> URL {
  URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
    .standardizedFileURL.resolvingSymlinksInPath()
}

func generationMilliseconds(_ duration: Duration) -> Double {
  let components = duration.components
  return Double(components.seconds) * 1_000 + Double(components.attoseconds) / 1e15
}
