import Foundation

/// Native reader and encoder for Mistral's `tekken.json` format.
///
/// Tekken uses the same byte-pair encoding rule as tiktoken: split text with
/// the checkpoint's ICU expression, then repeatedly merge the adjacent byte
/// pair whose resulting token has the lowest vocabulary rank. Text vocabulary
/// ranks are offset by the reserved special-token count.
final class VoxtralTekkenTokenizer: @unchecked Sendable {
  let voiceTokenCounts: [String: Int]

  private let expression: NSRegularExpression
  private let rankByBytes: [Data: Int]
  private let specialTokenIDs: [String: Int]
  private let specialTokenCount: Int
  private let vocabularySize: Int

  init(modelDirectory: URL) throws {
    let tokenizerURL = modelDirectory.appendingPathComponent("tekken.json")
    guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
      throw VoxtralTTSError.missingFile(tokenizerURL.path)
    }

    let root: Root
    do {
      root = try JSONDecoder().decode(Root.self, from: Data(contentsOf: tokenizerURL))
    } catch {
      throw VoxtralTTSError.invalidTokenizer(
        "could not decode \(tokenizerURL.lastPathComponent): \(error.localizedDescription)"
      )
    }

    guard root.configuration.defaultVocabularySize > 0,
      root.configuration.defaultSpecialTokenCount > 0,
      root.configuration.defaultVocabularySize > root.configuration.defaultSpecialTokenCount
    else {
      throw VoxtralTTSError.invalidTokenizer("default vocabulary sizes are inconsistent")
    }

    let textVocabularyCount =
      root.configuration.defaultVocabularySize - root.configuration.defaultSpecialTokenCount
    guard root.vocabulary.count >= textVocabularyCount else {
      throw VoxtralTTSError.invalidTokenizer(
        "tekken.json contains \(root.vocabulary.count) text ranks; expected at least \(textVocabularyCount)"
      )
    }

    do {
      expression = try NSRegularExpression(pattern: root.configuration.pattern)
    } catch {
      throw VoxtralTTSError.invalidTokenizer(
        "invalid ICU tokenization pattern: \(error.localizedDescription)"
      )
    }

    var ranks = [Data: Int](minimumCapacity: textVocabularyCount)
    for expectedRank in 0..<textVocabularyCount {
      let item = root.vocabulary[expectedRank]
      guard item.rank == expectedRank else {
        throw VoxtralTTSError.invalidTokenizer(
          "text vocabulary rank \(item.rank) appears at position \(expectedRank)"
        )
      }
      guard let bytes = Data(base64Encoded: item.tokenBytes) else {
        throw VoxtralTTSError.invalidTokenizer(
          "text vocabulary rank \(item.rank) has invalid base64 bytes"
        )
      }
      guard !bytes.isEmpty else {
        throw VoxtralTTSError.invalidTokenizer(
          "text vocabulary rank \(item.rank) is empty"
        )
      }
      guard ranks.updateValue(item.rank, forKey: bytes) == nil else {
        throw VoxtralTTSError.invalidTokenizer(
          "text vocabulary contains duplicate byte sequences"
        )
      }
    }

    // Tiktoken requires every raw byte to be representable before merges.
    for byte in UInt8.min...UInt8.max {
      guard ranks[Data([byte])] != nil else {
        throw VoxtralTTSError.invalidTokenizer(
          "text vocabulary is missing raw byte rank \(byte)"
        )
      }
    }

    var specials = [String: Int](minimumCapacity: root.specialTokens.count)
    var usedSpecialRanks = Set<Int>()
    for token in root.specialTokens {
      guard (0..<root.configuration.defaultSpecialTokenCount).contains(token.rank) else {
        throw VoxtralTTSError.invalidTokenizer(
          "special token \(token.tokenString) has out-of-range rank \(token.rank)"
        )
      }
      guard specials.updateValue(token.rank, forKey: token.tokenString) == nil,
        usedSpecialRanks.insert(token.rank).inserted
      else {
        throw VoxtralTTSError.invalidTokenizer("special tokens or ranks are duplicated")
      }
    }

    var voices = [String: Int](minimumCapacity: root.audio.voiceTokenCounts.count)
    for (voice, count) in root.audio.voiceTokenCounts {
      guard !voice.isEmpty, count > 0 else {
        throw VoxtralTTSError.invalidTokenizer(
          "voice audio-token counts must use non-empty names and positive values"
        )
      }
      voices[voice] = count
    }
    guard !voices.isEmpty else {
      throw VoxtralTTSError.invalidTokenizer("voice_num_audio_tokens must not be empty")
    }

    rankByBytes = ranks
    specialTokenIDs = specials
    specialTokenCount = root.configuration.defaultSpecialTokenCount
    vocabularySize = root.configuration.defaultVocabularySize
    voiceTokenCounts = voices
  }

  func encode(_ text: String) throws -> [Int] {
    guard !text.isEmpty else { return [] }

    let source = text as NSString
    let sourceRange = NSRange(location: 0, length: source.length)
    let matches = expression.matches(in: text, range: sourceRange)
    var cursor = 0
    var result: [Int] = []

    for match in matches {
      guard match.range.length > 0, match.range.location == cursor else {
        throw VoxtralTTSError.invalidTokenizer(
          "the ICU pattern did not partition the complete input at UTF-16 offset \(cursor)"
        )
      }
      let piece = source.substring(with: match.range)
      try appendEncodedPiece(Data(piece.utf8), to: &result)
      cursor = match.range.location + match.range.length
    }

    guard cursor == source.length else {
      throw VoxtralTTSError.invalidTokenizer(
        "the ICU pattern left input unmatched at UTF-16 offset \(cursor)"
      )
    }
    return result
  }

  func speechPrompt(
    text: String,
    voice: String,
    configuration: VoxtralTTSConfiguration,
    voiceEmbeddingRows: Int
  ) throws -> [Int] {
    guard let expectedRows = voiceTokenCounts[voice] else {
      throw VoxtralTTSError.voiceNotFound(voice)
    }
    guard voiceEmbeddingRows == expectedRows else {
      throw VoxtralTTSError.invalidTokenizer(
        "voice \(voice) declares \(expectedRows) audio tokens but its embedding has \(voiceEmbeddingRows) rows"
      )
    }

    let bos = try specialTokenID("<s>")
    let audio = try specialTokenID("[AUDIO]")
    let beginAudio = try specialTokenID("[BEGIN_AUDIO]")
    let nextAudioText = try specialTokenID("[NEXT_AUDIO_TEXT]")
    let repeatAudioText = try specialTokenID("[REPEAT_AUDIO_TEXT]")

    guard bos == configuration.beginningOfSequenceTokenID,
      audio == configuration.audioTokenID,
      beginAudio == configuration.beginAudioTokenID
    else {
      throw VoxtralTTSError.invalidTokenizer(
        "config.json audio prompt token IDs do not match tekken.json"
      )
    }

    let textTokens = try encode(text)
    var prompt: [Int] = []
    prompt.reserveCapacity(5 + expectedRows + textTokens.count)
    prompt.append(bos)
    prompt.append(beginAudio)
    prompt.append(contentsOf: repeatElement(audio, count: expectedRows))
    prompt.append(nextAudioText)
    prompt.append(contentsOf: textTokens)
    prompt.append(repeatAudioText)
    prompt.append(beginAudio)
    return prompt
  }

  private func specialTokenID(_ token: String) throws -> Int {
    guard let id = specialTokenIDs[token] else {
      throw VoxtralTTSError.invalidTokenizer("required special token \(token) is missing")
    }
    return id
  }

  private func appendEncodedPiece(_ bytes: Data, to output: inout [Int]) throws {
    guard !bytes.isEmpty else { return }
    if let rank = rankByBytes[bytes] {
      try appendTextRank(rank, to: &output)
      return
    }

    var parts = bytes.map { Data([$0]) }
    while parts.count > 1 {
      var bestIndex: Int?
      var bestRank = Int.max
      for index in 0..<(parts.count - 1) {
        var candidate = parts[index]
        candidate.append(parts[index + 1])
        if let rank = rankByBytes[candidate], rank < bestRank {
          bestRank = rank
          bestIndex = index
        }
      }
      guard let bestIndex else { break }
      parts[bestIndex].append(parts[bestIndex + 1])
      parts.remove(at: bestIndex + 1)
    }

    for part in parts {
      guard let rank = rankByBytes[part] else {
        throw VoxtralTTSError.invalidTokenizer(
          "byte-pair encoding produced a sequence absent from the vocabulary"
        )
      }
      try appendTextRank(rank, to: &output)
    }
  }

  private func appendTextRank(_ rank: Int, to output: inout [Int]) throws {
    let tokenID = rank + specialTokenCount
    guard tokenID >= specialTokenCount, tokenID < vocabularySize else {
      throw VoxtralTTSError.invalidTokenizer(
        "text rank \(rank) maps outside the configured vocabulary"
      )
    }
    output.append(tokenID)
  }
}

private extension VoxtralTekkenTokenizer {
  struct Root: Decodable {
    let configuration: Configuration
    let vocabulary: [VocabularyItem]
    let specialTokens: [SpecialToken]
    let audio: Audio

    enum CodingKeys: String, CodingKey {
      case configuration = "config"
      case vocabulary = "vocab"
      case specialTokens = "special_tokens"
      case audio
    }
  }

  struct Configuration: Decodable {
    let pattern: String
    let defaultVocabularySize: Int
    let defaultSpecialTokenCount: Int

    enum CodingKeys: String, CodingKey {
      case pattern
      case defaultVocabularySize = "default_vocab_size"
      case defaultSpecialTokenCount = "default_num_special_tokens"
    }
  }

  struct VocabularyItem: Decodable {
    let rank: Int
    let tokenBytes: String

    enum CodingKeys: String, CodingKey {
      case rank
      case tokenBytes = "token_bytes"
    }
  }

  struct SpecialToken: Decodable {
    let rank: Int
    let tokenString: String

    enum CodingKeys: String, CodingKey {
      case rank
      case tokenString = "token_str"
    }
  }

  struct Audio: Decodable {
    let voiceTokenCounts: [String: Int]

    enum CodingKeys: String, CodingKey {
      case voiceTokenCounts = "voice_num_audio_tokens"
    }
  }
}
