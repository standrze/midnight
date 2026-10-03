import Foundation
import Testing
@testable import ModelRunnerCore

@Suite("Voxtral native Tekken tokenizer")
struct VoxtralTekkenTokenizerTests {
  @Test("Uses ICU splitting, lowest-rank byte merges, and the special-token offset")
  func encodesText() throws {
    let directory = try makeFixture()
    defer { try? FileManager.default.removeItem(at: directory) }
    let tokenizer = try VoxtralTekkenTokenizer(modelDirectory: directory)

    #expect(tokenizer.voiceTokenCounts == ["test_voice": 2])
    #expect(try tokenizer.encode("hello hello") == [265, 38, 265])
    #expect(try tokenizer.encode("é") == [201, 175])

    // Rank 260 exists in the JSON fixture but is beyond the configured inner
    // vocabulary and must not become an available token.
    #expect(try tokenizer.encode("world") == [125, 117, 120, 114, 106])
  }

  @Test("Builds the exact preset-voice speech prompt")
  func buildsSpeechPrompt() throws {
    let directory = try makeFixture()
    defer { try? FileManager.default.removeItem(at: directory) }
    let tokenizer = try VoxtralTekkenTokenizer(modelDirectory: directory)
    let configuration = try VoxtralTTSConfiguration(modelDirectory: directory)

    let prompt = try tokenizer.speechPrompt(
      text: "hello",
      voice: "test_voice",
      configuration: configuration,
      voiceEmbeddingRows: 2
    )

    #expect(prompt == [1, 3, 2, 2, 4, 265, 5, 3])
  }

  @Test("Rejects a voice embedding whose row count differs from Tekken metadata")
  func validatesVoiceEmbeddingRows() throws {
    let directory = try makeFixture()
    defer { try? FileManager.default.removeItem(at: directory) }
    let tokenizer = try VoxtralTekkenTokenizer(modelDirectory: directory)
    let configuration = try VoxtralTTSConfiguration(modelDirectory: directory)

    do {
      _ = try tokenizer.speechPrompt(
        text: "hello",
        voice: "test_voice",
        configuration: configuration,
        voiceEmbeddingRows: 1
      )
      Issue.record("Expected mismatched voice embedding rows to be rejected")
    } catch {
      #expect(error.localizedDescription.contains("declares 2 audio tokens"))
      #expect(error.localizedDescription.contains("embedding has 1 rows"))
    }
  }

  private func makeFixture() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

    var vocabulary: [[String: Any]] = (0...255).map { rank in
      [
        "rank": rank,
        "token_bytes": Data([UInt8(rank)]).base64EncodedString(),
        "token_str": NSNull(),
      ]
    }
    for (rank, token) in ["he", "ll", "hell", "hello"].enumerated() {
      vocabulary.append([
        "rank": rank + 256,
        "token_bytes": Data(token.utf8).base64EncodedString(),
        "token_str": token,
      ])
    }
    vocabulary.append([
      "rank": 260,
      "token_bytes": Data("world".utf8).base64EncodedString(),
      "token_str": "world",
    ])

    let tekken: [String: Any] = [
      "config": [
        "pattern": "\\S+|\\s+",
        "default_vocab_size": 266,
        "default_num_special_tokens": 6,
      ],
      "vocab": vocabulary,
      "special_tokens": [
        ["rank": 0, "token_str": "<unk>", "is_control": true],
        ["rank": 1, "token_str": "<s>", "is_control": true],
        ["rank": 2, "token_str": "[AUDIO]", "is_control": true],
        ["rank": 3, "token_str": "[BEGIN_AUDIO]", "is_control": true],
        ["rank": 4, "token_str": "[NEXT_AUDIO_TEXT]", "is_control": true],
        ["rank": 5, "token_str": "[REPEAT_AUDIO_TEXT]", "is_control": true],
      ],
      "audio": ["voice_num_audio_tokens": ["test_voice": 2]],
    ]
    try JSONSerialization.data(withJSONObject: tekken)
      .write(to: directory.appendingPathComponent("tekken.json"))

    let configuration: [String: Any] = [
      "model_type": "voxtral_tts",
      "dim": 8,
      "n_layers": 1,
      "head_dim": 4,
      "hidden_dim": 16,
      "n_heads": 2,
      "n_kv_heads": 1,
      "vocab_size": 266,
      "rope_theta": 1_000_000,
      "norm_eps": 0.00001,
      "use_biases": false,
      "max_position_embeddings": 128,
      "multimodal": [
        "bos_token_id": 1,
        "audio_model_args": [
          "semantic_codebook_size": 8,
          "acoustic_codebook_size": 3,
          "n_acoustic_codebook": 2,
          "audio_token_id": 2,
          "begin_audio_token_id": 3,
          "audio_encoding_args": ["sampling_rate": 24_000, "frame_rate": 12.5],
          "acoustic_transformer_args": [
            "dim": 8,
            "n_layers": 1,
            "head_dim": 4,
            "hidden_dim": 16,
            "n_heads": 2,
            "n_kv_heads": 1,
            "sigma_max": 1.0,
          ],
        ],
        "audio_tokenizer_args": [
          "pretransform_patch_size": 4,
          "patch_proj_kernel_size": 3,
          "semantic_dim": 4,
          "acoustic_dim": 2,
          "dim": 8,
          "hidden_dim": 16,
          "head_dim": 4,
          "n_heads": 2,
          "n_kv_heads": 1,
          "qk_norm_eps": 0.000001,
          "norm_eps": 0.01,
          "decoder_transformer_lengths_str": "1",
          "decoder_convs_kernels_str": "3",
          "decoder_convs_strides_str": "1",
        ],
      ],
    ]
    try JSONSerialization.data(withJSONObject: configuration)
      .write(to: directory.appendingPathComponent("config.json"))
    return directory
  }
}
