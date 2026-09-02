import Foundation
import Testing
@testable import ModelRunnerCore

@Suite("Voxtral PCM and WAV encoding")
struct VoxtralAudioEncodingTests {
  @Test("Writes a valid mono 24 kHz signed-16 WAV")
  func signed16Wave() throws {
    let data = try VoxtralAudioEncoding.encode(
      samples: [-1, 0, 1],
      sampleRate: 24_000,
      format: .wav,
      pcmEncoding: .signedInt16LittleEndian
    )

    #expect(String(decoding: data[0 ..< 4], as: UTF8.self) == "RIFF")
    #expect(String(decoding: data[8 ..< 12], as: UTF8.self) == "WAVE")
    #expect(String(decoding: data[12 ..< 16], as: UTF8.self) == "fmt ")
    #expect(data.littleEndianUInt16(at: 20) == 1)
    #expect(data.littleEndianUInt16(at: 22) == 1)
    #expect(data.littleEndianUInt32(at: 24) == 24_000)
    #expect(data.littleEndianUInt16(at: 34) == 16)
    #expect(String(decoding: data[36 ..< 40], as: UTF8.self) == "data")
    #expect(data.littleEndianUInt32(at: 40) == 6)
    #expect(data.count == 50)
    #expect(Int16(bitPattern: data.littleEndianUInt16(at: 44)) == -32_767)
    #expect(Int16(bitPattern: data.littleEndianUInt16(at: 46)) == 0)
    #expect(Int16(bitPattern: data.littleEndianUInt16(at: 48)) == 32_767)
  }

  @Test("Writes raw float32 PCM and rejects mislabeled compressed formats")
  func floatPCMAndUnsupportedFormats() throws {
    let data = try VoxtralAudioEncoding.encode(
      samples: [0.25, -0.5],
      sampleRate: 24_000,
      format: .pcm,
      pcmEncoding: .float32LittleEndian
    )
    #expect(data.count == 8)
    #expect(Float(bitPattern: data.littleEndianUInt32(at: 0)) == 0.25)
    #expect(Float(bitPattern: data.littleEndianUInt32(at: 4)) == -0.5)

    #expect(throws: VoxtralTTSError.self) {
      try VoxtralAudioEncoding.encode(
        samples: [0],
        sampleRate: 24_000,
        format: .mp3,
        pcmEncoding: .signedInt16LittleEndian
      )
    }
  }
}

private extension Data {
  func littleEndianUInt16(at offset: Int) -> UInt16 {
    UInt16(self[offset]) | UInt16(self[offset + 1]) << 8
  }

  func littleEndianUInt32(at offset: Int) -> UInt32 {
    UInt32(self[offset])
      | UInt32(self[offset + 1]) << 8
      | UInt32(self[offset + 2]) << 16
      | UInt32(self[offset + 3]) << 24
  }
}
