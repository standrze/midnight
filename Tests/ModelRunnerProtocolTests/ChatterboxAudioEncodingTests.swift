#if os(macOS)
import Foundation
@testable import ModelRunnerCore
import Testing

@Suite("Chatterbox OpenAI audio encodings")
struct ChatterboxAudioEncodingTests {
    @Test func rawPCMContract() throws {
        let request = LocalSpeechSynthesisRequest(input: "test", voiceID: "default", format: .pcm,
            pcmEncoding: .signedInt16LittleEndian)
        let data = try ChatterboxAudioEncoding.encode(samples: [0, 1, -1], sampleRate: 24000, request: request, ffmpeg: nil)
        #expect(Array(data) == [0, 0, 255, 127, 1, 128])
    }

    @Test(arguments: LocalSpeechAudioFormat.allCases)
    func codecs(_ format: LocalSpeechAudioFormat) throws {
        guard let ffmpeg = try ChatterboxAudioEncoding.executable(configuredPath: nil) else { return }
        let samples = (0..<2400).map { Float(sin(Double($0) * 2 * .pi * 440 / 24000)) * 0.2 }
        let request = LocalSpeechSynthesisRequest(input: "test", voiceID: "default", format: format,
            pcmEncoding: .signedInt16LittleEndian, speed: 0.25)
        let data = try ChatterboxAudioEncoding.encode(samples: samples, sampleRate: 24000, request: request, ffmpeg: ffmpeg)
        #expect(!data.isEmpty)
        switch format {
        case .wav: #expect(String(decoding: data.prefix(4), as: UTF8.self) == "RIFF")
        case .opus: #expect(String(decoding: data.prefix(4), as: UTF8.self) == "OggS")
        case .flac: #expect(String(decoding: data.prefix(4), as: UTF8.self) == "fLaC")
        case .mp3: #expect(String(decoding: data.prefix(3), as: UTF8.self) == "ID3")
        case .aac: #expect(data.first == 0xff)
        case .pcm: #expect(data.count % 2 == 0)
        }
    }
}
#endif
