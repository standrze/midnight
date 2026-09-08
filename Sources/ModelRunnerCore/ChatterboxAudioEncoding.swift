#if os(macOS)
import Foundation

/// Optional codec support for Chatterbox only. WAV/PCM remain dependency-free.
enum ChatterboxAudioEncoding {
    static func executable(configuredPath: String?) throws -> URL? {
        if let configuredPath {
            let path = NSString(string: configuredPath).expandingTildeInPath
            guard FileManager.default.isExecutableFile(atPath: path) else {
                throw Failure("ffmpeg_path is not executable: \(path)")
            }
            return URL(fileURLWithPath: path)
        }
        return ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]
            .first(where: { FileManager.default.isExecutableFile(atPath: $0) })
            .map { URL(fileURLWithPath: $0) }
    }

    static func encode(samples: [Float], sampleRate: Int, request: LocalSpeechSynthesisRequest,
                       ffmpeg: URL?) throws -> Data {
        if request.speed == 1, request.format == .wav || request.format == .pcm {
            return try VoxtralAudioEncoding.encode(samples: samples, sampleRate: sampleRate,
                format: request.format, pcmEncoding: request.pcmEncoding)
        }
        guard let ffmpeg else { throw Failure("This format or speed requires FFmpeg") }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("midnight-audio-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = directory.appendingPathComponent("input.wav")
        let output = directory.appendingPathComponent("output")
        try VoxtralAudioEncoding.encode(samples: samples, sampleRate: sampleRate,
            format: .wav, pcmEncoding: .float32LittleEndian).write(to: input)
        let process = Process()
        process.executableURL = ffmpeg
        var args = ["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", input.path]
        if request.speed != 1 {
            let filter = request.speed < 0.5 ? "atempo=0.5,atempo=\(request.speed * 2)" : "atempo=\(request.speed)"
            args += ["-af", filter]
        }
        // OpenAI raw PCM is signed 16-bit mono, 24 kHz, without a header.
        args += ["-ac", "1", "-ar", request.format == .opus ? "48000" : "24000"]
        switch request.format {
        case .mp3: args += ["-c:a", "libmp3lame", "-f", "mp3"]
        case .opus: args += ["-c:a", "libopus", "-f", "ogg"]
        case .aac: args += ["-c:a", "aac", "-f", "adts"]
        case .flac: args += ["-c:a", "flac", "-f", "flac"]
        case .wav: args += ["-c:a", request.pcmEncoding == .signedInt16LittleEndian ? "pcm_s16le" : "pcm_f32le", "-f", "wav"]
        case .pcm: args += ["-f", request.pcmEncoding == .signedInt16LittleEndian ? "s16le" : "f32le"]
        }
        process.arguments = args + [output.path]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw Failure("FFmpeg could not encode \(request.format.rawValue); check installed codecs")
        }
        return try Data(contentsOf: output)
    }

    private struct Failure: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}
#endif
