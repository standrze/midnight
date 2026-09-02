import Foundation

enum VoxtralAudioEncoding {
  static func encode(
    samples: [Float],
    sampleRate: Int,
    format: LocalSpeechAudioFormat,
    pcmEncoding: LocalSpeechPCMEncoding
  ) throws -> Data {
    guard format == .wav || format == .pcm else {
      throw VoxtralTTSError.unsupportedAudioFormat(format.rawValue)
    }

    let payload: Data
    let bitsPerSample: UInt16
    let waveFormat: UInt16
    switch pcmEncoding {
    case .float32LittleEndian:
      bitsPerSample = 32
      waveFormat = 3
      payload = float32Data(samples)
    case .signedInt16LittleEndian:
      bitsPerSample = 16
      waveFormat = 1
      payload = signedInt16Data(samples)
    }

    guard format == .wav else { return payload }
    return wave(
      payload: payload,
      sampleRate: sampleRate,
      bitsPerSample: bitsPerSample,
      format: waveFormat
    )
  }

  private static func float32Data(_ samples: [Float]) -> Data {
    var data = Data(capacity: samples.count * MemoryLayout<UInt32>.size)
    for sample in samples {
      let finite = sample.isFinite ? sample : 0
      data.appendLittleEndian(finite.bitPattern)
    }
    return data
  }

  private static func signedInt16Data(_ samples: [Float]) -> Data {
    var data = Data(capacity: samples.count * MemoryLayout<Int16>.size)
    for sample in samples {
      let finite = sample.isFinite ? sample : 0
      let clipped = min(max(finite, -1), 1)
      let scaled = Int32((clipped * Float(Int16.max)).rounded())
      data.appendLittleEndian(Int16(clamping: scaled))
    }
    return data
  }

  private static func wave(
    payload: Data,
    sampleRate: Int,
    bitsPerSample: UInt16,
    format: UInt16
  ) -> Data {
    let channels: UInt16 = 1
    let bytesPerSample = UInt16(bitsPerSample / 8)
    let blockAlignment = channels * bytesPerSample
    let byteRate = UInt32(sampleRate) * UInt32(blockAlignment)
    let riffSize = UInt32(clamping: 36 + payload.count)

    var data = Data(capacity: 44 + payload.count)
    data.append(contentsOf: "RIFF".utf8)
    data.appendLittleEndian(riffSize)
    data.append(contentsOf: "WAVE".utf8)
    data.append(contentsOf: "fmt ".utf8)
    data.appendLittleEndian(UInt32(16))
    data.appendLittleEndian(format)
    data.appendLittleEndian(channels)
    data.appendLittleEndian(UInt32(sampleRate))
    data.appendLittleEndian(byteRate)
    data.appendLittleEndian(blockAlignment)
    data.appendLittleEndian(bitsPerSample)
    data.append(contentsOf: "data".utf8)
    data.appendLittleEndian(UInt32(clamping: payload.count))
    data.append(payload)
    return data
  }
}

private extension Data {
  mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
    var value = value.littleEndian
    Swift.withUnsafeBytes(of: &value) { bytes in
      append(contentsOf: bytes)
    }
  }
}
