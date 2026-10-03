import Foundation

/// Test-only import boundary for one contiguous row slice. This is not a model loader.
enum Q4ZeroPrototypeError: Error, Equatable {
    case unsupportedFormat
    case invalidDimensions
    case invalidRange
    case sizeOverflow
    case budgetExceeded
    case truncatedPayload
    case nonfiniteScale
}

struct Q4ZeroPrototypeDescriptor: Equatable {
    let rows: Int
    let columns: Int
    let sourceOffset: UInt64
    let sourceByteCount: Int

    init(
        rows: Int, columns: Int, sourceOffset: UInt64 = 0, sourceByteCount: Int,
        ggmlType: Int = 2, groupSize: Int = 32, bits: Int = 4,
        scaleType: String = "F16", implicitOffset: Int = 8, hasBiasTensor: Bool = false
    ) throws {
        guard ggmlType == 2, groupSize == 32, bits == 4, scaleType == "F16",
            implicitOffset == 8, !hasBiasTensor
        else { throw Q4ZeroPrototypeError.unsupportedFormat }
        guard rows > 0, columns > 0, columns.isMultiple(of: 32) else {
            throw Q4ZeroPrototypeError.invalidDimensions
        }
        let (blocks, blockOverflow) = rows.multipliedReportingOverflow(by: columns / 32)
        let (bytes, byteOverflow) = blocks.multipliedReportingOverflow(by: 18)
        guard !blockOverflow, !byteOverflow else {
            throw Q4ZeroPrototypeError.sizeOverflow
        }
        guard sourceByteCount == bytes else {
            throw Q4ZeroPrototypeError.truncatedPayload
        }
        guard !sourceOffset.addingReportingOverflow(UInt64(bytes)).overflow else {
            throw Q4ZeroPrototypeError.sizeOverflow
        }
        self.rows = rows
        self.columns = columns
        self.sourceOffset = sourceOffset
        self.sourceByteCount = sourceByteCount
    }

    func slice(_ range: Range<Int>, maximumBytes: Int) throws -> Q4ZeroPrototypeDescriptor {
        guard !range.isEmpty, range.lowerBound >= 0, range.upperBound <= rows else {
            throw Q4ZeroPrototypeError.invalidRange
        }
        // The complete tensor byte count was checked at initialization.
        let bytesPerRow = columns / 32 * 18
        let byteCount = range.count * bytesPerRow
        guard maximumBytes >= byteCount else {
            throw Q4ZeroPrototypeError.budgetExceeded
        }
        return try .init(
            rows: range.count, columns: columns,
            sourceOffset: sourceOffset + UInt64(range.lowerBound * bytesPerRow), sourceByteCount: byteCount)
    }
}

/// MLX-compatible consecutive nibbles and untouched IEEE F16 scale payloads.
struct Q4ZeroPrototypeGrid {
    let rows: Int
    let columns: Int
    let packed: [UInt32]
    let scaleBits: [UInt16]

    var storedByteCount: Int { packed.count * 4 + scaleBits.count * 2 }

    func code(row: Int, column: Int) -> UInt32 {
        let element = row * columns + column
        return (packed[element / 8] >> (4 * (element % 8))) & 15
    }

    func referenceValue(row: Int, column: Int) -> Float {
        let scale = Float(Float16(bitPattern: scaleBits[row * (columns / 32) + column / 32]))
        return scale * Float(Int(code(row: row, column: column)) - 8)
    }

    /// Reverse only the nibble permutation; every source scale bit remains intact.
    func sourceBytes() -> Data {
        var result = Data(capacity: storedByteCount)
        for block in scaleBits.indices {
            let scale = scaleBits[block]
            result.append(UInt8(truncatingIfNeeded: scale))
            result.append(UInt8(truncatingIfNeeded: scale >> 8))
            for index in 0..<16 {
                let first = block * 32 + index
                let second = first + 16
                let low = (packed[first / 8] >> (4 * (first % 8))) & 15
                let high = (packed[second / 8] >> (4 * (second % 8))) & 15
                result.append(UInt8(low | (high << 4)))
            }
        }
        return result
    }
}

enum Q4ZeroPrototypeCodec {
    static let maximumFixtureBytes = 4 * 1024 * 1024

    static func readRows(
        from url: URL, descriptor: Q4ZeroPrototypeDescriptor, rows: Range<Int>,
        maximumBytes: Int = maximumFixtureBytes
    ) throws -> Q4ZeroPrototypeGrid {
        guard maximumBytes <= maximumFixtureBytes else {
            throw Q4ZeroPrototypeError.budgetExceeded
        }
        let slice = try descriptor.slice(rows, maximumBytes: maximumBytes)
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let fileBytes = try handle.seekToEnd()
        guard descriptor.sourceOffset + UInt64(descriptor.sourceByteCount) <= fileBytes else {
            throw Q4ZeroPrototypeError.truncatedPayload
        }
        try handle.seek(toOffset: slice.sourceOffset)
        var payload = Data(capacity: slice.sourceByteCount)
        while payload.count < slice.sourceByteCount {
            let chunk = try handle.read(upToCount: slice.sourceByteCount - payload.count) ?? Data()
            guard !chunk.isEmpty else {
                throw Q4ZeroPrototypeError.truncatedPayload
            }
            payload.append(chunk)
        }
        return try repack(payload, descriptor: slice, maximumBytes: maximumBytes)
    }

    static func repack(
        _ source: Data, descriptor: Q4ZeroPrototypeDescriptor,
        maximumBytes: Int = maximumFixtureBytes
    ) throws -> Q4ZeroPrototypeGrid {
        guard maximumBytes <= maximumFixtureBytes, source.count <= maximumBytes else {
            throw Q4ZeroPrototypeError.budgetExceeded
        }
        guard source.count == descriptor.sourceByteCount else {
            throw Q4ZeroPrototypeError.truncatedPayload
        }
        let bytes = [UInt8](source)
        let blockCount = bytes.count / 18
        var packed = [UInt32](repeating: 0, count: blockCount * 4)
        var scales = [UInt16]()
        scales.reserveCapacity(blockCount)
        for block in 0..<blockCount {
            let start = block * 18
            let scale = UInt16(bytes[start]) | (UInt16(bytes[start + 1]) << 8)
            guard scale & 0x7c00 != 0x7c00 else {
                throw Q4ZeroPrototypeError.nonfiniteScale
            }
            scales.append(scale)
            // GGUF stores the first sixteen codes in low nibbles and the next sixteen in high nibbles.
            for index in 0..<32 {
                let byte = bytes[start + 2 + index % 16]
                let code = UInt32(index < 16 ? byte & 15 : byte >> 4)
                packed[block * 4 + index / 8] |= code << (4 * (index % 8))
            }
        }
        return .init(rows: descriptor.rows, columns: descriptor.columns, packed: packed, scaleBits: scales)
    }
}
