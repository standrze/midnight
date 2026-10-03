import Foundation

#if !Q4ZERO_STANDALONE
    import Testing

    @Suite("Q4_0 grid-preserving row import (CPU only)")
    struct Q4ZeroPrototypeTests {
        @Test func allCodesAndScaleBits() throws { try Q4ZeroPrototypeChecks.allCodesAndScaleBits() }
        @Test func exactFiniteScaleGrid() throws { try Q4ZeroPrototypeChecks.exactFiniteScaleGrid() }
        @Test func nonzeroRowSlice() throws { try Q4ZeroPrototypeChecks.nonzeroRowSlice() }
        @Test func unsupportedMetadata() throws { try Q4ZeroPrototypeChecks.unsupportedMetadata() }
        @Test func invalidSizesAndRanges() throws { try Q4ZeroPrototypeChecks.invalidSizesAndRanges() }
        @Test func invalidPayload() throws { try Q4ZeroPrototypeChecks.invalidPayload() }
    }
#endif

/// Shared checks let a Foundation-only interpreter run the exact same assertions without loading MLX.
enum Q4ZeroPrototypeChecks {
    private struct Failure: Error { let message: String }

    private static func require(_ condition: Bool, _ message: String) throws {
        guard condition else {
            throw Failure(message: message)
        }
    }

    private static func rejects(_ error: Q4ZeroPrototypeError, _ body: () throws -> Void) throws {
        do {
            try body()
        } catch let actual as Q4ZeroPrototypeError {
            try require(actual == error, "Expected \(error), got \(actual)")
            return
        }
        throw Failure(message: "Expected rejection: \(error)")
    }

    static func block(scale: UInt16, seed: Int = 0) -> Data {
        var bytes = [UInt8(truncatingIfNeeded: scale), UInt8(truncatingIfNeeded: scale >> 8)]
        for index in 0..<16 {
            bytes.append(UInt8((index + seed) % 16) | (UInt8((15 - index + seed) % 16) << 4))
        }
        return Data(bytes)
    }

    static func allCodesAndScaleBits() throws {
        let scales: [UInt16] = [0, 0x8000, 1, 0x8001, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff]
        let source = scales.reduce(into: Data()) { $0.append(block(scale: $1)) }
        let descriptor = try Q4ZeroPrototypeDescriptor(rows: scales.count, columns: 32, sourceByteCount: source.count)
        let grid = try Q4ZeroPrototypeCodec.repack(source, descriptor: descriptor)
        try require(grid.scaleBits == scales, "Scale payload bits changed")
        try require(grid.sourceBytes() == source, "Source bytes changed on round trip")
        try require(grid.storedByteCount == source.count, "Implicit offset gained a stored bias")
        for row in scales.indices {
            for column in 0..<32 {
                let expected = column < 16 ? column : 31 - column
                try require(grid.code(row: row, column: column) == UInt32(expected), "Nibble order mismatch")
            }
        }
    }

    static func exactFiniteScaleGrid() throws {
        // All 63,488 finite F16 bit patterns, including signs/subnormals. The source stays below 4 MiB.
        let scales = (0...UInt32(UInt16.max)).map(UInt16.init).filter { $0 & 0x7c00 != 0x7c00 }
        let source = scales.reduce(into: Data()) { $0.append(block(scale: $1)) }
        let descriptor = try Q4ZeroPrototypeDescriptor(rows: scales.count, columns: 32, sourceByteCount: source.count)
        let grid = try Q4ZeroPrototypeCodec.repack(source, descriptor: descriptor)
        try require(grid.scaleBits == scales && grid.sourceBytes() == source, "Finite-scale round trip failed")
        for row in scales.indices {
            let scale = Float(Float16(bitPattern: scales[row]))
            for code in 0..<16 {
                let expected = scale * Float(code - 8)
                try require(
                    grid.referenceValue(row: row, column: code).bitPattern == expected.bitPattern,
                    "Exact source grid changed at scale \(scales[row]), code \(code)")
            }
        }
    }

    static func nonzeroRowSlice() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("synthetic-q4zero.bin")
        var source = Data(repeating: 0xa5, count: 19)
        for index in 0..<12 {
            source.append(block(scale: UInt16(0x3000 + index), seed: index))
        }
        try source.write(to: url, options: .withoutOverwriting)
        let descriptor = try Q4ZeroPrototypeDescriptor(rows: 3, columns: 128, sourceOffset: 19, sourceByteCount: 216)
        let grid = try Q4ZeroPrototypeCodec.readRows(from: url, descriptor: descriptor, rows: 1..<3)
        try require(grid.rows == 2 && grid.columns == 128, "Wrong row geometry")
        try require(grid.sourceBytes() == source.subdata(in: 91..<235), "Incorrect file or row offset")
        try rejects(.budgetExceeded) {
            _ = try Q4ZeroPrototypeCodec.readRows(from: url, descriptor: descriptor, rows: 1..<3, maximumBytes: 143)
        }
        try Data(source.dropLast()).write(to: url)
        try rejects(.truncatedPayload) {
            _ = try Q4ZeroPrototypeCodec.readRows(from: url, descriptor: descriptor, rows: 0..<1)
        }
    }

    static func unsupportedMetadata() throws {
        for type in [0, 1, 3, 12] {
            try rejects(.unsupportedFormat) {
                _ = try Q4ZeroPrototypeDescriptor(rows: 1, columns: 32, sourceByteCount: 18, ggmlType: type)
            }
        }
        for size in [16, 64, 128] {
            try rejects(.unsupportedFormat) {
                _ = try Q4ZeroPrototypeDescriptor(rows: 1, columns: 32, sourceByteCount: 18, groupSize: size)
            }
        }
        for bits in [2, 3, 8] {
            try rejects(.unsupportedFormat) {
                _ = try Q4ZeroPrototypeDescriptor(rows: 1, columns: 32, sourceByteCount: 18, bits: bits)
            }
        }
        for dtype in ["BF16", "F32", "I8"] {
            try rejects(.unsupportedFormat) {
                _ = try Q4ZeroPrototypeDescriptor(rows: 1, columns: 32, sourceByteCount: 18, scaleType: dtype)
            }
        }
        try rejects(.unsupportedFormat) {
            _ = try Q4ZeroPrototypeDescriptor(rows: 1, columns: 32, sourceByteCount: 18, implicitOffset: 0)
        }
        try rejects(.unsupportedFormat) {
            _ = try Q4ZeroPrototypeDescriptor(rows: 1, columns: 32, sourceByteCount: 18, hasBiasTensor: true)
        }
    }

    static func invalidSizesAndRanges() throws {
        for columns in [0, -32, 31, 33] {
            try rejects(.invalidDimensions) {
                _ = try Q4ZeroPrototypeDescriptor(rows: 1, columns: columns, sourceByteCount: 18)
            }
        }
        try rejects(.invalidDimensions) {
            _ = try Q4ZeroPrototypeDescriptor(rows: 0, columns: 32, sourceByteCount: 0)
        }
        try rejects(.sizeOverflow) {
            _ = try Q4ZeroPrototypeDescriptor(rows: Int.max, columns: 64, sourceByteCount: 0)
        }
        try rejects(.sizeOverflow) {
            _ = try Q4ZeroPrototypeDescriptor(rows: 1, columns: 32, sourceOffset: .max, sourceByteCount: 18)
        }
        let descriptor = try Q4ZeroPrototypeDescriptor(rows: 2, columns: 32, sourceByteCount: 36)
        for range in [0..<0, -1..<1, 1..<3] {
            try rejects(.invalidRange) { _ = try descriptor.slice(range, maximumBytes: 100) }
        }
        try rejects(.budgetExceeded) { _ = try descriptor.slice(0..<2, maximumBytes: 35) }
    }

    static func invalidPayload() throws {
        let descriptor = try Q4ZeroPrototypeDescriptor(rows: 1, columns: 32, sourceByteCount: 18)
        for source in [Data(repeating: 0, count: 17), Data(repeating: 0, count: 19)] {
            try rejects(.truncatedPayload) { _ = try Q4ZeroPrototypeCodec.repack(source, descriptor: descriptor) }
        }
        for scale: UInt16 in [0x7c00, 0xfc00, 0x7c01, 0xffff] {
            try rejects(.nonfiniteScale) {
                _ = try Q4ZeroPrototypeCodec.repack(block(scale: scale), descriptor: descriptor)
            }
        }
    }
}
