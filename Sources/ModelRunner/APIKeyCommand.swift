import ArgumentParser
import Crypto

struct APIKeyCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "api-key",
        abstract: "Generate a key for Midnight HTTP API access.",
        subcommands: [Generate.self]
    )

    struct Generate: ParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Print a new 256-bit API key as 64 hexadecimal characters."
        )

        mutating func run() throws {
            print(Self.generate())
        }

        static func generate() -> String {
            let key = SymmetricKey(size: .bits256)
            let hexDigits = Array("0123456789abcdef".utf8)
            return key.withUnsafeBytes { bytes in
                var result = [UInt8]()
                result.reserveCapacity(bytes.count * 2)
                for byte in bytes {
                    result.append(hexDigits[Int(byte >> 4)])
                    result.append(hexDigits[Int(byte & 0x0F)])
                }
                return String(decoding: result, as: UTF8.self)
            }
        }
    }
}
