import Foundation

#if canImport(CryptoKit)
    import CryptoKit
#endif

/// Returns the lowercase hexadecimal SHA-256 digest of the supplied bytes.
public func corpusSHA256(_ data: Data) -> String {
    #if canImport(CryptoKit)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    #else
        return portableCorpusSHA256(data)
    #endif
}

func portableCorpusSHA256(_ data: Data) -> String {
    // Portable SHA-256 for Linux builds without CryptoKit; no external interpreter.
    let k: [UInt32] = [
        0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4, 0xab1c_5ed5,
        0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174,
        0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
        0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7, 0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967,
        0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85,
        0xa2bf_e8a1, 0xa81a_664b, 0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
        0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
        0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
    ]
    var h: [UInt32] = [
        0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a, 0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19,
    ]
    var bytes = Array(data)
    let bits = UInt64(bytes.count) * 8
    bytes.append(0x80)
    while bytes.count % 64 != 56 {
        bytes.append(0)
    }
    for shift in stride(from: 56, through: 0, by: -8) {
        bytes.append(UInt8(truncatingIfNeeded: bits >> shift))
    }
    func rotate(_ x: UInt32, _ n: UInt32) -> UInt32 { (x >> n) | (x << (32 - n)) }
    for start in stride(from: 0, to: bytes.count, by: 64) {
        var w = [UInt32](repeating: 0, count: 64)
        for i in 0..<16 {
            for j in 0..<4 {
                w[i] = (w[i] << 8) | UInt32(bytes[start + 4 * i + j])
            }
        }
        for i in 16..<64 {
            let a = w[i - 15]
            let b = w[i - 2]
            w[i] =
                w[i - 16] &+ (rotate(a, 7) ^ rotate(a, 18) ^ (a >> 3)) &+ w[i - 7]
                &+ (rotate(b, 17) ^ rotate(b, 19) ^ (b >> 10))
        }
        var a = h[0]
        var b = h[1]
        var c = h[2]
        var d = h[3]
        var e = h[4]
        var f = h[5]
        var g = h[6]
        var z = h[7]
        for i in 0..<64 {
            let t1 = z &+ (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) &+ ((e & f) ^ (~e & g)) &+ k[i] &+ w[i]
            let t2 = (rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) &+ ((a & b) ^ (a & c) ^ (b & c))
            z = g
            g = f
            f = e
            e = d &+ t1
            d = c
            c = b
            b = a
            a = t1 &+ t2
        }
        for (i, value) in [a, b, c, d, e, f, g, z].enumerated() {
            h[i] = h[i] &+ value
        }
    }
    return h.map { String(format: "%08x", $0) }.joined()
}
