#if !os(macOS)
    import Foundation
    import Testing
    import VisionProtocol

    @Test func nativeVisionIsExplicitlyUnsupported() throws {
        let body = Data(
            #"{"model":"test","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jWZkAAAAASUVORK5CYII="}}]}]}"#
                .utf8)
        do {
            _ = try VisionRequest.parse(body, expectedModel: "test")
            Issue.record("Unsupported platform accepted image decoding")
        } catch let error as VisionError {
            #expect(error.code == "unsupported_backend")
        }
    }
#endif
