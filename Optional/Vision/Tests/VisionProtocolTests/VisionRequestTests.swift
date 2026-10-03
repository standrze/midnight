#if os(macOS)
    import Foundation
    import Testing
    import VisionProtocol

    @Suite("Optional vision request boundary")
    struct VisionRequestTests {
        static let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jWZkAAAAASUVORK5CYII="
        func body(_ modify: (inout [String: Any]) -> Void = { _ in }) throws -> Data {
            var value: [String: Any] = [
                "model": "vision-test", "stream": false,
                "messages": [
                    [
                        "role": "user",
                        "content": [
                            ["type": "text", "text": "Read this screenshot"],
                            ["type": "image_url", "image_url": ["url": "data:image/png;base64," + Self.png]],
                        ],
                    ]
                ],
            ]
            modify(&value)
            return try JSONSerialization.data(withJSONObject: value)
        }
        @Test func acceptsBoundedImageWithoutLoadingModel() throws {
            let request = try VisionRequest.parse(body(), expectedModel: "vision-test")
            #expect(request.image.width == 1)
            #expect(request.image.height == 1)
            #expect(request.maximumTokens == 512)
            #expect(request.image.longestEdgeLimit == 1024)
        }
        @Test func doesNotFallBackToTextModel() throws {
            #expect(throws: VisionError.self) { try VisionRequest.parse(body(), expectedModel: "text-model") }
        }
        @Test(arguments: [
            "https://example.com/image.png", "file:///tmp/image.png", "data:image/gif;base64,R0lG",
            "data:image/png;base64,not-base64",
        ])
        func rejectsUnsupportedImageLocations(_ url: String) throws {
            let data = try body {
                $0["messages"] = [["role": "user", "content": [["type": "image_url", "image_url": ["url": url]]]]]
            }
            #expect(throws: VisionError.self) { try VisionRequest.parse(data, expectedModel: "vision-test") }
        }
        @Test func rejectsMalformedJSONAsRequestError() {
            #expect(throws: VisionError.self) { try VisionRequest.parse(Data("{".utf8), expectedModel: "vision-test") }
        }
        @Test func rejectsTextOnly() throws {
            let data = try body { $0["messages"] = [["role": "user", "content": "Hello"]] }
            #expect(throws: VisionError.self) { try VisionRequest.parse(data, expectedModel: "vision-test") }
        }
        @Test func rejectsTwoImages() throws {
            let part: [String: Any] = ["type": "image_url", "image_url": ["url": "data:image/png;base64," + Self.png]]
            let data = try body { $0["messages"] = [["role": "user", "content": [part, part]]] }
            #expect(throws: VisionError.self) { try VisionRequest.parse(data, expectedModel: "vision-test") }
        }
        @Test(arguments: ["tools", "response_format", "previous_response_id"])
        func rejectsUnsupportedFeatures(_ name: String) throws {
            let data = try body { $0[name] = ["unsupported": true] }
            #expect(throws: VisionError.self) { try VisionRequest.parse(data, expectedModel: "vision-test") }
        }
        @Test func rejectsStreamingAndInvalidLimits() throws {
            for replacement: [String: Any] in [
                ["stream": true], ["stream": 0], ["max_tokens": true], ["max_tokens": 0], ["max_tokens": 1025],
                ["max_tokens": 1.5], ["temperature": false], ["temperature": 3],
            ] {
                let data = try body { $0.merge(replacement) { _, new in new } }
                #expect(throws: VisionError.self) { try VisionRequest.parse(data, expectedModel: "vision-test") }
            }
        }
        @Test func rejectsMimeMismatch() throws {
            let data = try body {
                $0["messages"] = [
                    [
                        "role": "user",
                        "content": [["type": "image_url", "image_url": ["url": "data:image/jpeg;base64," + Self.png]]],
                    ]
                ]
            }
            #expect(throws: VisionError.self) { try VisionRequest.parse(data, expectedModel: "vision-test") }
        }
        @Test func capsTextAndBody() throws {
            let data = try body {
                $0["messages"] = [["role": "user", "content": String(repeating: "x", count: 65 * 1024)]]
            }
            #expect(throws: VisionError.self) { try VisionRequest.parse(data, expectedModel: "vision-test") }
            #expect(throws: VisionError.self) {
                try VisionRequest.parse(Data(count: VisionRequest.maximumBodyBytes + 1), expectedModel: "vision-test")
            }
        }
    }

#endif
