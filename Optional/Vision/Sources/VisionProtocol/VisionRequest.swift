import CoreFoundation
import Foundation

#if canImport(ImageIO)
    import ImageIO
#endif

/// A bounded vision request validation error with a client-visible code.
public struct VisionError: Error, LocalizedError, Codable, Sendable {
    public let message: String
    public let code: String
    public let type: String
    /// Creates a client-visible vision error with a stable code.
    public init(_ code: String, _ message: String) {
        self.code = code
        self.message = message
        self.type = "invalid_request_error"
    }
    /// User-facing explanation for this error.
    public var errorDescription: String? { message }
}

/// A deliberately bounded subset of OpenAI Chat Completions. Parsing never starts a model.
public struct VisionRequest: Sendable {
    public static let maximumBodyBytes = 12 * 1024 * 1024
    public static let maximumImageBytes = 8 * 1024 * 1024
    public static let maximumPixels = 16_000_000
    public let model: String
    public let messages: [Message]
    public let image: Image
    public let maximumTokens: Int
    public let temperature: Float

    /// One text message and whether it contains the request image.
    public struct Message: Sendable {
        public let role: String
        public let text: String
        public let containsImage: Bool
    }
    /// Decoded image bytes and pixel dimensions after request validation.
    public struct Image: Sendable {
        public let data: Data
        public let mimeType: String
        public let detail: String
        public let width: Int
        public let height: Int
        /// Maximum image edge in pixels for the requested detail level.
        public var longestEdgeLimit: Int { detail == "low" ? 512 : 1024 }
    }

    /// Validates bounded JSON, image data, and the selected model before generation.
    public static func parse(_ data: Data, expectedModel: String) throws -> Self {
        guard data.count <= maximumBodyBytes else {
            throw VisionError("request_too_large", "Vision request exceeds 12 MiB.")
        }
        let decoded: Any
        do { decoded = try JSONSerialization.jsonObject(with: data) } catch {
            throw VisionError("invalid_json", "Vision request contains invalid JSON.")
        }
        guard let object = decoded as? [String: Any] else {
            throw VisionError("invalid_request", "Expected a JSON object.")
        }
        let accepted: Set<String> = [
            "model", "messages", "stream", "max_tokens", "max_completion_tokens", "temperature",
        ]
        guard Set(object.keys).isSubset(of: accepted) else {
            throw VisionError(
                "unsupported_parameter",
                "Vision currently supports model, messages, stream:false, max_tokens or max_completion_tokens, and temperature only."
            )
        }
        guard let model = object["model"] as? String, model == expectedModel else {
            throw VisionError(
                "vision_model_mismatch",
                "Select the configured vision model explicitly; the text model will not be replaced.")
        }
        if let stream = object["stream"] {
            guard let number = stream as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID(), !number.boolValue
            else {
                throw VisionError("unsupported_stream", "Vision currently supports stream:false only.")
            }
        }
        guard object["max_tokens"] == nil || object["max_completion_tokens"] == nil else {
            throw VisionError("invalid_token_limit", "Use only one output token limit.")
        }
        let maximumTokens = try integer(object["max_tokens"] ?? object["max_completion_tokens"], default: 512)
        guard (1...1024).contains(maximumTokens) else {
            throw VisionError("invalid_token_limit", "Vision output limit must be between 1 and 1024 tokens.")
        }
        let temperature: Float
        if let value = object["temperature"] {
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                number.doubleValue.isFinite,
                (0...2).contains(number.doubleValue)
            else {
                throw VisionError("invalid_temperature", "Temperature must be a number between 0 and 2.")
            }
            temperature = number.floatValue
        } else {
            temperature = 0
        }
        guard let source = object["messages"] as? [[String: Any]], !source.isEmpty, source.count <= 32 else {
            throw VisionError("invalid_messages", "Provide between 1 and 32 messages.")
        }
        var messages = [Message]()
        var images = [Image]()
        var textBytes = 0
        for message in source {
            guard Set(message.keys).isSubset(of: ["role", "content"]),
                let role = message["role"] as? String, ["system", "user", "assistant"].contains(role)
            else {
                throw VisionError(
                    "unsupported_message", "Vision accepts system, user, and assistant messages with role/content only."
                )
            }
            var texts = [String]()
            var containsImage = false
            if let text = message["content"] as? String {
                texts = [text]
            } else if let parts = message["content"] as? [[String: Any]], !parts.isEmpty {
                for part in parts {
                    switch part["type"] as? String {
                    case "text":
                        guard Set(part.keys) == ["type", "text"], let text = part["text"] as? String else {
                            throw VisionError("invalid_content", "Malformed text content part.")
                        }
                        texts.append(text)
                    case "image_url":
                        guard role == "user", Set(part.keys) == ["type", "image_url"], images.isEmpty,
                            let value = part["image_url"] as? [String: Any],
                            Set(value.keys).isSubset(of: ["url", "detail"]),
                            let url = value["url"] as? String
                        else {
                            throw VisionError("invalid_image", "Provide exactly one image_url part in a user message.")
                        }
                        let detail = value["detail"] as? String ?? "auto"
                        guard ["auto", "low", "high"].contains(detail),
                            value["detail"] == nil || value["detail"] is String
                        else {
                            throw VisionError("invalid_image_detail", "Image detail must be auto, low, or high.")
                        }
                        images.append(try parseImage(url, detail: detail))
                        containsImage = true
                    default:
                        throw VisionError(
                            "unsupported_content", "Vision accepts text and image_url content parts only.")
                    }
                }
            } else {
                throw VisionError("invalid_content", "Message content must be text or content parts.")
            }
            let text = texts.joined(separator: "\n")
            textBytes += text.utf8.count
            guard textBytes <= 64 * 1024 else {
                throw VisionError("text_too_large", "Vision message text exceeds 64 KiB.")
            }
            messages.append(Message(role: role, text: text, containsImage: containsImage))
        }
        guard let image = images.first else {
            throw VisionError("image_required", "An explicit image is required; send ordinary text to your text model.")
        }
        guard messages.last?.role == "user" else {
            throw VisionError("invalid_messages", "The final vision message must have the user role.")
        }
        return Self(
            model: model, messages: messages, image: image, maximumTokens: maximumTokens, temperature: temperature)
    }

    private static func integer(_ value: Any?, default fallback: Int) throws -> Int {
        guard let value else {
            return fallback
        }
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite,
            number.doubleValue.rounded() == number.doubleValue, number.doubleValue >= 0, number.doubleValue <= 1024
        else {
            throw VisionError("invalid_token_limit", "Output token limit must be an integer between 1 and 1024.")
        }
        return number.intValue
    }

    private static func parseImage(_ url: String, detail: String) throws -> Image {
        let mimeType: String
        let prefix: String
        if url.hasPrefix("data:image/png;base64,") {
            mimeType = "image/png"
            prefix = "data:image/png;base64,"
        } else if url.hasPrefix("data:image/jpeg;base64,") {
            mimeType = "image/jpeg"
            prefix = "data:image/jpeg;base64,"
        } else {
            throw VisionError(
                "unsupported_image_url",
                "Use an inline base64 PNG or JPEG data URL; remote and file URLs are not fetched.")
        }
        let payload = url.dropFirst(prefix.count)
        guard payload.utf8.count <= ((maximumImageBytes + 2) / 3) * 4,
            let bytes = Data(base64Encoded: String(payload)), !bytes.isEmpty, bytes.count <= maximumImageBytes
        else {
            throw VisionError("invalid_image", "Image must contain valid base64 and be no larger than 8 MiB.")
        }
        #if canImport(ImageIO)
            guard
                let source = CGImageSourceCreateWithData(
                    bytes as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
                CGImageSourceGetCount(source) == 1, let type = CGImageSourceGetType(source) as String?,
                (mimeType == "image/png" && type == "public.png")
                    || (mimeType == "image/jpeg" && type == "public.jpeg"),
                let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any],
                let width = properties[kCGImagePropertyPixelWidth as String] as? Int,
                let height = properties[kCGImagePropertyPixelHeight as String] as? Int,
                width > 0, height > 0, width <= 8192, height <= 8192, width <= maximumPixels / height
            else {
                throw VisionError(
                    "invalid_image",
                    "Image must be a single PNG or JPEG, no larger than 8192 pixels per edge or 16 megapixels.")
            }
            return Image(data: bytes, mimeType: mimeType, detail: detail, width: width, height: height)
        #else
            throw VisionError(
                "unsupported_backend",
                "The native vision worker currently requires macOS with Apple silicon. CUDA vision is not implemented.")
        #endif
    }
}
