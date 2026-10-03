#if os(macOS)
    import Foundation
    import CoreImage
    import ImageIO
    import MLX
    import MLXLMCommon
    import MLXVLM
    import MLXHuggingFace
    import HuggingFace
    import Tokenizers
    import VisionProtocol
    import VisionHTTP

    actor NativeVisionModel: VisionGenerating {
        private let container: ModelContainer
        private var busy = false
        private var idleWaiters: [CheckedContinuation<Void, Never>] = []
        private let admission: FastVLMAdmission
        init(directory: URL, memoryBytes: Int) async throws {
            admission = try FastVLMAdmission(
                configuration: Data(contentsOf: directory.appendingPathComponent("config.json")),
                processor: Data(contentsOf: directory.appendingPathComponent("preprocessor_config.json")))
            let files = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.fileSizeKey])
            let weightsBytes = try files.filter { $0.pathExtension == "safetensors" }.reduce(0) { total, file in
                total + (try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
            }
            guard weightsBytes > 0, weightsBytes <= memoryBytes - 256 * 1024 * 1024 else {
                throw VisionError(
                    "insufficient_memory_budget",
                    "Vision memory limit must cover the checkpoint weights plus at least 256 MiB of working space.")
            }
            Memory.memoryLimit = memoryBytes
            Memory.cacheLimit = min(memoryBytes / 16, 128 * 1024 * 1024)
            guard Memory.memoryLimit == memoryBytes else {
                throw VisionError("memory_limit_failed", "Could not apply the explicit vision memory limit.")
            }
            container = try await VLMModelFactory.shared.loadContainer(
                from: directory, using: #huggingFaceTokenizerLoader())
        }
        func answer(_ request: VisionRequest) async throws -> VisionAnswer {
            guard !busy else {
                throw VisionError("vision_busy", "Another vision request is active. Retry when it completes.")
            }
            busy = true
            defer {
                busy = false
                Memory.clearCache()
                let waiters = idleWaiters
                idleWaiters = []
                for waiter in waiters {
                    waiter.resume()
                }
            }
            try Task.checkCancellation()
            let admission = admission
            return try await container.perform(values: request) { context, request in
                // Also drain partially submitted preprocessing/prefill on thrown errors before
                // releasing admission. All worker operations use MLX's default GPU stream.
                defer { Stream.gpu.synchronize() }
                guard let source = CGImageSourceCreateWithData(request.image.data as CFData, nil),
                    let thumbnail = CGImageSourceCreateThumbnailAtIndex(
                        source, 0,
                        [
                            kCGImageSourceCreateThumbnailFromImageAlways: true,
                            kCGImageSourceCreateThumbnailWithTransform: true,
                            kCGImageSourceThumbnailMaxPixelSize: request.image.longestEdgeLimit,
                            kCGImageSourceShouldCacheImmediately: true,
                        ] as CFDictionary)
                else {
                    throw VisionError("invalid_image", "Image pixels could not be decoded.")
                }
                let image = UserInput.Image.ciImage(CIImage(cgImage: thumbnail))
                let messages = request.messages.map { message in
                    Chat.Message(
                        role: Chat.Message.Role(rawValue: message.role)!, content: message.text,
                        images: message.containsImage ? [image] : [])
                }
                var processing = UserInput.Processing()
                processing.maxPixels = request.image.longestEdgeLimit * request.image.longestEdgeLimit
                let input = try await context.processor.prepare(
                    input: UserInput(chat: messages, processing: processing))
                try Task.checkCancellation()
                guard input.image?.pixels.shape == [1, 3, 1024, 1024] else {
                    throw VisionError(
                        "unsupported_image_layout",
                        "FastVLM produced an unexpected image layout; refusing unaccounted image positions.")
                }
                let promptPositions = try admission.promptPositions(
                    tokenIDs: input.text.tokens.asArray(Int.self), maximumTokens: request.maximumTokens)
                let progress = PreparedPositions()
                let parameters = GenerateParameters(
                    maxTokens: request.maximumTokens, temperature: request.temperature,
                    prefill: .init(progress: { _, total in progress.record(total) }))
                let iterator = try TokenIterator(input: input, model: context.model, parameters: parameters)
                guard progress.value == promptPositions else {
                    throw VisionError(
                        "image_position_mismatch",
                        "Actual image-expanded prefill positions differed from admission accounting.")
                }
                let (stream, producer) = MLXLMCommon.generateTask(
                    promptTokenCount: promptPositions,
                    modelConfiguration: context.configuration, tokenizer: context.tokenizer, iterator: iterator)
                return try await withTaskCancellationHandler {
                    var text = ""
                    var completion: GenerateCompletionInfo?
                    var unsupportedToolOutput = false
                    for await event in stream {
                        if Task.isCancelled {
                            producer.cancel()
                            break
                        }
                        switch event {
                        case .chunk(let chunk): text += chunk
                        case .info(let info): completion = info
                        case .toolCall, .rejectedToolCall:
                            unsupportedToolOutput = true
                            producer.cancel()
                        }
                        if unsupportedToolOutput {
                            break
                        }
                    }
                    // Admission remains occupied until the MLX producer releases its arrays, including cancellation.
                    await producer.value
                    try Task.checkCancellation()
                    guard !unsupportedToolOutput else {
                        throw VisionError(
                            "unsupported_tool_output", "This vision endpoint does not support tool output.")
                    }
                    guard let completion else {
                        throw VisionError("vision_failed", "Vision generation ended without completion information.")
                    }
                    return VisionAnswer(
                        text: text, promptTokens: completion.promptTokenCount,
                        completionTokens: completion.generationTokenCount,
                        finishReason: completion.stopReason == .length ? "length" : "stop")
                } onCancel: {
                    producer.cancel()
                }
            }
        }
        func isBusy() -> Bool { busy }
        func memorySnapshot() -> [String: Int] {
            [
                "active_bytes": Memory.activeMemory, "cache_bytes": Memory.cacheMemory,
                "peak_bytes": Memory.peakMemory, "limit_bytes": Memory.memoryLimit,
            ]
        }
        func waitUntilIdle() async {
            if busy {
                await withCheckedContinuation { idleWaiters.append($0) }
            }
        }
    }
    private final class PreparedPositions: @unchecked Sendable {
        private let lock = NSLock()
        private var positions: Int?
        func record(_ value: Int) {
            lock.lock()
            defer { lock.unlock() }
            positions = value
        }
        var value: Int? {
            lock.lock()
            defer { lock.unlock() }
            return positions
        }
    }
#endif
