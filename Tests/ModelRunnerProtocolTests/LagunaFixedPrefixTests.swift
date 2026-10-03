#if os(macOS)
    import Foundation
    import HuggingFace
    import MLX
    import MLXHuggingFace
    import MLXLLM
    import MLXLMCommon
    import Testing
    import Tokenizers
    @testable import ModelRunnerCore

    struct LagunaFixedPrefixTests {
        @Test(
            "Actual Laguna state emission preserves fixed-prefix logits",
            .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_TEST_LAGUNA_TARGET"] != nil))
        func fixedPrefix() async throws {
            let path = try #require(ProcessInfo.processInfo.environment["MIDNIGHT_TEST_LAGUNA_TARGET"])
            let report = try #require(ProcessInfo.processInfo.environment["MIDNIGHT_TEST_LAGUNA_DIAGNOSTIC_OUTPUT"])
            let float32 = ProcessInfo.processInfo.environment["MIDNIGHT_TEST_LAGUNA_FLOAT32"] == "1"
            let runtime = MLXPinnedRuntime.shared
            try await runtime.run {
                try await Device.withDefaultDevice(.gpu) {
                    let stream = MLX.Stream()
                    try await MLX.Stream.withDefaultStream(stream) {
                        await LagunaModelRegistration.register()
                        let automaticImplementation = "native Laguna"
                        let container = try await LLMModelFactory.shared.loadContainer(
                            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                            configuration: ModelConfiguration(directory: URL(fileURLWithPath: path)))
                        let prompt =
                            "Write a long, detailed technical tutorial about implementing a lock-free work-stealing scheduler in Swift. Continue with implementation details and code examples until the output limit; do not conclude or summarize early."
                        let input = try await container.prepare(
                            input: UserInput(
                                prompt: .messages([["role": "user", "content": prompt]]),
                                additionalContext: ["enable_thinking": false]))
                        try await container.perform(nonSendable: input) { context, input in
                            let target = try #require(context.model as? LagunaModel)
                            let draftPath = try #require(
                                ProcessInfo.processInfo.environment["MIDNIGHT_TEST_LAGUNA_DRAFT"])
                            let config = try JSONDecoder().decode(
                                LagunaDFlashConfiguration.self,
                                from: Data(
                                    contentsOf: URL(fileURLWithPath: draftPath).appendingPathComponent("config.json")))
                            let draft = LagunaDFlashModel(config)
                            try target.configureDFlash(draft.targetDescriptor)
                            if float32 {
                                // Preserve packed quantized weights; increase arithmetic precision
                                // by widening only floating-point parameters in this diagnostic.
                                context.model.update(
                                    parameters: context.model.parameters().mapValues {
                                        $0.dtype.isFloatingPoint ? $0.asType(.float32) : $0
                                    })
                                eval(context.model)
                            }
                            let ordinaryCache = try context.model.newCache(parameters: nil)
                            let stateCache = try context.model.newCache(parameters: nil)
                            var state = LMOutput.State()
                            state[mtpEmitFlagKey] = true
                            // Use the exact checkpoint template for both paths.
                            let promptIDs = input.text.tokens.asArray(Int.self)

                            var tokens = MLXArray(promptIDs).reshaped(1, -1)
                            var rows: [[String: Any]] = []
                            var generated: [Int] = []
                            var referenceLogits: [MLXArray] = []
                            for index in 0..<192 {
                                let text = LMInput(tokens: tokens).text
                                let a = context.model(text, cache: ordinaryCache, state: nil)
                                let b = context.model(text, cache: stateCache, state: state)
                                state = try #require(b.state)
                                let x = a.logits[0, -1, 0...]
                                let y = b.logits[0, -1, 0...]
                                eval(x, y)
                                let token = argMax(x).item(Int.self)
                                let other = argMax(y).item(Int.self)
                                let error = MLX.max(abs(x.asType(.float32) - y.asType(.float32))).item(Float.self)
                                rows.append([
                                    "step": index, "ordinary_token": token, "state_token": other,
                                    "max_logit_error": error, "cache_offset": ordinaryCache[0].offset,
                                ])
                                generated.append(token)
                                referenceLogits.append(x)
                                tokens = MLXArray([token]).reshaped(1, 1)
                            }
                            var batchRows: [[String: Any]] = []
                            for block in [2, 4] {
                                let cache = try context.model.newCache(parameters: nil)
                                var requestState = LMOutput.State()
                                requestState[mtpEmitFlagKey] = true
                                let prefill = context.model(
                                    LMInput(tokens: MLXArray(promptIDs).reshaped(1, -1)).text,
                                    cache: cache, state: requestState)
                                eval(prefill.logits)
                                requestState = try #require(prefill.state)
                                for start in stride(from: 0, to: generated.count - 1, by: block) {
                                    let end = min(start + block, generated.count - 1)
                                    let batch = LMInput(tokens: MLXArray(Array(generated[start..<end])).reshaped(1, -1))
                                        .text
                                    let result = context.model(batch, cache: cache, state: requestState)
                                    eval(result.logits)
                                    requestState = try #require(result.state)
                                    for index in start..<end {
                                        let actual = result.logits[0, index - start, 0...]
                                        let expected = referenceLogits[index + 1]
                                        batchRows.append([
                                            "block_size": block, "step": index + 1,
                                            "ordinary_token": generated[index + 1],
                                            "batch_token": argMax(actual).item(Int.self),
                                            "reference_margin_over_batch_choice":
                                                (expected[generated[index + 1]].asType(.float32)
                                                - expected[argMax(actual).item(Int.self)].asType(.float32)).item(
                                                    Float.self),
                                            "max_logit_error": MLX.max(
                                                abs(actual.asType(.float32) - expected.asType(.float32))
                                            ).item(Float.self),
                                        ])
                                    }
                                }
                            }
                            let result: [String: Any] = [
                                "model_path": path,
                                "floating_point_parameters": float32 ? "float32" : "checkpoint",
                                "automatic_model_implementation": automaticImplementation,
                                "model_implementation": String(reflecting: type(of: context.model)),
                                "prompt_tokens": promptIDs,
                                "generated_tokens": generated, "rows": rows, "batch_rows": batchRows,
                                "ordinary_text": context.tokenizer.decode(tokenIds: generated),
                            ]
                            guard !FileManager.default.fileExists(atPath: report) else {
                                throw CocoaError(.fileWriteFileExists)
                            }
                            try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
                                .write(to: URL(fileURLWithPath: report), options: .atomic)
                            #expect(rows.allSatisfy { ($0["ordinary_token"] as? Int) == ($0["state_token"] as? Int) })
                        }
                    }
                }
            }
        }
    }
#endif
