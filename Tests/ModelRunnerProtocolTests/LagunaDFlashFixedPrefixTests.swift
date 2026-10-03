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

    /// Opt-in real-checkpoint diagnostic. Replays identical prefixes so numerical
    /// differences can be separated from free-running continuation divergence.
    struct LagunaDFlashFixedPrefixTests {
        @Test(
            "Laguna state capture and batched verification at identical prefixes",
            .enabled(if: ProcessInfo.processInfo.environment["MIDNIGHT_TEST_LAGUNA_TARGET"] != nil))
        func fixedPrefix() async throws {
            let environment = ProcessInfo.processInfo.environment
            let path = try #require(environment["MIDNIGHT_TEST_LAGUNA_TARGET"])
            let output = try #require(environment["MIDNIGHT_TEST_LAGUNA_REPORT"])
            let widen = environment["MIDNIGHT_TEST_LAGUNA_FLOAT32"] == "1"
            let count = Int(environment["MIDNIGHT_TEST_LAGUNA_TOKENS"] ?? "128") ?? 128
            try await MLXPinnedRuntime.shared.run {
                try await Device.withDefaultDevice(.gpu) {
                    await LagunaModelRegistration.register()
                    let container = try await LLMModelFactory.shared.loadContainer(
                        from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                        configuration: ModelConfiguration(directory: URL(fileURLWithPath: path)))
                    let input = try await container.prepare(
                        input: UserInput(
                            prompt: .messages([
                                [
                                    "role": "user",
                                    "content":
                                        "Write a long, detailed technical tutorial about implementing a lock-free work-stealing scheduler in Swift. Continue with implementation details and code examples until the output limit; do not conclude or summarize early.",
                                ]
                            ])))
                    try await container.perform(nonSendable: input) { context, input in
                        let target = try #require(context.model as? LagunaModel)
                        try target.configureDFlash(
                            .init(
                                hiddenSize: 2048, vocabularySize: 100352,
                                numberOfTargetLayers: 40, targetLayerIDs: [1, 13, 25, 33, 39], captureWindow: 512,
                                blockSize: 16))
                        if widen {
                            target.update(
                                parameters: target.parameters().mapValues {
                                    $0.dtype.isFloatingPoint ? $0.asType(.float32) : $0
                                })
                            eval(target)
                        }
                        let promptIDs = input.text.tokens.asArray(Int.self)
                        let ordinaryCache = try target.newCache(parameters: nil)
                        let stateCache = try target.newCache(parameters: nil)
                        var emit = LMOutput.State()
                        emit[mtpEmitFlagKey] = true
                        var tokens = MLXArray(promptIDs).reshaped(1, -1)
                        var reference: [MLXArray] = []
                        var generated: [Int] = []
                        var margins: [Float] = []
                        var stateRows: [[String: Any]] = []
                        for step in 0..<count {
                            let text = LMInput(tokens: tokens).text
                            let ordinary = target(text, cache: ordinaryCache, state: nil).logits[0, -1, 0...]
                            let captured = target(text, cache: stateCache, state: emit).logits[0, -1, 0...]
                            eval(ordinary, captured)
                            let token = argMax(ordinary).item(Int.self)
                            let other = argMax(captured).item(Int.self)
                            stateRows.append([
                                "step": step, "ordinary_token": token, "state_token": other,
                                "max_logit_error": MLX.max(abs(ordinary.asType(.float32) - captured.asType(.float32)))
                                    .item(Float.self),
                            ])
                            reference.append(ordinary)
                            generated.append(token)
                            var first = -Float.infinity
                            var second = -Float.infinity
                            for value in ordinary.asType(.float32).asArray(Float.self) {
                                if value > first {
                                    second = first
                                    first = value
                                } else if value > second {
                                    second = value
                                }
                            }
                            margins.append(first - second)
                            tokens = MLXArray([token]).reshaped(1, 1)
                        }
                        var batchRows: [[String: Any]] = []
                        var fusionRows: [[String: Any]] = []
                        for block in [2, 4, 16] {
                            let cache = try target.newCache(parameters: nil)
                            let fusedCache = try target.newCache(parameters: nil)
                            let prefill = target(
                                LMInput(tokens: MLXArray(promptIDs).reshaped(1, -1)).text, cache: cache, state: emit)
                            let fusedPrefill = target(
                                LMInput(tokens: MLXArray(promptIDs).reshaped(1, -1)).text, cache: fusedCache,
                                state: emit)
                            eval(prefill.logits, fusedPrefill.logits, cache, fusedCache)
                            for start in stride(from: 0, to: generated.count - 1, by: block) {
                                let end = min(start + block, generated.count - 1)
                                let batch = LMInput(tokens: MLXArray(Array(generated[start..<end])).reshaped(1, -1))
                                    .text
                                let result = LagunaRuntimeTuning.$useCompiledVerifyTail.withValue(false) {
                                    target(batch, cache: cache, state: emit)
                                }
                                let fused = LagunaRuntimeTuning.$useCompiledVerifyTail.withValue(true) {
                                    target(batch, cache: fusedCache, state: emit)
                                }
                                eval(result.logits, fused.logits, cache, fusedCache)
                                let error = MLX.max(abs(result.logits.asType(.float32) - fused.logits.asType(.float32)))
                                    .item(Float.self)
                                let hidden = try #require(result.state?[mtpLastHiddenStatesKey])
                                let fusedHidden = try #require(fused.state?[mtpLastHiddenStatesKey])
                                let hiddenError = MLX.max(abs(hidden.asType(.float32) - fusedHidden.asType(.float32)))
                                    .item(Float.self)
                                fusionRows.append([
                                    "block_size": block, "start": start,
                                    "max_logit_error": error, "max_hidden_error": hiddenError,
                                ])
                                #expect(
                                    error == 0, "Compiled verification changed logits at block \(block), start \(start)"
                                )
                                #expect(hiddenError == 0, "Compiled verification changed captured hidden states")
                                for index in start..<end {
                                    let actual = result.logits[0, index - start, 0...]
                                    let expected = reference[index + 1]
                                    let actualToken = argMax(actual).item(Int.self)
                                    batchRows.append([
                                        "block_size": block, "step": index + 1,
                                        "ordinary_token": generated[index + 1], "batch_token": actualToken,
                                        "ordinary_top2_margin": margins[index + 1],
                                        "max_logit_error": MLX.max(
                                            abs(actual.asType(.float32) - expected.asType(.float32))
                                        ).item(Float.self),
                                    ])
                                }
                            }
                        }
                        let report: [String: Any] = [
                            "model_path": path, "float32": widen,
                            "prompt_tokens": promptIDs, "generated_tokens": generated,
                            "state_rows": stateRows, "batch_rows": batchRows, "fusion_rows": fusionRows,
                            "ordinary_text": context.tokenizer.decode(tokenIds: generated),
                        ]
                        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                            .write(to: URL(fileURLWithPath: output), options: .withoutOverwriting)
                        #expect(stateRows.allSatisfy { ($0["ordinary_token"] as? Int) == ($0["state_token"] as? Int) })
                    }
                }
            }
        }
    }
#endif
