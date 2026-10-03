import ArgumentParser
import Foundation
import MLX
import MLXLLM
import ModelQualityCore
import ModelRunnerCore
import ModelRunnerProtocol

private struct RecordedMetrics: Encodable {
    var promptTokenCount: Int
    var prefilledPromptTokenCount: Int
    var cachedPromptTokenCount: Int
    var generationTokenCount: Int
    var promptTokensPerSecond: Double
    var tokensPerSecond: Double
    var stopReason: String
    var proposedDraftTokens: Int?
    var acceptedDraftTokens: Int?
    var speculativePassthroughReason: String?

    init(_ metrics: LocalModelRunnerMetrics) {
        promptTokenCount = metrics.promptTokenCount
        prefilledPromptTokenCount = metrics.prefilledPromptTokenCount
        cachedPromptTokenCount = metrics.cachedPromptTokenCount
        generationTokenCount = metrics.generationTokenCount
        promptTokensPerSecond = metrics.promptTokensPerSecond
        tokensPerSecond = metrics.tokensPerSecond
        stopReason = metrics.stopReason
        proposedDraftTokens = metrics.proposedDraftTokens
        acceptedDraftTokens = metrics.acceptedDraftTokens
        speculativePassthroughReason = metrics.speculativePassthroughReason
    }

    enum CodingKeys: String, CodingKey {
        case promptTokenCount = "prompt_token_count"
        case prefilledPromptTokenCount = "prefilled_prompt_token_count"
        case cachedPromptTokenCount = "cached_prompt_token_count"
        case generationTokenCount = "generation_token_count"
        case promptTokensPerSecond = "prompt_tokens_per_second"
        case tokensPerSecond = "tokens_per_second"
        case stopReason = "stop_reason"
        case proposedDraftTokens = "proposed_draft_tokens"
        case acceptedDraftTokens = "accepted_draft_tokens"
        case speculativePassthroughReason = "speculative_passthrough_reason"
    }
}

private struct RecordedTrial: Encodable {
    var sequence: Int
    var mode: String
    var metrics: RecordedMetrics
    var timeToFirstTokenMilliseconds: Double
    var totalMilliseconds: Double
    var peakActiveMemoryBytes: Int
    var promptTokenIDFingerprint: String
    var content: String
    var reasoning: String

    enum CodingKeys: String, CodingKey {
        case sequence, mode, metrics, content, reasoning
        case totalMilliseconds = "total_milliseconds"
        case peakActiveMemoryBytes = "peak_active_memory_bytes"
        case promptTokenIDFingerprint = "prompt_token_id_fingerprint"
        case timeToFirstTokenMilliseconds = "time_to_first_token_milliseconds"
    }
}

private struct DFlashABComparison: Encodable {
    var trialsPerMode: Int
    var targetOnlyMedianDecodeTokensPerSecond: Double
    var dflashMedianDecodeTokensPerSecond: Double
    var dflashSpeedupPercent: Double
    var outputsMatchExactly: Bool
    var firstOutputDivergenceUTF8Offset: Int?

    enum CodingKeys: String, CodingKey {
        case trialsPerMode = "trials_per_mode"
        case targetOnlyMedianDecodeTokensPerSecond =
            "target_only_median_decode_tokens_per_second"
        case dflashMedianDecodeTokensPerSecond = "dflash_median_decode_tokens_per_second"
        case dflashSpeedupPercent = "dflash_speedup_percent"
        case outputsMatchExactly = "outputs_match_exactly"
        case firstOutputDivergenceUTF8Offset = "first_output_divergence_utf8_offset"
    }
}

private struct GemmaAssistantABComparison: Encodable {
    var targetMedian: Double
    var assistantMedian: Double
    var speedupPercent: Double
    var outputsMatchExactly: Bool
    enum CodingKeys: String, CodingKey {
        case targetMedian = "target_only_median_decode_tokens_per_second"
        case assistantMedian = "assistant_median_decode_tokens_per_second"
        case speedupPercent = "speedup_percent"
        case outputsMatchExactly = "outputs_match_exactly"
    }
}

private struct LagunaFusionABComparison: Encodable {
    var trialsPerMode: Int
    var unfusedMedianDecodeTokensPerSecond: Double
    var fusedMedianDecodeTokensPerSecond: Double
    var fusedSpeedupPercent: Double
    var outputsMatchExactly: Bool

    enum CodingKeys: String, CodingKey {
        case trialsPerMode = "trials_per_mode"
        case unfusedMedianDecodeTokensPerSecond =
            "unfused_median_decode_tokens_per_second"
        case fusedMedianDecodeTokensPerSecond = "fused_median_decode_tokens_per_second"
        case fusedSpeedupPercent = "fused_speedup_percent"
        case outputsMatchExactly = "outputs_match_exactly"
    }
}

private struct LagunaAttentionGateABComparison: Encodable {
    var trialsPerMode: Int
    var eagerMedianDecodeTokensPerSecond: Double
    var compiledMedianDecodeTokensPerSecond: Double
    var compiledSpeedupPercent: Double
    var outputsMatchExactly: Bool

    enum CodingKeys: String, CodingKey {
        case trialsPerMode = "trials_per_mode"
        case eagerMedianDecodeTokensPerSecond =
            "eager_median_decode_tokens_per_second"
        case compiledMedianDecodeTokensPerSecond =
            "compiled_median_decode_tokens_per_second"
        case compiledSpeedupPercent = "compiled_speedup_percent"
        case outputsMatchExactly = "outputs_match_exactly"
    }
}

private struct LagunaBlockTailABComparison: Encodable {
    var trialsPerMode: Int
    var eagerMedianDecodeTokensPerSecond: Double
    var compiledMedianDecodeTokensPerSecond: Double
    var compiledSpeedupPercent: Double
    var outputsMatchExactly: Bool

    enum CodingKeys: String, CodingKey {
        case trialsPerMode = "trials_per_mode"
        case eagerMedianDecodeTokensPerSecond =
            "eager_median_decode_tokens_per_second"
        case compiledMedianDecodeTokensPerSecond =
            "compiled_median_decode_tokens_per_second"
        case compiledSpeedupPercent = "compiled_speedup_percent"
        case outputsMatchExactly = "outputs_match_exactly"
    }
}

private struct LagunaRouterTopKABComparison: Encodable {
    var trialsPerMode: Int
    var legacyMedianDecodeTokensPerSecond: Double
    var fusedMedianDecodeTokensPerSecond: Double
    var fusedSpeedupPercent: Double
    var outputsMatchExactly: Bool

    enum CodingKeys: String, CodingKey {
        case trialsPerMode = "trials_per_mode"
        case legacyMedianDecodeTokensPerSecond =
            "legacy_median_decode_tokens_per_second"
        case fusedMedianDecodeTokensPerSecond =
            "fused_median_decode_tokens_per_second"
        case fusedSpeedupPercent = "fused_speedup_percent"
        case outputsMatchExactly = "outputs_match_exactly"
    }
}

private struct PromptCacheComparison: Encodable {
    var trialsPerMode: Int
    var cachedMedianTimeToFirstTokenMilliseconds: Double
    var coldMedianTimeToFirstTokenMilliseconds: Double
    var cachedTTFTReductionPercent: Double
    var cachedMedianTotalPromptTokenCount: Double
    var cachedMedianPrefilledPromptTokenCount: Double
    var cachedMedianReusedPromptTokenCount: Double
    var coldMedianTotalPromptTokenCount: Double
    var coldMedianPrefilledPromptTokenCount: Double
    var coldMedianReusedPromptTokenCount: Double
    var outputsMatchExactly: Bool

    enum CodingKeys: String, CodingKey {
        case trialsPerMode = "trials_per_mode"
        case cachedMedianTimeToFirstTokenMilliseconds =
            "cached_median_time_to_first_token_milliseconds"
        case coldMedianTimeToFirstTokenMilliseconds =
            "cold_median_time_to_first_token_milliseconds"
        case cachedTTFTReductionPercent = "cached_ttft_reduction_percent"
        case cachedMedianTotalPromptTokenCount =
            "cached_median_total_prompt_token_count"
        case cachedMedianPrefilledPromptTokenCount =
            "cached_median_prefilled_prompt_token_count"
        case cachedMedianReusedPromptTokenCount =
            "cached_median_reused_prompt_token_count"
        case coldMedianTotalPromptTokenCount = "cold_median_total_prompt_token_count"
        case coldMedianPrefilledPromptTokenCount =
            "cold_median_prefilled_prompt_token_count"
        case coldMedianReusedPromptTokenCount =
            "cold_median_reused_prompt_token_count"
        case outputsMatchExactly = "outputs_match_exactly"
    }
}

private struct MistralHotCacheABComparison: Encodable {
    var trialsPerMode: Int
    var cachedMedianTimeToFirstTokenMilliseconds: Double
    var coldMedianTimeToFirstTokenMilliseconds: Double
    var cachedTTFTReductionPercent: Double
    var cachedMedianTotalPromptTokenCount: Double
    var cachedMedianPrefilledPromptTokenCount: Double
    var cachedMedianReusedPromptTokenCount: Double
    var coldMedianTotalPromptTokenCount: Double
    var coldMedianPrefilledPromptTokenCount: Double
    var coldMedianReusedPromptTokenCount: Double
    var outputsMatchExactly: Bool
    var firstOutputDivergenceUTF8Offset: Int?

    enum CodingKeys: String, CodingKey {
        case trialsPerMode = "trials_per_mode"
        case cachedMedianTimeToFirstTokenMilliseconds =
            "cached_median_time_to_first_token_milliseconds"
        case coldMedianTimeToFirstTokenMilliseconds =
            "cold_median_time_to_first_token_milliseconds"
        case cachedTTFTReductionPercent = "cached_ttft_reduction_percent"
        case cachedMedianTotalPromptTokenCount =
            "cached_median_total_prompt_token_count"
        case cachedMedianPrefilledPromptTokenCount =
            "cached_median_prefilled_prompt_token_count"
        case cachedMedianReusedPromptTokenCount =
            "cached_median_reused_prompt_token_count"
        case coldMedianTotalPromptTokenCount = "cold_median_total_prompt_token_count"
        case coldMedianPrefilledPromptTokenCount =
            "cold_median_prefilled_prompt_token_count"
        case coldMedianReusedPromptTokenCount =
            "cold_median_reused_prompt_token_count"
        case outputsMatchExactly = "outputs_match_exactly"
        case firstOutputDivergenceUTF8Offset = "first_output_divergence_utf8_offset"
    }
}

private struct GenerationMode {
    var label: String
    var useDFlash: Bool
    var useLagunaFusion: Bool
    var useCompiledAttentionGate: Bool
    var useCompiledBlockTail: Bool? = nil
    var useFusedRouterTopK: Bool? = nil
    var useFusedGateUpSilu: Bool? = nil
    var usePromptCache = true
    var useGemmaDenseFusion: Bool? = nil
    var useGemmaWindowSlicing: Bool? = nil
}

private struct HotCacheModeLabels {
    let cached: String
    let cold: String
    let seedBeforeCold: String
    let seedBeforeCached: String

    init(isMistral: Bool) {
        if isMistral {
            cached = "mistral_hot_cache_cached"
            cold = "mistral_hot_cache_cold"
            seedBeforeCold = "mistral_hot_cache_seed_before_cold"
            seedBeforeCached = "mistral_hot_cache_seed_before_cached"
        } else {
            cached = "hot_cache_cached"
            cold = "hot_cache_cold"
            seedBeforeCold = "hot_cache_seed_before_cold"
            seedBeforeCached = "hot_cache_seed_before_cached"
        }
    }
}

private struct TrialExecution {
    let warmups: [RecordedTrial]
    let trials: [RecordedTrial]
    var session: BenchmarkWiredMemorySessionReport? = nil
    var beforeMeasured: BenchmarkWiredMemoryState? = nil
    var afterMeasured: BenchmarkWiredMemoryState? = nil
}

private struct RuntimeBenchmarkReport: Encodable {
    var format = 1
    var wiredMemoryScope = "request"
    var wiredMemoryWarmupScope = "request"
    var wiredMemorySessionPhase: String?
    var wiredMemoryInitialPlan: BenchmarkWiredMemoryPlan?
    var wiredMemoryFinalPlan: BenchmarkWiredMemoryPlan?
    var wiredMemoryInitialState: BenchmarkWiredMemoryState?
    var wiredMemoryBeforeMeasured: BenchmarkWiredMemoryState?
    var wiredMemoryAfterMeasured: BenchmarkWiredMemoryState?
    var wiredMemorySession: BenchmarkWiredMemorySessionReport?
    var wiredMemoryRequests: [BenchmarkWiredMemoryRequest] = []
    var wiredMemorySessionEligible: Bool?
    var wiredMemorySessionFailures: [String] = []
    var wiredMemoryCacheLimitBytes: Int?
    var status = "measured"
    var createdAt: String
    var modelPath: String
    var servedModelName: String
    var modelImplementation: String
    var dflashModelPath: String?
    var dflashBlockSize: Int?
    var temperature: Float
    var topP: Float
    var gemmaAssistantModelPath: String?
    var gemmaAssistantBlockSize: Int?
    var gemmaAssistantQuantizationBits: Int?
    var gemmaAssistantQuantizedModuleCount: Int?
    var engine: String
    var prompt: String
    var continuationPrompt: String
    var contextLength: Int
    var prefillStepSize: Int
    var kvCompression: String
    var reasoningEffort: String?
    var memoryLimitBytes: Int
    var allowEarlyStop: Bool
    var promptReusePolicy: String
    var requestedTokens: Int
    var warmupCount: Int
    var measuredTrials: Int
    var medianPromptTokensPerSecond: Double?
    var medianDecodeTokensPerSecond: Double?
    var dflashABComparison: DFlashABComparison?
    var gemmaAssistantABComparison: GemmaAssistantABComparison?
    var lagunaFusionABComparison: LagunaFusionABComparison?
    var lagunaAttentionGateABComparison: LagunaAttentionGateABComparison?
    var lagunaBlockTailABComparison: LagunaBlockTailABComparison?
    var lagunaRouterTopKABComparison: LagunaRouterTopKABComparison?
    var lagunaGatherSiluABComparison: LagunaRouterTopKABComparison?
    var lagunaGatherSiluTraceCount: Int?
    var promptCacheComparison: PromptCacheComparison?
    var mistralHotCacheABComparison: MistralHotCacheABComparison?
    var hotCacheABComparison: MistralHotCacheABComparison?
    var warmups: [RecordedTrial]
    var trials: [RecordedTrial]

    enum CodingKeys: String, CodingKey {
        case format, status, engine, prompt, warmups, trials
        case wiredMemoryScope = "wired_memory_scope"
        case wiredMemoryWarmupScope = "wired_memory_warmup_scope"
        case wiredMemorySessionPhase = "wired_memory_session_phase"
        case wiredMemoryInitialPlan = "wired_memory_initial_plan"
        case wiredMemoryFinalPlan = "wired_memory_final_plan"
        case wiredMemoryInitialState = "wired_memory_initial_state"
        case wiredMemoryBeforeMeasured = "wired_memory_before_measured"
        case wiredMemoryAfterMeasured = "wired_memory_after_measured"
        case wiredMemorySession = "wired_memory_session"
        case wiredMemoryRequests = "wired_memory_requests"
        case wiredMemorySessionEligible = "wired_memory_session_eligible"
        case wiredMemorySessionFailures = "wired_memory_session_failures"
        case wiredMemoryCacheLimitBytes = "wired_memory_cache_limit_bytes"
        case createdAt = "created_at"
        case modelPath = "model_path"
        case servedModelName = "served_model_name"
        case modelImplementation = "model_implementation"
        case dflashModelPath = "dflash_model_path"
        case dflashBlockSize = "dflash_block_size"
        case temperature
        case topP = "top_p"
        case gemmaAssistantModelPath = "gemma_assistant_model_path"
        case gemmaAssistantBlockSize = "gemma_assistant_block_size"
        case gemmaAssistantQuantizationBits = "gemma_assistant_quantization_bits"
        case gemmaAssistantQuantizedModuleCount = "gemma_assistant_quantized_module_count"
        case continuationPrompt = "continuation_prompt"
        case contextLength = "context_length"
        case prefillStepSize = "prefill_step_size"
        case kvCompression = "kv_compression"
        case memoryLimitBytes = "memory_limit_bytes"
        case allowEarlyStop = "allow_early_stop"
        case promptReusePolicy = "prompt_reuse_policy"
        case requestedTokens = "requested_tokens"
        case warmupCount = "warmup_count"
        case measuredTrials = "measured_trials"
        case medianPromptTokensPerSecond = "median_prompt_tokens_per_second"
        case medianDecodeTokensPerSecond = "median_decode_tokens_per_second"
        case dflashABComparison = "dflash_ab_comparison"
        case gemmaAssistantABComparison = "gemma_assistant_ab_comparison"
        case lagunaFusionABComparison = "laguna_fusion_ab_comparison"
        case lagunaAttentionGateABComparison = "laguna_attention_gate_ab_comparison"
        case lagunaBlockTailABComparison = "laguna_block_tail_ab_comparison"
        case lagunaRouterTopKABComparison = "laguna_router_topk_ab_comparison"
        case lagunaGatherSiluABComparison = "laguna_gather_silu_ab_comparison"
        case lagunaGatherSiluTraceCount = "laguna_gather_silu_trace_count"
        case promptCacheComparison = "prompt_cache_comparison"
        case mistralHotCacheABComparison = "mistral_hot_cache_ab_comparison"
        case hotCacheABComparison = "hot_cache_ab_comparison"
        case reasoningEffort = "reasoning_effort"
    }
}

private enum BenchmarkError: Error, LocalizedError {
    case invalidInput(String)
    case incompleteGeneration(Int, Int)
    case missingMetrics(Int)
    case unexpectedToolCall(Int)
    case dflashPassthrough(Int, String)
    case promptCacheNotUsed(Int)
    case coldPromptUnexpectedlyCached(Int, Int)
    case unsupportedMistralHotCacheModel
    case mistralHotCacheSeedUnexpectedlyCached(Int, Int)
    case mistralHotCacheNotUsed(Int)
    case mistralHotCacheColdUnexpectedlyCached(Int, Int)
    case mistralHotCachePromptCountMismatch(Int, Int, Int)
    case mistralHotCacheDidNotReducePrefill(Int, Int, Int, Int)
    case mistralHotCachePartialReuse(Int, Int, Int)
    case mistralHotCachePrefillAccountingMismatch(Int, Int, Int, Int)
    case mistralHotCacheSeedOutputMismatch(Int, Int, Int?)

    var errorDescription: String? {
        switch self {
        case .invalidInput(let message): "Invalid runtime benchmark input: \(message)"
        case .incompleteGeneration(let actual, let expected):
            "Generation produced \(actual)/\(expected) requested tokens."
        case .missingMetrics(let sequence):
            "Generation \(sequence) completed without runtime metrics."
        case .unexpectedToolCall(let sequence):
            "Generation \(sequence) unexpectedly produced a tool call."
        case .dflashPassthrough(let sequence, let reason):
            "Speculative benchmark generation \(sequence) entered target-only passthrough: \(reason)."
        case .promptCacheNotUsed(let sequence):
            "Prompt-cache generation \(sequence) reused no prompt tokens; this mode requires native Laguna prompt-cache support."
        case .coldPromptUnexpectedlyCached(let sequence, let count):
            "Cold prompt-cache replay \(sequence) unexpectedly reused \(count) prompt tokens."
        case .unsupportedMistralHotCacheModel:
            "--hot-cache-ab requires a Mistral-family or GPT-OSS text checkpoint."
        case .mistralHotCacheSeedUnexpectedlyCached(let sequence, let count):
            "Hot-cache seed \(sequence) unexpectedly reused \(count) prompt tokens."
        case .mistralHotCacheNotUsed(let sequence):
            "Hot-cache continuation \(sequence) reused no prompt tokens."
        case .mistralHotCacheColdUnexpectedlyCached(let sequence, let count):
            "Cold continuation \(sequence) unexpectedly reused \(count) prompt tokens."
        case .mistralHotCachePromptCountMismatch(let cachedSequence, let coldSequence, let delta):
            "Hot/cold continuations \(cachedSequence)/\(coldSequence) rendered different prompt lengths (delta \(delta))."
        case .mistralHotCacheDidNotReducePrefill(
            let cachedSequence, let coldSequence, let cachedCount, let coldCount):
            "Hot/cold continuations \(cachedSequence)/\(coldSequence) did not reduce prefill tokens (\(cachedCount) versus \(coldCount))."
        case .mistralHotCachePartialReuse(let sequence, let actual, let expected):
            "Mistral hot-cache continuation \(sequence) reused \(actual)/\(expected) seed tokens; a full append-only reuse is required."
        case .mistralHotCachePrefillAccountingMismatch(
            let sequence, let actual, let total, let reused):
            "Hot-cache continuation \(sequence) reported \(actual) prefilled prompt tokens; expected \(total - reused) (\(total) total minus \(reused) reused)."
        case .mistralHotCacheSeedOutputMismatch(let firstSequence, let secondSequence, let offset):
            "Hot-cache seeds \(firstSequence)/\(secondSequence) produced different greedy text"
                + (offset.map { " at UTF-8 byte \($0)." } ?? ".")
        }
    }
}

@main
private struct RuntimeBenchmark: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "model-runner-runtime-bench",
        abstract: "Measure native LocalModelRunner prefill and decode rates without HTTP overhead."
    )

    @Argument(help: "Local MLX checkpoint directory.")
    var model: String

    @Argument(help: "New JSON report path.")
    var output: String

    @Option(help: "Execution engine: auto, metal, cuda, or cpu.")
    var engine = "auto"

    @Option(help: "Generated tokens per warm-up and measured trial.")
    var tokens = 256

    @Option(help: "Prompt plus output context ceiling; cannot exceed model metadata.")
    var contextLength: Int?

    @Option(help: "Prefill chunk size (1...8192).")
    var prefillStepSize = 512

    @Option(help: "KV compression: none, affine8, affine4, or turbo8v4.")
    var kvCompression = "none"

    @Option(help: "Poolside Laguna DFlash drafter checkpoint directory.")
    var dflashModel: String?

    @Option(help: "DFlash verification block size (2...checkpoint maximum; default Laguna 3, Muse 4).")
    var dflashBlockSize: Int?

    @Flag(
        name: .customLong("dflash-ab"),
        help:
            "Alternate target-only and DFlash generations on one loaded target; --trials is the count per mode."
    )
    var dflashAB = false

    @Option(help: "Gemma 4 shared-KV assistant checkpoint directory (experimental benchmark).")
    var gemmaAssistantModel: String?

    @Option(help: "Gemma assistant verification block size (2...checkpoint maximum).")
    var gemmaAssistantBlockSize: Int?

    @Flag(help: "Alternate ordinary and Gemma assistant decoding on the same loaded A4B target.")
    var gemmaAssistantAB = false

    @Option(help: "Experimentally quantize only the assistant in memory (4 or 8 bits, group size 64).")
    var gemmaAssistantQuantizationBits: Int?

    @Flag(
        name: .customLong("laguna-fusion-ab"),
        help:
            "Alternate unfused and compiled-fusion Laguna generations on one loaded target; --trials is the count per mode."
    )
    var lagunaFusionAB = false

    @Flag(help: "Alternate Gemma 4 dense activation fusion on one loaded model; trials is the count per mode.")
    var gemmaDenseFusionAB = false

    @Flag(help: "Alternate Gemma 4 sliding-window decode reads on one loaded model.")
    var gemmaWindowSlicingAB = false

    @Flag(
        name: .customLong("laguna-attention-gate-ab"),
        help:
            "Alternate eager and compiled Laguna per-head attention gates with compiled MoE enabled; --trials is the count per mode."
    )
    var lagunaAttentionGateAB = false

    @Flag(
        name: .customLong("laguna-block-tail-ab"),
        help:
            "Alternate eager and compiled Laguna decoder block tails on one loaded target; --trials is the count per mode."
    )
    var lagunaBlockTailAB = false

    @Flag(
        name: .customLong("laguna-router-topk-ab"),
        help:
            "Alternate legacy and fused Laguna decode router top-k tails on one loaded target; both arms use compiled block tails and --trials is the count per mode."
    )
    var lagunaRouterTopKAB = false

    @Flag(
        name: .customLong("laguna-gather-silu-ab"),
        help:
            "Alternate stock and fused Q4 gate/up-SiLU on one loaded Laguna target; both arms use compiled tails and fused routers. Experimental; --trials is per mode."
    )
    var lagunaGatherSiluAB = false

    @Flag(
        name: .customLong("prompt-cache"),
        help:
            "Alternate a cached sibling continuation and forced-cold replay, using conversation snapshots or exact-token checkpoints; reports TTFT, parity and cache/prefill counts. Use a prompt longer than 128 tokens for Gemma."
    )
    var promptCache = false

    @Flag(
        name: [.customLong("hot-cache-ab"), .customLong("mistral-hot-cache-ab")],
        help:
            "Alternate cached and forced-cold continuations on one loaded Mistral-family or GPT-OSS model; requires measured cache reuse and reports TTFT/prefill counts."
    )
    var mistralHotCacheAB = false

    @Option(help: "GPT-OSS reasoning effort: low, medium, or high. Omission preserves the checkpoint template default.")
    var reasoningEffort: String?

    @Flag(
        name: .customLong("wired-memory-session"),
        help:
            "Retain the measured wired-memory budget across measured trials only; warmups stay request-scoped. Metal only."
    )
    var wiredMemorySession = false

    @Option(help: "Warm-up generations excluded from the medians.")
    var warmups = 1

    @Option(help: "Measured generations.")
    var trials = 5

    @Option(help: "Sampling temperature (0...2). Default 0 keeps performance comparisons greedy.")
    var temperature: Float = 0

    @Option(help: "Nucleus sampling probability (greater than 0, at most 1).")
    var topP: Float = 1

    @Option(help: "Deterministic user prompt.")
    var prompt =
        "Write a long, detailed technical tutorial about implementing a lock-free work-stealing scheduler in Swift. Continue with implementation details and code examples until the output limit; do not conclude or summarize early."

    @Option(
        help:
            "Second deterministic user turn used by --prompt-cache and --hot-cache-ab."
    )
    var continuationPrompt =
        "Continue from exactly where you stopped, adding new implementation details and code without repeating the earlier response."

    @Flag(help: "Permit EOS before the requested token count.")
    var allowEarlyStop = false

    @Flag(help: "Disable prompt-cache reuse for every warmup and measured request.")
    var disablePromptReuse = false

    mutating func validate() throws {
        guard temperature.isFinite, (0...2).contains(temperature),
            topP.isFinite, topP > 0, topP <= 1
        else {
            throw ValidationError("--temperature must be in 0...2 and --top-p in (0, 1].")
        }
        guard tokens >= 1, tokens <= 2_048 else {
            throw ValidationError("--tokens must be in 1...2048.")
        }
        guard warmups >= 0, warmups <= 20 else {
            throw ValidationError("--warmups must be in 0...20.")
        }
        guard trials >= 1, trials <= 50 else {
            throw ValidationError("--trials must be in 1...50.")
        }
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError("--prompt must be nonblank.")
        }
        guard !continuationPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError("--continuation-prompt must be nonblank.")
        }
        if gemmaAssistantBlockSize != nil || gemmaAssistantAB || gemmaAssistantQuantizationBits != nil {
            guard gemmaAssistantModel != nil else {
                throw ValidationError("Gemma assistant options require --gemma-assistant-model.")
            }
        }
        if gemmaAssistantModel != nil {
            guard
                dflashModel == nil && !dflashAB && !lagunaFusionAB && !lagunaAttentionGateAB
                    && !lagunaBlockTailAB && !lagunaRouterTopKAB && !lagunaGatherSiluAB
                    && !promptCache && !gemmaDenseFusionAB && !gemmaWindowSlicingAB && !mistralHotCacheAB
            else {
                throw ValidationError("Gemma assistant benchmarks cannot be combined with other A/B or cache modes.")
            }
        }
        guard !disablePromptReuse || (!promptCache && !mistralHotCacheAB) else {
            throw ValidationError("--disable-prompt-reuse cannot be combined with cache comparison modes.")
        }
        if dflashBlockSize != nil, dflashModel == nil {
            throw ValidationError("--dflash-block-size requires --dflash-model.")
        }
        if let reasoningEffort,
            ChatCompletionRequest.ReasoningEffort(rawValue: reasoningEffort) == nil
        {
            throw ValidationError("--reasoning-effort must be low, medium, or high.")
        }
        if dflashAB, dflashModel == nil {
            throw ValidationError("--dflash-ab requires --dflash-model.")
        }
        if lagunaFusionAB, dflashModel != nil {
            throw ValidationError("--laguna-fusion-ab cannot be combined with --dflash-model.")
        }
        if lagunaAttentionGateAB, dflashModel != nil {
            throw ValidationError(
                "--laguna-attention-gate-ab cannot be combined with --dflash-model.")
        }
        if lagunaBlockTailAB, dflashModel != nil {
            throw ValidationError(
                "--laguna-block-tail-ab cannot be combined with --dflash-model.")
        }
        if lagunaRouterTopKAB, dflashModel != nil {
            throw ValidationError(
                "--laguna-router-topk-ab cannot be combined with --dflash-model.")
        }
        if lagunaGatherSiluAB, dflashModel != nil {
            throw ValidationError("--laguna-gather-silu-ab cannot be combined with --dflash-model.")
        }
        if promptCache, dflashModel != nil {
            throw ValidationError("--prompt-cache cannot be combined with --dflash-model.")
        }
        if gemmaDenseFusionAB || gemmaWindowSlicingAB, dflashModel != nil {
            throw ValidationError("--gemma-dense-fusion-ab cannot be combined with --dflash-model.")
        }
        if mistralHotCacheAB, dflashModel != nil {
            throw ValidationError("--hot-cache-ab cannot be combined with --dflash-model.")
        }
        if mistralHotCacheAB, allowEarlyStop {
            throw ValidationError(
                "--hot-cache-ab cannot be combined with --allow-early-stop."
            )
        }
        let exclusiveModes = [
            dflashAB, lagunaFusionAB, gemmaDenseFusionAB, gemmaWindowSlicingAB, lagunaAttentionGateAB,
            lagunaBlockTailAB,
            lagunaRouterTopKAB, lagunaGatherSiluAB, promptCache, mistralHotCacheAB,
        ].filter { $0 }.count
        if wiredMemorySession && (warmups == 0 || exclusiveModes > 0 || gemmaAssistantAB) {
            throw ValidationError(
                "--wired-memory-session requires warmups and cannot be combined with other A/B or cache modes.")
        }
        if exclusiveModes > 1 {
            throw ValidationError(
                "A/B and cache benchmark modes, including --gemma-dense-fusion-ab, are mutually exclusive."
            )
        }
    }

    mutating func run() async throws {
        defer { clearModelRunnerMLXStreams() }

        let modelURL = URL(fileURLWithPath: model).standardizedFileURL
        let outputURL = URL(fileURLWithPath: output).standardizedFileURL
        let runner = try await makeRunner(modelURL: modelURL, outputURL: outputURL)

        if wiredMemorySession {
            _ = try await runner.makeBenchmarkWiredMemorySession()
        }
        let initialPlan = await runner.benchmarkWiredMemoryPlan
        let initialState = await BenchmarkWiredMemoryState.capture()
        await runner.beginBenchmarkWiredMemoryDiagnostics()
        let execution: TrialExecution
        do {
            execution = try await runTrials(runner: runner)
        } catch {
            await runner.waitUntilIdle()
            _ = await runner.takeBenchmarkWiredMemoryDiagnostics()
            if let failure = error as? BenchmarkWiredMemorySession.ExecutionFailure,
                let encoded = try? JSONEncoder().encode(failure.report),
                let text = String(data: encoded, encoding: .utf8)
            {
                print("wired_memory_session_failure=\(text)")
            }
            throw error
        }
        var (report, dflashComparison) = await makeReport(
            runner: runner, warmupRecords: execution.warmups, measuredRecords: execution.trials
        )
        report.wiredMemoryScope = wiredMemorySession ? "session" : "request"
        report.wiredMemorySessionPhase = wiredMemorySession ? "measured_trials" : nil
        report.wiredMemoryInitialPlan = initialPlan
        report.wiredMemoryFinalPlan = await runner.benchmarkWiredMemoryPlan
        report.wiredMemoryInitialState = initialState
        report.wiredMemoryBeforeMeasured = execution.beforeMeasured
        report.wiredMemoryAfterMeasured = execution.afterMeasured
        report.wiredMemorySession = execution.session
        report.wiredMemoryRequests = await runner.takeBenchmarkWiredMemoryDiagnostics()
        report.wiredMemoryCacheLimitBytes = Memory.cacheLimit
        if wiredMemorySession {
            report.wiredMemorySessionFailures = wiredSessionFailures(report)
            report.wiredMemorySessionEligible = report.wiredMemorySessionFailures.isEmpty
            if !report.wiredMemorySessionFailures.isEmpty {
                report.status = "failed_wired_memory_session"
            }
        }
        try writeReport(report, dflashComparison: dflashComparison, to: outputURL)
        if !report.wiredMemorySessionFailures.isEmpty {
            throw BenchmarkError.invalidInput(report.wiredMemorySessionFailures.joined(separator: "; "))
        }
    }

    private func wiredSessionFailures(_ report: RuntimeBenchmarkReport) -> [String] {
        guard let session = report.wiredMemorySession else {
            return ["Session residency diagnostics are missing."]
        }
        var failures = [String]()
        if !session.confirmsRestoredBaseline || report.wiredMemoryAfterMeasured?.confirmsUnwiredBaseline != true {
            failures.append("Session residency restoration was not confirmed by successful setter history.")
        }
        if report.wiredMemoryFinalPlan?.limitBytes != session.requestedLimitBytes {
            failures.append("The measured wired-memory plan changed during the session.")
        }
        if report.wiredMemoryCacheLimitBytes != report.wiredMemoryFinalPlan?.cacheReserveBytes {
            failures.append("The allocator cache limit differs from the measured plan reserve.")
        }
        if report.wiredMemoryRequests.count != report.warmups.count + report.trials.count {
            failures.append("A generation lacks a wired-memory request observation.")
        }
        for request in report.wiredMemoryRequests.suffix(report.trials.count) {
            let state = request.activeState
            if request.requestedLimitBytes != session.requestedLimitBytes
                || request.startReturnedLimitBytes != session.requestedLimitBytes
                || state.currentLimitBytes != session.requestedLimitBytes
                || state.lastSuccessfulBackendLimitBytes != session.requestedLimitBytes
                || state.lastAttemptSucceeded != true || state.baselineBytes != 0
                || !state.backendSupported || state.activeTicketCount != 2 || state.ticketCount != 2
                || state.backendFailureCount != 0 || state.backendSuccessCount != session.started.backendSuccessCount
            {
                failures.append("A measured request lacks an unchanged, successfully applied session capacity.")
                break
            }
        }
        return failures
    }

    private func makeReport(
        runner: LocalModelRunner,
        warmupRecords: [RecordedTrial],
        measuredRecords: [RecordedTrial]
    ) async -> (RuntimeBenchmarkReport, DFlashABComparison?) {
        let speculativeLabel = gemmaAssistantModel == nil ? "dflash" : "gemma_assistant"

        let dflashRecords = measuredRecords.filter { $0.mode == speculativeLabel }
        let targetOnlyRecords = measuredRecords.filter { $0.mode == "target_only" }
        let fusionOffRecords = measuredRecords.filter { $0.mode == "laguna_fusion_off" }
        let fusionOnRecords = measuredRecords.filter { $0.mode == "laguna_fusion_on" }
        let attentionGateEagerRecords = measuredRecords.filter {
            $0.mode == "laguna_attention_gate_eager"
        }
        let attentionGateCompiledRecords = measuredRecords.filter {
            $0.mode == "laguna_attention_gate_compiled"
        }
        let blockTailEagerRecords = measuredRecords.filter {
            $0.mode == "laguna_block_tail_eager"
        }
        let blockTailCompiledRecords = measuredRecords.filter {
            $0.mode == "laguna_block_tail_compiled"
        }
        let routerTopKLegacyRecords = measuredRecords.filter {
            $0.mode == "laguna_router_topk_legacy"
        }
        let routerTopKFusedRecords = measuredRecords.filter {
            $0.mode == "laguna_router_topk_fused"
        }
        let gatherSiluStockRecords = measuredRecords.filter { $0.mode == "laguna_gather_silu_stock" }
        let gatherSiluFusedRecords = measuredRecords.filter { $0.mode == "laguna_gather_silu_fused" }
        let promptCacheCachedRecords = measuredRecords.filter {
            $0.mode == "prompt_cache_cached"
        }
        let promptCacheColdRecords = measuredRecords.filter {
            $0.mode == "prompt_cache_cold"
        }
        let hotCacheLabels = HotCacheModeLabels(isMistral: runner.supportsMistralHotConversationCache)
        let mistralHotCacheCachedRecords = measuredRecords.filter {
            $0.mode == hotCacheLabels.cached
        }
        let mistralHotCacheColdRecords = measuredRecords.filter {
            $0.mode == hotCacheLabels.cold
        }
        let primaryRecords: [RecordedTrial]
        if dflashAB || gemmaAssistantAB {
            primaryRecords = dflashRecords
        } else if lagunaFusionAB {
            primaryRecords = fusionOnRecords
        } else if gemmaDenseFusionAB {
            primaryRecords = measuredRecords.filter { $0.mode == "gemma_dense_fusion_on" }
        } else if gemmaWindowSlicingAB {
            primaryRecords = measuredRecords.filter { $0.mode == "gemma_window_slicing_on" }
        } else if lagunaAttentionGateAB {
            primaryRecords = attentionGateCompiledRecords
        } else if lagunaBlockTailAB {
            primaryRecords = blockTailCompiledRecords
        } else if lagunaRouterTopKAB {
            primaryRecords = routerTopKFusedRecords
        } else if lagunaGatherSiluAB {
            primaryRecords = gatherSiluFusedRecords
        } else if promptCache || mistralHotCacheAB {
            primaryRecords = []
        } else {
            primaryRecords = measuredRecords
        }
        let dflashComparison = makeDFlashComparison(targetOnlyRecords: targetOnlyRecords, dflashRecords: dflashRecords)
        let lagunaFusionComparison = makeLagunaFusionComparison(
            fusionOffRecords: fusionOffRecords, fusionOnRecords: fusionOnRecords)
        let lagunaAttentionGateComparison = makeLagunaAttentionGateComparison(
            attentionGateEagerRecords: attentionGateEagerRecords,
            attentionGateCompiledRecords: attentionGateCompiledRecords)
        let lagunaBlockTailComparison = makeLagunaBlockTailComparison(
            blockTailEagerRecords: blockTailEagerRecords, blockTailCompiledRecords: blockTailCompiledRecords)
        let lagunaRouterTopKComparison = makeLagunaRouterTopKComparison(
            routerTopKLegacyRecords: routerTopKLegacyRecords, routerTopKFusedRecords: routerTopKFusedRecords)
        let lagunaGatherSiluComparison = makeLagunaGatherSiluComparison(
            gatherSiluStockRecords: gatherSiluStockRecords, gatherSiluFusedRecords: gatherSiluFusedRecords)
        let promptCacheModeComparison =
            promptCache
            ? makePromptCacheComparison(cachedRecords: promptCacheCachedRecords, coldRecords: promptCacheColdRecords)
            : nil
        let mistralHotCacheModeComparison =
            mistralHotCacheAB
            ? makeHotCacheComparison(
                cachedRecords: mistralHotCacheCachedRecords,
                coldRecords: mistralHotCacheColdRecords,
                measuredRecords: measuredRecords, labels: hotCacheLabels)
            : nil

        let report = RuntimeBenchmarkReport(
            status: ((gemmaAssistantAB || dflashAB) && dflashComparison?.outputsMatchExactly == false)
                || (promptCache && promptCacheModeComparison?.outputsMatchExactly == false)
                ? "failed_output_parity" : "measured",
            createdAt: ISO8601DateFormatter().string(from: Date()),
            modelPath: runner.modelPath,
            servedModelName: runner.servedModelName,
            modelImplementation: runner.modelImplementation,
            dflashModelPath: runner.dflashModelPath,
            dflashBlockSize: runner.dflashBlockSize,
            temperature: temperature,
            topP: topP,
            gemmaAssistantModelPath: runner.gemmaAssistantModelPath,
            gemmaAssistantBlockSize: runner.gemmaAssistantBlockSize,
            gemmaAssistantQuantizationBits: runner.gemmaAssistantQuantizationBits,
            gemmaAssistantQuantizedModuleCount: runner.gemmaAssistantQuantizedModuleCount,
            engine: runner.engine.rawValue,
            prompt: prompt,
            continuationPrompt: continuationPrompt,
            contextLength: runner.contextLength,
            prefillStepSize: runner.prefillStepSize,
            kvCompression: runner.kvCompression,
            reasoningEffort: reasoningEffort,
            memoryLimitBytes: runner.memoryLimitBytes,
            allowEarlyStop: allowEarlyStop,
            promptReusePolicy: disablePromptReuse ? "disabled" : "mode_default",
            requestedTokens: tokens,
            warmupCount: warmupRecords.count,
            measuredTrials: measuredRecords.count,
            medianPromptTokensPerSecond: promptCache || mistralHotCacheAB
                ? nil : median(primaryRecords.map(\.metrics.promptTokensPerSecond)),
            medianDecodeTokensPerSecond: promptCache || mistralHotCacheAB
                ? nil : median(primaryRecords.map(\.metrics.tokensPerSecond)),
            dflashABComparison: dflashAB ? dflashComparison : nil,
            gemmaAssistantABComparison: gemmaAssistantAB
                ? dflashComparison.map {
                    GemmaAssistantABComparison(
                        targetMedian: $0.targetOnlyMedianDecodeTokensPerSecond,
                        assistantMedian: $0.dflashMedianDecodeTokensPerSecond,
                        speedupPercent: $0.dflashSpeedupPercent, outputsMatchExactly: $0.outputsMatchExactly)
                } : nil,

            lagunaFusionABComparison: lagunaFusionComparison,
            lagunaAttentionGateABComparison: lagunaAttentionGateComparison,
            lagunaBlockTailABComparison: lagunaBlockTailComparison,
            lagunaRouterTopKABComparison: lagunaRouterTopKComparison,
            lagunaGatherSiluABComparison: lagunaGatherSiluComparison,
            lagunaGatherSiluTraceCount: lagunaGatherSiluAB
                ? (await runner.lagunaFusedGateUpSiluCoverage()).traces : nil,
            promptCacheComparison: promptCacheModeComparison,
            mistralHotCacheABComparison: runner.supportsMistralHotConversationCache
                ? mistralHotCacheModeComparison : nil,
            hotCacheABComparison: mistralHotCacheModeComparison,
            warmups: warmupRecords,
            trials: measuredRecords
        )
        return (report, dflashComparison)
    }

    private func makeDFlashComparison(targetOnlyRecords: [RecordedTrial], dflashRecords: [RecordedTrial])
        -> DFlashABComparison?
    {
        guard dflashAB || gemmaAssistantAB else {
            return nil
        }
        let targetMedian = median(targetOnlyRecords.map(\.metrics.tokensPerSecond))
        let dflashMedian = median(dflashRecords.map(\.metrics.tokensPerSecond))
        let referenceContent = targetOnlyRecords.first?.content
        let outputsMatch =
            referenceContent != nil
            && targetOnlyRecords.allSatisfy { $0.content == referenceContent }
            && dflashRecords.allSatisfy { $0.content == referenceContent }
        let firstDivergence = referenceContent.flatMap { reference in
            (targetOnlyRecords.dropFirst().map(\.content) + dflashRecords.map(\.content))
                .compactMap { firstUTF8DivergenceOffset(reference, $0) }
                .min()
        }
        return DFlashABComparison(
            trialsPerMode: trials,
            targetOnlyMedianDecodeTokensPerSecond: targetMedian,
            dflashMedianDecodeTokensPerSecond: dflashMedian,
            dflashSpeedupPercent: 100 * (dflashMedian / targetMedian - 1),
            outputsMatchExactly: outputsMatch,
            firstOutputDivergenceUTF8Offset: firstDivergence
        )
    }

    private func makeLagunaFusionComparison(fusionOffRecords: [RecordedTrial], fusionOnRecords: [RecordedTrial])
        -> LagunaFusionABComparison?
    {
        guard lagunaFusionAB else {
            return nil
        }
        let unfusedMedian = median(fusionOffRecords.map(\.metrics.tokensPerSecond))
        let fusedMedian = median(fusionOnRecords.map(\.metrics.tokensPerSecond))
        let referenceContent = fusionOffRecords.first?.content
        let outputsMatch =
            referenceContent != nil
            && fusionOffRecords.allSatisfy { $0.content == referenceContent }
            && fusionOnRecords.allSatisfy { $0.content == referenceContent }
        return LagunaFusionABComparison(
            trialsPerMode: trials,
            unfusedMedianDecodeTokensPerSecond: unfusedMedian,
            fusedMedianDecodeTokensPerSecond: fusedMedian,
            fusedSpeedupPercent: 100 * (fusedMedian / unfusedMedian - 1),
            outputsMatchExactly: outputsMatch)
    }

    private func makeLagunaAttentionGateComparison(
        attentionGateEagerRecords: [RecordedTrial], attentionGateCompiledRecords: [RecordedTrial]
    ) -> LagunaAttentionGateABComparison? {
        guard lagunaAttentionGateAB else {
            return nil
        }
        let eagerMedian = median(attentionGateEagerRecords.map(\.metrics.tokensPerSecond))
        let compiledMedian = median(
            attentionGateCompiledRecords.map(\.metrics.tokensPerSecond))
        let referenceContent = attentionGateEagerRecords.first?.content
        let outputsMatch =
            referenceContent != nil
            && attentionGateEagerRecords.allSatisfy { $0.content == referenceContent }
            && attentionGateCompiledRecords.allSatisfy { $0.content == referenceContent }
        return LagunaAttentionGateABComparison(
            trialsPerMode: trials,
            eagerMedianDecodeTokensPerSecond: eagerMedian,
            compiledMedianDecodeTokensPerSecond: compiledMedian,
            compiledSpeedupPercent: 100 * (compiledMedian / eagerMedian - 1),
            outputsMatchExactly: outputsMatch
        )
    }

    private func makeLagunaBlockTailComparison(
        blockTailEagerRecords: [RecordedTrial], blockTailCompiledRecords: [RecordedTrial]
    ) -> LagunaBlockTailABComparison? {
        guard lagunaBlockTailAB else {
            return nil
        }
        let eagerMedian = median(blockTailEagerRecords.map(\.metrics.tokensPerSecond))
        let compiledMedian = median(
            blockTailCompiledRecords.map(\.metrics.tokensPerSecond))
        let referenceContent = blockTailEagerRecords.first?.content
        let outputsMatch =
            referenceContent != nil
            && blockTailEagerRecords.allSatisfy { $0.content == referenceContent }
            && blockTailCompiledRecords.allSatisfy { $0.content == referenceContent }
        return LagunaBlockTailABComparison(
            trialsPerMode: trials,
            eagerMedianDecodeTokensPerSecond: eagerMedian,
            compiledMedianDecodeTokensPerSecond: compiledMedian,
            compiledSpeedupPercent: 100 * (compiledMedian / eagerMedian - 1),
            outputsMatchExactly: outputsMatch
        )
    }

    private func makeLagunaRouterTopKComparison(
        routerTopKLegacyRecords: [RecordedTrial], routerTopKFusedRecords: [RecordedTrial]
    ) -> LagunaRouterTopKABComparison? {
        guard lagunaRouterTopKAB else {
            return nil
        }
        let legacyMedian = median(
            routerTopKLegacyRecords.map(\.metrics.tokensPerSecond))
        let fusedMedian = median(
            routerTopKFusedRecords.map(\.metrics.tokensPerSecond))
        let referenceContent = routerTopKLegacyRecords.first?.content
        let outputsMatch =
            referenceContent != nil
            && routerTopKLegacyRecords.allSatisfy { $0.content == referenceContent }
            && routerTopKFusedRecords.allSatisfy { $0.content == referenceContent }
        return LagunaRouterTopKABComparison(
            trialsPerMode: trials,
            legacyMedianDecodeTokensPerSecond: legacyMedian,
            fusedMedianDecodeTokensPerSecond: fusedMedian,
            fusedSpeedupPercent: 100 * (fusedMedian / legacyMedian - 1),
            outputsMatchExactly: outputsMatch
        )
    }

    private func makeLagunaGatherSiluComparison(
        gatherSiluStockRecords: [RecordedTrial], gatherSiluFusedRecords: [RecordedTrial]
    ) -> LagunaRouterTopKABComparison? {
        guard lagunaGatherSiluAB else {
            return nil
        }
        let stockMedian = median(gatherSiluStockRecords.map(\.metrics.tokensPerSecond))
        let fusedMedian = median(gatherSiluFusedRecords.map(\.metrics.tokensPerSecond))
        let reference = gatherSiluStockRecords.first?.content
        return LagunaRouterTopKABComparison(
            trialsPerMode: trials, legacyMedianDecodeTokensPerSecond: stockMedian,
            fusedMedianDecodeTokensPerSecond: fusedMedian,
            fusedSpeedupPercent: 100 * (fusedMedian / stockMedian - 1),
            outputsMatchExactly: reference != nil
                && (gatherSiluStockRecords + gatherSiluFusedRecords).allSatisfy { $0.content == reference })
    }

    private func makePromptCacheComparison(
        cachedRecords: [RecordedTrial], coldRecords: [RecordedTrial]
    ) -> PromptCacheComparison {
        let cachedTTFT = median(
            cachedRecords.map(\.timeToFirstTokenMilliseconds))
        let coldTTFT = median(coldRecords.map(\.timeToFirstTokenMilliseconds))
        let outputsMatch =
            !cachedRecords.isEmpty && cachedRecords.count == coldRecords.count
            && zip(cachedRecords, coldRecords).allSatisfy { cached, cold in
                cached.content == cold.content && cached.reasoning == cold.reasoning
                    && cached.metrics.generationTokenCount == cold.metrics.generationTokenCount
                    && cached.metrics.stopReason == cold.metrics.stopReason
                    && cached.promptTokenIDFingerprint == cold.promptTokenIDFingerprint
            }
        return PromptCacheComparison(
            trialsPerMode: trials,
            cachedMedianTimeToFirstTokenMilliseconds: cachedTTFT,
            coldMedianTimeToFirstTokenMilliseconds: coldTTFT,
            cachedTTFTReductionPercent: 100 * (1 - cachedTTFT / coldTTFT),
            cachedMedianTotalPromptTokenCount: median(
                cachedRecords.map { Double($0.metrics.promptTokenCount) }),
            cachedMedianPrefilledPromptTokenCount: median(
                cachedRecords.map { Double($0.metrics.prefilledPromptTokenCount) }),
            cachedMedianReusedPromptTokenCount: median(
                cachedRecords.map { Double($0.metrics.cachedPromptTokenCount) }),
            coldMedianTotalPromptTokenCount: median(
                coldRecords.map { Double($0.metrics.promptTokenCount) }),
            coldMedianPrefilledPromptTokenCount: median(
                coldRecords.map { Double($0.metrics.prefilledPromptTokenCount) }),
            coldMedianReusedPromptTokenCount: median(
                coldRecords.map { Double($0.metrics.cachedPromptTokenCount) }),
            outputsMatchExactly: outputsMatch
        )
    }

    private func makeHotCacheComparison(
        cachedRecords: [RecordedTrial], coldRecords: [RecordedTrial],
        measuredRecords: [RecordedTrial], labels: HotCacheModeLabels
    ) -> MistralHotCacheABComparison {
        let cachedTTFT = median(
            cachedRecords.map(\.timeToFirstTokenMilliseconds))
        let coldTTFT = median(
            coldRecords.map(\.timeToFirstTokenMilliseconds))
        let outputsMatch = mistralHotCacheOutputsMatchExactly(measuredRecords, labels: labels)
        let firstDivergence = mistralHotCacheFirstOutputDivergence(measuredRecords, labels: labels)
        return MistralHotCacheABComparison(
            trialsPerMode: trials,
            cachedMedianTimeToFirstTokenMilliseconds: cachedTTFT,
            coldMedianTimeToFirstTokenMilliseconds: coldTTFT,
            cachedTTFTReductionPercent: 100 * (1 - cachedTTFT / coldTTFT),
            cachedMedianTotalPromptTokenCount: median(
                cachedRecords.map { Double($0.metrics.promptTokenCount) }),
            cachedMedianPrefilledPromptTokenCount: median(
                cachedRecords.map { Double($0.metrics.prefilledPromptTokenCount) }),
            cachedMedianReusedPromptTokenCount: median(
                cachedRecords.map { Double($0.metrics.cachedPromptTokenCount) }),
            coldMedianTotalPromptTokenCount: median(
                coldRecords.map { Double($0.metrics.promptTokenCount) }),
            coldMedianPrefilledPromptTokenCount: median(
                coldRecords.map { Double($0.metrics.prefilledPromptTokenCount) }),
            coldMedianReusedPromptTokenCount: median(
                coldRecords.map { Double($0.metrics.cachedPromptTokenCount) }),
            outputsMatchExactly: outputsMatch,
            firstOutputDivergenceUTF8Offset: firstDivergence
        )
    }

    private func writeReport(
        _ report: RuntimeBenchmarkReport,
        dflashComparison: DFlashABComparison?,
        to outputURL: URL
    ) throws {
        let lagunaGatherSiluComparison = report.lagunaGatherSiluABComparison
        let lagunaFusionComparison = report.lagunaFusionABComparison
        let lagunaAttentionGateComparison = report.lagunaAttentionGateABComparison
        let lagunaBlockTailComparison = report.lagunaBlockTailABComparison
        let lagunaRouterTopKComparison = report.lagunaRouterTopKABComparison
        let promptCacheModeComparison = report.promptCacheComparison
        let mistralHotCacheModeComparison = report.hotCacheABComparison

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(report).write(to: outputURL, options: .atomic)
        if let comparison = lagunaGatherSiluComparison {
            guard (report.lagunaGatherSiluTraceCount ?? 0) > 0 else {
                throw BenchmarkError.invalidInput(
                    "No fused gate/up-SiLU graph was built; stock fallback report retained.")
            }
            print(
                "Gather/SiLU A/B: stock \(comparison.legacyMedianDecodeTokensPerSecond) tok/s, fused \(comparison.fusedMedianDecodeTokensPerSecond) tok/s; output text matches: \(comparison.outputsMatchExactly)"
            )
            guard comparison.outputsMatchExactly else {
                throw BenchmarkError.invalidInput(
                    "Fused gate/up-SiLU changed greedy output text; diagnostic report retained at \(outputURL.path).")
            }
        }
        if let comparison = dflashComparison {
            print(
                String(
                    format: "A/B median: target %.2f tok/s, assistant %.2f tok/s (%+.2f%%)",
                    comparison.targetOnlyMedianDecodeTokensPerSecond,
                    comparison.dflashMedianDecodeTokensPerSecond,
                    comparison.dflashSpeedupPercent
                )
            )
            print("A/B outputs match exactly: \(comparison.outputsMatchExactly)")
            if let offset = comparison.firstOutputDivergenceUTF8Offset {
                print("A/B first output divergence: UTF-8 byte \(offset)")
            }
            if gemmaAssistantAB && !comparison.outputsMatchExactly {
                throw BenchmarkError.invalidInput(
                    "Gemma assistant changed greedy output; diagnostic report retained at \(outputURL.path).")
            }
        } else if let comparison = lagunaFusionComparison {
            print(
                String(
                    format: "A/B median: unfused %.2f tok/s, fused %.2f tok/s (%+.2f%%)",
                    comparison.unfusedMedianDecodeTokensPerSecond,
                    comparison.fusedMedianDecodeTokensPerSecond,
                    comparison.fusedSpeedupPercent
                )
            )
            print("A/B outputs match exactly: \(comparison.outputsMatchExactly)")
        } else if let comparison = lagunaAttentionGateComparison {
            print(
                String(
                    format: "A/B median: eager gate %.2f tok/s, compiled gate %.2f tok/s (%+.2f%%)",
                    comparison.eagerMedianDecodeTokensPerSecond,
                    comparison.compiledMedianDecodeTokensPerSecond,
                    comparison.compiledSpeedupPercent
                )
            )
            print("A/B outputs match exactly: \(comparison.outputsMatchExactly)")
        } else if let comparison = lagunaBlockTailComparison {
            print(
                String(
                    format: "A/B median: eager block tail %.2f tok/s, compiled block tail %.2f tok/s (%+.2f%%)",
                    comparison.eagerMedianDecodeTokensPerSecond,
                    comparison.compiledMedianDecodeTokensPerSecond,
                    comparison.compiledSpeedupPercent
                )
            )
            print("A/B outputs match exactly: \(comparison.outputsMatchExactly)")
        } else if let comparison = lagunaRouterTopKComparison {
            print(
                String(
                    format: "A/B median: legacy router %.2f tok/s, fused router %.2f tok/s (%+.2f%%)",
                    comparison.legacyMedianDecodeTokensPerSecond,
                    comparison.fusedMedianDecodeTokensPerSecond,
                    comparison.fusedSpeedupPercent
                )
            )
            print("A/B outputs match exactly: \(comparison.outputsMatchExactly)")
        } else if let comparison = promptCacheModeComparison {
            print(
                String(
                    format: "prompt-cache median TTFT: cached %.2f ms, cold %.2f ms (%+.2f%% reduction)",
                    comparison.cachedMedianTimeToFirstTokenMilliseconds,
                    comparison.coldMedianTimeToFirstTokenMilliseconds,
                    comparison.cachedTTFTReductionPercent
                )
            )
            print(
                String(
                    format: "prompt tokens (cached): %.0f total, %.0f reused, %.0f prefilled",
                    comparison.cachedMedianTotalPromptTokenCount,
                    comparison.cachedMedianReusedPromptTokenCount,
                    comparison.cachedMedianPrefilledPromptTokenCount
                )
            )
            print("continuation outputs match exactly: \(comparison.outputsMatchExactly)")
        } else if let comparison = mistralHotCacheModeComparison {
            print(
                String(
                    format:
                        "Hot-cache median TTFT: cached %.2f ms, cold %.2f ms (%+.2f%% reduction)",
                    comparison.cachedMedianTimeToFirstTokenMilliseconds,
                    comparison.coldMedianTimeToFirstTokenMilliseconds,
                    comparison.cachedTTFTReductionPercent
                )
            )
            print(
                String(
                    format: "prompt tokens (hot): %.0f total, %.0f reused, %.0f prefilled",
                    comparison.cachedMedianTotalPromptTokenCount,
                    comparison.cachedMedianReusedPromptTokenCount,
                    comparison.cachedMedianPrefilledPromptTokenCount
                )
            )
            print("continuation outputs match exactly: \(comparison.outputsMatchExactly)")
            if let offset = comparison.firstOutputDivergenceUTF8Offset {
                print("continuation first output divergence: UTF-8 byte \(offset)")
            }
        } else {
            guard let medianDecode = report.medianDecodeTokensPerSecond else {
                throw BenchmarkError.invalidInput("missing decode median")
            }
            print(
                "median decode: "
                    + String(format: "%.2f tok/s", medianDecode)
            )
        }
        print("Wrote \(outputURL.path)")
    }

    private func runTrials(runner: LocalModelRunner) async throws -> TrialExecution {
        let messages = [OpenAIMessage(role: "user", content: prompt)]
        var warmupRecords = [RecordedTrial]()
        var sequence = 0
        let speculativeLabel = gemmaAssistantModel == nil ? "dflash" : "gemma_assistant"
        let ordinaryMode = GenerationMode(
            label: dflashModel == nil && gemmaAssistantModel == nil ? "target_only" : speculativeLabel,
            useDFlash: dflashModel != nil || gemmaAssistantModel != nil,
            useLagunaFusion: true,
            useCompiledAttentionGate: true)
        if promptCache {
            let records = try await runPromptCacheTrials(runner: runner, messages: messages, mode: ordinaryMode)
            return TrialExecution(warmups: records.0, trials: records.1)
        }
        if mistralHotCacheAB {
            let records = try await runHotCacheTrials(runner: runner, messages: messages, mode: ordinaryMode)
            return TrialExecution(warmups: records.0, trials: records.1)
        }

        let comparisonModes = selectComparisonModes(speculativeLabel: speculativeLabel)
        let warmupModes = comparisonModes.map { [$0.0, $0.1] } ?? [ordinaryMode]
        for warmupIndex in 0..<warmups {
            for mode in warmupModes {
                let record = try await runGeneration(
                    runner: runner,
                    messages: messages,
                    sequence: sequence,
                    mode: mode
                )
                sequence += 1
                warmupRecords.append(record)
                print(
                    "warmup \(warmupIndex + 1)/\(warmups) [\(record.mode)]: "
                        + trialSummary(record)
                )
            }
        }

        await runner.waitUntilIdle()
        let beforeMeasured = await BenchmarkWiredMemoryState.capture()
        let measuredRecords: [RecordedTrial]
        var sessionReport: BenchmarkWiredMemorySessionReport?
        if wiredMemorySession {
            guard beforeMeasured.confirmsUnwiredBaseline else {
                throw BenchmarkError.invalidInput(
                    "Session residency requires a known zero baseline restored by successful warmup setter calls.")
            }
            let session = try await runner.makeBenchmarkWiredMemorySession()
            (measuredRecords, sessionReport) = try await session.run(
                cleanup: { await runner.waitUntilIdle() },
                operation: {
                    try await runMeasuredTrials(
                        runner: runner, messages: messages, firstSequence: sequence,
                        ordinaryMode: ordinaryMode, comparisonModes: comparisonModes)
                })
        } else {
            measuredRecords = try await runMeasuredTrials(
                runner: runner, messages: messages, firstSequence: sequence,
                ordinaryMode: ordinaryMode, comparisonModes: comparisonModes)
            await runner.waitUntilIdle()
        }
        let afterMeasured = await BenchmarkWiredMemoryState.capture()
        return TrialExecution(
            warmups: warmupRecords, trials: measuredRecords, session: sessionReport,
            beforeMeasured: beforeMeasured, afterMeasured: afterMeasured)
    }

    private func runMeasuredTrials(
        runner: LocalModelRunner, messages: [OpenAIMessage], firstSequence: Int,
        ordinaryMode: GenerationMode, comparisonModes: (GenerationMode, GenerationMode)?
    ) async throws -> [RecordedTrial] {
        var sequence = firstSequence
        var measuredRecords = [RecordedTrial]()
        for trialIndex in 0..<trials {
            let measuredModes =
                comparisonModes.map { modes in
                    trialIndex.isMultiple(of: 2) ? [modes.0, modes.1] : [modes.1, modes.0]
                } ?? [ordinaryMode]
            for mode in measuredModes {
                let record = try await runGeneration(
                    runner: runner,
                    messages: messages,
                    sequence: sequence,
                    mode: mode
                )
                sequence += 1
                measuredRecords.append(record)
                print(
                    "trial \(trialIndex + 1)/\(trials) [\(record.mode)]: "
                        + trialSummary(record)
                )
            }
        }

        return measuredRecords
    }

    private func selectComparisonModes(speculativeLabel: String) -> (GenerationMode, GenerationMode)? {
        let targetOnlyMode = GenerationMode(
            label: "target_only", useDFlash: false, useLagunaFusion: true,
            // Compare against ordinary production decoding, including its Metal
            // single-token fast paths. Disabling them understates DFlash's cost.
            useCompiledAttentionGate: true)
        let dflashMode = GenerationMode(
            label: speculativeLabel, useDFlash: true, useLagunaFusion: true,
            useCompiledAttentionGate: true, useCompiledBlockTail: false,
            useFusedRouterTopK: false)
        let fusionOffMode = GenerationMode(
            label: "laguna_fusion_off", useDFlash: false, useLagunaFusion: false,
            useCompiledAttentionGate: true, useCompiledBlockTail: false,
            useFusedRouterTopK: false)
        let fusionOnMode = GenerationMode(
            label: "laguna_fusion_on", useDFlash: false, useLagunaFusion: true,
            useCompiledAttentionGate: true, useCompiledBlockTail: false,
            useFusedRouterTopK: false)
        let gemmaFusionOffMode = GenerationMode(
            label: "gemma_dense_fusion_off", useDFlash: false, useLagunaFusion: true,
            useCompiledAttentionGate: true, useGemmaDenseFusion: false)
        let gemmaFusionOnMode = GenerationMode(
            label: "gemma_dense_fusion_on", useDFlash: false, useLagunaFusion: true,
            useCompiledAttentionGate: true, useGemmaDenseFusion: true)
        let gemmaWindowOffMode = GenerationMode(
            label: "gemma_window_slicing_off", useDFlash: false, useLagunaFusion: true,
            useCompiledAttentionGate: true, useGemmaWindowSlicing: false)
        let gemmaWindowOnMode = GenerationMode(
            label: "gemma_window_slicing_on", useDFlash: false, useLagunaFusion: true,
            useCompiledAttentionGate: true, useGemmaWindowSlicing: true)
        let attentionGateEagerMode = GenerationMode(
            label: "laguna_attention_gate_eager", useDFlash: false,
            useLagunaFusion: true, useCompiledAttentionGate: false,
            useCompiledBlockTail: false, useFusedRouterTopK: false)
        let attentionGateCompiledMode = GenerationMode(
            label: "laguna_attention_gate_compiled", useDFlash: false,
            useLagunaFusion: true, useCompiledAttentionGate: true,
            useCompiledBlockTail: false, useFusedRouterTopK: false)
        let blockTailEagerMode = GenerationMode(
            label: "laguna_block_tail_eager", useDFlash: false,
            useLagunaFusion: true, useCompiledAttentionGate: true,
            useCompiledBlockTail: false, useFusedRouterTopK: false)
        let blockTailCompiledMode = GenerationMode(
            label: "laguna_block_tail_compiled", useDFlash: false,
            useLagunaFusion: true, useCompiledAttentionGate: true,
            useCompiledBlockTail: true, useFusedRouterTopK: false)
        let routerTopKLegacyMode = GenerationMode(
            label: "laguna_router_topk_legacy", useDFlash: false,
            useLagunaFusion: true, useCompiledAttentionGate: true,
            useCompiledBlockTail: true, useFusedRouterTopK: false)
        let routerTopKFusedMode = GenerationMode(
            label: "laguna_router_topk_fused", useDFlash: false,
            useLagunaFusion: true, useCompiledAttentionGate: true,
            useCompiledBlockTail: true, useFusedRouterTopK: true)

        let gatherSiluStockMode = GenerationMode(
            label: "laguna_gather_silu_stock", useDFlash: false,
            useLagunaFusion: true, useCompiledAttentionGate: true,
            useCompiledBlockTail: true, useFusedRouterTopK: true, useFusedGateUpSilu: false)
        let gatherSiluFusedMode = GenerationMode(
            label: "laguna_gather_silu_fused", useDFlash: false,
            useLagunaFusion: true, useCompiledAttentionGate: true,
            useCompiledBlockTail: true, useFusedRouterTopK: true, useFusedGateUpSilu: true)

        let comparisonModes: (GenerationMode, GenerationMode)?
        if dflashAB || gemmaAssistantAB {
            comparisonModes = (targetOnlyMode, dflashMode)
        } else if lagunaFusionAB {
            comparisonModes = (fusionOffMode, fusionOnMode)
        } else if gemmaDenseFusionAB {
            comparisonModes = (gemmaFusionOffMode, gemmaFusionOnMode)
        } else if gemmaWindowSlicingAB {
            comparisonModes = (gemmaWindowOffMode, gemmaWindowOnMode)
        } else if lagunaAttentionGateAB {
            comparisonModes = (attentionGateEagerMode, attentionGateCompiledMode)
        } else if lagunaBlockTailAB {
            comparisonModes = (blockTailEagerMode, blockTailCompiledMode)
        } else if lagunaRouterTopKAB {
            comparisonModes = (routerTopKLegacyMode, routerTopKFusedMode)
        } else if lagunaGatherSiluAB {
            comparisonModes = (gatherSiluStockMode, gatherSiluFusedMode)
        } else {
            comparisonModes = nil
        }
        return comparisonModes
    }

    private func runPromptCacheTrials(
        runner: LocalModelRunner, messages: [OpenAIMessage], mode: GenerationMode
    ) async throws -> ([RecordedTrial], [RecordedTrial]) {
        var warmupRecords = [RecordedTrial]()
        var measuredRecords = [RecordedTrial]()
        var sequence = 0
        for warmupIndex in 0..<warmups {
            let records = try await runPromptCacheProbe(
                runner: runner,
                sequence: sequence,
                seedMessages: messages,
                mode: mode,
                coldFirst: warmupIndex.isMultiple(of: 2)
            )
            sequence += records.count
            warmupRecords.append(contentsOf: records)
            printPromptCacheProbe(records, prefix: "warmup \(warmupIndex + 1)/\(warmups)")
        }
        for trialIndex in 0..<trials {
            let records = try await runPromptCacheProbe(
                runner: runner,
                sequence: sequence,
                seedMessages: messages,
                mode: mode,
                coldFirst: trialIndex.isMultiple(of: 2)
            )
            sequence += records.count
            measuredRecords.append(contentsOf: records)
            printPromptCacheProbe(records, prefix: "trial \(trialIndex + 1)/\(trials)")
        }
        return (warmupRecords, measuredRecords)
    }

    private func runHotCacheTrials(
        runner: LocalModelRunner, messages: [OpenAIMessage], mode: GenerationMode
    ) async throws -> ([RecordedTrial], [RecordedTrial]) {
        var warmupRecords = [RecordedTrial]()
        var measuredRecords = [RecordedTrial]()
        var sequence = 0
        for warmupIndex in 0..<warmups {
            let records = try await runMistralHotCacheProbe(
                runner: runner,
                sequence: sequence,
                seedMessages: messages,
                mode: mode,
                coldFirst: warmupIndex.isMultiple(of: 2)
            )
            sequence += records.count
            warmupRecords.append(contentsOf: records)
            printMistralHotCacheProbe(
                records, prefix: "warmup \(warmupIndex + 1)/\(warmups)")
        }
        for trialIndex in 0..<trials {
            let records = try await runMistralHotCacheProbe(
                runner: runner,
                sequence: sequence,
                seedMessages: messages,
                mode: mode,
                coldFirst: trialIndex.isMultiple(of: 2)
            )
            sequence += records.count
            measuredRecords.append(contentsOf: records)
            printMistralHotCacheProbe(
                records, prefix: "trial \(trialIndex + 1)/\(trials)")
        }
        return (warmupRecords, measuredRecords)
    }

    private func makeRunner(modelURL: URL, outputURL: URL) async throws -> LocalModelRunner {
        if gemmaDenseFusionAB || gemmaWindowSlicingAB {
            let config =
                try JSONSerialization.jsonObject(
                    with: Data(contentsOf: modelURL.appendingPathComponent("config.json"))) as? [String: Any]
            guard let type = config?["model_type"] as? String,
                ["gemma4", "gemma4_text", "gemma4_unified"].contains(type)
            else {
                throw ValidationError("--gemma-dense-fusion-ab requires a Gemma 4 checkpoint.")
            }
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelURL.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw BenchmarkError.invalidInput("model directory does not exist: \(modelURL.path)")
        }
        guard !FileManager.default.fileExists(atPath: outputURL.path) else {
            throw BenchmarkError.invalidInput("output already exists: \(outputURL.path)")
        }
        if lagunaRouterTopKAB || lagunaGatherSiluAB {
            try validateFusedRouterCheckpoint(modelURL)
        }
        let requestedEngine: ModelEngine
        do {
            requestedEngine = try ModelEngine(argument: engine)
        } catch {
            throw ValidationError(error.localizedDescription)
        }

        let runner = try await LocalModelRunner(
            modelPath: modelURL.path,
            servedModelName: "runtime-benchmark",
            engine: requestedEngine,
            maximumTokens: tokens,
            dflashModelPath: dflashModel,
            dflashBlockSize: dflashBlockSize,
            gemmaAssistantModelPath: gemmaAssistantModel,
            gemmaAssistantBlockSize: gemmaAssistantBlockSize,
            gemmaAssistantQuantizationBits: gemmaAssistantQuantizationBits,
            longContext: try LongContextOptions(
                contextLength: contextLength,
                prefillStepSize: prefillStepSize, kvCompression: kvCompression)
        )
        if mistralHotCacheAB, !runner.supportsHotConversationCache {
            throw BenchmarkError.unsupportedMistralHotCacheModel
        }
        if lagunaRouterTopKAB || lagunaGatherSiluAB, runner.engine != .metal {
            throw BenchmarkError.invalidInput(
                "--laguna-router-topk-ab requires the Metal engine; resolved \(runner.engine.rawValue)."
            )
        }
        if lagunaGatherSiluAB {
            let coverage = await runner.lagunaFusedGateUpSiluCoverage()
            guard coverage.sparse > 0, coverage.eligible == coverage.sparse else {
                throw BenchmarkError.invalidInput(
                    "Fused gate/up-SiLU requires every sparse layer to match BF16 affine Q4/G64 E256/top8/K2048/hidden512; eligible \(coverage.eligible)/\(coverage.sparse)."
                )
            }
        }
        return runner
    }

    private func validateFusedRouterCheckpoint(_ modelURL: URL) throws {
        let configURL = modelURL.appendingPathComponent("config.json")
        let data: Data
        do {
            data = try Data(contentsOf: configURL)
        } catch {
            throw BenchmarkError.invalidInput(
                "--laguna-router-topk-ab requires a readable config.json: \(error.localizedDescription)"
            )
        }
        guard
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            object["model_type"] as? String == "laguna"
        else {
            throw BenchmarkError.invalidInput(
                "--laguna-router-topk-ab requires a native Laguna checkpoint.")
        }
        guard
            let quantization = object["quantization"] as? [String: Any],
            let bits = quantization["bits"] as? NSNumber,
            bits.intValue == 4
        else {
            throw BenchmarkError.invalidInput(
                "--laguna-router-topk-ab requires a Q4 checkpoint (quantization.bits == 4).")
        }
        let experts = (object["num_experts"] as? NSNumber)?.intValue
        let topK = (object["num_experts_per_tok"] as? NSNumber)?.intValue
        let scoreFunction = object["moe_router_score_func"] as? String ?? "sigmoid"
        let softcap = (object["moe_router_logit_softcapping"] as? NSNumber)?.doubleValue ?? 0
        guard experts == 256, topK == 8, scoreFunction == "sigmoid", softcap == 0 else {
            throw BenchmarkError.invalidInput(
                "--laguna-router-topk-ab requires the 256-expert/top-8 sigmoid router with zero softcap."
            )
        }
    }

    private func runGeneration(
        runner: LocalModelRunner,
        messages: [OpenAIMessage],
        sequence: Int,
        mode: GenerationMode
    ) async throws -> RecordedTrial {
        let clock = ContinuousClock()
        Memory.peakMemory = 0
        let startedAt = clock.now
        let effort = reasoningEffort.flatMap(ChatCompletionRequest.ReasoningEffort.init(rawValue:))
        let prepared = try await runner.preparePrompt(
            messages: messages, maximumTokens: tokens,
            reasoningEffort: effort)
        let promptFingerprint = ModelQualityCore.tokenIDFingerprint(prepared.promptTokenIDs)
        let events = await Gemma4RuntimeTuning.$useDenseFusion.withValue(mode.useGemmaDenseFusion) {
            await Gemma4RuntimeTuning.$useWindowSlicing.withValue(mode.useGemmaWindowSlicing) {
                await LagunaRuntimeTuning.$useCompiledMoEFusion.withValue(
                    mode.useLagunaFusion
                ) {
                    await LagunaRuntimeTuning.$useCompiledAttentionGate.withValue(
                        mode.useCompiledAttentionGate
                    ) {
                        await LagunaRuntimeTuning.$useCompiledBlockTail.withValue(
                            mode.useCompiledBlockTail
                        ) {
                            await LagunaRuntimeTuning.$useFusedRouterTopK.withValue(
                                mode.useFusedRouterTopK
                            ) {
                                await LagunaRuntimeTuning.$useFusedGateUpSilu.withValue(mode.useFusedGateUpSilu) {
                                    await runner.stream(
                                        messages: messages,
                                        maximumTokens: tokens,
                                        temperature: Double(temperature),
                                        topP: Double(topP),
                                        reasoningEffort: effort,
                                        enablePromptCache: mode.usePromptCache && !disablePromptReuse,
                                        enableSpeculativeDecoding: mode.useDFlash,
                                        preparedPrompt: prepared
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
        var content = ""
        var reasoning = ""
        var finalMetrics: LocalModelRunnerMetrics?
        var firstTokenAt: ContinuousClock.Instant?
        for try await event in events {
            switch event {
            case .reasoning(let text): reasoning += text
            case .content(let text):
                if firstTokenAt == nil {
                    firstTokenAt = clock.now
                }
                content += text
            case .metrics(let metrics): finalMetrics = metrics
            case .toolCall: throw BenchmarkError.unexpectedToolCall(sequence)
            }
        }
        guard let finalMetrics else {
            throw BenchmarkError.missingMetrics(sequence)
        }
        if mode.useDFlash, let reason = finalMetrics.speculativePassthroughReason {
            throw BenchmarkError.dflashPassthrough(sequence, reason)
        }
        if mode.useDFlash, (finalMetrics.proposedDraftTokens ?? 0) == 0 {
            throw BenchmarkError.invalidInput("Assistant produced no draft proposals in generation \(sequence).")
        }
        if !allowEarlyStop, finalMetrics.generationTokenCount != tokens {
            throw BenchmarkError.incompleteGeneration(finalMetrics.generationTokenCount, tokens)
        }
        guard let firstTokenAt else {
            throw BenchmarkError.incompleteGeneration(0, tokens)
        }
        return RecordedTrial(
            sequence: sequence,
            mode: mode.label,
            metrics: RecordedMetrics(finalMetrics),
            timeToFirstTokenMilliseconds: milliseconds(startedAt.duration(to: firstTokenAt)),
            totalMilliseconds: milliseconds(startedAt.duration(to: clock.now)),
            peakActiveMemoryBytes: Memory.peakMemory,
            promptTokenIDFingerprint: promptFingerprint,
            content: content,
            reasoning: reasoning
        )
    }

    private func runPromptCacheProbe(
        runner: LocalModelRunner,
        sequence: Int,
        seedMessages: [OpenAIMessage],
        mode: GenerationMode,
        coldFirst: Bool
    ) async throws -> [RecordedTrial] {
        // Reset between probes so warmup/earlier branches cannot inflate hits.
        try await runner.resetKVCache()
        var seedMode = mode
        seedMode.label = "prompt_cache_seed"
        let seed = try await runGeneration(
            runner: runner, messages: seedMessages, sequence: sequence, mode: seedMode)
        let continuationMessages =
            seedMessages + [
                OpenAIMessage(
                    role: "assistant", content: seed.content,
                    reasoningContent: seed.reasoning.isEmpty ? nil : seed.reasoning),
                OpenAIMessage(role: "user", content: continuationPrompt),
            ]
        // Move the hot session down a sibling branch first. The measured cached
        // continuation must then come from the immutable completed-prefix LRU,
        // rather than the zero-copy linear hot path.
        let displacementMessages =
            seedMessages + [
                OpenAIMessage(
                    role: "assistant", content: seed.content,
                    reasoningContent: seed.reasoning.isEmpty ? nil : seed.reasoning),
                OpenAIMessage(
                    role: "user",
                    content: continuationPrompt + "\n\nBegin instead with memory reclamation."
                ),
            ]
        var displacementMode = mode
        displacementMode.label = "prompt_cache_displacement"
        let displacement = try await runGeneration(
            runner: runner, messages: displacementMessages, sequence: sequence + 1,
            mode: displacementMode)
        guard displacement.metrics.cachedPromptTokenCount > 0 else {
            throw BenchmarkError.promptCacheNotUsed(displacement.sequence)
        }
        var cachedMode = mode
        cachedMode.label = "prompt_cache_cached"
        var coldMode = mode
        coldMode.label = "prompt_cache_cold"
        coldMode.usePromptCache = false
        let modes = coldFirst ? [coldMode, cachedMode] : [cachedMode, coldMode]
        var paired: [RecordedTrial] = []
        for (index, selectedMode) in modes.enumerated() {
            paired.append(
                try await runGeneration(
                    runner: runner, messages: continuationMessages, sequence: sequence + 2 + index,
                    mode: selectedMode))
        }
        let cached = paired.first { $0.mode == cachedMode.label }!
        let cold = paired.first { $0.mode == coldMode.label }!
        guard cached.metrics.cachedPromptTokenCount > 0 else {
            throw BenchmarkError.promptCacheNotUsed(cached.sequence)
        }
        guard cold.metrics.cachedPromptTokenCount == 0 else {
            throw BenchmarkError.coldPromptUnexpectedlyCached(
                cold.sequence, cold.metrics.cachedPromptTokenCount)
        }
        return [seed, displacement] + paired
    }

    private func runMistralHotCacheProbe(
        runner: LocalModelRunner,
        sequence: Int,
        seedMessages: [OpenAIMessage],
        mode: GenerationMode,
        coldFirst: Bool
    ) async throws -> [RecordedTrial] {
        let labels = HotCacheModeLabels(isMistral: runner.supportsMistralHotConversationCache)
        var cachedMode = mode
        cachedMode.label = labels.cached
        var coldMode = mode
        coldMode.label = labels.cold
        coldMode.usePromptCache = false

        let cachedSeed: RecordedTrial
        let coldSeed: RecordedTrial
        let cached: RecordedTrial
        let cold: RecordedTrial
        if coldFirst {
            coldSeed = try await runMistralHotCacheSeed(
                runner: runner, sequence: sequence, messages: seedMessages, mode: mode,
                label: labels.seedBeforeCold)
            cold = try await runGeneration(
                runner: runner, messages: continuationMessages(seedMessages, coldSeed),
                sequence: sequence + 1,
                mode: coldMode)
            cachedSeed = try await runMistralHotCacheSeed(
                runner: runner, sequence: sequence + 2, messages: seedMessages, mode: mode,
                label: labels.seedBeforeCached)
            cached = try await runGeneration(
                runner: runner, messages: continuationMessages(seedMessages, cachedSeed),
                sequence: sequence + 3,
                mode: cachedMode)
        } else {
            cachedSeed = try await runMistralHotCacheSeed(
                runner: runner, sequence: sequence, messages: seedMessages, mode: mode,
                label: labels.seedBeforeCached)
            cached = try await runGeneration(
                runner: runner, messages: continuationMessages(seedMessages, cachedSeed),
                sequence: sequence + 1,
                mode: cachedMode)
            coldSeed = try await runMistralHotCacheSeed(
                runner: runner, sequence: sequence + 2, messages: seedMessages, mode: mode,
                label: labels.seedBeforeCold)
            cold = try await runGeneration(
                runner: runner, messages: continuationMessages(seedMessages, coldSeed),
                sequence: sequence + 3,
                mode: coldMode)
        }

        guard cachedSeed.content == coldSeed.content else {
            throw BenchmarkError.mistralHotCacheSeedOutputMismatch(
                cachedSeed.sequence,
                coldSeed.sequence,
                firstUTF8DivergenceOffset(cachedSeed.content, coldSeed.content)
            )
        }
        guard cached.metrics.cachedPromptTokenCount > 0 else {
            throw BenchmarkError.mistralHotCacheNotUsed(cached.sequence)
        }
        let expectedCachedPromptTokenCount =
            cachedSeed.metrics.promptTokenCount + cachedSeed.metrics.generationTokenCount
        // Harmony can remove private analysis from a completed assistant turn.
        // GPT-OSS may safely reuse a shorter prefix; require actual reuse and valid
        // accounting below without assigning Mistral's full-prefix contract to it.
        if runner.supportsMistralHotConversationCache,
            cached.metrics.cachedPromptTokenCount != expectedCachedPromptTokenCount
        {
            throw BenchmarkError.mistralHotCachePartialReuse(
                cached.sequence,
                cached.metrics.cachedPromptTokenCount,
                expectedCachedPromptTokenCount
            )
        }
        guard
            cached.metrics.prefilledPromptTokenCount
                == cached.metrics.promptTokenCount - cached.metrics.cachedPromptTokenCount
        else {
            throw BenchmarkError.mistralHotCachePrefillAccountingMismatch(
                cached.sequence,
                cached.metrics.prefilledPromptTokenCount,
                cached.metrics.promptTokenCount,
                cached.metrics.cachedPromptTokenCount
            )
        }
        guard cold.metrics.cachedPromptTokenCount == 0 else {
            throw BenchmarkError.mistralHotCacheColdUnexpectedlyCached(
                cold.sequence, cold.metrics.cachedPromptTokenCount)
        }
        guard cold.metrics.prefilledPromptTokenCount == cold.metrics.promptTokenCount else {
            throw BenchmarkError.mistralHotCachePrefillAccountingMismatch(
                cold.sequence,
                cold.metrics.prefilledPromptTokenCount,
                cold.metrics.promptTokenCount,
                cold.metrics.cachedPromptTokenCount
            )
        }
        guard cached.metrics.promptTokenCount == cold.metrics.promptTokenCount else {
            throw BenchmarkError.mistralHotCachePromptCountMismatch(
                cached.sequence,
                cold.sequence,
                cached.metrics.promptTokenCount - cold.metrics.promptTokenCount
            )
        }
        guard
            cached.metrics.prefilledPromptTokenCount
                < cold.metrics.prefilledPromptTokenCount
        else {
            throw BenchmarkError.mistralHotCacheDidNotReducePrefill(
                cached.sequence,
                cold.sequence,
                cached.metrics.prefilledPromptTokenCount,
                cold.metrics.prefilledPromptTokenCount
            )
        }
        return coldFirst
            ? [coldSeed, cold, cachedSeed, cached]
            : [cachedSeed, cached, coldSeed, cold]
    }

    private func runMistralHotCacheSeed(
        runner: LocalModelRunner,
        sequence: Int,
        messages: [OpenAIMessage],
        mode: GenerationMode,
        label: String
    ) async throws -> RecordedTrial {
        var seedMode = mode
        seedMode.label = label
        let seed = try await runGeneration(
            runner: runner, messages: messages, sequence: sequence, mode: seedMode)
        guard seed.metrics.cachedPromptTokenCount == 0 else {
            throw BenchmarkError.mistralHotCacheSeedUnexpectedlyCached(
                seed.sequence, seed.metrics.cachedPromptTokenCount)
        }
        return seed
    }

    private func continuationMessages(
        _ seedMessages: [OpenAIMessage], _ seed: RecordedTrial
    ) -> [OpenAIMessage] {
        seedMessages + [
            OpenAIMessage(role: "assistant", content: seed.content),
            OpenAIMessage(role: "user", content: continuationPrompt),
        ]
    }

    private func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }

    private func mistralHotCacheOutputsMatchExactly(
        _ records: [RecordedTrial], labels: HotCacheModeLabels
    ) -> Bool {
        guard records.count.isMultiple(of: 4) else {
            return false
        }
        return stride(from: 0, to: records.count, by: 4).allSatisfy { start in
            let probe = records[start..<(start + 4)]
            guard
                let cached = probe.first(where: { $0.mode == labels.cached }),
                let cold = probe.first(where: { $0.mode == labels.cold })
            else {
                return false
            }
            return cached.content == cold.content
        }
    }

    private func mistralHotCacheFirstOutputDivergence(
        _ records: [RecordedTrial], labels: HotCacheModeLabels
    ) -> Int? {
        guard records.count.isMultiple(of: 4) else {
            return nil
        }
        return stride(from: 0, to: records.count, by: 4).compactMap { start in
            let probe = records[start..<(start + 4)]
            guard
                let cached = probe.first(where: { $0.mode == labels.cached }),
                let cold = probe.first(where: { $0.mode == labels.cold })
            else {
                return nil
            }
            return firstUTF8DivergenceOffset(cached.content, cold.content)
        }.min()
    }

    private func firstUTF8DivergenceOffset(_ lhs: String, _ rhs: String) -> Int? {
        let lhsBytes = Array(lhs.utf8)
        let rhsBytes = Array(rhs.utf8)
        let commonCount = min(lhsBytes.count, rhsBytes.count)
        if let mismatch = (0..<commonCount).first(where: { lhsBytes[$0] != rhsBytes[$0] }) {
            return mismatch
        }
        return lhsBytes.count == rhsBytes.count ? nil : commonCount
    }

    private func milliseconds(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds) * 1_000
            + Double(components.attoseconds) / 1_000_000_000_000_000
    }

    private func printPromptCacheProbe(_ records: [RecordedTrial], prefix: String) {
        guard records.count == 4 else {
            return
        }
        for record in records {
            print("\(prefix) [\(record.mode)]: " + trialSummary(record))
        }
    }

    private func printMistralHotCacheProbe(_ records: [RecordedTrial], prefix: String) {
        guard records.count == 4 else {
            return
        }
        for record in records {
            print("\(prefix) [\(record.mode)]: " + trialSummary(record))
        }
    }

    private func trialSummary(_ record: RecordedTrial) -> String {
        let metrics = record.metrics
        var summary = String(
            format: "%.2f tok/s, TTFT %.2f ms", metrics.tokensPerSecond,
            record.timeToFirstTokenMilliseconds)
        if metrics.cachedPromptTokenCount > 0 {
            summary +=
                ", prompt \(metrics.cachedPromptTokenCount) cached/"
                + "\(metrics.prefilledPromptTokenCount) prefilled"
        }
        if let proposed = metrics.proposedDraftTokens,
            let accepted = metrics.acceptedDraftTokens,
            proposed > 0
        {
            summary += String(
                format: ", draft acceptance %.1f%% (%d/%d)",
                100 * Double(accepted) / Double(proposed), accepted, proposed)
        }
        if let reason = metrics.speculativePassthroughReason {
            summary += ", speculative passthrough: \(reason)"
        }
        return summary
    }
}
