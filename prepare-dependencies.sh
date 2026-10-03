#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"
# A sibling package may reuse the reviewed patches with its own dependency
# graph and build directory. Patch ownership remains in Midnight.
DEPENDENCY_PACKAGE_ROOT="${MODEL_RUNNER_DEPENDENCY_PACKAGE_ROOT:-$PACKAGE_ROOT}"
if [[ ! -f "$DEPENDENCY_PACKAGE_ROOT/Package.swift" ]]; then
  echo "Dependency package root must contain Package.swift: $DEPENDENCY_PACKAGE_ROOT" >&2
  exit 2
fi
DEPENDENCY_PACKAGE_ROOT="$(cd "$DEPENDENCY_PACKAGE_ROOT" && pwd -P)"
source "$PACKAGE_ROOT/Scripts/swiftpm-scratch-path.sh"
source "$PACKAGE_ROOT/Scripts/optional-dependency-patch.sh"
source "$PACKAGE_ROOT/Scripts/cuda-expert-patch.sh"
source "$PACKAGE_ROOT/Scripts/compile-cache-lifetime-patch.sh"
source "$PACKAGE_ROOT/Scripts/gate-up-slices-patch.sh"
source "$PACKAGE_ROOT/Scripts/gemma-grouped-expert-patch.sh"
source "$PACKAGE_ROOT/Scripts/gemma3-compiled-tail-patch.sh"
source "$PACKAGE_ROOT/Scripts/mtp-adaptive-drafts-patch.sh"
source "$PACKAGE_ROOT/Scripts/extra-eos-token-identity-patch.sh"
source "$PACKAGE_ROOT/Scripts/generation-progress-patch.sh"
source "$PACKAGE_ROOT/Scripts/added-token-regex-patch.sh"
source "$PACKAGE_ROOT/Scripts/gemma-cleanup-default-patch.sh"
source "$PACKAGE_ROOT/Scripts/affine-q4-qmv-tail-patch.sh"
source "$PACKAGE_ROOT/Scripts/metal-sdpa-d512-patch.sh"
source "$PACKAGE_ROOT/Scripts/metal-command-timing-patch.sh"
source "$PACKAGE_ROOT/Scripts/gemma4-window-cache-patch.sh"
HOST_OS="$(uname -s)"
model_runner_configure_swiftpm_scratch "$DEPENDENCY_PACKAGE_ROOT" "$HOST_OS"
MLX_SWIFT_CHECKOUT="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/checkouts/mlx-swift"
MLX_SWIFT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-cuda-linux.patch"
MLX_SWIFT_GENERATED_HEADER_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-cuda-generated-header.patch"
MLX_SWIFT_MLX32_LINK_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-mlx32-cuda-link.patch"
MLX_SWIFT_CROSS_THREAD_STREAM_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-cross-thread-stream.patch"
MLX_SWIFT_CLEAR_STREAMS_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-clear-streams.patch"
MLX_SWIFT_EXISTING_DEFAULT_STREAM_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-existing-default-stream.patch"
MLX_SWIFT_CUDA_KERNEL_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-cuda-kernel.patch"
MLX_SWIFT_DIRECT_SLICE_UPDATE_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-direct-slice-update.patch"
MLX_SWIFT_WIRED_MEMORY_SNAPSHOT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-wired-memory-snapshot.patch"
MLX_SWIFT_AFFINE_Q4_QMV_JIT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-affine-q4-qmv-jit.patch"
MLX_SWIFT_SORTED_GATHER_QMM_NAX_JIT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-sorted-gather-qmm-nax-row-bounds-jit.patch"
MLX_SWIFT_DARWIN_EXPECTED_REVISION="72f3c3ad8aeee39bfc94f8fbeb446cac89e3a798"
MLX_SWIFT_LINUX_EXPECTED_REVISION="2d2724006b62855c6c2a71df633baf4ee4ad8a0f"
MLX_SOURCE_CHECKOUT="$MLX_SWIFT_CHECKOUT/Source/Cmlx/mlx"
MLX_SOURCE_PATCH="$PACKAGE_ROOT/Patches/mlx-cuda-half-fmod.patch"
MLX_SOURCE_GLOBAL_STREAM_CLEANUP_PATCH="$PACKAGE_ROOT/Patches/mlx-global-stream-cleanup.patch"
MLX_SOURCE_AFFINE_Q4_QMV_PATCH="$PACKAGE_ROOT/Patches/mlx-affine-q4-qmv-specialization.patch"
MLX_SOURCE_SORTED_GATHER_QMM_NAX_PATCH="$PACKAGE_ROOT/Patches/mlx-sorted-gather-qmm-nax-row-bounds.patch"
MLX_SOURCE_DARWIN_EXPECTED_REVISION="1f8e74e3f12f31365464a6867c6579f0e9b29d85"
MLX_SOURCE_LINUX_EXPECTED_REVISION="7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247"
MLX_C_SOURCE_CHECKOUT="$MLX_SWIFT_CHECKOUT/Source/Cmlx/mlx-c"
MLX_C_SOURCE_CLEAR_STREAMS_PATCH="$PACKAGE_ROOT/Patches/mlx-c-clear-streams.patch"
MLX_C_SOURCE_CLEAR_GLOBAL_STREAMS_PATCH="$PACKAGE_ROOT/Patches/mlx-c-clear-global-streams.patch"
MLX_C_SOURCE_DARWIN_EXPECTED_REVISION="c74db5307cc8ce122f48d97ef951b30578674e7f"
MLX_C_SOURCE_LINUX_EXPECTED_REVISION="fba4470b89073180056c9ea46c443051375f7399"
SWIFT_TRANSFORMERS_CHECKOUT="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/checkouts/swift-transformers"
SWIFT_TRANSFORMERS_INCREMENTAL_BYTELEVEL_PATCH="$PACKAGE_ROOT/Patches/swift-transformers-incremental-bytelevel-decoder.patch"
SWIFT_TRANSFORMERS_EXPECTED_REVISION="2fa33e1f5e7131a7fc64c28e6d161dcec0d24820"
MLX_SWIFT_LM_CHECKOUT="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/checkouts/mlx-swift-lm"
MLX_SWIFT_LM_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-ignore-readmes.patch"
MLX_SWIFT_LM_COREFOUNDATION_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-corefoundation-linux.patch"
MLX_SWIFT_LM_GEMMA4_LORA_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-gemma4-lora-layers.patch"
MLX_SWIFT_LM_GEMMA4_CACHE_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-gemma4-nonrotating-cache.patch"
MLX_SWIFT_LM_BACKEND_TOKEN_EVAL_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-backend-token-evaluation.patch"
MLX_SWIFT_LM_TASK_EXECUTOR_PREFERENCE_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-task-executor-preference.patch"
MLX_SWIFT_LM_MTP_PROMPT_WINDOW_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-mtp-prompt-hidden-window.patch"
MLX_SWIFT_LM_MTP_DECODE_SCHEDULING_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-mtp-decode-scheduling.patch"
MLX_SWIFT_LM_MTP_FIRST_REJECTION_DIAGNOSTIC_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-mtp-first-rejection-diagnostic.patch"
MLX_SWIFT_LM_CHAT_SESSION_SNAPSHOT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-chat-session-snapshot.patch"
MLX_SWIFT_LM_DIRECT_KV_SLICE_UPDATE_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-direct-kv-slice-update.patch"
MLX_SWIFT_LM_INCREMENTAL_BYTELEVEL_STREAMING_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-incremental-bytelevel-streaming.patch"
MLX_SWIFT_LM_Q4_AFFINE_SCALE_SEARCH_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-affine-scale-search.patch"
MLX_SWIFT_LM_Q4_AFFINE_CENTERED_SCALE_SEARCH_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-affine-centered-scale-search.patch"
MLX_SWIFT_LM_Q4_AFFINE_BIAS_REFINEMENT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-affine-bias-refinement.patch"
MLX_SWIFT_LM_Q4_AFFINE_JOINT_FIT_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-affine-joint-fit.patch"
MLX_SWIFT_LM_Q4_AFFINE_GROUP_SIZE_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-q4-affine-group-size.patch"
MLX_SWIFT_LM_MISTRAL_HYBRID_ATTENTION_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-mistral-hybrid-attention.patch"
MLX_SWIFT_LM_MIXTRAL_FUSED_ROUTER_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-mixtral-fused-router.patch"
MLX_SWIFT_LM_MUSE_GLIMMER_PREFILL_LOGITS_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-muse-glimmer-prefill-logits.patch"
MLX_SWIFT_LM_MUSE_GLIMMER_DFLASH_TARGET_PATCH="$PACKAGE_ROOT/Patches/mlx-swift-lm-muse-glimmer-dflash-target.patch"
MLX_SWIFT_LM_EXPECTED_REVISION="14414441fa44f45eee35a61e9fa0bab577cf9734"
MLX_AUDIO_SWIFT_CHECKOUT="$MODEL_RUNNER_SWIFTPM_SCRATCH_PATH/checkouts/mlx-audio-swift"
MLX_AUDIO_SWIFT_EXPECTED_REVISION="bf14ae0c26e4e85553dd989571cae29d70fa6735"
MLX_AUDIO_SWIFT_QWEN_TTS_PATCH="$PACKAGE_ROOT/Patches/mlx-audio-swift-qwen3-tts-only.patch"

case "$HOST_OS" in
  Darwin)
    MLX_SWIFT_EXPECTED_REVISION="$MLX_SWIFT_DARWIN_EXPECTED_REVISION"
    MLX_SOURCE_EXPECTED_REVISION="$MLX_SOURCE_DARWIN_EXPECTED_REVISION"
    MLX_C_SOURCE_EXPECTED_REVISION="$MLX_C_SOURCE_DARWIN_EXPECTED_REVISION"
    APPLY_LINUX_DEPENDENCY_PATCHES=0
    MLX_SWIFT_CROSS_THREAD_STREAM_OVERLAY_MODE="$(
      model_runner_mlx_cross_thread_stream_overlay_mode "$HOST_OS"
    )"
    ;;
  Linux)
    MLX_SWIFT_EXPECTED_REVISION="$MLX_SWIFT_LINUX_EXPECTED_REVISION"
    MLX_SOURCE_EXPECTED_REVISION="$MLX_SOURCE_LINUX_EXPECTED_REVISION"
    MLX_C_SOURCE_EXPECTED_REVISION="$MLX_C_SOURCE_LINUX_EXPECTED_REVISION"
    APPLY_LINUX_DEPENDENCY_PATCHES=1
    MLX_SWIFT_CROSS_THREAD_STREAM_OVERLAY_MODE="$(
      model_runner_mlx_cross_thread_stream_overlay_mode "$HOST_OS"
    )"
    ;;
  *)
    echo "Unsupported host for dependency preparation: $HOST_OS" >&2
    exit 1
    ;;
esac

cd "$DEPENDENCY_PACKAGE_ROOT"
# Bash 3.2 treats an empty array expansion as unbound under `set -u`; keep the
# no-scratch path explicit so this preparation step also remains usable on the
# Mac client.
if [[ -n "${MODEL_RUNNER_SWIFT_PACKAGE_SCRATCH_ARGS+configured}" ]]; then
  swift package "${MODEL_RUNNER_SWIFT_PACKAGE_SCRATCH_ARGS[@]}" resolve --quiet
else
  swift package resolve --quiet
fi

verify_checkout_revision() {
  local label="$1"
  local checkout="$2"
  local expected_revision="$3"
  local actual_revision

  if [[ ! -d "$checkout" ]]; then
    echo "$label checkout was not created at $checkout" >&2
    return 1
  fi
  actual_revision="$(git -C "$checkout" rev-parse HEAD)"
  if [[ "$actual_revision" != "$expected_revision" ]]; then
    echo "Refusing to patch unexpected $label revision: $actual_revision" >&2
    echo "Expected: $expected_revision" >&2
    return 1
  fi
}

apply_dependency_patch() {
  local label="$1"
  local checkout="$2"
  local patch_file="$3"

  if [[ "$label" == "swift-transformers incremental ByteLevel decoder" ]]; then
    source "$PACKAGE_ROOT/Scripts/incremental-bytelevel-patch.sh"
    if model_runner_bytelevel_present_with_gemma_cleanup "$PACKAGE_ROOT" "$checkout"; then
      echo "$label patch already applied (complete replay with Gemma cleanup)."
      return 0
    fi
  fi

  # This checkout may also carry the CUDA diagnostic overlay used on the
  # Linux hosts. That overlay preserves the cache fix while changing its
  # explanatory comment, so a byte-for-byte reverse patch check is not a
  # reliable idempotence test for this one semantic change.
  if [[ "$label" == "mlx-swift-lm Gemma 4 non-rotating cache" ]] \
    && grep -Fq 'try makeAttentionKVCache(parameters: parameters)' \
      "$checkout/Libraries/MLXLLM/Models/Gemma4Text.swift"; then
    echo "$label patch already applied."
    return 0
  fi

  # Early Linux bring-up used the same safe CUDA decode boundary without the
  # later macOS-only async branch. Accept that equivalent state so existing
  # server checkouts can converge without reverting generated dependencies.
  if [[ "$label" == "mlx-swift-lm backend-aware token evaluation" ]] \
    && grep -Fq 'eval([token] + cache.flatMap { $0.state })' \
      "$checkout/Libraries/MLXLMCommon/Evaluate.swift" \
    && ! grep -Fq 'if tokenCount % 256 == 0' \
      "$checkout/Libraries/MLXLMCommon/Evaluate.swift"; then
    echo "$label patch already applied."
    return 0
  fi

  # The centered-grid follow-up intentionally refines the original Q4 affine
  # scale-search hunk. Recognize both semantic states directly because the
  # follow-up means reverse-applying the first patch is no longer byte-exact.
  if [[ "$label" == "mlx-swift-lm Q4 affine scale search" ]] \
    && grep -Fq 'case q4AffineScaleSearch' \
      "$checkout/Libraries/MLXLMCommon/ModelConversion.swift"; then
    echo "$label patch already applied."
    return 0
  fi
  if [[ "$label" == "mlx-swift-lm Q4 affine centered scale search" ]] \
    && grep -Fq 'q4AffineScaleSearchFactors' \
      "$checkout/Libraries/MLXLMCommon/ModelConversion.swift"; then
    echo "$label patch already applied."
    return 0
  fi
  if [[ "$label" == "mlx-swift-lm Q4 affine bias refinement" ]] \
    && grep -Fq 'least-squares affine bias' \
      "$checkout/Libraries/MLXLMCommon/ModelConversion.swift"; then
    echo "$label patch already applied."
    return 0
  fi
  if [[ "$label" == "mlx-swift-lm Q4 affine joint fit" ]] \
    && grep -Fq 'let secondCodeValues' \
      "$checkout/Libraries/MLXLMCommon/ModelConversion.swift"; then
    echo "$label patch already applied."
    return 0
  fi

  # Repository formatting removes spaces around these four inserted ranges.
  # Validate the complete patch (including its tests) with only that exact
  # formatting difference; do not treat a matching source substring as proof.
  if [[ "$label" == "mlx-swift-lm direct KV slice update" ]] \
    && sed \
      -e '/^+.*updateKVCacheSlice/s/previous \.\.< self.offset/previous..<self.offset/' \
      -e '/^+.*updateKVCacheSlice/s/idx \.\.< (idx + S)/idx..<(idx + S)/' \
      "$patch_file" | git -C "$checkout" apply --reverse --check - >/dev/null 2>&1; then
    echo "$label patch already applied (formatted ranges)."
    return 0
  fi

  # The diagnostic follow-up inserts calls inside the scheduling patch's
  # verifier hunk. Recognize both durable semantic states directly so an
  # already fully overlaid checkout remains idempotent in forward order.
  if [[ "$label" == "mlx-swift-lm MTP decode scheduling" ]] \
    && grep -Fq 'if processor == nil, sampler is ArgMaxSampler {' \
      "$checkout/Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift" \
    && grep -Fq 'private var passthroughPipelinePrimed = false' \
      "$checkout/Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift"; then
    echo "$label patch already applied."
    return 0
  fi
  if [[ "$label" == "mlx-swift-lm MTP first-rejection diagnostic" ]] \
    && grep -Fq 'MODEL_RUNNER_DFLASH_FIRST_REJECTION_DIAGNOSTIC' \
      "$checkout/Libraries/MLXLMCommon/MTPSpeculativeTokenIterator.swift"; then
    echo "$label patch already applied."
    return 0
  fi

  # The Foundation Models follow-up extends the reasoning-stream switch with
  # richer protocol routing. Its extra cases make reverse-applying the original
  # patch non-exact even though the public Generation.reasoning event and its
  # decoder emission are already present.
  if [[ "$label" == "mlx-swift-lm reasoning stream events" ]] \
    && grep -Fq 'case reasoning(String)' \
      "$checkout/Libraries/MLXLMCommon/Evaluate.swift" \
    && grep -Fq 'emit(.reasoning(text))' \
      "$checkout/Libraries/MLXLMCommon/Evaluate.swift" \
    && grep -Fq 'ReasoningEventEmitter' \
      "$checkout/Libraries/MLXLMCommon/Tool/TokenStreamDecoder.swift"; then
    echo "$label patch already applied."
    return 0
  fi

  # The global-cleanup follow-up intentionally extends the same mlx-c function
  # hunk, so reverse-applying the first patch is no longer a valid idempotence
  # test after both are present. Recognize each exported behavior directly.
  if [[ "$label" == "mlx-c clear streams API" ]] \
    && grep -Fq 'extern "C" int mlx_clear_streams(void)' \
      "$checkout/mlx/c/stream.cpp"; then
    echo "$label patch already applied."
    return 0
  fi
  if [[ "$label" == "mlx-c clear global streams API" ]] \
    && grep -Fq 'mlx::core::gpu::clear_global_streams();' \
      "$checkout/mlx/c/stream.cpp"; then
    echo "$label patch already applied."
    return 0
  fi

  if git -C "$checkout" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "$label patch already applied."
    return 0
  fi
  if ! git -C "$checkout" apply --check "$patch_file" >/dev/null 2>&1; then
    echo "Could not apply $label patch cleanly: $patch_file" >&2
    return 1
  fi
  git -C "$checkout" apply "$patch_file"
  echo "Applied $label patch."
}

# Verify every checkout that this host will patch before mutating any of them.
# The conditional manifest keeps macOS on the synchronized official MLX 0.32
# update revision and Linux on its exact CUDA compatibility revision.
verify_checkout_revision \
  "mlx-swift" "$MLX_SWIFT_CHECKOUT" "$MLX_SWIFT_EXPECTED_REVISION"
verify_checkout_revision \
  "mlx-swift-lm" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_EXPECTED_REVISION"
verify_checkout_revision \
  "mlx source" "$MLX_SOURCE_CHECKOUT" "$MLX_SOURCE_EXPECTED_REVISION"
verify_checkout_revision \
  "mlx-c source" "$MLX_C_SOURCE_CHECKOUT" "$MLX_C_SOURCE_EXPECTED_REVISION"
verify_checkout_revision \
  "swift-transformers" "$SWIFT_TRANSFORMERS_CHECKOUT" "$SWIFT_TRANSFORMERS_EXPECTED_REVISION"
verify_checkout_revision \
  "mlx-audio-swift" "$MLX_AUDIO_SWIFT_CHECKOUT" "$MLX_AUDIO_SWIFT_EXPECTED_REVISION"

if [[ "$APPLY_LINUX_DEPENDENCY_PATCHES" == "1" ]]; then
  apply_dependency_patch \
    "mlx-swift CUDA Linux integration" "$MLX_SWIFT_CHECKOUT" "$MLX_SWIFT_PATCH"
  apply_dependency_patch \
    "mlx-swift CUDA generated header" "$MLX_SWIFT_CHECKOUT" "$MLX_SWIFT_GENERATED_HEADER_PATCH"
  apply_dependency_patch \
    "mlx-swift MLX 0.32 CUDA link" "$MLX_SWIFT_CHECKOUT" "$MLX_SWIFT_MLX32_LINK_PATCH"
  model_runner_reconcile_optional_dependency_patch \
    "mlx-swift cross-thread stream" \
    "$MLX_SWIFT_CHECKOUT" \
    "$MLX_SWIFT_CROSS_THREAD_STREAM_PATCH" \
    "$MLX_SWIFT_CROSS_THREAD_STREAM_OVERLAY_MODE"
  apply_dependency_patch \
    "mlx-swift clear streams API" \
    "$MLX_SWIFT_CHECKOUT" \
    "$MLX_SWIFT_CLEAR_STREAMS_PATCH"
  apply_dependency_patch \
    "mlx CUDA half fmod" "$MLX_SOURCE_CHECKOUT" "$MLX_SOURCE_PATCH"
  apply_dependency_patch \
    "mlx global stream terminal cleanup" \
    "$MLX_SOURCE_CHECKOUT" \
    "$MLX_SOURCE_GLOBAL_STREAM_CLEANUP_PATCH"
  apply_dependency_patch \
    "mlx-c clear streams API" \
    "$MLX_C_SOURCE_CHECKOUT" \
    "$MLX_C_SOURCE_CLEAR_STREAMS_PATCH"
  apply_dependency_patch \
    "mlx-c clear global streams API" \
    "$MLX_C_SOURCE_CHECKOUT" \
    "$MLX_C_SOURCE_CLEAR_GLOBAL_STREAMS_PATCH"
  model_runner_prepare_cuda_experts "$HOST_OS" "$PACKAGE_ROOT" "$MLX_SWIFT_CHECKOUT"
fi

if [[ "$HOST_OS" == "Darwin" ]]; then
  apply_dependency_patch \
    "mlx-swift sparse module array updates" "$MLX_SWIFT_CHECKOUT" \
    "$PACKAGE_ROOT/Patches/mlx-swift-sparse-module-array-update.patch"
  model_runner_prepare_compile_cache_lifetime "$HOST_OS" "$PACKAGE_ROOT" "$MLX_SWIFT_CHECKOUT"
  model_runner_prepare_gate_up_slices "$HOST_OS" "$PACKAGE_ROOT" "$MLX_SWIFT_LM_CHECKOUT"
  model_runner_prepare_gemma_grouped_expert "$HOST_OS" "$PACKAGE_ROOT" "$MLX_SWIFT_LM_CHECKOUT"
  # Upstream d73eb752: clamp large sorted expert-row counts before narrowing.
  # Keep compiled Metal headers and generated JIT shader sources synchronized.
  apply_dependency_patch \
    "mlx sorted gather QMM NAX row bounds" \
    "$MLX_SOURCE_CHECKOUT" \
    "$MLX_SOURCE_SORTED_GATHER_QMM_NAX_PATCH"
  apply_dependency_patch \
    "mlx-swift sorted gather QMM NAX generated JIT" \
    "$MLX_SWIFT_CHECKOUT" \
    "$MLX_SWIFT_SORTED_GATHER_QMM_NAX_JIT_PATCH"
  # Own the "mlx affine Q4 QMV specialization" and
  # "mlx-swift affine Q4 QMV generated JIT" prerequisites with their tail overlay.
  model_runner_prepare_affine_q4_qmv_tail "$HOST_OS" "$PACKAGE_ROOT" "$MLX_SWIFT_CHECKOUT"
  model_runner_prepare_metal_sdpa_d512 "$HOST_OS" "$PACKAGE_ROOT" "$MLX_SWIFT_CHECKOUT"
  model_runner_prepare_metal_command_timing "$HOST_OS" "$PACKAGE_ROOT" "$MLX_SOURCE_CHECKOUT"
fi

apply_dependency_patch \
  "mlx-swift existing default stream API" \
  "$MLX_SWIFT_CHECKOUT" \
  "$MLX_SWIFT_EXISTING_DEFAULT_STREAM_PATCH"
apply_dependency_patch \
  "mlx-swift custom CUDA kernel API" \
  "$MLX_SWIFT_CHECKOUT" \
  "$MLX_SWIFT_CUDA_KERNEL_PATCH"
if [[ "$HOST_OS" == "Darwin" ]]; then
  apply_dependency_patch \
    "mlx-swift wired-memory snapshot" \
    "$MLX_SWIFT_CHECKOUT" \
    "$MLX_SWIFT_WIRED_MEMORY_SNAPSHOT_PATCH"
fi
apply_dependency_patch \
  "mlx-swift direct slice update" \
  "$MLX_SWIFT_CHECKOUT" \
  "$MLX_SWIFT_DIRECT_SLICE_UPDATE_PATCH"
apply_dependency_patch \
  "swift-transformers incremental ByteLevel decoder" \
  "$SWIFT_TRANSFORMERS_CHECKOUT" \
  "$SWIFT_TRANSFORMERS_INCREMENTAL_BYTELEVEL_PATCH"
model_runner_prepare_added_token_regex "$PACKAGE_ROOT" "$SWIFT_TRANSFORMERS_CHECKOUT"
model_runner_prepare_gemma_cleanup_default "$PACKAGE_ROOT" "$SWIFT_TRANSFORMERS_CHECKOUT"
apply_dependency_patch \
  "mlx-audio-swift Qwen3-TTS-only target" \
  "$MLX_AUDIO_SWIFT_CHECKOUT" \
  "$MLX_AUDIO_SWIFT_QWEN_TTS_PATCH"

apply_dependency_patch \
  "mlx-swift-lm README warning fix" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_PATCH"
if [[ "$APPLY_LINUX_DEPENDENCY_PATCHES" == "1" ]]; then
  apply_dependency_patch \
    "mlx-swift-lm CoreFoundation import" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_COREFOUNDATION_PATCH"
fi
apply_dependency_patch \
  "mlx-swift-lm Gemma 4 LoRA layer coverage" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_GEMMA4_LORA_PATCH"
# The stack owns "mlx-swift-lm Gemma 4 non-rotating cache" and later overlapping overlays.
model_runner_prepare_gemma4_window_cache "$PACKAGE_ROOT" "$MLX_SWIFT_LM_CHECKOUT"
apply_dependency_patch \
  "mlx-swift-lm Mistral hybrid attention" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$MLX_SWIFT_LM_MISTRAL_HYBRID_ATTENTION_PATCH"
apply_dependency_patch \
  "mlx-swift-lm Mixtral fused decode router" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$MLX_SWIFT_LM_MIXTRAL_FUSED_ROUTER_PATCH"
apply_dependency_patch \
  "mlx-swift-lm Muse Glimmer prefill logits" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$MLX_SWIFT_LM_MUSE_GLIMMER_PREFILL_LOGITS_PATCH"
apply_dependency_patch \
  "mlx-swift-lm Muse Glimmer DFlash target state" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$MLX_SWIFT_LM_MUSE_GLIMMER_DFLASH_TARGET_PATCH"
apply_dependency_patch \
  "mlx-swift-lm backend-aware token evaluation" "$MLX_SWIFT_LM_CHECKOUT" "$MLX_SWIFT_LM_BACKEND_TOKEN_EVAL_PATCH"
apply_dependency_patch \
  "mlx-swift-lm task executor preference" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$MLX_SWIFT_LM_TASK_EXECUTOR_PREFERENCE_PATCH"
# Own the overlapping MTP prompt, scheduling, diagnostic, and adaptive overlays.
model_runner_prepare_mtp_adaptive_drafts "$PACKAGE_ROOT" "$MLX_SWIFT_LM_CHECKOUT"
model_runner_prepare_extra_eos_token_identity "$PACKAGE_ROOT" "$MLX_SWIFT_LM_CHECKOUT"
apply_dependency_patch \
  "mlx-swift-lm ChatSession in-memory snapshot" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$MLX_SWIFT_LM_CHAT_SESSION_SNAPSHOT_PATCH"
apply_dependency_patch \
  "mlx-swift-lm direct KV slice update" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$MLX_SWIFT_LM_DIRECT_KV_SLICE_UPDATE_PATCH"
apply_dependency_patch \
  "mlx-swift-lm incremental ByteLevel streaming" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$MLX_SWIFT_LM_INCREMENTAL_BYTELEVEL_STREAMING_PATCH"
apply_dependency_patch \
  "mlx-swift-lm Q4 affine scale search" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$MLX_SWIFT_LM_Q4_AFFINE_SCALE_SEARCH_PATCH"
apply_dependency_patch \
  "mlx-swift-lm Q4 affine centered scale search" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$MLX_SWIFT_LM_Q4_AFFINE_CENTERED_SCALE_SEARCH_PATCH"
apply_dependency_patch \
  "mlx-swift-lm Q4 affine bias refinement" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$MLX_SWIFT_LM_Q4_AFFINE_BIAS_REFINEMENT_PATCH"
apply_dependency_patch \
  "mlx-swift-lm Q4 affine joint fit" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$MLX_SWIFT_LM_Q4_AFFINE_JOINT_FIT_PATCH"
apply_dependency_patch \
  "mlx-swift-lm Q4 affine group size" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$MLX_SWIFT_LM_Q4_AFFINE_GROUP_SIZE_PATCH"

apply_dependency_patch \
  "mlx-swift-lm self-contained conversion metadata" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$PACKAGE_ROOT/Patches/mlx-swift-lm-self-contained-conversion.patch"

apply_dependency_patch \
  "mlx-swift-lm cache memory accounting" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$PACKAGE_ROOT/Patches/mlx-swift-lm-chat-cache-memory.patch"

apply_dependency_patch \
  "mlx-swift-lm reasoning stream events" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$PACKAGE_ROOT/Patches/mlx-swift-lm-reasoning-stream.patch"

apply_dependency_patch \
  "mlx-swift-lm Laguna tool calls and reasoning history" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$PACKAGE_ROOT/Patches/mlx-swift-lm-laguna-tools.patch"


apply_dependency_patch \
  "mlx-swift-lm nested Gemma marker values" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$PACKAGE_ROOT/Patches/mlx-swift-lm-gemma-nested-marker-values.patch"

apply_dependency_patch \
  "mlx-swift-lm Gemma 3 attention layout" \
  "$MLX_SWIFT_LM_CHECKOUT" \
  "$PACKAGE_ROOT/Patches/mlx-swift-lm-gemma3-attention-layout.patch"

model_runner_prepare_gemma3_compiled_tail \
  "$HOST_OS" "$PACKAGE_ROOT" "$MLX_SWIFT_LM_CHECKOUT"

model_runner_prepare_generation_progress "$PACKAGE_ROOT" "$MLX_SWIFT_LM_CHECKOUT"
