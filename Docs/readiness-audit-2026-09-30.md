# Midnight readiness audit — 30 September 2026

## Verdict

Midnight is ready for authenticated, single-user local model serving and for the
serving side of an external abliteration workflow. Its API, lifecycle manager,
installed-model discovery, text inference, and speech backends are implemented
and exercised. A live Metal smoke test loaded the installed Talkie 13B
checkpoint and completed requests through both Chat Completions and Responses.

Midnight does not currently alter or ablate model weights. ABSlayer and other
tools own that transformation step; Midnight provides bounded inspection and
serves compatible exported MLX `safetensors` checkpoints. Calling the whole
workflow “cyber-ready” still requires behavior and task-quality evidence for
each promoted checkpoint. Weight integrity and a successful API response do not
establish cybersecurity correctness.

This checkout has no root Git metadata, so the audit could not establish a
trustworthy baseline diff, branch, commit, or dirty-tree report.

## Verified behavior

- The full root Swift test suite passes. Hardware-specific CUDA and unavailable
  Metal cases skip through their documented gates.
- The optional Vision package passes its 19 tests.
- Swift formatting checks pass for the audited source tree.
- The CLI launches, displays grouped help, discovers installed models, and the
  console smoke suite covers idle startup, lifecycle controls, shutdown, bind
  failure, and terminal restoration.
- The interactive dashboard reports active and queued requests, live generated
  tokens, output limits, prompt-cache usage, first-token latency, throughput,
  and MLX memory. A live Talkie request generated 256 tokens while those values
  were painted and the terminal restored cleanly afterward.
- Every HTTP route requires a 32–256 character bearer key. Authentication is
  checked before request bodies are accumulated, and bodies are capped at
  32 MiB.
- The managed listener exposes 24 method/path pairs: 8 OpenAI-compatible, 10
  Midnight-specific, and 6 Mistral-style voice operations.
- One model is resident at a time. Named inference requests can drain the active
  selection, load a preflight-valid installed checkpoint, and lease that exact
  selection for the request.

## Supported model paths

The support label matters: “best effort” means the pinned upstream MLX registry
contains an architecture, not that Midnight has qualified a particular
checkpoint on every backend.

| Model family | Midnight path | Current claim |
| --- | --- | --- |
| Mistral, Ministral, Mixtral, Codestral, text Devstral | Native Swift + MLX | Primary text path |
| Poolside Laguna XS | Native Swift hybrid attention/MoE | Primary text path; DFlash remains experimental |
| GPT-OSS | Pinned upstream MLX Swift architecture | Supported |
| Talkie | Native Swift + MLX | Validated on Metal; 2,048-token checkpoint limit |
| Liquid LFM2 / LFM2.5 | Native MLX loaders | Included; checkpoint-level validation still applies |
| Muse Glimmer | Native text/tool path | Text and tool calls; image input deliberately excluded |
| Gemma 1–4, Qwen text, Llama, Phi, DeepSeek, Granite, Cohere, and other registered MLX families | Pinned upstream registry | Best effort unless a Midnight guide records validation |
| Voxtral TTS | Native Swift + MLX speech | Preset voices; WAV/PCM |
| Qwen3-TTS Base | Native Swift + MLX on Metal | Reference audio plus required reference transcript |
| Qwen3-TTS CustomVoice | Native Swift + MLX on Metal | Preset speakers and optional instructions |
| VibeVoice 1.5B / 7B | Optional local Python/PyTorch worker | Metal or CPU; reference-audio conditioning |
| FastVLM / LLaVA-Qwen2 vision | Optional managed worker | macOS/Metal, non-streaming image chat, checkpoint-dependent |

Text LoRA adapters are supported when the bundle contains
`adapter_config.json` and `adapters.safetensors`. Speech and vision backends do
not accept text adapters. Metal is the default on macOS, CUDA is the default in
the Linux CUDA package, and MLX CPU remains a fallback.

## Models found on the audit host

The catalog found ten standalone candidates after excluding the Laguna DFlash
assistant from primary-model discovery:

| Installed ID | Size | Type / status |
| --- | ---: | --- |
| `Laguna-XS-2.1-midnight` | 18 GB | Laguna text; quantization verified, task quality unbenchmarked |
| `Qwen3.8-27B-midnight` | 14 GB | Qwen 3.5 text; best-effort upstream runtime |
| `gemma-4-26B-A4B-it-midnight` | 13 GB | Gemma 4 text; experimental local conversion |
| `gemma-4-31B-it-midnight` | 16 GB | Gemma 4 text; task quality unbenchmarked |
| `mlx-community--gpt-oss-20b-MXFP4-Q8` | 11 GB | GPT-OSS text |
| `talkie-1930-13b-it-hf-midnight` | 9.8 GB | Talkie text; live API smoke passed |
| `mlx-community--Voxtral-4B-TTS-2603-mlx-4bit` | 2.4 GB | Voxtral speech |
| `Qwen3-TTS-12Hz-0.6B-Base-8bit` | 1.9 GB | Qwen Base speech and reference conditioning |
| `VibeVoice-1.5B-hf` | 5.0 GB | VibeVoice speech |
| `VibeVoice-7B-hf` | 17 GB | VibeVoice speech |

`poolside--Laguna-XS-2.1-DFlash` is an 881 MB speculative assistant. It is no
longer advertised as a standalone inference model.

## Corrections made during the audit

- Qwen3-TTS generation now participates in producer-lifetime draining, so model
  unload cannot race its unstructured producer task.
- Qwen capability metadata is subtype-specific: Base advertises reference audio;
  CustomVoice does not accept and silently ignore it.
- VibeVoice cancellation now interrupts blocking worker I/O, terminates the
  affected worker, drains promptly, and permits a clean reload.
- Chat Completions rejects unknown or misspelled top-level fields instead of
  silently ignoring controls such as `seed` or `presence_penalty`.
- Both OpenAI and Mistral speech dialects enforce the documented 1–4,096 input
  character limit.
- Auxiliary DFlash and Muse assistant checkpoints are rejected as primary model
  selections and omitted from installed-model discovery.
- The terminal dashboard now uses passive, request-local token observation and
  distinguishes requests waiting for the model execution slot from active work.
- API and model-capability documentation now matches installed-model discovery,
  Qwen reference conditioning, VibeVoice's optional Python worker, and the
  current 24-route inventory.

## Remaining work, in priority order

1. **Define a cyber model promotion gate.** Keep paired base and transformed
   checkpoints, version the transformation recipe, and preserve raw outputs.
   Test refusal reduction, substantive answer rate, technical correctness,
   false-positive behavior, tool-use safety, and regression prompts. Add
   sandboxed executable oracles for tasks that can be checked automatically.
2. **Align abliteration and inspection architectures.** Midnight's bounded
   residual capture currently covers Laguna, Mistral 3 / Ministral 3, and
   GPT-OSS, while the current ABSlayer intervention backend is Gemma 4. Add and
   version Gemma 4 inspection points or document the direct-load handoff used by
   ABSlayer.
3. **Repair the environment-specific default selection.** The current
   `model-stack.local.json` points at a missing Muse symlink. Select a real
   installed checkpoint before relying on a no-argument plain launch.
4. **Distinguish preflight from proven loadability.** Catalog preflight checks
   metadata and required files. Add architecture-registry validation and a
   `preflight_only`, `validated`, or `failed` status with a cached failure reason
   so clients can distinguish discovery from a successful tensor load.
5. **Harden remote/shared deployment before offering it.** The present security
   model is appropriate for loopback, single-user use: one bearer key, no TLS,
   no identity, roles, quotas, or audit trail. Add TLS termination, scoped keys,
   rate limits, and separately gated inspector access before serving an
   untrusted network or multiple users.
6. **Add repository provenance and CI.** Restore root Git metadata, add
   `SECURITY.md`, continuous Swift format/test/API-guide checks, dependency and
   model provenance, and an SBOM/release-signing path.
7. **Reduce maintenance risk at natural seams.** Split checkpoint loading,
   resource planning, and generation helpers out of the 2,600-line
   `LocalModelRunner.swift`; separate chat routing, transport, handler state, and
   wire schemas in `ModelHTTPServer.swift`; then split protocol-only tests from
   MLX-heavy core tests.
