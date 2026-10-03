# API documentation maintenance

Keep this folder focused on Midnight serving, inspection, runtime evaluation,
and their tests. New model development, decision training, calibration, and quantization belong in
`../afterglow`; existing quantization sources in `../wick` and experiments in
`../training` are preserved, and website code in
`../midnight-site` or `../midnight-nite`. Do not add forwarding tools or duplicate
those implementations here. See `Docs/project-layout.md` for current ownership.

New distillation development belongs in the independent `../midnight-moonshine`
project. Midnight Activation Capture is a public serving/runtime feature owned
here; ABSlayer, midnight-moonshine, and other authorized clients consume the same
API. Keep consumer-specific training, analysis, and transformations outside
Midnight.

For Swift source and package layout work, follow [Docs/SwiftProjectStyle.md](Docs/SwiftProjectStyle.md). Format and check each touched Swift file with `Scripts/format-swift.sh` and `Scripts/check-swift-format.sh`. See [Docs/SwiftStyleAdoption.md](Docs/SwiftStyleAdoption.md) for existing exceptions and verification scope.

When changing Midnight API routes, request or response fields, provider compatibility, model discovery, or client-visible behavior, update `Docs/api-field-guide/index.html` in the same change. Preserve the user's Field Guide starter design; the deliverable is HTML, not PDF.

Keep `Docs/api-field-guide/routes.json` and route counts aligned when the route inventory changes. Refresh `source-manifest.json` hashes for changed source files. Mark each operation as OpenAI-compatible, Midnight-specific, or Mistral-style, and document limitations and unsupported operations explicitly.

The published copy lives in `standrze/field-guides` under `midnight-api/` (local checkout: `/Users/stephen/Documents/ChatGPT/field-guide/github-pages`). Keep that copy synchronized with API guide changes. Its public URL is `https://standrze.github.io/field-guides/midnight-api/`. Preserve unrelated publishing edits; do not start a temporary localhost server unless requested.

# Model downloads and priorities

For model download requests, follow [Docs/model-support-scope.md](Docs/model-support-scope.md).
Use Hugging Face repositories owned by the original model maker only, with
approximately 4-bit quantized target weights compatible with Midnight/MLX.
Verify current repository metadata and resolve immutable revisions before
transferring weights. Do not silently substitute community conversions,
third-party assistants, full-precision weights, or incompatible formats.

Tier 1 and tier 2 downloads include the exact matching publisher-owned assistant
when available. Keep assistants at the publisher's supported precision; the
approximately 4-bit preference applies to target weights. Report both artifacts
and their total size, and verify the target/assistant pairing. If a compliant
target or assistant is unavailable or unsupported, explain that concrete gap
before downloading an alternative; never claim a target-only download completed
an assistant bundle. A download request authorizes the eligible model and its
matching assistant without a separate companion confirmation. The CLI and loom
console use the shared native Swift download service and bundle designated
companions.

Use the native Swift Hugging Face downloader with its existing token support
(`midnight auth login`, `HF_TOKEN`, or the shared Hugging Face token file).
Never request a token in chat or put it in command arguments or logs.
Muse Glimmer is deferred from active scope until a maker-owned approximately
4-bit native MLX release is verified. Our own quantized Hugging Face releases
are a future phase; do not publish or substitute them during the current
publisher-owned sourcing phase.
