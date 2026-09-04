# Changelog

## 0.2.0-beta.1

- Admit text requests using exact prompt tokens, requested output, conservative
  KV/workspace estimates, and the process memory budget before prefill.
- Expose context length and prefill chunk size through CLI/settings and model
  discovery; preserve the model's declared context ceiling.
- Scale Metal memory limits with physical RAM, host reserve, and the device's
  recommended working set; retain explicit bounded overrides.
- Budget hot conversations and branch snapshots together, skip snapshots that
  cannot be retained, and evict before copying or admitting a request.
- Integrate opt-in affine8, affine4, and turbo8v4 KV compression for compatible
  Laguna/Mistral-family layers, preserving native sliding windows.
- Fix macOS release launch selection and make optimized release the default.
- Add memory, context, cache, compression, and launcher regression tests.

This remains a beta. KV compression is experimental and does not imply a
quality or speed improvement. Paged attention and continuous batching are not
included. See `Docs/memory-and-context.md` for configuration and boundaries.
