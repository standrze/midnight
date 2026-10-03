# Unknown-token fallback fixture

This minimal synthetic BPE vocabulary uses the real swift-transformers
`GemmaTokenizer` loader. It preserves the relevant Gemma token identities:
EOS 1, unknown 3, and end-of-turn 106. `<turn|>` is deliberately absent.
No model weights, downloaded vocabulary or generation are needed.

The regression checks the actual tokenizer's fallback behavior rather than a
mock that returns nil for unknown token strings.
