# Talkie tokenizer compatibility fixture

This small fixture is derived from the Apache-2.0
[warshanks/talkie-1930-13b-it-mlx-4bit-DWQ checkpoint](https://huggingface.co/warshanks/talkie-1930-13b-it-mlx-4bit-DWQ/tree/1cde1b1becc99e097923ae6dfe0c51b89014fa7f),
revision `1cde1b1becc99e097923ae6dfe0c51b89014fa7f`, checked September 13, 2026.
The original Talkie authors are Nick Levine, David Duvenaud and Alec Radford;
this community conversion is by warshanks, via lewtun's Transformers conversion.

`config.json`, `tokenizer_config.json`, and `chat_template.jinja` are copied
unchanged. `tokenizer.json` retains the original tokenizer processing settings
and special tokens, with a reduced BPE vocabulary and merge list. The complete
original tokenizer has SHA-256
`cc3813d9d674cf0e86e4171579ba276879c66c2171d993e5776fc5615756a03b`.

The reduction keeps all one-character byte vocabulary entries plus every
vocabulary string that is a substring of a pretokenized segment of either
`Hello` or `Café naïve résumé — 1930\nDon't change 123456.` (with a real newline).
It keeps original IDs and the relative order of merges whose two inputs and
combined output remain in that vocabulary. Special tokens are also included
in the sparse vocabulary to preserve their original IDs when loading with the
Python backend. There are 323 vocabulary entries and 103 merges.

Expected token IDs were generated independently with Python `tokenizers` 0.22.2
and the full original tokenizer. The reduced fixture was checked against those
IDs and exact decode round trips for both sample texts and the single/multiple
turn chat strings used in `TalkieTokenizerTests`. The fixture tests the pinned
Swift loader's `TokenizersBackend`, separate Jinja file, role markers, no-BOS
framing and Unicode BPE compatibility. It is not a full vocabulary, model
checkpoint, or claim of general tokenization/model inference parity.
