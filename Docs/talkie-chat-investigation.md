# Talkie conversational failure investigation

Status: unresolved. No verified inference fix has been identified or applied.

## Online findings

- The [author report](https://talkie-lm.com/introducing-talkie) explicitly describes
  multi-turn preference training and a final multi-turn supervised training round.
  It does not support treating broken conversation as an intentional limitation.
- [Official source](https://github.com/talkie-lm/talkie) uses temperature 0.7,
  the role/end delimiters already used by Midnight, and full-sequence inference.
- [Official PR 6](https://github.com/talkie-lm/talkie/pull/6) addresses macOS
  installation and peak loading memory, not repetitive answers. Our isolated
  original-weight diagnostic uses equivalent meta/assign memory loading.
- [llama.cpp issue 23953](https://github.com/ggml-org/llama.cpp/issues/23953)
  reports intermittent garbage for Talkie Q8 on Vulkan. It is unconfirmed and
  closed without a demonstrated fix; it is not evidence of the same MLX defect.
- The [Q8 publisher](https://huggingface.co/warshanks/talkie-1930-13b-it-mlx-8bit)
  documents plain-Q4 degradation. That alone cannot explain the Q8 and original
  checkpoint observations here.

## Evidence and limits

The upstream-chat-comparison report records 16/16 identical outputs between
Midnight and upstream Python MLX using the public Q8 weights, with a documented
packed-head loading adapter. Temperature 0.7 did not fix the observed conversation.
A short fixed-prefix cache comparison preserved all selected next tokens.

The official-pytorch-chat report records three fresh prompts with original
weights and author PyTorch computation on CPU. Two had odd interpretations;
this is not a reproduction of the complete multi-turn repetition and does not
prove that the original model has the same failure in its intended CUDA setup.

The remaining decisive work is a matched multi-turn original-weight reference
and token/logit comparison, including numerical execution differences between
the reference CPU path and intended CUDA autocast path. CUDA/4090 work remains
paused by the user. Do not change general sampling defaults, suppress repeated
answers, or modify model equations based only on similar-looking issue reports.
Any eventual runtime repair must be Talkie-specific and update the API Field
Guide and its published copy if client-visible behavior changes.

Evidence: benchmark-results/talkie-20260913/upstream-chat-comparison/ and
benchmark-results/talkie-20260913/official-pytorch-chat/.
