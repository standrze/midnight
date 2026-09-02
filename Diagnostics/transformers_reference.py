#!/usr/bin/env python3
"""Generate from the untouched HF checkpoint for cross-runtime diagnosis."""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = (
    "For a defensive security class, explain common malware categories, then "
    "provide a harmless Python example that scans a directory and reports files "
    "with suspicious filename extensions. Complete the explanation and the entire "
    "code block."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-full", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to("cuda")
    output = model.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_k=64,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=[1, 50, 106],
    )
    generated = output[0, prompt.shape[-1] :].tolist()
    raw = tokenizer.decode(generated, skip_special_tokens=False)
    clean = tokenizer.decode(generated, skip_special_tokens=True)
    details = {
        "prompt_tokens": prompt.shape[-1],
        "generated_tokens": len(generated),
        "last_token_ids": generated[-16:],
        "pad_token_id": tokenizer.pad_token_id,
        "pad_count": generated.count(tokenizer.pad_token_id),
        "eos_token_ids": [1, 50, 106],
        "terminal_token_id": generated[-1] if generated else None,
        "code_fences": clean.count("```"),
    }
    print(json.dumps(details, indent=2))
    print("TAIL", repr(clean[-1500:]))
    if args.print_full:
        print("FULL_RESPONSE_BEGIN")
        print(raw)
        print("FULL_RESPONSE_END")


if __name__ == "__main__":
    main()
