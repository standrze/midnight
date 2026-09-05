#!/usr/bin/env python3
"""Prepare question-only generation tasks and a separate private scoring key.

Example (offline, using the existing pinned source cache):
  python3 Scripts/prepare-generated-eval-corpus.py --cache-dir /absolute/corpora \
    --output-dir /tmp/generated-pilot --math-count 16 --code-count 16

Retrieval sizes are estimates, not tokenizer-certified context lengths. The
native generation report supplies actual rendered prompt-token counts. Public
benchmark overlap with training data is unknown; this is a paired regression
suite, not an uncontaminated general-capability estimate.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import sys
import warnings

_path = Path(__file__).with_name("prepare-quantization-corpus.py")
_spec = importlib.util.spec_from_file_location("pinned_reference_corpus", _path)
pinned = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pinned)


def payload(records):
    return ("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for r in records)).encode("utf-8")


def signatures(item):
    """Use test-call names to expose required APIs, never source function bodies."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        module = ast.parse(item["code"])
        test_tree = ast.parse("\n".join(item["test_list"]))
    called = {node.func.id for node in ast.walk(test_tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    result = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in called:
            # Defaults/annotations can contain reference expressions. Publish only
            # argument names and API shape; the model supplies the implementation.
            args = node.args
            names = [a.arg for a in args.posonlyargs]
            if names:
                names.append("/")
            names.extend(a.arg for a in args.args)
            if args.vararg:
                names.append("*" + args.vararg.arg)
            elif args.kwonlyargs:
                names.append("*")
            names.extend(a.arg for a in args.kwonlyargs)
            if args.kwarg:
                names.append("**" + args.kwarg.arg)
            result.append(f"def {node.name}({', '.join(names)}):")
    if not result:
        raise ValueError(f"No supported tested function API in MBPP task {item['task_id']}")
    return result


def build_records(source_data, math_count, code_count, retrieval_tokens, seed):
    tasks, answers = [], []
    if code_count:
        rows = [json.loads(s) for s in source_data["mbpp"].decode().splitlines() if s.strip()]
        chosen = [r for r in rows if 11 <= r["task_id"] < 11 + code_count]
        if [r["task_id"] for r in chosen] != list(range(11, 11 + code_count)):
            raise ValueError("MBPP selection is incomplete, duplicated or out of order")
        for row in chosen:
            identifier = f"mbpp-test-{row['task_id']}"
            prompt = ("Write Python 3 code for the following task. Return only the complete code, "
                      "with any needed imports. Implement the required function API.\n\n"
                      + row["text"] + "\n\nRequired API:\n" + "\n".join(signatures(row)))
            tasks.append({"id": identifier, "category": "code", "prompt": prompt})
            tests = row.get("test_list", [])
            if not tests or not all(isinstance(t, str) and t.strip() for t in tests):
                raise ValueError(f"Missing tests for {identifier}")
            answers.append({"id": identifier, "kind": "python-tests",
                            "setup": row.get("test_setup_code", ""), "tests": tests})
    if math_count:
        rows = [json.loads(s) for s in source_data["gsm8k"].decode().splitlines() if s.strip()]
        if len(rows) < math_count:
            raise ValueError("GSM8K selection is incomplete")
        for index, row in enumerate(rows[:math_count]):
            match = re.search(r"^####\s*(\S[^\n]*)\s*$", row["answer"], re.M)
            if not match:
                raise ValueError(f"Missing GSM8K final answer at index {index}")
            identifier = f"gsm8k-test-{index}"
            tasks.append({"id": identifier, "category": "math", "prompt":
                          "Solve the following problem. End with a separate line containing "
                          "#### followed by only the final number.\n\n" + row["question"]})
            answers.append({"id": identifier, "kind": "numeric", "answer": match.group(1).strip()})
    rng = random.Random(seed)
    for budget in retrieval_tokens:
        # Random hex pairs are deliberately independent of the answer location.
        # Roughly 32 tokens/record is an estimate; model tokenizers vary.
        count = max(16, budget // 32)
        for position in (0.1, 0.5, 0.9):
            keys = rng.sample(range(1 << 40), count)
            values = [f"V{rng.getrandbits(48):012x}" for _ in range(count)]
            target = round((count - 1) * position)
            lines = [f"record {i:04d}: key K{key:010x}; value {value}."
                     for i, (key, value) in enumerate(zip(keys, values))]
            identifier = f"retrieval-est{budget}-pos{int(position * 100)}"
            tasks.append({"id": identifier, "category": "retrieval", "prompt":
                          "Read the records below and retrieve the requested value exactly.\n\n"
                          + "\n".join(lines) + f"\n\nWhat value belongs to key K{keys[target]:010x}? "
                          "Return only that value, without explanation or punctuation.",
                          "metadata": {"requested_context_tokens_estimate": budget,
                                       "record_count": count, "target_record_index": target,
                                       "target_position_fraction": position, "seed": seed}})
            answers.append({"id": identifier, "kind": "exact", "answer": values[target]})
    if not tasks:
        raise ValueError("Select at least one evaluation task")
    return tasks, answers


def prepare(output, cache, math_count=16, code_count=16,
            retrieval_tokens=(4096, 16384, 32768), seed=20260904, sources=pinned.SOURCES):
    if type(math_count) is not int or not 0 <= math_count <= 64 or type(code_count) is not int or not 0 <= code_count <= 64:
        raise ValueError("Math/code counts must be in 0...64")
    if len(set(retrieval_tokens)) != len(retrieval_tokens) or any(type(x) is not int or not 512 <= x <= 131072 for x in retrieval_tokens):
        raise ValueError("Retrieval token estimates must be unique integers in 512...131072")
    output, cache = Path(output), Path(cache)
    data, provenance = {}, []
    needed = {name for name, count in (("mbpp", code_count), ("gsm8k", math_count)) if count}
    for source in sources:
        if source["name"] not in needed:
            continue
        path = cache / f"{source['name']}-source.jsonl"
        data[source["name"]] = pinned.verify(path.read_bytes(), source["source_sha256"], str(path))
        provenance.append({k: source[k] for k in ("name", "repo", "revision", "path", "source_sha256")})
    tasks, answers = build_records(data, math_count, code_count, retrieval_tokens, seed)
    files = {"tasks.jsonl": payload(tasks), "answers.jsonl": payload(answers)}
    meta = {"format": 1, "sources": provenance, "seed": seed,
            "selection": {"math_count": math_count, "code_count": code_count,
                          "retrieval_context_token_estimates": list(retrieval_tokens)},
            "task_count": len(tasks), "files": {n: hashlib.sha256(b).hexdigest() for n, b in files.items()},
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": ["Only tasks.jsonl is passed to generation. Gold answers, tests and reference bodies are excluded from prompts.",
                            "Code APIs are extracted from reference signatures; private tests stay in the scoring key. Three public MBPP test cases are used, not challenge tests or hidden HumanEval tests.",
                            "Public benchmark training contamination and reuse in prior reference evaluation are possible. Do not use these records for fitting or candidate selection.",
                            "Retrieval contains its supporting fact in the context by design; the scorer key is separate. Context lengths are estimates until measured by the model tokenizer."]}
    files["provenance.json"] = (json.dumps(meta, indent=2) + "\n").encode()
    for name, raw in files.items():
        path = output / name
        if path.exists() and (not path.is_file() or path.read_bytes() != raw):
            raise ValueError(f"Refusing to overwrite differing output: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in files.items():
        path = output / name
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(raw)
            if name == "answers.jsonl":
                path.chmod(0o600)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, required=True, help="Verified *-source.jsonl cache; never downloads")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--math-count", type=int, default=16)
    parser.add_argument("--code-count", type=int, default=16)
    parser.add_argument("--retrieval-token-estimates", type=int, nargs="*", default=[4096, 16384, 32768])
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(args.output_dir, args.cache_dir, args.math_count, args.code_count,
                                 args.retrieval_token_estimates, args.seed), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
