#!/usr/bin/env python3
import json
import sys

chunks = []
finish_reason = None
for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    payload = json.loads(line[6:])
    choices = payload.get("choices", [])
    if not choices:
        continue
    choice = choices[0]
    content = choice.get("delta", {}).get("content")
    if content is not None:
        chunks.append(content)
    if choice.get("finish_reason") is not None:
        finish_reason = choice["finish_reason"]

text = "".join(chunks)
print(
    f"chars={len(text)} chunks={len(chunks)} fences={text.count('```')} "
    f"finish_reason={finish_reason}"
)
print("--- tail ---")
print(text[-3000:])
