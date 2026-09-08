#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BINARY="${CORPUS_PREPARATION_BINARY:-$ROOT/.build/debug/model-runner-prepare-corpus}"
[[ -x "$BINARY" ]] || { echo "Build model-runner-prepare-corpus first." >&2; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
"$BINARY" --help > "$WORK/help"
"$BINARY" text --help > "$WORK/text-help"
"$BINARY" reference --help > "$WORK/reference-help"
printf ' first paragraph\n\nsecond paragraph ' > "$WORK/source.txt"
# The built tool needs no Python, shell subprocess, or external hashing command.
env PATH=/nonexistent "$BINARY" text "$WORK/source.txt" "$WORK/corpus.jsonl" --samples 1 > "$WORK/report"
printf '{"id":"heldout-001","category":"language-modeling","text":"first paragraph\\n\\nsecond paragraph"}\n' > "$WORK/expected"
cmp "$WORK/expected" "$WORK/corpus.jsonl"
if "$BINARY" text "$WORK/source.txt" "$WORK/invalid" --samples 0 > "$WORK/error" 2>&1; then
    echo 'Invalid sample count accepted' >&2; exit 1
fi
[[ ! -e "$WORK/invalid" ]]
if "$BINARY" text "$WORK/source.txt" "$WORK/source.txt" --samples 1 > "$WORK/error" 2>&1; then
    echo 'Same source and output accepted' >&2; exit 1
fi
printf 'preserved' > "$WORK/protected"
printf '[{"text":42}]' > "$WORK/bad.json"
if "$BINARY" text "$WORK/bad.json" "$WORK/protected" --samples 1 > "$WORK/error" 2>&1; then
    echo 'Invalid JSON record accepted' >&2; exit 1
fi
[[ "$(cat "$WORK/protected")" == preserved ]]
if env PATH=/nonexistent "$BINARY" reference --output-dir "$WORK/missing" --offline > "$WORK/error" 2>&1; then
    echo 'Missing offline sources accepted' >&2; exit 1
fi
[[ ! -e "$WORK/missing" ]]
echo 'Corpus preparation CLI checks passed.'
