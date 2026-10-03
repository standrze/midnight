#!/usr/bin/env bash
# Requires an already-built executable; never builds or loads an MLX model.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
generation_bench="${MODEL_GENERATION_BENCH:-$repo_root/.build/release/model-runner-generation-bench}"
if [[ ! -x "$generation_bench" ]]; then
  echo "Build model-runner-generation-bench first, or set MODEL_GENERATION_BENCH." >&2
  exit 1
fi
fixture_dir="$(mktemp -d "${TMPDIR:-/tmp}/midnight-generation-cli.XXXXXX")"
trap 'rm -rf "$fixture_dir"' EXIT
python3 - "$fixture_dir" <<'PY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1])
records=[
 {'id':'math-0','category':'math','prompt':'Compute the requested value.\nUse #### for the final number.'},
 {'id':'code-0','category':'code','prompt':'Write a Python function.','metadata':{'source_index':11}},
 {'id':'retrieval-0','category':'retrieval','prompt':'Key: café\nValue: Λ\nRetrieve the requested value.','metadata':{'requested_context_tokens':4096}},
]
def write(name,values):
 (root/name).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
write('valid.jsonl',records)
(root/'reformatted.jsonl').write_text('\n'.join(json.dumps(v,ensure_ascii=False,separators=(',',':')) for v in records))
write('duplicate.jsonl',[records[0],records[0]])
write('private-answer.jsonl',[dict(records[0],answer='private')])
write('private-tests.jsonl',[dict(records[1],tests=['assert hidden()'])])
write('blank-prompt.jsonl',[dict(records[0],prompt=' \t\n')])
write('wrong-type.jsonl',[dict(records[0],prompt=7)])
write('missing-category.jsonl',[{'id':'x','prompt':'Prompt'}])
(root/'empty.jsonl').write_text('\n  \n')
(root/'malformed.jsonl').write_text('{"id":')
(root/'invalid-utf8.jsonl').write_bytes(b'\xff\n')
(root/'existing.json').write_bytes(b'preserve existing report exactly\n')
PY
MLX_MAX_OPS_PER_BUFFER=200 MLX_MAX_MB_PER_BUFFER=256 \
  "$generation_bench" "$fixture_dir/missing-model" "$fixture_dir/valid.jsonl" \
  "$fixture_dir/valid.json" --engine cpu --tokens 17 --context-length 4096 \
  --prefill-step-size 128 --validate-only > "$fixture_dir/valid.log" 2>&1
"$generation_bench" "$fixture_dir/missing-model" "$fixture_dir/reformatted.jsonl" \
  "$fixture_dir/reformatted.json" --engine cpu --validate-only > "$fixture_dir/reformatted.log" 2>&1
expect_failure() {
  local label="$1"
  shift
  if "$@" > "$fixture_dir/$label.log" 2>&1; then
    echo "Expected failure: $label" >&2
    exit 1
  fi
}
for case_name in duplicate private-answer private-tests blank-prompt wrong-type missing-category empty malformed invalid-utf8 missing; do
  expect_failure "$case_name" "$generation_bench" "$fixture_dir/missing-model" \
    "$fixture_dir/$case_name.jsonl" "$fixture_dir/$case_name.json" --engine cpu --validate-only
done
expect_failure existing "$generation_bench" "$fixture_dir/missing-model" \
  "$fixture_dir/valid.jsonl" "$fixture_dir/existing.json" --validate-only
expect_failure tokens "$generation_bench" "$fixture_dir/missing-model" \
  "$fixture_dir/valid.jsonl" "$fixture_dir/tokens.json" --tokens 0 --validate-only
expect_failure prefill "$generation_bench" "$fixture_dir/missing-model" \
  "$fixture_dir/valid.jsonl" "$fixture_dir/prefill.json" --prefill-step-size 0 --validate-only
expect_failure modes "$generation_bench" "$fixture_dir/missing-model" \
  "$fixture_dir/valid.jsonl" "$fixture_dir/modes.json" --prepare-only --validate-only
python3 - "$fixture_dir" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]); valid=json.loads((root/'valid.json').read_text())
assert valid['status']=='validated' and valid['format']==1
assert valid['model_load_count']==0 and valid['input_sample_count']==3
assert valid['completed_sample_count']==0 and valid['prepared_sample_count']==0
assert valid['requested_tokens']==17 and valid['context_length']==4096
assert valid['prefill_step_size']==128 and valid['kv_compression']=='none'
assert valid['temperature']==0 and valid['top_p']==1
assert not valid['prompt_cache'] and not valid['speculative_decoding']
assert valid['runtime_environment']['MLX_MAX_OPS_PER_BUFFER']=='200'
assert valid['runtime_environment']['MLX_MAX_MB_PER_BUFFER']=='256'
assert valid['corpus_bytes']==(root/'valid.jsonl').stat().st_size
if sys.platform=='darwin':
 assert valid['corpus_sha256']==hashlib.sha256((root/'valid.jsonl').read_bytes()).hexdigest()
assert valid['corpus_fingerprint'].startswith('fnv1a64:')
assert [x['id'] for x in valid['samples']]==['math-0','code-0','retrieval-0']
inputs=[json.loads(line) for line in (root/'valid.jsonl').read_text().splitlines()]
for source,sample in zip(inputs,valid['samples']):
 assert sample['category']==source['category'] and sample['status']=='validated'
 assert sample['generated_text']=='' and not sample['prompt_truncated']
 assert 'prompt_token_count' not in sample and 'prompt_token_id_fingerprint' not in sample
 if sys.platform=='darwin':
  assert sample['prompt_sha256']==hashlib.sha256(source['prompt'].encode()).hexdigest()
reformatted=json.loads((root/'reformatted.json').read_text())
assert valid['corpus_fingerprint']==reformatted['corpus_fingerprint']
if sys.platform=='darwin': assert valid['corpus_sha256']!=reformatted['corpus_sha256']
for case in ['duplicate','private-answer','private-tests','blank-prompt','wrong-type','missing-category','empty','malformed','invalid-utf8','missing']:
 failure=json.loads((root/(case+'.json')).read_text())
 assert failure['status']=='failed' and failure['failure_status']=='corpus_validation',case
 assert failure['error'] and failure['model_load_count']==0 and not failure['samples'],case
assert 'duplicate sample id' in json.loads((root/'duplicate.json').read_text())['error']
assert 'unexpected fields: answer' in json.loads((root/'private-answer.json').read_text())['error']
assert 'unexpected fields: tests' in json.loads((root/'private-tests.json').read_text())['error']
assert (root/'existing.json').read_bytes()==b'preserve existing report exactly\n'
for case in ['tokens','prefill','modes']: assert not (root/(case+'.json')).exists(),case
print('Generation benchmark public-corpus, provenance and failure-persistence tests passed (no model loaded).')
PY
