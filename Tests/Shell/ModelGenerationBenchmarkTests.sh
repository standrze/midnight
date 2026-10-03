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
def adapter(name, *, scale=16, dropout=None, fine_tune_type='lora', keys=None):
 directory=root/name
 directory.mkdir()
 parameters={'rank':4,'scale':scale}
 if dropout is not None: parameters['dropout']=dropout
 if keys is not None: parameters['keys']=keys
 configuration={'fine_tune_type':fine_tune_type,'num_layers':1,'lora_parameters':parameters}
 (directory/'adapter_config.json').write_text(json.dumps(configuration))
 # Intentionally not a tensor archive: --validate-only must inspect only files/config.
 (directory/'adapters.safetensors').write_bytes(b'metadata-only adapter fixture\n')
 return directory
adapter('adapter-default')
adapter('adapter-negative-config-scale',scale=-2)
adapter('adapter-dora',dropout=0.25,fine_tune_type='dora',keys=['self_attn.q_proj'])
missing_config=adapter('adapter-missing-config')
(missing_config/'adapter_config.json').unlink()
missing_weights=adapter('adapter-missing-weights')
(missing_weights/'adapters.safetensors').unlink()
malformed_adapter=adapter('adapter-malformed')
(malformed_adapter/'adapter_config.json').write_text('{"fine_tune_type":')
missing_field=adapter('adapter-missing-field')
(missing_field/'adapter_config.json').write_text('{"fine_tune_type":"lora","num_layers":1}')
adapter('adapter-dropout-negative',dropout=-0.1)
adapter('adapter-dropout-one',dropout=1)
PY
MLX_MAX_OPS_PER_BUFFER=200 MLX_MAX_MB_PER_BUFFER=256 \
  "$generation_bench" "$fixture_dir/missing-model" "$fixture_dir/valid.jsonl" \
  "$fixture_dir/valid.json" --engine cpu --tokens 17 --context-length 4096 \
  --prefill-step-size 128 --validate-only > "$fixture_dir/valid.log" 2>&1
"$generation_bench" "$fixture_dir/missing-model" "$fixture_dir/reformatted.jsonl" \
  "$fixture_dir/reformatted.json" --engine cpu --validate-only > "$fixture_dir/reformatted.log" 2>&1
for adapter_case in adapter-default adapter-negative-config-scale adapter-dora; do
  "$generation_bench" "$fixture_dir/missing-model" "$fixture_dir/valid.jsonl" \
    "$fixture_dir/$adapter_case.json" --engine cpu --adapter "$fixture_dir/$adapter_case" \
    --validate-only > "$fixture_dir/$adapter_case.log" 2>&1
done
"$generation_bench" "$fixture_dir/missing-model" "$fixture_dir/valid.jsonl" \
  "$fixture_dir/adapter-zero.json" --engine cpu --adapter "$fixture_dir/adapter-default" \
  --adapter-scale 0 --validate-only > "$fixture_dir/adapter-zero.log" 2>&1
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
expect_failure adapter-blank "$generation_bench" "$fixture_dir/missing-model" \
  "$fixture_dir/valid.jsonl" "$fixture_dir/adapter-blank.json" --adapter '   ' --validate-only
expect_failure adapter-scale-only "$generation_bench" "$fixture_dir/missing-model" \
  "$fixture_dir/valid.jsonl" "$fixture_dir/adapter-scale-only.json" --adapter-scale 0.5 --validate-only
for scale_case in negative:-1 nan:nan infinite:inf negative-infinite:-inf; do
  label="adapter-scale-${scale_case%%:*}"
  expect_failure "$label" "$generation_bench" "$fixture_dir/missing-model" \
    "$fixture_dir/valid.jsonl" "$fixture_dir/$label.json" --adapter "$fixture_dir/adapter-default" \
    --adapter-scale="${scale_case#*:}" --validate-only
done
for adapter_case in adapter-missing-directory adapter-missing-config adapter-missing-weights adapter-malformed adapter-missing-field adapter-dropout-negative adapter-dropout-one; do
  expect_failure "$adapter_case" "$generation_bench" "$fixture_dir/missing-model" \
    "$fixture_dir/valid.jsonl" "$fixture_dir/$adapter_case.json" --engine cpu \
    --adapter "$fixture_dir/$adapter_case" --validate-only
done
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
assert 'adapter' not in valid
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
assert 'adapter' not in reformatted
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
adapter_cases=[
 ('adapter-default','adapter-default',16,None),
 ('adapter-negative-config-scale','adapter-negative-config-scale',-2,None),
 ('adapter-dora','adapter-dora',16,None),
 ('adapter-zero','adapter-default',16,0),
]
for case,directory,configured_scale,requested_scale in adapter_cases:
 report=json.loads((root/(case+'.json')).read_text())
 assert report['status']=='validated' and report['model_load_count']==0,case
 assert report['input_sample_count']==3 and report['completed_sample_count']==0,case
 assert report['prepared_sample_count']==0 and len(report['samples'])==3,case
 assert all(sample['status']=='validated' for sample in report['samples']),case
 observed=report['adapter']
 # Foundation and pathlib may spell the same macOS directory as /var or /private/var.
 expected_path=(root/directory).absolute()
 observed_path=Path(observed['path'])
 assert observed_path.is_absolute() and observed_path.samefile(expected_path),case
 assert observed['validation']=='files_and_config_only' and observed['loaded'] is False,case
 assert observed['configured_scale']==configured_scale,case
 if requested_scale is None:
  assert 'requested_scale' not in observed,case
 else:
  assert observed['requested_scale']==requested_scale,case
 assert observed['effective_scale']==(configured_scale if requested_scale is None else requested_scale),case
 assert 'files_unchanged_after_load' not in observed and 'files_unchanged_after_run' not in observed,case
 expected_provenance='observed_preload' if sys.platform=='darwin' else 'observed_preload_only_sha256_unavailable'
 assert observed['provenance_verification']==expected_provenance,case
 for prefix,filename in [('config','adapter_config.json'),('weights','adapters.safetensors')]:
  data=(root/directory/filename).read_bytes()
  assert observed[prefix+'_bytes']==len(data),case
  if sys.platform=='darwin': assert observed[prefix+'_sha256']==hashlib.sha256(data).hexdigest(),case
for case in ['adapter-missing-directory','adapter-missing-config','adapter-missing-weights','adapter-malformed','adapter-missing-field','adapter-dropout-negative','adapter-dropout-one']:
 failure=json.loads((root/(case+'.json')).read_text())
 assert failure['status']=='failed' and failure['failure_status']=='adapter_validation',case
 assert failure['error'] and failure['model_load_count']==0 and not failure['samples'],case
 assert failure['input_sample_count']==3,case
for case in ['adapter-blank','adapter-scale-only','adapter-scale-negative','adapter-scale-nan','adapter-scale-infinite','adapter-scale-negative-infinite']:
 assert not (root/(case+'.json')).exists(),case
print('Generation benchmark public-corpus, adapter preflight, provenance and failure-persistence tests passed (no model loaded).')
PY
