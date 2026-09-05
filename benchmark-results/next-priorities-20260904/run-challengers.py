from pathlib import Path
import subprocess, json, hashlib, datetime, runpy
root=Path('/private/tmp/midnight-priorities-20260904');repo=Path('/private/tmp/midnight-memory-release.ccwHpH/repo');out=root/'challengers';out.mkdir(exist_ok=True)
base=Path('/Users/stephen/Documents/ChatGPT/midnight/tmp/models');bin=repo/'.build/release';source=base/'Laguna-XS-2.1-BF16-c5f36269';template=base/'Laguna-XS-2.1-Q4R8-c42e0a8f'
info=runpy.run_path(str(repo/'Scripts/benchmark-campaign.py'))['model_info'];manifest={'binaries':{},'runs':[],'models':[]}
for name in ['model-runner-laguna-q4r8-rescore','model-runner-generation-bench','model-runner-quality-bench']:
 manifest['binaries'][name]=hashlib.sha256((bin/name).read_bytes()).hexdigest()
def save(): (out/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
def run(label,command):
 print('Starting',label,flush=True);item={'label':label,'command':command,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};manifest['runs'].append(item);save()
 with (out/(label+'.log')).open('x') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 item['exit_code']=result.returncode;item['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
 if result.returncode:raise SystemExit('Failed: '+label)
 print('Completed',label,flush=True)
for label,name in [('standard','Laguna-XS-2.1-Q4R8-c42e0a8f'),('ls2','Laguna-XS-2.1-Q4R8-LS2-c5f36269'),('awss','Laguna-XS-2.1-Q4R8-AWSS-c5f36269')]:
 run(label+'-nll',[str(bin/'model-runner-quality-bench'),str(base/name),str(base.parent/'calibration/laguna-20260904/corpora/laguna-heldout-192.jsonl'),str(out/(label+'-nll.json')),'--max-tokens-per-sample','2048','--prefill-step-size','512'])
for label,suffix,flags in [('g128-standard','Standard',['--standard-q4']),('g128-ls2','LS2',[])]:
 model=base/('Laguna-XS-2.1-Q4R8-G128-'+suffix+'-c5f36269')
 run(label+'-conversion',[str(bin/'model-runner-laguna-q4r8-rescore'),str(source),str(template),str(model),'--expert-batch','16','--group-size','128']+flags)
 manifest['models'].append(info({'label':label,'path':str(model)},full_hash=True));save()
 run(label+'-generation',[str(bin/'model-runner-generation-bench'),str(model),str(root/'generated-corpus/tasks.jsonl'),str(out/(label+'-generation.json')),'--engine','metal','--tokens','1024','--prefill-step-size','512','--kv-compression','none'])
 run(label+'-nll',[str(bin/'model-runner-quality-bench'),str(model),str(base.parent/'calibration/laguna-20260904/corpora/laguna-heldout-192.jsonl'),str(out/(label+'-nll.json')),'--max-tokens-per-sample','2048','--prefill-step-size','512'])
for kv in ['none','affine8','turbo8v4']:
 run('kv-'+kv,[str(bin/'model-runner-generation-bench'),str(base/'Laguna-XS-2.1-Q4R8-AWSS-c5f36269'),str(root/'retrieval-corpus/tasks.jsonl'),str(out/('kv-'+kv+'.json')),'--engine','metal','--tokens','1024','--prefill-step-size','512','--kv-compression',kv])
