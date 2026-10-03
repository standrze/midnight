from pathlib import Path
import subprocess, json, hashlib, datetime, runpy
root=Path('/private/tmp/midnight-priorities-20260904');repo=Path('/private/tmp/midnight-memory-release.ccwHpH/repo');out=root/'generated';out.mkdir(exist_ok=True)
base=Path('/Users/stephen/Documents/ChatGPT/midnight/tmp/models'); binary=repo/'.build/release/model-runner-generation-bench'
models=[('standard','Laguna-XS-2.1-Q4R8-c42e0a8f'),('ls2','Laguna-XS-2.1-Q4R8-LS2-c5f36269'),('awss','Laguna-XS-2.1-Q4R8-AWSS-c5f36269')]
info=runpy.run_path(str(repo/'Scripts/benchmark-campaign.py'))['model_info']
manifest={'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'metallib_sha256':hashlib.sha256((binary.parent/'mlx.metallib').read_bytes()).hexdigest(),'models':[]}
for label,name in models:
 model=base/name
 command=[str(binary),str(model),str(root/'generated-corpus/tasks.jsonl'),str(out/(label+'.json')),'--engine','metal','--tokens','1024','--prefill-step-size','512','--kv-compression','none']
 print('Starting',label,flush=True)
 item={'label':label,'command':command,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':info({'label':label,'path':str(model)},full_hash=True)}
 manifest['models'].append(item);(out/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
 with (out/(label+'.log')).open('x') as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 item['exit_code']=result.returncode;item['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(out/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
 if result.returncode:raise SystemExit('Generation failed: '+label)
 print('Completed',label,flush=True)
