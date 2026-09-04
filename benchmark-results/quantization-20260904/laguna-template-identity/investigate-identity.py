import mlx.core as mx
mx.set_default_device(mx.cpu)
import json,pathlib
base=pathlib.Path('/Users/stephen/Documents/ChatGPT/midnight/tmp/models')
s=base/'Laguna-XS-2.1-BF16-c5f36269'; t=base/'Laguna-XS-2.1-Q4R8-c42e0a8f'
si=json.loads((s/'model.safetensors.index.json').read_text())['weight_map'];ti=json.loads((t/'model.safetensors.index.json').read_text())['weight_map']
def read(root,index,key):return mx.load(str(root/index[key]))[key]
result={'mlx_version':mx.__version__,'device':str(mx.default_device()),'tensors':[]}
checks=[('model.layers.1.mlp.shared_expert.down_proj.weight','language_model.model.layers.1.mlp.shared_expert.down_proj',4,None),('model.layers.1.mlp.gate.weight','language_model.model.layers.1.mlp.gate.proj',8,None),('model.layers.1.mlp.experts.0.down_proj.weight','language_model.model.layers.1.mlp.switch_mlp.down_proj',4,0)]
for sk,m,bits,expert in checks:
 source=read(s,si,sk)
 target=[read(t,ti,m+'.'+suffix) for suffix in ['weight','scales','biases']]
 if expert is not None:target=[v[expert] for v in target]
 standard=mx.quantize(source,group_size=64,bits=bits)
 record={'source':sk,'shape':list(source.shape),'dtype':str(source.dtype),'target_dtypes':[str(v.dtype) for v in target],'differences':{}}
 for suffix,a,b in zip(['weight','scales','biases'],standard,target):
  record['differences'][suffix]={'count':int(mx.sum(a!=b).item()),'total':a.size,'max_abs':float(mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))).item())}
 standard_d=mx.dequantize(*standard,group_size=64,bits=bits).astype(mx.float32)
 target_d=mx.dequantize(*target,group_size=64,bits=bits).astype(mx.float32)
 record['mse_standard']=float(mx.mean(mx.square(standard_d-source.astype(mx.float32))).item())
 record['mse_public']=float(mx.mean(mx.square(target_d-source.astype(mx.float32))).item())
 result['tensors'].append(record)
result['unquantized_norms']=[]
for key in ['model.layers.0.input_layernorm.weight','model.layers.1.input_layernorm.weight','model.layers.1.self_attn.q_norm.weight']:
 a=read(s,si,key);b=read(t,ti,'language_model.'+key)
 result['unquantized_norms'].append({'key':key,'equal':bool(mx.array_equal(a,b).item()),'max_abs':float(mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))).item())})
print(json.dumps(result,indent=2))
pathlib.Path('/private/tmp/midnight-optimization-20260904/identity-investigation.json').write_text(json.dumps(result,indent=2))
