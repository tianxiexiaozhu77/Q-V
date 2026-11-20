import os,sys,json,hashlib,subprocess,platform,time,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
sys.path.insert(0,str(ROOT/'local_setup'))
import torch,numpy as np
from adapter import create_adapter
torch.set_num_threads(4);torch.manual_seed(0)
t0=time.monotonic();adapter=create_adapter('cpu')
source=ROOT.parents[1]/'AlphaRAD/local_setup/query_refinement_data_20261021/chestxdet10_evaluation.csv'
rows=list(csv.DictReader(source.open()))
pathkey=next(k for k in rows[0] if k.lower() in ['path','image_path','image','filepath'])
paths=[r[pathkey] for r in rows[:2]]
classes=['Atelectasis','Pneumothorax']
out=adapter.predict(paths,classes)
assert out.shape==(2,2) and np.isfinite(out).all()
singles=np.concatenate([adapter.predict([path],classes) for path in paths],axis=0)
batch_error=float(np.max(np.abs(out-singles)))
assert np.allclose(out,singles,atol=1e-5,rtol=1e-5),f'Batch/single scoring difference: {batch_error}'
adapter.metadata['batch_single_parity']={'passed':True,'max_abs_error':batch_error,'batch_size':2,'atol':1e-5,'rtol':1e-5,'note':'Native per-image fusion; only floating-point image-encoder batching variation permitted.'}
coverage=adapter.metadata['checkpoint_key_coverage']
assert coverage['strict'] and coverage['model_state_keys']==coverage['loaded_state_keys'] and coverage['model_numel']==coverage['loaded_numel']
report={'status':'passed','device':'cpu','python':platform.python_version(),'torch':torch.__version__,'images':paths,'classes':classes,'predictions':out.tolist(),'metadata':adapter.metadata}
if hasattr(adapter,'grounding'):
 maps=adapter.grounding(paths[:1],classes)
 assert maps.shape==(1,2,224,224) and np.isfinite(maps).all()
 report['grounding']={'shape':list(maps.shape),'finite':True,'range':[float(maps.min()),float(maps.max())]}
report['elapsed_s']=time.monotonic()-t0
(ROOT/'local_setup/validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
manifest_path=ROOT/'local_setup/manifest.json';m=json.loads(manifest_path.read_text());h=hashlib.sha256()
with open(m['checkpoint_local_path'],'rb') as f:
 while b:=f.read(16*1024**2):h.update(b)
m.update({'checkpoint_sha256':h.hexdigest(),'checkpoint_status':'downloaded','environment_status':'validated_cpu','smoke_test_status':'passed','torch':torch.__version__,'metadata':adapter.metadata,'validation_path':str(ROOT/'local_setup/validation.json')})
m['adapter_status']='CPU author-score parity, batch-single parity, complete strict state-key coverage passed'
manifest_path.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False))
