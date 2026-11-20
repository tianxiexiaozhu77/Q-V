import concurrent.futures,hashlib,json,os,pathlib,subprocess,time
BASE=pathlib.Path(__file__).resolve().parents[2]

def run(name):
 root=BASE/name
 manifest=root/'local_setup/manifest.json'
 ckpt=pathlib.Path(json.loads(manifest.read_text())['checkpoint_local_path'])
 while not ckpt.is_file():time.sleep(10)
 print(f'{name}: checkpoint complete, starting CPU verification',flush=True)
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4';env['PYTHONDONTWRITEBYTECODE']='1'
 log=root/'local_setup/validation.log'
 with log.open('w') as f:
  proc=subprocess.run(['bash',str(root/'local_setup/python_local.sh'),str(root/'local_setup/validate.py')],cwd=BASE.parent,env=env,stdout=f,stderr=subprocess.STDOUT)
 if proc.returncode:
  (root/'local_setup/validation_failure.json').write_text(json.dumps({'status':'failed','returncode':proc.returncode,'log':str(log)},indent=2)+'\n')
  print(f'{name}: CPU verification failed; see {log}',flush=True);return
 validation=json.loads((root/'local_setup/validation.json').read_text())
 assert validation['status']=='passed' and validation['metadata']['native_score_parity']['passed']
 if name=='MaCo':assert validation['grounding_native_parity']['passed']
 ready={'ready':True,'model':name,'adapter_sha256':hashlib.sha256((root/'local_setup/adapter.py').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'launcher_sha256':hashlib.sha256((root/'local_setup/python_local.sh').read_bytes()).hexdigest(),'cpu_validation':validation,'validation_file':str(root/'local_setup/validation.json')}
 tmp=root/'local_setup/READY.json.tmp';tmp.write_text(json.dumps(ready,indent=2,ensure_ascii=False)+'\n');tmp.replace(root/'local_setup/READY.json')
 print(f'{name}: READY, source hashes frozen',flush=True)
with concurrent.futures.ThreadPoolExecutor(2) as ex:list(ex.map(run,['MaCo','CARZero']))
