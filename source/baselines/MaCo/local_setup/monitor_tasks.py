"""Read-only watch of the two assigned baseline runs; never starts GPU work."""
import json,pathlib,time,math
BASE=pathlib.Path(__file__).resolve().parents[2]
RUNS={'MaCo':BASE/'runs/matched_20261022_183356_2424149','CARZero':BASE/'runs/gpu45_20261022_1918'}
CAT=json.loads((BASE/'evaluation/matched_phase6_20261022/catalog.json').read_text())['datasets']
seen={};seen_result=set()
def finite(x):
 if isinstance(x,float):return math.isfinite(x)
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 return True
while True:
 done=True
 for name,run in RUNS.items():
  status=run/'status.json'
  if not status.is_file():
   done=False
   if seen.get(name)!='waiting_directory':print(json.dumps({'model':name,'status':'waiting_directory','run':str(run)}),flush=True);seen[name]='waiting_directory'
   continue
  try:s=json.loads(status.read_text())
  except (OSError,json.JSONDecodeError):done=False;continue
  model=s['models'][name];state=model['status'];attempts=model.get('attempts',[]);signature=(state,len(attempts))
  if seen.get(name)!=signature:print(json.dumps({'model':name,'status':state,'attempts':len(attempts),'updated_at':s.get('updated_at')}),flush=True);seen[name]=signature
  for attempt in attempts:
   out=pathlib.Path(attempt['output']);stage=attempt['stage']
   for key,ent in CAT.items():
    path=out/key/'result.json'
    if str(path) in seen_result or not path.is_file():continue
    try:r=json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):continue
    seen_result.add(str(path));record={'model':name,'stage':stage,'dataset':key,'status':r['status'],'rows':r.get('rows')}
    if r['status'] in ['complete','smoke_passed']:
     assert r['rows']==(ent['count'] if stage=='full' else 2),(name,key,'row count')
     assert r['class_names']==ent['class_names'],(name,key,'class names')
     assert finite(r['metrics']),(name,key,'nonfinite')
     if stage=='full' and ent['task']=='classification':assert r['metrics']['valid_class_count']==len(ent['class_names']),(name,key,'missing class')
     record['checked']='row count, exact classes, finite metrics'
    elif r['status']=='failed':record['error']=r.get('error');record['result_path']=str(path)
    print(json.dumps(record,ensure_ascii=False),flush=True)
  if state not in ['complete','unsupported','failed']:done=False
  if state=='failed':
   print(json.dumps({'worker_failed':name,'attempts':attempts},ensure_ascii=False),flush=True);raise SystemExit(2)
 if done:print('Both models complete; completed task manifests and metrics verified',flush=True);break
 time.sleep(45)
