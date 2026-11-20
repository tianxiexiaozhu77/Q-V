"""Read-only preprocessing timings; no model, GPU, labels, or adapter changes."""
import io,json,time,statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image
from open_clip.transform import image_transform
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
cfg=json.loads((ROOT/'checkpoints/open_clip_config.json').read_text())
transform=image_transform(224,is_train=False,mean=cfg['preprocess_cfg']['mean'],std=cfg['preprocess_cfg']['std'])
rows=json.loads((ROOT.parent/'evaluation/matched_phase6_20261022/chest14.json').read_text())['rows']
paths=[r['image'] for r in rows[10000:10016]]
other=[r['image'] for r in rows[15000:15016]]
def measure(paths):
 out=[]
 for path in paths:
  t=time.perf_counter();data=Path(path).read_bytes();t1=time.perf_counter()
  with Image.open(io.BytesIO(data)) as im:im=im.convert('RGB');im.load()
  t2=time.perf_counter();x=transform(im);t3=time.perf_counter()
  out.append(dict(path=path,bytes=len(data),read=t1-t,decode=t2-t1,transform=t3-t2,total=t3-t))
 return out
cold=measure(paths);warm=measure(paths)
t=time.perf_counter()
with ThreadPoolExecutor(8) as pool:sizes=list(pool.map(lambda p:len(Path(p).read_bytes()),other))
prefetch_seconds=time.perf_counter()-t
prefetched=measure(other)
def summary(rows):return {k:dict(total_seconds=sum(r[k] for r in rows),median_seconds=statistics.median(r[k] for r in rows),max_seconds=max(r[k] for r in rows)) for k in ['read','decode','transform','total']}
result=dict(status='complete',count=16,transform=repr(transform),sequential_first=summary(cold),sequential_second=summary(warm),prefetch_workers=8,prefetch_seconds=prefetch_seconds,after_prefetch=summary(prefetched),records=dict(first=cold,second=warm,prefetched=prefetched))
(ROOT/'local_setup/chest14_io_profile.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
