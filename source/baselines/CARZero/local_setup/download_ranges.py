"""Resume validated HTTP byte ranges from an official public artifact URL."""
import argparse, concurrent.futures, os, pathlib, time, requests, hashlib, json
p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('output');p.add_argument('--workers',type=int,default=8);a=p.parse_args()
out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);partdir=out.parent/(out.name+'.parts');partdir.mkdir(exist_ok=True)
r=requests.get(a.url,headers={'Range':'bytes=0-0'},timeout=(20,90));r.raise_for_status()
if r.status_code!=206:raise RuntimeError('Range unsupported')
size=int(r.headers['Content-Range'].split('/')[-1]);chunk=32*1024**2
print(json.dumps({'total_bytes':size,'url':a.url,'workers':a.workers}),flush=True)
def one(start):
 end=min(size,start+chunk)-1; dest=partdir/f'{start:012d}'; length=end-start+1
 if dest.exists() and dest.stat().st_size==length:return dest
 for attempt in range(30):
  try:
   tmp=pathlib.Path(str(dest)+'.tmp')
   got=tmp.stat().st_size if tmp.exists() else 0
   if got==length:
    os.replace(tmp,dest);return dest
   if got>length:raise RuntimeError('oversized partial')
   offset=start+got
   request_end=min(end,offset+4*1024**2-1)
   with requests.get(a.url,headers={'Range':f'bytes={offset}-{request_end}'},stream=True,timeout=(20,120)) as rr:
    rr.raise_for_status()
    if rr.status_code!=206 or rr.headers.get('Content-Range')!=f'bytes {offset}-{request_end}/{size}':raise RuntimeError(('bad range',rr.status_code,rr.headers))
    with open(str(dest)+'.tmp','ab') as f:
     for data in rr.iter_content(256*1024):f.write(data)
    if os.path.getsize(str(dest)+'.tmp')!=length:continue
    os.replace(str(dest)+'.tmp',dest)
    print(f'completed {start}-{end}',flush=True);return dest
  except Exception as e:
   print(f'retry {start} {attempt} {type(e).__name__}: {e}',flush=True);time.sleep(min(attempt*2,10))
 raise RuntimeError(f'Failed chunk {start}')
with concurrent.futures.ThreadPoolExecutor(a.workers) as ex:files=list(ex.map(one,range(0,size,chunk)))
h=hashlib.sha256()
with open(str(out)+'.assembling','wb') as f:
 for file in files:
  with open(file,'rb') as src:
   while data:=src.read(8*1024*1024):f.write(data);h.update(data)
 f.flush();os.fsync(f.fileno())
os.replace(str(out)+'.assembling',out)
print(json.dumps({'path':str(out),'bytes':size,'sha256':h.hexdigest()}),flush=True)
for file in files:file.unlink()
partdir.rmdir()
