"""CPU parity audit against the author's inference helper; no labels involved."""
import json,sys,hashlib,os,time
from pathlib import Path
import numpy as np
import torch
import adapter
from PIL import Image
ROOT=Path(adapter.__file__).resolve().parents[1]
torch.set_num_threads(4)
image=sys.argv[1]
classes=['Pneumonia','Pneumothorax']
start=time.time()
a=adapter.create_adapter('cpu')
scores=a.predict([image],classes)
assert scores.shape==(1,2) and np.isfinite(scores).all()
report={'model':ROOT.name,'status':'passed','device':'cpu','image':image,'class_names':classes,'scores':scores.tolist(),'metadata':a.metadata}
if ROOT.name=='BiomedCLIP':
 with Image.open(image) as im:inputs=a.preprocess(im.convert('RGB'))[None]
 texts=a.tokenizer([f'There is {c}.' for c in classes])
 with torch.no_grad():
  feats,txt,scale=a.model(inputs,texts)
  expected=(scale*feats@txt.T).numpy()
 report['native_logits_max_abs_diff']=float(np.max(np.abs(expected-scores)))
 assert np.allclose(scores,expected,atol=1e-5,rtol=1e-5)
elif ROOT.name=='BioViL-T':
 expected=np.array([[a.engine.get_similarity_score_from_raw_data(Path(image),f'There is {c}.') for c in classes]])
 report['native_cosine_max_abs_diff']=float(np.max(np.abs(expected-scores)))
 assert np.allclose(scores,expected,atol=1e-6,rtol=1e-5)
 ours=a.grounding([image],classes)[0]
 native=np.stack([a.engine.get_similarity_map_from_raw_data(Path(image),f'There is {c}.') for c in classes])
 report['native_grounding_max_abs_diff']=float(np.nanmax(np.abs(ours-native)))
 report['map_shape']=list(ours.shape)
 report['valid_map_fraction']=float(np.isfinite(ours).mean())
 assert np.allclose(ours,native,atol=1e-6,rtol=1e-5,equal_nan=True)
else:
 import zero_shot,data_process,cv2
 zero_shot.tqdm=lambda xs:xs
 if not hasattr(Image,'ANTIALIAS'):Image.ANTIALIAS=Image.Resampling.LANCZOS
 im=Image.fromarray(cv2.cvtColor(cv2.imread(image),cv2.COLOR_BGR2RGB))
 im=data_process.preprocess(im,desired_size=320)
 image_tensor=torch.from_numpy(np.asarray(im,dtype=np.float32).copy())[None].repeat(3,1,1)
 image_tensor=a.resize(a.normalize(image_tensor))
 assert torch.equal(a._image(image),image_tensor)
 loader=[{'img':image_tensor[None]}]
 expected=[]
 for model in a.models:
  expected.append(zero_shot.run_softmax_eval(model,loader,classes,('There is {}.','There is no {}.')))
 expected=np.stack(expected).mean(0)
 report['native_ensemble_max_abs_diff']=float(np.max(np.abs(expected-scores)))
 assert np.allclose(scores,expected,atol=1e-6,rtol=1e-5)
report['elapsed_seconds']=time.time()-start
(ROOT/'local_setup/cpu_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
