"""Early native inference audit while the other nine ensemble members download."""
import sys,json,hashlib
from pathlib import Path
import torch,numpy as np
import adapter
from torchvision.transforms import Normalize,Resize,InterpolationMode
import clip,zero_shot
from model import build_model
zero_shot.tqdm=lambda xs:xs
torch.set_num_threads(4)
root=Path(adapter.__file__).resolve().parents[1]
p=next(root.joinpath('checkpoints').glob('*.pt'))
model=build_model(torch.load(p,map_location='cpu')).float().eval().requires_grad_(False)
a=adapter.Adapter.__new__(adapter.Adapter)
a.device=torch.device('cpu');a.models=[model];a.tokenize=clip.tokenize;a._text={}
a.normalize=Normalize((101.48761,)*3,(83.43944,)*3)
a.resize=Resize(224,interpolation=InterpolationMode.BICUBIC,antialias=False)
image=sys.argv[1];classes=['Pneumonia','Pneumothorax']
scores=a.predict([image],classes)
native=zero_shot.run_softmax_eval(model,[{'img':a._image(image)[None]}],classes,('There is {}.','There is no {}.'))
diff=float(np.max(np.abs(scores-native)));assert diff<1e-6
report=dict(status='passed',checkpoint=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scores=scores.tolist(),native_max_abs_diff=diff,ensemble_complete=False)
(root/'local_setup/cpu_single_checkpoint_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
