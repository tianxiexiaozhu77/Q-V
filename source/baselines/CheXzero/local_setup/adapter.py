"""Official 10-checkpoint CheXzero ensemble with original pixel normalization."""
from pathlib import Path
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Normalize, Resize, InterpolationMode
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'official'))

def prompt_disease_name(name):
    """Render the OpenI metadata identifier as the author's natural disease phrase."""
    return 'pleural thickening' if name.lower() == 'pleural_thickening' else name

class Adapter:
    def __init__(self,device='cuda'):
        import clip
        from model import build_model
        self.device=torch.device(device)
        self.tokenize=clip.tokenize
        listing=json.loads((ROOT/'local_setup/checkpoint_listing.json').read_text())
        self.checkpoints=[ROOT/'checkpoints'/d['name'] for d in sorted(listing,key=lambda x:x['name'])]
        self.models=[]
        for path in self.checkpoints:
            state=torch.load(path,map_location='cpu')
            model=build_model(state).float().eval().requires_grad_(False).to(self.device)
            self.models.append(model)
        self.normalize=Normalize((101.48761,)*3,(83.43944,)*3)
        # torchvision 0.11.3 official code uses tensor Resize with antialias=False.
        self.resize=Resize(224,interpolation=InterpolationMode.BICUBIC,antialias=False)
        self._text={}
        self.metadata=dict(model='CheXzero',score_kind='mean_10_model_positive_negative_softmax_probability',
            prompt_templates=['There is {disease}.','There is no {disease}.'],
            prompt_protocol='AlphaRAD section4.3 positive prefix; original CheXzero binary mechanism. Negative prefix adapted consistently, not specified by AlphaRAD.',
            preprocess='Official data_process.preprocess long-edge320 Lanczos, zero-pad grayscale320; pixel-domain mean101.48761 std83.43944; tensor bicubic Resize224 antialiasFalse',
            checkpoint=[str(p) for p in self.checkpoints], checkpoint_count=10,
            grounding_available=False,training=False,precision='float32',
            class_mapping={'Pleural_Thickening':'pleural thickening','otherwise':'identity'},
            class_mapping_source='AlphaRAD/cxrclip/evaluator_utils.py disease_name_mapping; output column names and ordering preserved',
            protocol='Official all-ten released checkpoint ensemble. Per-model softmax of raw positive/negative cosine (no logit_scale), then model mean. Does not use standard OpenAI CLIP normalization.')

    def _image(self,path):
        # Match official cv2 BGR read -> RGB -> PIL Lanczos -> grayscale padded image.
        import cv2
        im=cv2.imread(str(path))
        if im is None: raise ValueError(f'Unreadable image {path}')
        im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        ratio=320/max(im.size)
        size=tuple(int(v*ratio) for v in im.size)
        im=im.resize(size,Image.Resampling.LANCZOS)
        padded=Image.new('L',(320,320))
        padded.paste(im,((320-size[0])//2,(320-size[1])//2))
        image=torch.from_numpy(np.asarray(padded,dtype=np.float32).copy()).unsqueeze(0).repeat(3,1,1)
        return self.resize(self.normalize(image))

    @torch.inference_mode()
    def predict(self,image_paths,class_names):
        key=tuple(class_names)
        if not key or any(not isinstance(s,str) or not s.strip() for s in key):raise ValueError('Nonempty disease names required')
        if key not in self._text:
            phrases=[prompt_disease_name(s) for s in key]
            tokens=self.tokenize([f'There is {s}.' for s in phrases]+[f'There is no {s}.' for s in phrases],context_length=77).to(self.device)
            self._text[key]=[F.normalize(m.encode_text(tokens),dim=-1) for m in self.models]
        images=torch.stack([self._image(p) for p in image_paths]).to(self.device)
        predictions=[]
        c=len(key)
        for model,queries in zip(self.models,self._text[key]):
            feats=F.normalize(model.encode_image(images),dim=-1)
            similarity=feats @ queries.T
            predictions.append(torch.sigmoid(similarity[:,:c]-similarity[:,c:]))
        return torch.stack(predictions).mean(0).float().cpu().numpy()

def create_adapter(device='cuda'):
    return Adapter(device)
