"""Check adapter against author phrase-grounding methods and pretrained model.
AST extraction avoids importing the author's unrelated dataset/plotting dependencies.
Only hardcoded CUDA placement is changed to the requested CPU device.
"""
import ast,math,random,types
from pathlib import Path
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from einops import rearrange
from scipy import ndimage

def check(adapter,image_path,class_name):
 root=Path(__file__).resolve().parents[1]
 import model_MaCo
 source=ast.parse((root/'Phrase Grounding.py').read_text())
 engine_class=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='Engine')
 base_source=ast.parse((root/'eval/common.py').read_text())
 base_class=next(n for n in base_source.body if isinstance(n,ast.ClassDef) and n.name=='ImageTextInferenceEngine')
 base_class.body=[n for n in base_class.body if isinstance(n,ast.FunctionDef) and n.name in ['_get_similarity_map_from_embeddings','convert_similarity_to_image_size']]
 engine_class.body=[n for n in engine_class.body if isinstance(n,ast.FunctionDef) and n.name in ['pil_loader','_text_process','get_emb','get_similarity_map_from_raw_data']]
 class Placement(ast.NodeTransformer):
  def visit_Call(self,node):
   self.generic_visit(node)
   if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
    node.func.attr='to';node.args=[ast.Name(id='device',ctx=ast.Load())]
   return node
 engine_class=Placement().visit(engine_class)
 module=ast.fix_missing_locations(ast.Module(body=[base_class,engine_class],type_ignores=[]))
 namespace={'torch':torch,'nn':nn,'F':F,'Path':Path,'np':np,'Image':Image,'random':random,'ndimage':ndimage,'ceil':math.ceil,'floor':math.floor,'Optional':__import__('typing').Optional,'rearrange':rearrange,'trans':lambda:transforms.Compose([transforms.Grayscale(3),transforms.ToTensor(),transforms.Normalize([.4978],[.2449])])}
 exec(compile(module,'author_grounding_device_port','exec'),namespace)
 engine=namespace['Engine']()
 sd=torch.load(adapter.checkpoint,map_location='cpu')['model']
 sd={k:v for k,v in sd.items() if k!='WCE.weight'}
 sr=1 if sd['decoder_pred.weight'].shape[0]==(16*2)**2*3 else 0
 engine.model=model_MaCo.MaCo(img_size=224,patch_size=16,in_chans=3,embed_dim=768,depth=12,num_heads=12,decoder_embed_dim=768,decoder_depth=4,decoder_num_heads=6,mlp_ratio=4,norm_layer=partial(nn.LayerNorm,eps=1e-6),T=.07,SR=sr).eval()
 out_dim,in_dim=sd['bert_encoder.model.bert.pooler.dense.weight'].shape
 engine.model.bert_encoder.model.bert.pooler.dense=nn.Linear(in_dim,out_dim)
 engine.model.load_state_dict(sd,strict=True)
 engine.tokenizer=adapter.model.bert_encoder.tokenizer
 native=engine.get_similarity_map_from_raw_data(Path(image_path),f'There is {class_name}.','cpu')
 with Image.open(image_path) as im:hw=(im.height,im.width)
 native=F.interpolate(torch.from_numpy(native).reshape(1,1,224,224),size=hw,mode='nearest')[0,0].numpy()
 actual=adapter.grounding([image_path],[class_name])[0][0]
 err=float(np.max(np.abs(native-actual)))
 if not np.allclose(native,actual,atol=1e-6,rtol=1e-6):raise RuntimeError(f'Author grounding parity failed: {err}')
 return {'passed':True,'max_abs_error':err,'shape':list(actual.shape),'comparison':'Original Engine methods (CPU placement only), original full MaCo model with checkpoint-shaped pooler and strict checkpoint; native 224 maps nearest restored to original H/W'}
