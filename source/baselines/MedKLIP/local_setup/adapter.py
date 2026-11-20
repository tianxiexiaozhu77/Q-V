"""Inference-only bridge to the released MedKLIP model (no fitted parameters)."""
from pathlib import Path
import sys, json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
ROOT=Path(__file__).resolve().parents[1]
AUTHOR=ROOT/'Sample_zero-shot_Classification_CXR14'
sys.path.insert(0,str(AUTHOR))
from models.model_MedKLIP import MedKLIP
from models.tokenization_bert import BertTokenizer as AuthorBertTokenizer, load_vocab, BasicTokenizer
class BertTokenizer(AuthorBertTokenizer):
 """Compatibility only: modern transformers asks for vocab during base __init__."""
 def __init__(self,vocab_file,*args,**kwargs):
  self.vocab=load_vocab(vocab_file)
  self.basic_tokenizer=BasicTokenizer(do_lower_case=kwargs.get('do_lower_case',True))
  super().__init__(vocab_file,*args,**kwargs)


# MedKLIP paper Appendix A, Table A1; also reproduced verbatim in the MAVL
# authors' observation explanation.json and configs/covid_medklip.yaml.
COVID_DESCRIPTION='It is a contagious disease caused by a virus. Ground-glass opacities, consolidation, thickening, pleural effusions commonly appear in infection.'

# Published CXR14 mapping and direct aliases to the author's original concepts.
ALIASES={'covid':'covid19','covid-19':'covid19','pneumonia':'eumonia','no finding':'normal','no findings':'normal','enlarged cardiomediastinum':'enlarge',
 'lung opacity':'opacity','lung lesion':'lesion','pleural effusion':'effusion','infiltration':'infiltrate',
 'fibrosis':'tail_abnorm_obs','pleural thickening':'thicken','pleural_thickening':'thicken',
 'lung mass':'mass','lung nodule':'nodule','support devices':'device'}
class Adapter:
 def __init__(self,device='cuda'):
  self.device=torch.device(device)
  self.book=json.loads((ROOT/'PreTrain_MedKLIP/data_file/observation explanation.json').read_text())
  self.base_classes=list(self.book);self.classes=self.base_classes
  self.extra_book={'covid19':COVID_DESCRIPTION};self.covid_query=None;self.active_mode='base75'
  self.checkpoint=ROOT/'local_assets/checkpoint_final.pth'
  bert=str(ROOT/'local_assets/Bio_ClinicalBERT')
  tok=BertTokenizer.from_pretrained(bert,local_files_only=True)
  self.tokenizer=tok
  book=tok(list(self.book.values()),padding='max_length',truncation=True,max_length=64,return_tensors='pt').to(self.device)
  config=dict(d_model=256,res_base_model='resnet50',H=4,N=4,dropout=.1,attribute_set_size=2,num_queries=75,text_encoder=bert)
  torch.manual_seed(42)
  self.model=MedKLIP(config,book)
  state=torch.load(self.checkpoint,map_location='cpu')['model']
  state={k.removeprefix('module.'):v for k,v in state.items()}
  self.model.load_state_dict(state,strict=True)
  self.model.to(self.device).eval().requires_grad_(False)
  self.base_queries=self.model.disease_book
  self.transform=transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
  self.metadata={'model':'MedKLIP','score_kind':'positive_class_softmax_probability','checkpoint':str(self.checkpoint),
   'preprocess':'RGB; bilinear resize 224x224; ImageNet mean/std; no TTA',
   'prompts':'author 75 disease descriptions; COVID tasks append the author Appendix A COVID description; other tasks retain exactly the original75 context; Bio_ClinicalBERT CLS, max_length=64, author tokenizer',
   'query_extension':{'class':'covid19','definition':COVID_DESCRIPTION,'source':'https://arxiv.org/html/2301.02228#A1','protocol':'MedKLIP Section3.5: unseen disease description appended to the original entity query set; shared classifier and released checkpoint unchanged.','native_example':'https://github.com/hieuvmphan/CVPR2024_MAVL/blob/main/Zero-shot_classification/configs/covid_medklip.yaml'},
   'label_mapping':ALIASES,'author_key_correction':'Official JSON key eumonia is the pneumonia query at index20 in author test.py; its original description and sequence position are unchanged.','limitations':['TB/Granuloma queries lack verified author definitions in this frozen protocol; these tasks are rejected. This is a description-resource/protocol limitation, not an architectural closed-vocabulary restriction.',
   'Fibrosis uses author CXR14 mapping tail_abnorm_obs, a broad abnormality concept.'],
   'original_code':str(AUTHOR/'test.py')}
 @torch.inference_mode()
 def _activate_queries(self,covid):
  mode='base75_plus_covid' if covid else 'base75'
  if self.active_mode==mode:return
  if covid:
   if self.covid_query is None:
    bert=str(ROOT/'local_assets/Bio_ClinicalBERT')
    tokens=self.tokenizer([COVID_DESCRIPTION],padding='max_length',truncation=True,max_length=64,return_tensors='pt').to(self.device)
    text_encoder=self.model._get_bert_basemodel(bert,freeze_layers=None).to(self.device).eval()
    self.covid_query=text_encoder(input_ids=tokens['input_ids'],attention_mask=tokens['attention_mask']).last_hidden_state[:,0,:].detach()
   self.model.disease_book=torch.cat([self.base_queries,self.covid_query],dim=0)
   self.classes=self.base_classes+['covid19']
  else:
   self.model.disease_book=self.base_queries;self.classes=self.base_classes
  self.active_mode=mode
 def _indices(self,names):
  keys=[ALIASES.get(x.lower(),x.lower()) for x in names]
  self.check_classes(names)
  self._activate_queries('covid19' in keys)
  return [self.classes.index(k) for k in keys]
 def check_classes(self,class_names):
  keys=[ALIASES.get(x.lower(),x.lower()) for x in class_names]
  missing=[x for x,k in zip(class_names,keys) if k not in self.book and k not in self.extra_book]
  if missing:raise NotImplementedError('MedKLIP verified description resources lack: '+', '.join(missing))
  return True
 @torch.inference_mode()
 def predict(self,image_paths,class_names):
  idx=self._indices(class_names)
  images=[]
  for p in image_paths:
   with Image.open(p) as im:images.append(self.transform(im.convert('RGB')))
  x=torch.stack(images).to(self.device)
  return self.model(x).softmax(-1)[:,idx,1].cpu().numpy()
 @torch.inference_mode()
 def grounding(self,image_paths,class_names):
  idx=self._indices(class_names);images=[];sizes=[]
  for p in image_paths:
   with Image.open(p) as im:
    sizes.append((im.height,im.width));images.append(self.transform(im.convert('RGB')))
  captured=[]
  hook=self.model.decoder.register_forward_hook(lambda m,args,out:captured.append(out[1]))
  try:self.model(torch.stack(images).to(self.device))
  finally:hook.remove()
  ws=captured[0];maps=(ws[-4]+ws[-3]+ws[-2]+ws[-1])/4
  maps=maps[:,idx].reshape(len(images),len(idx),14,14)
  maps=maps.repeat_interleave(16,-2).repeat_interleave(16,-1)
  self.metadata['grounding']={'source':str(ROOT/'Sample_Zero-Shot_Grounding_RSNA/test.py'),
   'method':'mean of four cross-attention layers, 14x14 grid repeated16 to224 as author; nearest inverse-resize to full original image extent',
   'values':'raw attention, no minmax normalization','crop':False}
  return [torch.nn.functional.interpolate(m[None],size=sz,mode='nearest')[0].cpu().numpy() for m,sz in zip(maps,sizes)]
def create_adapter(device='cuda'):return Adapter(device)
