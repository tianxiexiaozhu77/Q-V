"""Released MAVL zero-shot classifier; preserves full native disease-query context."""
from pathlib import Path
import json, sys, gc, csv, hashlib, importlib.util
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as tvmodels
ROOT=Path(__file__).resolve().parents[1]
AUTHOR=ROOT/'Zero-shot_classification'
sys.path.insert(0,str(AUTHOR))
from models.model_MAVL import MAVL
from models.tokenization_bert import BertTokenizer as AuthorBertTokenizer, load_vocab, BasicTokenizer
class BertTokenizer(AuthorBertTokenizer):
 """Compatibility only: modern transformers asks for vocab during base __init__."""
 def __init__(self,vocab_file,*args,**kwargs):
  self.vocab=load_vocab(vocab_file)
  self.basic_tokenizer=BasicTokenizer(do_lower_case=kwargs.get('do_lower_case',True))
  super().__init__(vocab_file,*args,**kwargs)

ALIASES={'no finding':'normal','no findings':'normal','enlarged cardiomediastinum':'enlarge',
 'lung opacity':'opacity','lung lesion':'lesion','pleural effusion':'effusion','infiltration':'infiltrate',
 'fibrosis':'tail_abnorm_obs','pleural thickening':'thicken','pleural_thickening':'thicken',
 'lung mass':'mass','lung nodule':'nodule','support devices':'device',
 'covid':'covid19','covid-19':'covid19'}
ANATOMIES=['trachea','left_hilar','right_hilar','hilar_unspec','left_pleural','right_pleural','pleural_unspec','heart_size','heart_border','left_diaphragm','right_diaphragm','diaphragm_unspec','retrocardiac','lower_left_lobe','upper_left_lobe','lower_right_lobe','middle_right_lobe','upper_right_lobe','left_lower_lung','left_mid_lung','left_upper_lung','left_apical_lung','left_lung_unspec','right_lower_lung','right_mid_lung','right_upper_lung','right_apical_lung','right_lung_unspec','lung_apices','lung_bases','left_costophrenic','right_costophrenic','costophrenic_unspec','cardiophrenic_sulcus','mediastinal','spine','clavicle','rib','stomach','right_atrium','right_ventricle','aorta','svc','interstitium','parenchymal','cavoatrial_junction','cardiopulmonary','pulmonary','lung_volumes','unspecified','other']
class Adapter:
 def __init__(self,device='cuda'):
  self.device=torch.device(device);self.model=None;self.active_mode=None;self.dataset=None;self.native_images={}
  self.book=json.loads((AUTHOR/'observation explanation.json').read_text())
  self.concepts=json.loads((ROOT/'Pretrain/concept_gen/gpt4_mimic.json').read_text())
  self.base_classes=[k for k in self.book if k!='covid19']
  assert len(self.base_classes)==75
  self.transform=transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
  self.metadata={'model':'MAVL','score_kind':'positive_class_softmax_probability; COVID uses author text mode',
   'checkpoint':{'classification':str(ROOT/'local_assets/checkpoint_full_46.pth'),'covid':str(ROOT/'local_assets/checkpoint_full_40.pth')},
   'preprocess':'RGB; bilinear 224x224; ImageNet normalization; no TTA',
   'prompts':'author disease explanations and eight GPT4 visual concepts plus description; original 75 query context, +covid19 only for COVID',
   'label_mapping':ALIASES,'limitations':['Author COVID config uses epoch40 text mode with normal/clear references; other classifications use epoch46 feature mode.',
   'No author tuberculosis disease description/concept entry; TB tasks rejected.',
   'Fibrosis mapped to author broad tail_abnorm_obs concept.'],
   'initialization_only_patch':'Instantiate only chosen ResNet50 with random initialization, then strict released checkpoint; skip unused ViT downloads.'}
 def set_dataset(self,entry):
  self.dataset=entry['dataset'];self.native_images={}
  if self.dataset=='rsna_pneumonia':
   medical=ROOT.parent.parent
   csv_path=medical/'RadZero_i2cl/local_assets/splits/native_public5_local_seed42/RSNA/test.csv'
   with csv_path.open(newline='') as f:source=list(csv.DictReader(f))
   by_patient={r['patient_id']:r for r in source}
   assert len(by_patient)==len(source)
   for row in entry['rows']:
    image=row['image'];pid=Path(image).stem
    if pid not in by_patient:raise ValueError('RSNA patient absent from frozen source: '+pid)
    native=medical/'data/radzero/prepared'/by_patient[pid]['image_path']
    if not native.is_file():raise FileNotFoundError(native)
    self.native_images[image]=str(native)
   self.metadata['native_source_remap']={'dataset':self.dataset,'count':len(self.native_images),
    'csv':str(csv_path),'csv_sha256':hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    'match_key':'AlphaRAD PNG filename = original DICOM PatientID; original target image/label/order unchanged',
    'preprocess':'original DICOM pixel_array.astype(float)/255 -> skimage.exposure.equalize_hist -> (255*x).astype(uint8) -> RGB -> 224 bilinear -> ImageNet normalization'}
 def _read_image(self,path):
  if self.dataset=='rsna_pneumonia':
   import pydicom
   from skimage import exposure
   dcm=pydicom.dcmread(self.native_images[path])
   if str(dcm.PatientID)!=Path(path).stem:raise ValueError('DICOM patient identity mismatch')
   pixels=dcm.pixel_array.astype(float)/255.
   return Image.fromarray((255*exposure.equalize_hist(pixels)).astype(np.uint8)).convert('RGB')
  with Image.open(path) as im:return im.convert('RGB')
 def _setup(self,mode):
  if self.active_mode==mode:return
  self.model=None;gc.collect()
  if self.device.type=='cuda':torch.cuda.empty_cache()
  is_ground=mode.startswith('ground')
  is_covid=mode in ('covid','ground_covid')
  self.classes=self.base_classes+(['covid19'] if is_covid else [])
  if is_ground:
   model_file=ROOT/'Zero-shot_grounding/models/model_MAVL.py'
   spec=importlib.util.spec_from_file_location('models.mavl_ground_native',model_file)
   module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   model_class=module.MAVL
   book=json.loads((ROOT/'Zero-shot_grounding/observation explanation.json').read_text())
   concepts_book=json.loads((ROOT/'Pretrain/concept_gen'/('gpt4_mimic_covidr.json' if is_covid else 'gpt4_mimic.json')).read_text())
  else:model_class=MAVL;book=self.book;concepts_book=self.concepts
  bert=str(ROOT/'local_assets/Bio_ClinicalBERT')
  tokenizer=BertTokenizer.from_pretrained(bert,local_files_only=True)
  def tok(text):return tokenizer(text,padding='max_length',truncation=True,max_length=64,return_tensors='pt').to(self.device)
  disease=tok([book[k] for k in self.classes])
  ana=tok(['It is located at '+k for k in ANATOMIES])
  concepts=tok([text for k in self.classes for text in concepts_book[k]])
  config=dict(d_model=256,base_model='resnet50',decoder='cross',num_queries=75,dropout=.1,attribute_set_size=2,N=4,H=4,self_attention=True,text_encoder=bert,pretrained=False)
  old=model_class._get_basemodel
  def selected_resnet(obj,model_name,pretrained=False,layers=None):
   assert model_name=='resnet50'
   res=tvmodels.resnet50(weights=None)
   return torch.nn.Sequential(*list(res.children())[:-3]),int(res.fc.in_features/2)
  model_class._get_basemodel=selected_resnet
  torch.manual_seed(42)
  try:self.model=model_class(config,disease,concepts) if is_ground else model_class(config,ana,disease,concepts)
  finally:model_class._get_basemodel=old
  ckpt=ROOT/'local_assets'/('checkpoint_full_40.pth' if is_covid or is_ground else 'checkpoint_full_46.pth')
  state=torch.load(ckpt,map_location='cpu')['model']
  state={k.removeprefix('module.'):v for k,v in state.items() if 'temp' not in k}
  self.model.load_state_dict(state,strict=True);self.model.to(self.device).eval().requires_grad_(False)
  self.active_mode=mode
 def check_classes(self,class_names):
  keys=[ALIASES.get(n.lower(),n.lower()) for n in class_names]
  missing=[k for k in keys if k not in self.book or k not in self.concepts]
  if missing:raise NotImplementedError('MAVL official book/concepts missing '+', '.join(missing))
  return True
 @torch.inference_mode()
 def predict(self,image_paths,class_names):
  keys=[ALIASES.get(n.lower(),n.lower()) for n in class_names]
  missing=[k for k in keys if k not in self.book or k not in self.concepts]
  if missing:raise NotImplementedError('MAVL official book/concepts missing '+', '.join(missing))
  mode='covid' if 'covid19' in keys else 'feature'
  self._setup(mode);idx=[self.classes.index(k) for k in keys]
  images=[]
  for p in image_paths:
   images.append(self.transform(self._read_image(p)))
  pred_class,_,_,pred_global,_=self.model(torch.stack(images).to(self.device))
  if mode=='feature':return pred_class.softmax(-1)[:,idx,1].cpu().numpy()
  # Author Zero-shot_classification/test.py: average normal and clear references,
  # then average the nine aspect probabilities, without adding a temperature.
  references=[]
  for normal_idx in [0,1]:
   normal=pred_global[:,:,[normal_idx]].repeat(1,1,len(self.classes))
   references.append(torch.stack([normal,pred_global],dim=-1).softmax(-1))
  return torch.stack(references).mean(0)[:,:,idx,1].mean(1).cpu().numpy()
 @torch.inference_mode()
 def grounding(self,image_paths,class_names):
  self.check_classes(class_names)
  keys=[ALIASES.get(n.lower(),n.lower()) for n in class_names]
  self._setup('ground_covid' if 'covid19' in keys else 'ground')
  idx=[self.classes.index(k) for k in keys];images=[];sizes=[]
  for p in image_paths:
   with Image.open(p) as im:sizes.append((im.height,im.width))
   images.append(self.transform(self._read_image(p)))
  _,ws=self.model(torch.stack(images).to(self.device))
  ws=(ws[-4]+ws[-3]+ws[-2]+ws[-1])/4
  side=int(ws.shape[-1]**.5)
  if side*side!=ws.shape[-1]:raise ValueError('Non-square MAVL attention grid')
  maps=ws.reshape(len(images),len(self.classes),9,-1)[:,:,[0,6,7,8]].mean(2)
  maps=maps[:,idx].reshape(len(images),len(idx),side,side)
  maps=torch.nn.functional.interpolate(maps,size=(224,224),mode='bilinear',align_corners=True)
  self.metadata['grounding']={'source':str(ROOT/'Zero-shot_grounding/test.py'),
   'checkpoint':str(ROOT/'local_assets/checkpoint_full_40.pth'),
   'method':'author grounding model; mean4 decoder layers; mean concepts [0,6,7,8]; bilinear align_corners=True to224 and full-original-image inverse resize',
   'values':'raw attention, no normalization','crop':False,
   'native_segmentation_thresholds':'author uses 0.0001:0.00001:0.1; caller must document any changed threshold/normalization protocol'}
  return [torch.nn.functional.interpolate(m[None],size=sz,mode='bilinear',align_corners=True)[0].cpu().numpy() for m,sz in zip(maps,sizes)]
def create_adapter(device='cuda'):return Adapter(device)
