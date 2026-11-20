#!/usr/bin/env python3
"""Attach existing phase6 support sets to frozen baseline holdouts. No resampling."""
import argparse
import ast
import csv
import json
from pathlib import Path
from evaluate_baseline_buffered import sha, write

ROOT=Path(__file__).resolve().parents[1]
MEDICAL=ROOT.parent

def rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    oldpath=ROOT/'evaluation/matched_phase6_20261022/catalog.json'
    old=json.loads(oldpath.read_text())
    alpha=MEDICAL/'AlphaRAD/runs/query_refinement_20261021_1441/catalog.json'
    chex=MEDICAL/'AlphaRAD/runs/chexpert_phase6_20261022_170123_2309385/data/catalog.json'
    source=json.loads(alpha.read_text())['datasets']
    source.update(json.loads(chex.read_text())['datasets'])
    rz=MEDICAL/'RadZero_i2cl/local_assets/splits/native_public5_local_seed42'
    baselines=json.loads((ROOT/'RESULTS.json').read_text())['records']
    catalog={'status':'complete','version':'query-transfer-fixed-v1','eta':.3,
      'views':['identity','center80'],'negative_query_policy':'unchanged',
      'query_norm_policy':'preserve original norm after direction update',
      'scope':old['scope'],'test_catalog':str(oldpath),'test_catalog_sha256':sha(oldpath),
      'source_catalogs':{str(x):sha(x) for x in (alpha,chex)},'datasets':{}}
    for name,entry in old['datasets'].items():
        testpath=Path(entry['manifest'])
        if sha(testpath)!=entry['sha256']:raise ValueError('Changed frozen test')
        test=json.loads(testpath.read_text())
        classes=entry['class_names']
        sources={str(testpath):sha(testpath)}
        if name in source:
            src=source[name];sp=Path(src['support_csv_path'])
            if sha(sp)!=src['support_csv_sha256']:raise ValueError('Changed support CSV')
            assert src['class_names']==classes
            support=[]
            for r in rows(sp):
                y=ast.literal_eval(r['label']);y=y if isinstance(y,list) else [y]
                support.append({'image':r['image'],'labels':y,
                  **{k:r[k] for k in ('patient_id','study_id','group') if r.get(k)}})
            audit=Path(src['split_audit_path'])
            if sha(audit)!=src['split_audit_sha256']:raise ValueError('Changed split audit')
            sources[str(audit)]=sha(audit)
        else:
            split=rz/('OpenI' if name=='openi' else 'SIIM')
            sp=split/'support.csv';meta=json.loads((split/'manifest.json').read_text())
            assert meta['class_names']==classes
            sources[str(split/'manifest.json')]=sha(split/'manifest.json')
            full_siim=json.loads(Path(old['datasets']['siim_pneumothorax']['manifest']).read_text())
            by_uid={Path(r['image']).stem:r for r in full_siim['rows']}
            support=[]
            for r in rows(sp):
                y=json.loads(r['labels_json'])
                if name=='openi':image=MEDICAL/'data/radzero/prepared'/r['image_path']
                else:
                    png=by_uid[r['image_id']];assert png['labels']==y
                    image=Path(png['image'])
                support.append({'image':str(image),'labels':y,
                  **{k:r[k] for k in ('patient_id','study_id','group') if r.get(k)}})
        sources[str(sp)]=sha(sp)
        if {r['image'] for r in support}&{r['image'] for r in test['rows']}:
            raise ValueError('Support/test file overlap')
        # CheXpert filenames are view1_frontal.jpg for many unrelated patients;
        # its identity is the full patient/study/path, audited separately below.
        if name!='chexpert' and {Path(r['image']).stem for r in support}&{Path(r['image']).stem for r in test['rows']}:
            raise ValueError('Support/test image identity overlap')
        overlap={}
        for field in ('patient_id','study_id','group'):
            s={r[field] for r in support if r.get(field)}
            t={r[field] for r in test['rows'] if r.get(field)}
            overlap[field]={'available_on_both':bool(s and t),'overlap_count':len(s&t)}
            if s&t:raise ValueError(f'{name} overlap in {field}')
        for r in support:
            if len(r['labels'])!=len(classes) or any(y not in (0,1) for y in r['labels']):raise ValueError('Invalid support labels')
            image=Path(r['image'])
            if not image.is_file():raise FileNotFoundError(image)
            r['byte_sha256']=sha(image)
        counts=[sum(r['labels'][c] for r in support) for c in range(len(classes))]
        if min(counts)<1 or max(counts)>=len(support):raise ValueError('Support class missing outcome')
        record={**{k:v for k,v in test.items() if k!='rows'},'support_rows':support,
                'support_count':len(support),'support_positive_counts':counts,
                'test_manifest':str(testpath),'test_manifest_sha256':sha(testpath),
                'support_sources':sources,'identity_audit':overlap,
                'support_policy':'Exact previously fixed support images and labels; original view only; no resampling.',
                'baseline_references':{r['model']:{'status':r['status'],'result':r['result'],
                  'result_sha256':sha(r['result'])} for r in baselines
                  if r['dataset']==name and r['model'] in ['BiomedCLIP','BioViL-T','CheXzero','MedKLIP','MAVL','KAD','MaCo','CARZero']}}
        dest=args.output/(name+'.json');write(dest,record)
        catalog['datasets'][name]={**entry,'support_manifest':str(dest),
                                  'support_manifest_sha256':sha(dest),'support_count':len(support)}
        print(name,'support',len(support),'test',entry['count'],flush=True)
    write(args.output/'catalog.json',catalog)
    print('Catalog:',args.output/'catalog.json')

if __name__=='__main__':main()
