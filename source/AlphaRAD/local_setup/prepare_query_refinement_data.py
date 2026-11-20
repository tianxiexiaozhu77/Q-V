#!/usr/bin/env python3
"""Freeze label-only support sampling and matched AlphaRAD evaluation manifests.

No model predictions are inspected. Published zero-shot CSVs are never changed.
CXD/NIH/RSNA reuse the exact RadZero support and group holdouts. Other datasets
use seed-42 five-positive/five-negative support, preferentially independent
source images. Known patient/study overlap and exact decoded-pixel duplicates
are excluded; unavailable patient metadata is explicitly reported.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re

from prepare_storage_data import ROOT, ALPHA, MEDICAL, SHARED, atomic_json, read_csv, save_csv, sha, source_file, author_converter
from prepare_storage_additional import load_mha_converter

NATIVE = MEDICAL / 'RadZero_i2cl/local_assets/splits/native_public5_shot5_seed42'
SIIM_META = MEDICAL / 'RadZero_i2cl/local_assets/splits/siim_released_holdout_shot5_seed42/SIIM'
KEYS = ['chestxdet10','chestxdet10_gt','chest14','rsna_pneumonia','siim_pneumothorax','node21','qatacovid_seg','shenzhenxray','montgomery','covidkaggle']


def labels(row):
    val = ast.literal_eval(row['label'])
    return [int(x) for x in val] if isinstance(val, (list, tuple)) else [int(val)]


def support_row(image, values, classes, **metadata):
    return {'image': str(image), 'class': repr([c for c,y in zip(classes,values) if y]),
            'label': repr(values), **metadata}


def patient_qata(image):
    match = re.search(r'(sub-S\d+)_ses-(E\d+)', Path(image).name)
    return (match[1], f'{match[1]}/{match[2]}') if match else ('', '')


def pixel_signature(path):
    import cv2
    pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if pixels is None or pixels.ndim != 2:
        raise ValueError(f'Unreadable grayscale image: {path}')
    return hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()


def signatures(rows, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(pixel_signature, [r['image'] for r in rows]))


def header(path):
    import pydicom
    d = pydicom.dcmread(str(path), stop_before_pixels=True,
            specific_tags=['PatientID','StudyInstanceUID','SOPInstanceUID'])
    result = {k: str(getattr(d, k, '')).strip() for k in ('PatientID','StudyInstanceUID','SOPInstanceUID')}
    if not all(result.values()) or result['SOPInstanceUID'] != Path(path).stem:
        raise ValueError(f'Missing or inconsistent DICOM identity: {path}')
    return result


def select_binary(candidates, seed, eligible=lambda r: True):
    rng = random.Random(seed)
    selected, seen = [], set()
    for target in (0, 1):
        pool = sorted((r for r in candidates if labels(r) == [target]), key=lambda r:r['image'])
        rng.shuffle(pool)
        count = 0
        for row in pool:
            if row['image'] in seen or not eligible(row):
                continue
            selected.append({**row, 'anchor_class': f'positive:{target}'})
            seen.add(row['image']); count += 1
            if count == 5:
                break
        if count != 5:
            raise ValueError(f'Only {count} eligible examples for binary target {target}')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT/'local_setup/pretrained_catalog.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(args.catalog.read_text())['datasets']
    catalog = {'status':'preparing', 'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'datasets':{}, 'source_catalog':source_file(args.catalog), 'source_script':source_file(Path(__file__)),
        'protocol':{'seed':args.seed,'eta':0.3,'eta_selected_on_alpharad_test':False,
                    'support_only_for_query_direction':True,'baseline_and_refined_use_identical_evaluation_images':True,
                    'no_gradient_training':True,'independent_adaptation_per_dataset':True}}
    catalog_path=args.output/'catalog.json';atomic_json(catalog_path,catalog)
    originals={}
    for key in KEYS:
        entry=baseline[key]
        if sha(entry['csv_path']) != entry['csv_sha256']:
            raise ValueError(f'Baseline CSV changed: {key}')
        originals[key]=read_csv(entry['csv_path'])

    def publish(key, support, evaluated, excluded, audit, sources=(), support_files=None):
        entry=baseline[key];classes=entry['class_names']
        if support_files is None:
            support_path=args.output/f'{key}_support.csv'
            fields=list(dict.fromkeys(k for row in support for k in row))
            save_csv(support_path,support,fields)
            support_digest=sha(support_path)
        else:
            support_path,support_digest=support_files
        eval_path=args.output/f'{key}_evaluation.csv'
        save_csv(eval_path,evaluated,list(evaluated[0]))
        support_y=[labels(r) for r in support]
        positive=[sum(row[c] for row in support_y) for c in range(len(classes))]
        negative=[len(support)-v for v in positive]
        if min(positive+negative)<1:
            raise ValueError(f'Query refinement requires both outcomes in support for every class: {key}')
        if {r['image'] for r in support} & {r['image'] for r in evaluated}:
            raise ValueError(f'Support/evaluation image path overlap: {key}')
        for row in support+evaluated:
            if not Path(row['image']).is_file():
                raise FileNotFoundError(row['image'])
        audit.update({'support_count':len(support),'support_positive_count':positive,'support_negative_count':negative,
            'support_anchor_counts':dict(Counter(r.get('anchor_class','') for r in support)),
            'original_evaluation_count':len(originals[key]),'evaluation_count':len(evaluated),
            'removed_from_original_evaluation':len(excluded),'excluded_rows':excluded,
            'support_csv_path':str(support_path),'support_csv_sha256':support_digest,
            'evaluation_csv_path':str(eval_path),'evaluation_csv_sha256':sha(eval_path),
            'source_files':[source_file(p) for p in sources]})
        audit_path=args.output/f'{key}_split_audit.json';atomic_json(audit_path,audit)
        new={**entry, 'csv_path':str(eval_path),'csv_sha256':sha(eval_path),
            'expected_count':len(evaluated),'actual_count':len(evaluated),'unique_images':len({r['image'] for r in evaluated}),
            'support_csv_path':str(support_path),'support_csv_sha256':support_digest,'support_count':len(support),
            'support_positive_count':positive,'support_negative_count':negative,
            'original_csv_path':entry['csv_path'],'original_csv_sha256':entry['csv_sha256'],
            'original_evaluation_count':len(originals[key]),'removed_from_original_evaluation':len(excluded),
            'split_audit_path':str(audit_path),'split_audit_sha256':sha(audit_path),
            'scope':'matched baseline/refined evaluation after support-group and duplicate exclusion',
            'notes':list(entry.get('notes',[]))+audit.get('limitations',[])+[
                'Query eta=0.3 fixed before AlphaRAD evaluation; no test-driven eta selection.',
                'This uses labeled few-shot support; it is not zero-shot evaluation.'],
            'sources':entry.get('sources',[])+[source_file(p) for p in sources]+[source_file(audit_path)]}
        if key not in ('node21','qatacovid_seg'):
            new['positive_count']=[sum(labels(r)[c] for r in evaluated) for c in range(len(classes))]
        catalog['datasets'][key]=new;atomic_json(catalog_path,catalog)
        print(f'READY {key}: support={len(support)} evaluation={len(evaluated)} excluded={len(excluded)}',flush=True)
        return (support_path,support_digest)

    # Reuse all original selected examples; label vectors retain AlphaRAD's order.
    for key,family in [('chestxdet10','ChestXDet10'),('chest14','ChestXray14'),('rsna_pneumonia','RSNA')]:
        directory=NATIVE/family;manifest_path=directory/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        for filename in ['support.csv','inventory.csv','test.csv','excluded.csv']:
            if sha(directory/filename)!=manifest['csv_sha256'][filename]:
                raise ValueError(f'Native immutable split changed: {directory/filename}')
        old_support=read_csv(directory/'support.csv');inventory=read_csv(directory/'inventory.csv')
        map_key='patient_id' if family=='RSNA' else 'image_id'
        metadata={r[map_key]:r for r in inventory};by_id={Path(r['image']).stem:r for r in originals[key]}
        if set(metadata)!=set(by_id):
            raise ValueError(f'Native and AlphaRAD image IDs differ: {key}')
        blocked={r['group'] for r in old_support};support=[]
        for row in old_support:
            current=by_id[row[map_key]]
            if labels(current)!=json.loads(row['labels_json']):
                raise ValueError(f'Label order mismatch: {key} {row[map_key]}')
            original_path=SHARED/row['image_path']
            if sha(original_path)!=row['byte_sha256']:
                raise ValueError(f'Selected source image changed: {original_path}')
            support.append(support_row(current['image'],labels(current),baseline[key]['class_names'],
                patient_id=row['patient_id'],study_id=row['study_id'],group=row['group'],
                source_image=str(original_path),source_byte_sha256=row['byte_sha256'],
                source_split='reused_RadZero_support',anchor_class=row['anchor_class']))
        evaluated=[];excluded=[]
        for row in originals[key]:
            meta=metadata[Path(row['image']).stem]
            if labels(row)!=json.loads(meta['labels_json']):
                raise ValueError(f'Inventory labels changed: {key}')
            (excluded if meta['group'] in blocked else evaluated).append(row)
        limitations=[x for x in manifest['limitations'] if 'No separate local' not in x]
        audit={'support_source':'exact native_public5_shot5_seed42 support',
               'grouping':manifest['grouping'],'patient_identity_available':family!='ChestXDet10',
               'group_overlap_count':0,'limitations':limitations,
               'native_support_groups':sorted(blocked),'native_manifest':source_file(manifest_path),
               'support_input_pixel_sha256':signatures(support,args.workers),
               'duplicate_check':'source-byte groups from verified native inventory; no near-duplicate search'}
        files=publish(key,support,evaluated,excluded,audit,[manifest_path,directory/'support.csv',directory/'inventory.csv'])
        if key=='chestxdet10':
            removed={Path(r['image']).stem for r in excluded}
            rows=originals['chestxdet10_gt'];held=[r for r in rows if Path(r['image']).stem in removed]
            kept=[r for r in rows if Path(r['image']).stem not in removed]
            publish('chestxdet10_gt',support,kept,held,{**audit,'classification_and_grounding_share_support':True},
                    [manifest_path,directory/'support.csv',directory/'inventory.csv'],files)

    import cv2
    cv2.setNumThreads(1)
    # SIIM: newly available source training DICOMs; labels are union of all RLE rows.
    key='siim_pneumothorax';classes=baseline[key]['class_names']
    test_manifest=SIIM_META/'manifest.json';manifest=json.loads(test_manifest.read_text())
    if sha(SIIM_META/'inventory.csv')!=manifest['csv_sha256']['inventory.csv']:
        raise ValueError('SIIM native header inventory changed')
    old_test=read_csv(SIIM_META/'inventory.csv')
    if {r['image_id'] for r in old_test}!={Path(r['image']).stem for r in originals[key]}:
        raise ValueError('SIIM baseline no longer equals original image inventory')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        headers=list(pool.map(header,[SHARED/r['image_path'] for r in old_test]))
    for row,h in zip(old_test,headers):
        if (row['patient_id'],row['study_id'])!=(h['PatientID'],h['StudyInstanceUID']):
            raise ValueError('Current SIIM header differs from audited metadata')
    blocked_patients={h['PatientID'] for h in headers};blocked_studies={h['StudyInstanceUID'] for h in headers}
    test_ids={r['image_id'] for r in old_test};source=ALPHA/'SIIM/train-rle.csv'
    all_labels={}
    for row in read_csv(source):
        uid=row['ImageId'].strip();y=int(row[' EncodedPixels'].strip()!='-1')
        all_labels[uid]=max(all_labels.get(uid,0),y)
    paths={p.stem:p for p in (ALPHA/'SIIM/dicom-images-train').rglob('*.dcm')}
    candidates=[support_row(path,[all_labels[uid]],classes,source_split='independent_source_training',source_image=str(path))
                for uid,path in paths.items() if uid in all_labels and uid not in test_ids]
    test_sigs=set(signatures(originals[key],args.workers));convert=author_converter();dest=args.output/'siim_support_png';dest.mkdir()
    receipts=[]
    def siim_eligible(row):
        h=header(row['image'])
        if h['PatientID'] in blocked_patients or h['StudyInstanceUID'] in blocked_studies:
            return False
        target=dest/(h['SOPInstanceUID']+'.png')
        convert(row['image'],str(target),do_resize=True)
        signature=pixel_signature(target)
        if signature in test_sigs:
            target.unlink();return False
        row.update(image=str(target),patient_id=h['PatientID'],study_id=h['StudyInstanceUID'],pixel_sha256=signature)
        receipts.append({'source':row['source_image'],'source_sha256':sha(row['source_image']),
                         'png':str(target),'png_sha256':sha(target),'pixel_sha256':signature})
        return True
    support=select_binary(candidates,args.seed,siim_eligible)
    receipt_path=args.output/'siim_support_conversion.json';atomic_json(receipt_path,receipts)
    publish(key,support,originals[key],[],{'support_source':'SIIM original training DICOMs excluding all baseline test UIDs/patients/studies',
        'candidate_count':len(candidates),'patient_identity_available':True,'patient_overlap_count':0,'study_overlap_count':0,
        'exact_decoded_pixel_overlap_count':0,'grouping':'PatientID, StudyInstanceUID, SOPInstanceUID and decoded author PNG pixels',
        'test_dicom_headers_checked':len(headers),'limitations':['Near-duplicate visual similarity is not searched.']},
        [source,test_manifest,SIIM_META/'inventory.csv',receipt_path,ROOT/'cxrclip/data/dataformat_converter.py'])

    # The remaining tasks have either filename identities or no complete patient mapping.
    for key in ['node21','qatacovid_seg','shenzhenxray','montgomery','covidkaggle']:
        classes=baseline[key]['class_names'];rows=originals[key];sources=[];limitations=['Near-duplicate visual similarity is not searched.']
        candidate_count=0;candidate_excluded_groups=0;patient_available=False;identity=lambda r: ('','')
        eval_sigs=signatures(rows,args.workers);sig_set=set(eval_sigs)
        if key in ['shenzhenxray','montgomery']:
            def identity(r):
                return ('_'.join(Path(r['image']).stem.split('_')[:2]),'')
            candidates=[support_row(r['image'],labels(r),classes,patient_id=identity(r)[0],source_split='source_collection_holdout') for r in rows]
            support=select_binary(candidates,args.seed);source_description='five-positive/five-negative holdout from complete NLM collection'
            patient_available=True
            limitations.append('NLM public case ID is used as patient/case grouping; no additional linkage beyond released IDs is available.')
        elif key=='node21':
            source=ALPHA/'Node21/cxr_images/proccessed_data/metadata.csv';sources=[source]
            labelmap={}
            for r in read_csv(source):
                if r['img_name'] in labelmap and labelmap[r['img_name']]!=int(r['label']):
                    raise ValueError('Conflicting Node21 image labels')
                labelmap[r['img_name']]=int(r['label'])
            image_dir=source.parent/'images';candidates=[support_row(image_dir/name,[y],classes,
                source_image=str(image_dir/name),source_split='public_Node21_collection') for name,y in labelmap.items()]
            support=select_binary(candidates,args.seed);converter=load_mha_converter();dest=args.output/'node21_support_png';dest.mkdir()
            for r in support:
                target=dest/(Path(r['image']).stem+'.png');converter(r['image'],str(target),do_resize=True)
                r.update(image=str(target),source_byte_sha256=sha(r['source_image']))
            source_description='five positive and five negative real Node21 images; held-out positives removed from grounding evaluation'
            limitations.append('Node21 metadata has image IDs but no patient/study linkage; patient-level separation cannot be verified.')
        elif key=='qatacovid_seg':
            source_dir=ALPHA/'QaTaCOV19/QaTa-COV19';train=source_dir/'QaTa-COV19-v2/Train Set/Images'
            blocked={patient_qata(r['image'])[0] for r in rows};assert '' not in blocked
            # Only the sub-S images with explicit public subject IDs enter positive support.
            positive=[]
            for path in sorted(train.glob('sub-S*.png')):
                patient,study=patient_qata(path)
                if not patient: raise ValueError(f'Unknown QaTa subject format: {path}')
                if patient in blocked:
                    candidate_excluded_groups+=1;continue
                positive.append(support_row(path,[1],classes,patient_id=patient,study_id=study,source_split='QaTa_v2_train_known_subject'))
            negative=[support_row(path,[0],classes,source_split='QaTa_Control_Group_I_documented_normal')
                      for path in sorted((source_dir/'Control_Group/Control_Group_I').glob('*.png'))]
            candidates=positive+negative
            def eligible(row): return pixel_signature(row['image']) not in sig_set
            support=select_binary(candidates,args.seed,eligible);identity=lambda r:patient_qata(r['image'])
            source_description='QaTa v2 Train Set positives with known non-test subjects; documented normal Control_Group_I negatives'
            sources=[ALPHA/'QaTaCOV19/readme.txt'];patient_available='positive_subjects_only'
            limitations.append('QaTa selected positive support and every evaluation image have disjoint explicit sub-S subject IDs. Control_Group_I normal files have no patient mapping; their patient-level separation cannot be independently verified.')
        else:
            directory=ALPHA/'CovidKaggle/COVID-19_Radiography_Dataset';test_ids={Path(r['image']).name for r in rows}
            candidates=[support_row(path,[y],classes,source_split='outside_author_test_csv')
                        for family,y in [('Normal',0),('COVID',1)] for path in sorted((directory/family/'images').glob('*.png'))
                        if path.name not in test_ids]
            def eligible(row): return pixel_signature(row['image']) not in sig_set
            support=select_binary(candidates,args.seed,eligible);source_description='full COVID Radiography Normal/COVID collection outside all author test filenames'
            sources=[directory/'README.md.txt',directory/'COVID.metadata.xlsx',directory/'Normal.metadata.xlsx']
            limitations.append('COVID Radiography metadata provides generic source URLs rather than patient IDs; patient-level separation cannot be verified.')
        candidate_count=len(candidates)
        support_sigs=signatures(support,args.workers)
        if len(set(support_sigs))!=len(support_sigs):
            raise ValueError(f'Exact duplicate pixels within selected support: {key}')
        blocked_ids={Path(r['image']).stem for r in support}
        blocked_patients={r.get('patient_id','') for r in support}-{''}
        blocked_studies={r.get('study_id','') for r in support}-{''}
        kept=[];excluded=[]
        for r,sig in zip(rows,eval_sigs):
            patient,study=identity(r)
            if (Path(r['image']).stem in blocked_ids or sig in support_sigs or
                (patient and patient in blocked_patients) or (study and study in blocked_studies)):
                excluded.append(r)
            else:kept.append(r)
        if key in ['qatacovid_seg','covidkaggle'] and excluded:
            raise ValueError(f'Independent support unexpectedly overlaps evaluation: {key}')
        for r,sig in zip(support,support_sigs):
            r['pixel_sha256']=sig
        publish(key,support,kept,excluded,{'support_source':source_description,'candidate_count':candidate_count,
            'candidate_rows_excluded_by_test_subject':candidate_excluded_groups,
            'patient_identity_available':patient_available,'exact_decoded_pixel_overlap_count':0,
            'support_groups':[{'image':r['image'],'patient_id':r.get('patient_id',''),'study_id':r.get('study_id',''),
                               'source_image':r.get('source_image',r['image'])} for r in support],
            'limitations':limitations},sources)
    catalog['status']='complete';catalog['completed_at_utc']=datetime.now(timezone.utc).isoformat()
    atomic_json(catalog_path,catalog);print(f'COMPLETE {catalog_path}',flush=True)

if __name__=='__main__': main()
