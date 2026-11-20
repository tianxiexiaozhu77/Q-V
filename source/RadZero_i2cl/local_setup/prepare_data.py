"""Link downloaded image folders into the upstream layout; report coverage."""
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / 'radzero-assets'
DATA = ASSETS / 'data'
IMAGES = ASSETS / 'data_images'
FOLDERS = {
    'ChestXDet10': ['test_data'],
    'OpenI': ['iu_xray_kaggle'],
    'SIIM': ['dicom-images-train', 'dicom-images-test'],
    'RSNA': ['mdai_rsna_project_x9N20BZa_images_2018-07-20-153330'],
    'ChestXray14': [f'images_{index:03}' for index in range(1, 13)],
}
TABLES = {
    'ChestXDet10': [('chestXDet10_test_image.csv', 'Path')],
    'ChestXray14': [('chestxray14_test_image.csv', 'Path')],
    'Chexpert': [('chexpert5_test_image.csv', 'Path')],
    'OpenI': [('openi_multi_label_image.csv', 'Path')],
    'PadChest': [('padchest_multi_label_image.csv', 'Path')],
    'RSNA': [('test.csv', 'img_path')],
    'SIIM': [('train.csv', 'dicom_path'), ('valid.csv', 'dicom_path'), ('test.csv', 'dicom_path')],
}


def coverage(paths):
    names = sorted(set(paths))
    missing = [name for name in names if not (DATA / name).is_file()]
    return {
        'rows': len(paths), 'unique_images': len(names),
        'present_images': len(names) - len(missing), 'missing_images': len(missing),
        'complete': not missing, 'missing_examples': missing[:3],
    }


def main():
    for dataset, folders in FOLDERS.items():
        for folder in folders:
            source, target = IMAGES / dataset / folder, DATA / dataset / folder
            if not source.is_dir():
                continue
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    raise RuntimeError(f'Existing symlink points elsewhere: {target}')
            elif not target.exists():
                target.symlink_to(source, target_is_directory=True)

    report = {
        'checked_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'data_root': str(DATA), 'datasets': {},
    }
    for dataset, tables in TABLES.items():
        report['datasets'][dataset] = {}
        for filename, column in tables:
            with (DATA / dataset / filename).open(newline='') as handle:
                paths = [row[column] for row in csv.DictReader(handle)]
            report['datasets'][dataset][filename] = coverage(paths)
    for filename in ('train.json', 'validate.json', 'test.json'):
        entries = json.loads((DATA / 'MIMIC-CXR' / filename).read_text())
        paths = ['MIMIC-CXR/images/' + entry['dicom_id'] for entry in entries]
        report['datasets'].setdefault('MIMIC-CXR', {})[filename] = coverage(paths)
    entries = json.loads((DATA / 'MS-CXR/test.json').read_text())
    report['datasets']['MS-CXR'] = {'test.json': coverage([entry['image'] for entry in entries])}
    target = ROOT / 'local_setup/data_status.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for dataset, tables in report['datasets'].items():
        for name, result in tables.items():
            print(f'{dataset}/{name}: {result["present_images"]}/{result["unique_images"]} unique images')
    print(f'Wrote {target}')


if __name__ == '__main__':
    main()
