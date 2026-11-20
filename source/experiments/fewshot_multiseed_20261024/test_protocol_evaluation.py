"""CPU integration checks for inherited splits and the test-label unlock boundary."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('multiseed_evaluation_test_module', HERE/'evaluate_multiseed.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path':str(path), 'sha256':evaluation.sha(path)}


def unit(values):
    values = values.astype(np.float32)
    return values/np.linalg.norm(values, axis=-1, keepdims=True)


def test_frozen_supports_groups_and_folds_are_inherited_without_resampling():
    catalog = json.loads((HERE/'protocol_v1/catalog.json').read_text())
    original = json.loads(Path(catalog['source_catalog']['path']).read_text())
    for dataset, item in catalog['datasets'].items():
        old = json.loads(Path(original['datasets'][dataset]['path']).read_text())
        new = json.loads(Path(item['path']).read_text())
        assert old['test_rows'] == new['test_rows']
        assert old['test_groups'] == new['test_groups']
        assert old['candidate_pool'] == new['candidate_pool']
        for seed, ref in item['seeds'].items():
            p = json.loads(evaluation.checked(ref).read_text())
            assert p['support_rows'] == old['seeds'][seed]['support_rows']
            assert p['support_groups'] == old['seeds'][seed]['support_groups']
            assert p['support_cv_audit']['all_training_folds_have_both_outcomes']
            assert not p['support_cv_audit']['fallback_triggered']
            assert 'test_rows' not in p


def fixture(tmp_path):
    catalog = json.loads((HERE/'protocol_v1/catalog.json').read_text())
    source = catalog['datasets']['rsna_pneumonia']['seeds']
    paths = [f'/synthetic_test/{i}' for i in range(8)]
    test_ref = write(tmp_path/'test_manifest.json', {'rows':[{'image':path, 'labels':[i%2]} for i,path in enumerate(paths)]})
    entries = {}
    for seed in catalog['support_seeds']:
        rng = np.random.default_rng(seed)
        p = copy.deepcopy(json.loads(Path(source[str(seed)]['path']).read_text()))
        p.update(test_images=paths, test_count=8, test_groups=[f'test_{i}' for i in range(8)], test_manifest=test_ref)
        cache = tmp_path/'cache'/f'seed_{seed}'
        cache.mkdir(parents=True)
        common = {'class_names':np.asarray(p['class_names'])}
        support_features, test_features = unit(rng.normal(size=(10,7))), unit(rng.normal(size=(8,7)))
        positive, negative = unit(rng.normal(size=(1,7))), unit(rng.normal(size=(1,7)))
        support_logits = support_features @ (positive-negative).T
        test_logits = test_features @ (positive-negative).T
        np.savez(cache/'support.npz', **common, image_paths=np.asarray([r['image'] for r in p['support_rows']]),
                 labels=np.asarray([r['labels'] for r in p['support_rows']]), features=support_features,
                 local=support_features[:,None,:], logits=support_logits)
        np.savez(cache/'test.npz', **common, image_paths=np.asarray(paths), features=test_features,
                 local=test_features[:,None,:], logits=test_logits)
        np.savez(cache/'queries.npz', **common, positive=positive, negative=negative)
        scores = {key:test_logits+(i*.01) for i,key in enumerate(evaluation.NATIVE)}
        np.savez(cache/'native_predictions.npz', **common, image_paths=np.asarray(paths), **scores)
        reference = tmp_path/f'reference_{seed}.npz'
        np.savez(reference, **common, image_paths=np.asarray(paths), seed=seed,
                 **{old:scores[new] for new,old in evaluation.ALIASES.items()})
        p['stability_references']['BiomedCLIP']['predictions.npz'] = {'path':str(reference),'sha256':evaluation.sha(reference)}
        entries[str(seed)] = write(tmp_path/f'protocol_{seed}.json', p)
        write(cache/'result.json', {'status':'complete','model':'BiomedCLIP','dataset':'rsna_pneumonia',
                                   'protocol_sha256':entries[str(seed)]['sha256'],'test_count':8,
                                   'score_domain':'native binary margin',
                                   'artifact_sha256':{name:evaluation.sha(cache/name) for name in
                                       ('support.npz','test.npz','queries.npz','native_predictions.npz')}})
    catalog['datasets']['rsna_pneumonia']['seeds'] = entries
    catalog_path = tmp_path/'catalog.json'
    write(catalog_path, catalog)
    return SimpleNamespace(protocol=catalog_path, model='BiomedCLIP', dataset='rsna_pneumonia',
                           cache=tmp_path/'cache',output=tmp_path/'evaluation',threads=1), Path(test_ref['path'])


def test_all_five_real_fits_commit_before_labels_open_and_reference_is_same_seed(tmp_path, monkeypatch):
    args, test_manifest = fixture(tmp_path)
    checked = evaluation.checked
    label_opens = []
    def guarded(ref):
        if Path(ref['path']) == test_manifest:
            assert len(list(args.output.glob('seed_*/fit_commit.json'))) == 5
            label_opens.append(ref['path'])
        return checked(ref)
    monkeypatch.setattr(evaluation, 'checked', guarded)
    evaluation.run(args)
    assert len(label_opens) == 5
    assert json.loads((args.output/'status.json').read_text())['status'] == 'complete'
    for seed in range(42,47):
        output = args.output/f'seed_{seed}'
        result = json.loads((output/'result.json').read_text())
        assert len(result['methods']) == 11
        assert result['same_seed_stability_reference_audit']['passed']
        fit = json.loads((output/'fit_info.json').read_text())
        assert fit['simnl_feature_ovr']['epoch_candidates'] == [0,10,20,40]
        assert fit['simnl_feature_ovr']['config']['learning_rate'] == .0001
    # Resume rechecks cache bytes even when an unchanged result.json and fit
    # commit already exist; a valid old commit must not bypass authentication.
    cached_support = args.cache/'seed_42/support.npz'
    support_bytes = cached_support.read_bytes()
    cached_support.write_bytes(support_bytes+b'corruption')
    with pytest.raises(ValueError, match='Cache artifact SHA256 mismatch: support.npz'):
        evaluation.run(args)
    cached_support.write_bytes(support_bytes)
    # The fit commit also catches corruption before a resumed scoring pass.
    (args.output/'seed_42/fit_info.json').write_text('{}')
    with pytest.raises(ValueError, match='Committed fit/provenance changed'):
        evaluation.run(args)


def test_test_labels_in_fit_cache_are_rejected(tmp_path):
    args, _ = fixture(tmp_path)
    catalog = json.loads(args.protocol.read_text())
    ref = catalog['datasets']['rsna_pneumonia']['seeds']['42']
    ppath = Path(ref['path'])
    p = json.loads(ppath.read_text())
    cache = args.cache/'seed_42'
    arrays = evaluation.load_npz(cache/'test.npz')
    np.savez(cache/'test.npz', **arrays, labels=np.zeros((8,1)))
    # Authenticate this deliberately invalid export to exercise the independent
    # label-exclusion guard rather than stopping first at the byte-hash guard.
    record = json.loads((cache/'result.json').read_text())
    record['artifact_sha256']['test.npz'] = evaluation.sha(cache/'test.npz')
    write(cache/'result.json', record)
    with pytest.raises(ValueError, match='contains test labels'):
        evaluation.fit_seed(args, ppath, p, cache, args.output/'seed_42')


@pytest.mark.parametrize('artifact', ['support.npz','test.npz','queries.npz','native_predictions.npz'])
def test_mutated_cache_artifact_rejected_before_fit(tmp_path, artifact):
    args, _ = fixture(tmp_path)
    catalog = json.loads(args.protocol.read_text())
    ref = catalog['datasets']['rsna_pneumonia']['seeds']['42']
    ppath = Path(ref['path'])
    p = json.loads(ppath.read_text())
    cache = args.cache/'seed_42'
    path = cache/artifact
    path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(ValueError, match='Cache artifact SHA256 mismatch'):
        evaluation.fit_seed(args, ppath, p, cache, args.output/'seed_42')
    assert not (args.output/'seed_42/fit_info.json').exists()
