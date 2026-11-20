"""Check query information flow and prediction provenance without a GPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from infer_maps import fit_queries


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.original = np.asarray([[0., 2., 0.]], dtype=np.float32)
        self.labels = np.asarray([[1]] * 5 + [[0]] * 5)
        self.features = np.asarray([[[1., 0., 0.]]] * 5 + [[[0., 0., 1.]]] * 5, dtype=np.float32)

    def test_positive_query_does_not_consume_negative_features(self):
        first = fit_queries(self.features, self.labels, self.original)
        changed = self.features.copy()
        changed[5:] = [[[-1., 0., 0.]]]
        second = fit_queries(changed, self.labels, self.original)
        np.testing.assert_array_equal(first['positive_query'], second['positive_query'])
        self.assertFalse(np.allclose(first['query'], second['query']))
        direction = np.asarray([[1., 0., -1.]], dtype=np.float32) / np.sqrt(2)
        expected = np.asarray([[0., 1., 0.]]) + .3 * direction
        expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
        np.testing.assert_allclose(first['query'], expected, atol=1e-7)
        np.testing.assert_array_equal(first['queries'][0], self.original)

    def test_zero_difference_preserves_raw_original_query(self):
        self.features[5:] = self.features[:5]
        np.testing.assert_array_equal(fit_queries(self.features, self.labels, self.original)['query'], self.original)

    def test_nonbinary_or_missing_class_support_rejected(self):
        for labels in (np.ones_like(self.labels), np.full_like(self.labels, 2)):
            with self.assertRaises(ValueError):
                fit_queries(self.features, labels, self.original)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = self.root / 'protocol.json'
        self.protocol.write_text('{}')
        self.rows = [dict(image=str(self.root / 'image.png'), patient_id='sub-S1')]
        self.query = self.root / 'query.npz'
        self.support_rows = [dict(image=str(self.root / f'support_{i}.png'), label_vector=[int(i < 5)])
                             for i in range(10)]
        self.support_path = self.root / 'supports.json'
        self.support_path.write_text(json.dumps(self.support_rows))
        self.config = dict(support_sets={'42': dict(rows_path=str(self.support_path),
                                                   rows_sha256=common.sha256(self.support_path))})
        np.savez(self.query, query=np.ones((1, 3)), support_seed=42,
                 image_paths=np.asarray([r['image'] for r in self.support_rows]),
                 labels=np.asarray([r['label_vector'] for r in self.support_rows]),
                 protocol_sha256=common.sha256(self.protocol))
        self.maps = self.root / 'maps.npz'
        self.write_maps()
        self.manifest = dict(status='complete', smoke=0, seeds=list(common.SEEDS),
                             branches=list(common.BRANCHES), checkpoint_sha256=common.CHECKPOINT_SHA,
                             source_sha256=common.source_snapshot(),
                             protocol=dict(sha256=common.sha256(self.protocol)),
                             maps={'42': {'validation': dict(path=str(self.maps), sha256=common.sha256(self.maps))}},
                             query_fits={'42': dict(path=str(self.query), sha256=common.sha256(self.query))})
        self.write_manifest()

    def write_maps(self, seed=42, dtype=np.float32):
        np.savez(self.maps, **{b: np.zeros((1, 1, 1370), dtype=dtype) for b in common.BRANCHES},
                 image_paths=np.asarray([r['image'] for r in self.rows]), patient_ids=np.asarray(['sub-S1']),
                 class_names=np.asarray(['covid']), support_seed=seed,
                 protocol_sha256=common.sha256(self.protocol))

    def write_manifest(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))

    def load(self):
        with patch.object(common, 'load_rows', return_value=self.rows):
            return common.load_maps(self.root, self.protocol, self.config, 'validation', 42)

    def test_complete_identity_match_loads(self):
        maps, proof = self.load()
        self.assertEqual(tuple(maps), common.BRANCHES)
        self.assertEqual(proof['query_fit']['sha256'], common.sha256(self.query))

    def test_smoke_predictions_cannot_be_scored(self):
        self.manifest.update(status='smoke_complete', smoke=16)
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.load()

    def test_wrong_seed_and_dtype_rejected_even_with_matching_file_hash(self):
        for seed, dtype in ((43, np.float32), (42, np.float64)):
            self.write_maps(seed=seed, dtype=dtype)
            self.manifest['maps']['42']['validation']['sha256'] = common.sha256(self.maps)
            self.write_manifest()
            with self.assertRaises(ValueError):
                self.load()

    def test_changed_query_is_rejected(self):
        self.query.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.load()


if __name__ == '__main__':
    unittest.main()
