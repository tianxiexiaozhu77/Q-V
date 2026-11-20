"""Protocol invariants: patient/pixel isolation and reproducible support draws."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import prepare_protocol as p


def row(name, label, patient=None, pixel=None):
    return {'image': name, 'image_id': name, 'patient_id': patient,
            'label_vector': [label], 'pixel_sha256': pixel or name}


def pools():
    positive = [row(f'patient-{i}/image-{j}', 1, f'sub-S{i}')
                for i in range(20) for j in range(3)]
    negative = [row(f'normal-{i}', 0) for i in range(30)]
    return positive, negative


class SupportProtocolTests(unittest.TestCase):
    def test_independent_seed_reproducibility_and_input_order(self):
        positive, negative = pools()
        baseline = p.sample_support(positive, negative, 42)
        p.sample_support(positive, negative, 46)
        self.assertEqual(baseline, p.sample_support(positive[::-1], negative[::-1], 42))
        self.assertNotEqual(baseline, p.sample_support(positive, negative, 43))
        self.assertEqual(positive[0]['image'], 'patient-0/image-0')
        self.assertNotIn('support_seed', positive[0])

    def test_seed_sets_have_five_distinct_positive_patients_and_unknown_normals(self):
        positive, negative = pools()
        for seed in p.SEEDS:
            support = p.sample_support(positive, negative, seed)
            p.validate_support(support, [])
            self.assertEqual(len({r['patient_id'] for r in support if r['label_vector'] == [1]}), 5)
            self.assertTrue(all(r['patient_id'] is None for r in support if r['label_vector'] == [0]))

    def test_exclude_entire_known_patient_and_pixels_even_for_unknown_normals(self):
        positive, negative = pools()
        held = [row('held-known', 1, 'sub-S0'), row('held-pixel', 1, 'sub-S90', 'patient-1/image-0'),
                row('held-normal-duplicate', 1, 'sub-S91', 'normal-0')]
        pos, neg, audit = p.eligible_candidates(positive, negative, held)
        self.assertNotIn('sub-S0', {r['patient_id'] for r in pos})
        self.assertNotIn('sub-S1', {r['patient_id'] for r in pos})
        self.assertNotIn('normal-0', {r['pixel_sha256'] for r in neg})
        self.assertEqual(audit['positive_rows_removed_by_patient_exclusions'], 6)
        for seed in p.SEEDS:
            p.validate_support(p.sample_support(pos, neg, seed), held)

    def test_duplicate_pixels_and_conflicting_labels_are_removed_before_sampling(self):
        positive, negative = pools()
        positive.append(row('copy-positive', 1, 'sub-S19', 'patient-2/image-1'))
        negative.extend([row('zz-normal-copy', 0, pixel='normal-1'),
                         row('cross-label', 0, pixel='patient-3/image-1')])
        pos, neg, audit = p.eligible_candidates(positive, negative, [])
        self.assertEqual(audit['positive_exact_duplicate_rows_removed'], 1)
        self.assertEqual(audit['negative_exact_duplicate_rows_removed'], 1)
        self.assertNotIn('sub-S3', {r['patient_id'] for r in pos})
        self.assertNotIn('patient-3/image-1', {r['pixel_sha256'] for r in neg})
        self.assertEqual(audit['cross_label_pixel_conflicts_excluded'], ['patient-3/image-1'])
        self.assertEqual(len(pos), len({r['pixel_sha256'] for r in pos}))

    def test_never_invent_patient_ids_for_normals_or_unknown_positive_images(self):
        positive, negative = pools()
        bad = copy.deepcopy(positive)
        bad[0]['patient_id'] = None
        with self.assertRaises(ValueError):
            p.eligible_candidates(bad, negative, [])
        bad = copy.deepcopy(negative)
        bad[0]['patient_id'] = 'normal-0'
        with self.assertRaises(ValueError):
            p.eligible_candidates(positive, bad, [])
        self.assertIsNone(p.patient_id('normal_42.png'))
        self.assertEqual(p.patient_id('sub-S123_ses-E999_run-1.png'), 'sub-S123')

    def test_insufficient_groups_and_invalid_supports_fail_closed(self):
        positive, negative = pools()
        with self.assertRaises(ValueError):
            p.sample_support(positive[:3], negative, 42)
        with self.assertRaises(ValueError):
            p.sample_support(positive, negative[:4], 42)
        support = p.sample_support(positive, negative, 42)
        with self.assertRaises(ValueError):
            p.validate_support(support, [support[0]])
        support[-1] = dict(support[-2])
        with self.assertRaises(ValueError):
            p.validate_support(support, [])

    def test_frozen_file_hash_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.json'
            path.write_text(json.dumps([{'id': 1}]))
            frozen = p.ref(path)
            self.assertEqual(p.checked(frozen), [{'id': 1}])
            path.write_text(json.dumps([{'id': 2}]))
            with self.assertRaises(ValueError):
                p.checked(frozen)


if __name__ == '__main__':
    unittest.main()
