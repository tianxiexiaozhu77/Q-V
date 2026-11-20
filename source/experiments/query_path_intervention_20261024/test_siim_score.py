import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("siim_score", Path(__file__).with_name("fit_s_siim.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ScoreFoldTests(unittest.TestCase):
    def test_each_fold_has_positive_and_negative_and_disjoint_groups(self):
        y = np.asarray([1,0,1,0,1,0,1,0,1,0])[:,None]
        folds = module.folds_for_binary_support(y, list(range(10)))
        counts = np.zeros(10, dtype=int)
        for fold in folds:
            tr, va = fold["train"], fold["validation"]
            self.assertEqual(int(y[va].sum()), 1)
            self.assertEqual(int(y[tr].sum()), 4)
            self.assertFalse(set(tr) & set(va))
            counts[va] += 1
        np.testing.assert_array_equal(counts, np.ones(10))

    def test_duplicate_groups_rejected(self):
        with self.assertRaises(ValueError):
            module.folds_for_binary_support(np.asarray([0]*5+[1]*5)[:,None], [0]*10)


if __name__ == "__main__":
    unittest.main()
