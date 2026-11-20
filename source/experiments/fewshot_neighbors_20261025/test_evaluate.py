"""Integration checks for shared fits, score fusion, and test-feature isolation."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate import fit_task, COMPETITORS

class FairViewComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        rng=np.random.default_rng(31)
        unit=lambda a:a/np.linalg.norm(a,axis=-1,keepdims=True)
        support=unit(rng.normal(size=(10,8)))
        test=unit(rng.normal(size=(14,8)))
        positive=unit(rng.normal(size=(1,8)))
        negative=unit(rng.normal(size=(1,8)))
        direction=(positive-negative).T
        cls.kw=dict(support_features=support, support_logits=support@direction,
            labels=np.asarray([0]*5+[1]*5)[:,None],groups=[f'p{i}' for i in range(10)],
            folds=[dict(train=[j for j in range(10) if j not in (i,i+5)],validation=[i,i+5]) for i in range(5)],
            queries_positive=positive,queries_negative=negative,features_identity=test,
            logits_identity=test@direction,support_seed=42)
        cls.identical,cls.fit1,_=fit_task(**cls.kw,features_center=test,logits_center=test@direction)
        cls.swapped,cls.fit2,_=fit_task(**cls.kw,features_center=test[::-1],logits_center=(test@direction)[::-1])

    def test_identical_views_reproduce_single(self):
        for name in COMPETITORS:
            np.testing.assert_allclose(self.identical[name],self.identical[name+'_view'],atol=1e-10,rtol=1e-10)

    def test_center_cannot_change_fit_or_single_prediction(self):
        def without_timing(value):
            if isinstance(value,dict):
                return {k:without_timing(v) for k,v in value.items() if not k.endswith('_seconds')}
            if isinstance(value,list):return [without_timing(v) for v in value]
            return value
        self.assertEqual(without_timing(self.fit1),without_timing(self.fit2))
        for name in COMPETITORS:
            np.testing.assert_allclose(self.identical[name],self.swapped[name],atol=1e-10,rtol=1e-10)

    def test_two_view_scores_are_same_fit_equal_average(self):
        for name in COMPETITORS:
            np.testing.assert_allclose(self.swapped[name+'_view'],
                .5*(self.identical[name]+self.identical[name][::-1]),atol=1e-6,rtol=1e-6)
            self.assertTrue(self.fit1[name]['dual_view_parameters_identical'])
            self.assertFalse(self.fit1[name]['test_labels_received'])

if __name__=='__main__':unittest.main()
