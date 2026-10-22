"""CPU checks for train-only spectral fit and Ridge CV on synthetic proxies."""
import unittest
from unittest.mock import patch
import numpy as np
from scripts.vad_research.m0.spectral_core import fit_nystrom_spectral
from scripts.vad_research.m0.run_spectral_ridge import ALPHAS, select_alpha, spectral_pair


class SpectralRunnerTests(unittest.TestCase):
    def test_basis_fit_excludes_dev_in_both_arms(self):
        train=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[2.,1.]])
        unlabeled=np.array([[3.,0.],[3.,1.]])
        dev=np.array([[1000.,1000.]])
        seen=[]
        def capture(x,**kwargs):
            seen.append(x.copy())
            return fit_nystrom_spectral(x,**kwargs)
        with patch('scripts.vad_research.m0.run_spectral_ridge.fit_nystrom_spectral',side_effect=capture):
            spectral_pair('b1',train,dev,unlabeled,gamma=.3,rank=2,landmarks=3,seed=0)
            spectral_pair('b2',train,dev,unlabeled,gamma=.3,rank=2,landmarks=3,seed=0)
        self.assertEqual([len(x) for x in seen],[5,7])
        self.assertTrue(all(np.max(x)<1000 for x in seen))

    def test_cv_runs_without_dev_data(self):
        rng=np.random.default_rng(4)
        train=rng.normal(size=(30,4))
        unlabeled=rng.normal(size=(12,4))
        targets=rng.random((30,3))
        for mode in ('b0','b1','b2'):
            alpha,scores=select_alpha(mode,train,targets,unlabeled,gamma=.3,
                                      rank=2,landmarks=4,seed=0)
            self.assertIn(alpha,ALPHAS)
            self.assertEqual(len(scores),len(ALPHAS))
            self.assertTrue(all(np.isfinite(list(scores.values()))))


if __name__=='__main__':
    unittest.main()
