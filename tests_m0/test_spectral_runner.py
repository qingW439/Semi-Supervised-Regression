"""CPU checks for train-only spectral fit and Ridge CV on synthetic proxies."""
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.model_selection import KFold
from scripts.vad_research.m0.spectral_core import fit_nystrom_spectral
from scripts.vad_research.m0.run_spectral_ridge import ALPHAS, select_alpha, spectral_pair


class SpectralRunnerTests(unittest.TestCase):
    def test_basis_fit_excludes_dev_in_both_arms(self):
        train=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[2.,1.]])
        unlabeled=np.array([[3.,0.],[3.,1.]])
        dev=np.array([[1000.,1000.]])
        seen=[]
        def capture(x,**kwargs):
            seen.append((x.copy(),kwargs['external_landmarks'].copy()))
            return fit_nystrom_spectral(x,**kwargs)
        with patch('scripts.vad_research.m0.run_spectral_ridge.fit_nystrom_spectral',side_effect=capture):
            spectral_pair('b1',train,dev,unlabeled,gamma=.3,rank=2,landmarks=3,seed=0)
            spectral_pair('b2',train,dev,unlabeled,gamma=.3,rank=2,landmarks=3,seed=0)
        self.assertEqual([len(x) for x,_ in seen],[5,7])
        self.assertTrue(all(np.max(x)<1000 for x,_ in seen))
        expected=train[np.random.default_rng(0).choice(len(train),size=3,replace=False)]
        np.testing.assert_array_equal(seen[0][1],expected)
        np.testing.assert_array_equal(seen[1][1],expected)

    def test_cv_uses_shared_fit_only_anchors(self):
        train=np.arange(60,dtype=float).reshape(30,2)
        unlabeled=np.arange(1000,1024,dtype=float).reshape(12,2)
        targets=np.arange(90,dtype=float).reshape(30,3)/90
        folds=list(KFold(n_splits=5,shuffle=True,random_state=11).split(train))
        by_mode={}
        for mode in ('b1','b2'):
            seen=[]
            def capture(x,**kwargs):
                seen.append((x.copy(),kwargs['external_landmarks'].copy(),kwargs['seed']))
                return fit_nystrom_spectral(x,**kwargs)
            with patch('scripts.vad_research.m0.run_spectral_ridge.fit_nystrom_spectral',side_effect=capture):
                select_alpha(mode,train,targets,unlabeled,gamma=.3,rank=2,landmarks=4,seed=11)
            self.assertEqual(len(seen),5)
            by_mode[mode]=seen
            for fold,(fit_idx,val_idx) in enumerate(folds):
                basis_x,anchors,fit_seed=seen[fold]
                fit_x=train[fit_idx]
                expected=fit_x[np.random.default_rng(11+fold).choice(len(fit_x),size=4,replace=False)]
                np.testing.assert_array_equal(anchors,expected)
                np.testing.assert_array_equal(basis_x,fit_x if mode=='b1' else np.concatenate((fit_x,unlabeled)))
                self.assertFalse(any(np.any(np.all(train[val_idx]==anchor,axis=1)) for anchor in anchors))
                self.assertEqual(fit_seed,11+fold)
        for b1,b2 in zip(by_mode['b1'],by_mode['b2']):
            np.testing.assert_array_equal(b1[1],b2[1])

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