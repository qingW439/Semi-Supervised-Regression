"""Tiny CPU checks for the predeclared Nyström approximation."""
import unittest
import numpy as np
from scripts.vad_research.m0.spectral_core import rbf_kernel, fit_nystrom_spectral


class SpectralCoreTests(unittest.TestCase):
    def test_full_landmarks_reconstruct_kernel(self):
        x = np.array([[0., 0.], [1., 0.], [0., 1.], [2., 2.]])
        basis = fit_nystrom_spectral(x, gamma=0.4, rank=4, landmarks=4, seed=7)
        phi = basis.transform(x)
        np.testing.assert_allclose(phi @ phi.T, rbf_kernel(x, x, 0.4), atol=1e-6)

    def test_transform_does_not_refit_on_dev(self):
        x = np.array([[0., 0.], [1., 0.], [0., 1.], [2., 2.]])
        dev = np.array([[20., 20.]])
        basis = fit_nystrom_spectral(x, gamma=0.4, rank=2, landmarks=3, seed=7)
        before = basis.transform(x)
        self.assertEqual(tuple(basis.transform(dev).shape), (1, 2))
        np.testing.assert_array_equal(before, basis.transform(x))
        self.assertEqual(len(basis.landmarks), 3)

    def test_external_landmarks_are_used_and_copied(self):
        x = np.array([[0., 0.], [1., 0.], [0., 1.], [2., 2.]])
        anchors = x[[2, 0, 1]].copy()
        basis = fit_nystrom_spectral(x, gamma=0.4, rank=2, landmarks=3,
                                     seed=7, external_landmarks=anchors)
        np.testing.assert_array_equal(basis.landmarks, anchors)
        anchors[0] = 99.
        np.testing.assert_array_equal(basis.landmarks, x[[2, 0, 1]])
        self.assertEqual(basis.transform(x).shape, (4, 2))
        with self.assertRaises(ValueError):
            fit_nystrom_spectral(x, gamma=0.4, rank=2, landmarks=3, seed=7,
                                 external_landmarks=x[:2])
        with self.assertRaises(ValueError):
            fit_nystrom_spectral(x, gamma=0.4, rank=2, landmarks=3, seed=7,
                                 external_landmarks=np.full((3, 2), np.nan))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            fit_nystrom_spectral(np.ones((2, 2)), gamma=1., rank=3, landmarks=3, seed=1)
        with self.assertRaises(ValueError):
            rbf_kernel(np.ones((2, 2)), np.ones((2, 3)), gamma=1.)


if __name__ == '__main__':
    unittest.main()