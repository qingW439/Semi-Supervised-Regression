"""CPU-feasible Nyström spectral proxy features for an EmoBank M0 ablation.

This is an approximation to Kim & Kim (ICML 2026), not their full N x N
kernel eigendecomposition. Only fit on TRAIN proxy covariates.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Expected [samples, features] arrays with matching widths")
    sq = cdist(x, y, metric="sqeuclidean")
    return np.exp(-gamma * sq)


@dataclass
class NystromSpectralBasis:
    landmarks: np.ndarray
    whitening: np.ndarray
    directions: np.ndarray
    gamma: float

    def transform(self, features: np.ndarray) -> np.ndarray:
        return rbf_kernel(np.asarray(features, dtype=np.float64), self.landmarks, self.gamma) @ self.whitening @ self.directions


def fit_nystrom_spectral(
    train_features: np.ndarray,
    *,
    gamma: float,
    rank: int,
    landmarks: int,
    seed: int,
    jitter: float = 1e-8,
    external_landmarks: np.ndarray | None = None,
) -> NystromSpectralBasis:
    """Approximate top kernel eigenfeatures without a full N x N matrix.

    By default landmarks are sampled from training features with a fixed seed.
    External landmarks must also come from labeled TRAIN; neither DEV nor TEST
    features or labels may be used to construct the basis.
    """
    x = np.asarray(train_features, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite 2D training features")
    n = len(x)
    if not (1 <= rank <= landmarks <= n) or gamma <= 0 or jitter <= 0:
        raise ValueError("Invalid rank, landmark count, gamma, or jitter")
    if external_landmarks is None:
        indices = np.random.default_rng(seed).choice(n, size=landmarks, replace=False)
        anchor = x[indices].copy()
    else:
        anchor = np.asarray(external_landmarks, dtype=np.float64)
        if anchor.shape != (landmarks, x.shape[1]) or not np.isfinite(anchor).all():
            raise ValueError("Expected finite external landmarks with matching shape")
        anchor = anchor.copy()
    k_mm = rbf_kernel(anchor, anchor, gamma)
    eigval, eigvec = np.linalg.eigh(k_mm)
    eigval = np.maximum(eigval, jitter)
    whitening = (eigvec * eigval**-0.5) @ eigvec.T
    z = rbf_kernel(x, anchor, gamma) @ whitening
    spectral_value, spectral_vector = np.linalg.eigh(z.T @ z)
    top = spectral_vector[:, np.argsort(spectral_value)[-rank:][::-1]]
    return NystromSpectralBasis(anchor, whitening, top, gamma)
