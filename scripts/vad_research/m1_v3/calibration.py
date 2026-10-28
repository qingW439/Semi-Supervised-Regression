"""Fixed, train-only nearest-neighbor pseudo-label calibration.

This module is deliberately independent of model training. Input predictions and
features must carry provenance checked by ``audit_targets.py`` before use.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

K = 8
TEMPERATURE = 0.1
MODES = ('reference_target', 'simregmatch_calibration', 'residual_direct',
         'residual_consistency')


def normalize_features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Features must be a finite 2-D array')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError('Zero-norm feature')
    return x / norms


def check_targets(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError(f'{name} must be finite [N,3]')
    if ((x < 0) | (x > 1)).any():
        raise ValueError(f'{name} outside standardized VAD [0,1]')
    return x


@dataclass(frozen=True)
class NeighborLibrary:
    ids: tuple[str, ...]
    features: np.ndarray
    labels: np.ndarray
    oof_predictions: np.ndarray

    def __post_init__(self):
        if len(self.ids) != len(set(self.ids)) or len(self.ids) < K:
            raise ValueError('Library needs at least 8 distinct IDs')
        feats = normalize_features(self.features)
        labels = check_targets(self.labels, 'labels')
        oof = check_targets(self.oof_predictions, 'OOF predictions')
        if len(self.ids) != len(feats) or len(self.ids) != len(labels) or len(self.ids) != len(oof):
            raise ValueError('Library array length mismatch')
        object.__setattr__(self, 'features', feats)
        object.__setattr__(self, 'labels', labels)
        object.__setattr__(self, 'oof_predictions', oof)


def calibrate(reference: np.ndarray, query_features: np.ndarray,
              query_ids: list[str] | tuple[str, ...], library: NeighborLibrary,
              mode: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return clipped, detached numpy targets and diagnostic arrays.

    SimRegMatch-style arm is an explicitly fixed 50:50 prediction/neighbor-label
    interpolation; it is a controlled adaptation, not a literal copy of its
    uncertainty-filtered end-to-end training algorithm.
    """
    if mode not in MODES:
        raise ValueError(f'Unknown mode {mode}')
    base = check_targets(reference, 'reference')
    query = normalize_features(query_features)
    if len(base) != len(query) or len(base) != len(query_ids):
        raise ValueError('Query array length mismatch')
    if set(query_ids) & set(library.ids):
        raise ValueError('Query IDs overlap their own neighbor library')
    scores = query @ library.features.T
    order = np.argsort(-scores, axis=1, kind='stable')[:, :K]
    top = np.take_along_axis(scores, order, axis=1)
    scaled = (top - top.max(axis=1, keepdims=True)) / TEMPERATURE
    weights = np.exp(scaled)
    weights /= weights.sum(axis=1, keepdims=True)
    neighbor_labels = library.labels[order]
    neighbor_label_mean = np.sum(weights[..., None] * neighbor_labels, axis=1)
    residuals = library.labels - library.oof_predictions
    picked_residuals = residuals[order]
    correction = np.sum(weights[..., None] * picked_residuals, axis=1)
    signed_support = np.sum(weights[..., None] * np.sign(picked_residuals), axis=1)
    sign_gate = np.abs(signed_support)
    if mode == 'reference_target':
        raw = base.copy()
    elif mode == 'simregmatch_calibration':
        raw = 0.5 * base + 0.5 * neighbor_label_mean
    elif mode == 'residual_direct':
        raw = base + correction
    else:
        raw = base + sign_gate * correction
    target = np.clip(raw, 0.0, 1.0)
    if not np.isfinite(target).all():
        raise ValueError('Nonfinite calibrated target')
    return target, {'correction': target - base, 'raw_correction': correction,
                    'sign_gate': sign_gate, 'neighbor_ids': np.asarray(library.ids)[order],
                    'neighbor_weights': weights, 'neighbor_label_mean': neighbor_label_mean}
