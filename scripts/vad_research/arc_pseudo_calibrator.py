"""CPR: calibrated continuous pseudo-ranks for RankUp-3D ARC.

Only the unlabeled ARC pseudo-rank target/mask generator is replaced:

    teacher MC predictions (weak view)
      -> SimRegMatch uncertainty filter (previous-batch 95th percentile)
      -> labeled-bank similarity calibration (beta blend)
      -> per-dimension pairwise ordering labels [B,B,3] + mask [B,B,3]

The existing student ARC strong-view CE loss consumes the labels/mask unchanged.
"""
from __future__ import annotations

import torch

from scripts.vad_research.simregmatch3d.calibration import (
    mc_statistics,
    similarity_labels,
    calibrated_targets,
)
from scripts.vad_research.simregmatch3d.uncertainty_filter import (
    filter_mask,
    next_threshold,
)


@torch.no_grad()
def mc_predict(model, tokens, passes: int = 5):
    """K stochastic teacher forwards on the weak view; dropout on, no gradients."""
    if passes < 2:
        raise ValueError('passes must be >= 2')
    was_training = model.training
    model.train()  # enable dropout
    preds = []
    for _ in range(passes):
        pred, _ = model(**tokens)
        preds.append(pred.detach())
    model.train(was_training)
    return torch.stack(preds)  # [K,B,3]


def calibrated_pseudo_targets(mc_predictions, unlabeled_features, labeled_features,
                              labeled_targets, beta=0.6, temperature=1.0,
                              threshold=None, percentile=95.0):
    """SimRegMatch-3D filtering + calibration; returns calibrated VAD and mask."""
    mean, variance, uncertainty = mc_statistics(mc_predictions)
    thr = float(threshold) if threshold is not None else next_threshold(uncertainty, percentile)
    reliable = filter_mask(uncertainty, thr)
    neighbor, _ = similarity_labels(unlabeled_features, labeled_features,
                                    labeled_targets, temperature=temperature)
    calibrated = calibrated_targets(mean, neighbor, beta=beta)
    new_thr = next_threshold(uncertainty, percentile)
    return calibrated, reliable, thr, new_thr, uncertainty


def build_calibrated_pseudo_ranks(calibrated, reliable):
    """[B,3] calibrated targets + [B] reliability -> [B,B,3] class labels + mask."""
    if calibrated.ndim != 2 or calibrated.shape[1] != 3:
        raise ValueError('calibrated must be [B,3]')
    if reliable.shape != calibrated.shape[:1]:
        raise ValueError('reliable must be [B]')
    calib = calibrated.detach()
    rel = reliable.detach().to(torch.bool)
    b = calib.shape[0]
    diff = calib.unsqueeze(1) - calib.unsqueeze(0)      # [i,j] = calib_i - calib_j
    target = (diff > 0).long()                          # class 1 iff i > j
    pair_ok = rel.unsqueeze(1) & rel.unsqueeze(0)       # [B,B]
    valid = pair_ok.unsqueeze(-1) & (diff.abs() > 1e-6)  # exclude ties
    eye = torch.eye(b, dtype=torch.bool, device=calib.device)
    mask = valid & ~eye.unsqueeze(-1)
    return target, mask
