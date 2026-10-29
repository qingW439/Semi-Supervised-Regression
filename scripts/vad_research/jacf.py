"""JACF: Joint Anchor-Consistency Fusion.

One component that takes two *evidence sources* for an unlabeled text:

  teacher multi-view evidence   (mu_T, v_T)   from K stochastic teacher forwards
  labeled-anchor evidence       (mu_A, v_A)   from the labeled feature bank

and jointly outputs a calibrated target and a per-dimension reliability:

  alpha = v_T / (v_T + v_A)
  y_tilde = (1-alpha) mu_T + alpha mu_A
  D = (mu_T - mu_A)^2 / (v_T + v_A)
  c = exp(-D/2)
  v_F = v_T v_A / (v_T + v_A)
  r = c * v0 / (v0 + v_F)

This is precision-weighted fusion plus a conflict penalty, not a sequential
filter-then-calibrate pipeline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _unit(x):
    return F.normalize(x.detach().float(), dim=-1)


def jacf_fuse(mc_predictions, unlabeled_features, labeled_features, labeled_targets,
              k: int = 20, tau: float = 0.1,
              eps_t: float = 1e-4, eps_a: float = 1e-4, v0: float = 1e-3):
    if mc_predictions.ndim != 3 or mc_predictions.shape[-1] != 3:
        raise ValueError('mc_predictions must be [K,B,3]')
    if mc_predictions.shape[0] < 2:
        raise ValueError('need at least two views')
    if unlabeled_features.ndim != 2 or labeled_features.ndim != 2:
        raise ValueError('features must be [N,H]')
    if labeled_targets.shape != (labeled_features.shape[0], 3):
        raise ValueError('labeled_targets must be [L,3]')
    if k <= 0 or tau <= 0 or v0 <= 0:
        raise ValueError('k, tau, v0 must be positive')

    mc = mc_predictions.detach()
    mu_t = mc.mean(0)
    v_t = mc.var(0, unbiased=False) + eps_t

    sim = _unit(unlabeled_features) @ _unit(labeled_features).T
    kk = min(k, labeled_features.shape[0])
    topv, topi = sim.topk(kk, dim=1)
    w = torch.softmax(topv / tau, dim=1)
    nb = labeled_targets.detach()[topi]
    mu_a = (w.unsqueeze(-1) * nb).sum(1)
    v_a = (w.unsqueeze(-1) * (nb - mu_a.unsqueeze(1)).square()).sum(1) + eps_a

    alpha = v_t / (v_t + v_a)
    y_tilde = (1.0 - alpha) * mu_t + alpha * mu_a

    diff = (mu_t - mu_a).square() / (v_t + v_a)
    c = torch.exp(-0.5 * diff)
    v_f = v_t * v_a / (v_t + v_a)
    r = c * (v0 / (v0 + v_f))

    return y_tilde, r, {
        'alpha': alpha, 'v_t': v_t, 'v_a': v_a,
        'v_f': v_f, 'conflict': diff, 'support': c,
    }
