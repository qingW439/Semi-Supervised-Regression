"""Frozen labeled-feature bank used by local pseudo-label calibration.

Only labeled TRAIN features/labels may enter this bank. DEV/TEST IDs must never
be added; the caller is responsible for that invariant.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class RetrievalBank:
    def __init__(self, feat: torch.Tensor, labels: torch.Tensor):
        if feat.ndim != 2 or labels.ndim != 2 or feat.shape[0] != labels.shape[0]:
            raise ValueError('feat [L,H] and labels [L,3] must share L')
        if feat.shape[1] <= 0 or labels.shape[1] != 3:
            raise ValueError('expected [L,H] features and [L,3] labels')
        if not torch.isfinite(feat).all() or not torch.isfinite(labels).all():
            raise ValueError('bank contains non-finite values')
        self.feat = F.normalize(feat.detach().float(), dim=-1)
        self.labels = labels.detach().float()

    def __len__(self) -> int:
        return self.feat.shape[0]

    def query(self, query_feat: torch.Tensor, k: int = 20, tau: float = 0.1):
        if query_feat.ndim != 2 or query_feat.shape[1] != self.feat.shape[1]:
            raise ValueError('query feature width mismatch')
        if k <= 0 or tau <= 0:
            raise ValueError('k and tau must be positive')
        q = F.normalize(query_feat.float(), dim=-1)
        sim = q @ self.feat.T
        kk = min(k, self.feat.shape[0])
        topv, topi = sim.topk(kk, dim=1)
        w = torch.softmax(topv / tau, dim=1)
        knn = (w.unsqueeze(-1) * self.labels[topi]).sum(1)
        conf = topv[:, 0].clamp(0.0, 1.0)
        return knn, conf
