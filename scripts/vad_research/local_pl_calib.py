"""Feature-conditional local pseudo-label calibration (M1 component).

Replaces RankUp's global RDA with a local correction:

    p_cal = (1 - a_u) * p + a_u * sum_j w_uj y_j
    a_u   = alpha * clip((conf_u - c0) / (1 - c0), 0, 1)

where the neighbours j come from the frozen labeled TRAIN bank and conf_u is the
top-1 cosine similarity. Labels are per V/A/D dimension, so the same text can be
corrected differently on each dimension.
"""
from __future__ import annotations

import torch

from scripts.vad_research.retrieval_bank import RetrievalBank


def local_calibrate(pred: torch.Tensor,
                    query_feat: torch.Tensor,
                    bank: RetrievalBank,
                    k: int = 20,
                    tau: float = 0.1,
                    alpha: float = 0.4,
                    c0: float = 0.3) -> torch.Tensor:
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError('pred must be [B,3]')
    if pred.shape[0] != query_feat.shape[0]:
        raise ValueError('pred and query_feat batch mismatch')
    if alpha < 0 or not 0.0 <= c0 < 1.0:
        raise ValueError('alpha >= 0 and 0 <= c0 < 1 required')
    knn, conf = bank.query(query_feat, k=k, tau=tau)
    gate = ((conf - c0) / max(1e-6, 1.0 - c0)).clamp(0.0, 1.0)
    a = (alpha * gate).unsqueeze(-1)
    return (1.0 - a) * pred + a * knn.to(pred.dtype)
