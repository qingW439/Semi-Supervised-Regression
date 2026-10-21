"""Dimensionwise latent Close Mixup for EmoBank VAD.

The E1 backbone is kept unchanged so its checkpoint loads strictly.  Rank
class 0 means the first item has the larger target, as in the RankUp code.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from semilearn.nets.bert.bert_vad import BertVADRegressor


class BertVADRCUS3D(nn.Module):
    def __init__(self, backbone: BertVADRegressor):
        super().__init__()
        self.backbone = backbone
        self.rank_head = nn.Linear(backbone.num_features, 6)

    def rank(self, features: torch.Tensor) -> torch.Tensor:
        return self.rank_head(features).reshape(*features.shape[:-1], 3, 2)

    def forward(self, **tokens):
        features = self.backbone.encode(**tokens)
        return {"features": features,
                "regression": self.backbone.regress(features),
                "rank_logits": self.rank(features)}


def unique_pairs(n: int, device: torch.device):
    return torch.triu_indices(n, n, offset=1, device=device)


def pairwise_rank_loss(rank_logits: torch.Tensor, targets: torch.Tensor):
    """Head-only warmup on real V/A/D order, excluding self/duplicate pairs."""
    i, j = unique_pairs(len(targets), targets.device)
    binary = rank_logits[..., 0] - rank_logits[..., 1]
    logits = binary[i] - binary[j]
    delta = targets[i] - targets[j]
    gold = (delta > 0).to(logits.dtype) + 0.5 * (delta == 0).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, gold), (delta == 0).sum(dim=0)


def local_logits(model: BertVADRCUS3D, features: torch.Tensor,
                 i: torch.Tensor, j: torch.Tensor, lam: float):
    """Two endpoint/nearby-mix rankings, both oriented from i toward j."""
    left, right = features[i], features[j]
    near_left = lam * left + (1.0 - lam) * right
    near_right = (1.0 - lam) * left + lam * right
    scores = model.rank(torch.stack((left, near_left, near_right, right), dim=1))
    binary = scores[..., 0] - scores[..., 1]
    return binary[:, 0] - binary[:, 1], binary[:, 2] - binary[:, 3]


def masked_local_loss(first: torch.Tensor, second: torch.Tensor,
                      target: torch.Tensor, mask: torch.Tensor):
    """Masked sum / every unique pair and dimension; zero mask gives zero grad."""
    if first.shape != second.shape or first.shape != target.shape or first.shape != mask.shape:
        raise ValueError("Local logits, targets and mask must have identical [pairs,3] shapes")
    if first.numel() == 0:
        return (first.sum() + second.sum()) * 0.0
    terms = F.binary_cross_entropy_with_logits(first, target, reduction="none")
    terms += F.binary_cross_entropy_with_logits(second, target, reduction="none")
    return (terms * mask.to(terms.dtype)).sum() / mask.numel()


def supervised_close_mixup_loss(model: BertVADRCUS3D, features: torch.Tensor,
                                targets: torch.Tensor, lam: float):
    if targets.shape != (len(features), 3):
        raise ValueError("Expected VAD targets [batch,3]")
    i, j = unique_pairs(len(features), features.device)
    first, second = local_logits(model, features, i, j, lam)
    delta = targets[i] - targets[j]
    # The original rank head uses 0.5 for ties; retain that convention.
    gold = (delta > 0).to(first.dtype) + 0.5 * (delta == 0).to(first.dtype)
    loss = masked_local_loss(first, second, gold, torch.ones_like(gold, dtype=torch.bool))
    return loss, (delta == 0).sum(dim=0)


def unsupervised_close_mixup_loss(model: BertVADRCUS3D,
                                  teacher_logits: torch.Tensor,
                                  student_features: torch.Tensor,
                                  lam: float, cutoff: float | torch.Tensor,
                                  temperature: torch.Tensor | None = None,
                                  active: torch.Tensor | None = None):
    if teacher_logits.shape != (len(student_features), 3, 2):
        raise ValueError("Teacher logits must have shape [batch,3,2]")
    i, j = unique_pairs(len(student_features), student_features.device)
    first, second = local_logits(model, student_features, i, j, lam)
    with torch.no_grad():
        cutoffs = torch.as_tensor(cutoff, device=student_features.device, dtype=first.dtype).expand(3)
        if not bool(((cutoffs >= 0.5) & (cutoffs <= 1.0)).all()):
            raise ValueError("Confidence cutoffs must be in [0.5,1]")
        temperatures = (torch.ones(3, device=student_features.device, dtype=first.dtype)
                        if temperature is None else torch.as_tensor(
                            temperature, device=student_features.device, dtype=first.dtype).expand(3))
        if not bool((temperatures > 0).all()):
            raise ValueError("Temperatures must be positive")
        enabled = (torch.ones(3, device=student_features.device, dtype=torch.bool)
                   if active is None else torch.as_tensor(active, device=student_features.device,
                                                           dtype=torch.bool).expand(3))
        teacher_binary = teacher_logits[..., 0] - teacher_logits[..., 1]
        probability = torch.sigmoid((teacher_binary[i] - teacher_binary[j]) / temperatures)
        confidence = torch.maximum(probability, 1.0 - probability)
        mask = (confidence >= cutoffs) & enabled
        pseudo = (probability >= 0.5).to(first.dtype)
        kept = mask.sum(dim=0)
        possible = torch.full((3,), len(i), dtype=torch.long, device=student_features.device)
    return masked_local_loss(first, second, pseudo, mask), kept, possible
