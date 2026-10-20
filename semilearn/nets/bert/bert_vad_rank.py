"""Three-dimensional RankUp auxiliary head for the unchanged E1 VAD backbone."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from semilearn.nets.bert.bert_vad import BertVADRegressor


class BertVADRank3D(nn.Module):
    """Each V/A/D rank head emits two classes; class 0 means i > j."""

    def __init__(self, backbone: BertVADRegressor):
        super().__init__()
        self.backbone = backbone
        self.rank_head = nn.Linear(backbone.num_features, 6)

    def forward(self, **tokens):
        features = self.backbone.encode(**tokens)
        return {
            "regression": self.backbone.regress(features),
            "rank_logits": self.rank_head(features).reshape(-1, 3, 2),
        }

    @staticmethod
    def pair_logits(rank_logits: torch.Tensor) -> torch.Tensor:
        if rank_logits.ndim != 3 or rank_logits.shape[1:] != (3, 2):
            raise ValueError("rank_logits must have shape [batch, 3, 2]")
        return rank_logits[:, None, :, :] - rank_logits[None, :, :, :]

    @staticmethod
    def pair_probability(rank_logits: torch.Tensor) -> torch.Tensor:
        """Probability that item i has a larger label than item j: [B,B,3]."""
        return BertVADRank3D.pair_logits(rank_logits).softmax(dim=-1)[..., 0]


def _unique_pair_mask(n: int, device: torch.device) -> torch.Tensor:
    upper = torch.ones((n, n), dtype=torch.bool, device=device).triu(diagonal=1)
    return upper.unsqueeze(-1).expand(n, n, 3)


def _pair_binary_logit(rank_logits: torch.Tensor) -> torch.Tensor:
    pair = BertVADRank3D.pair_logits(rank_logits)
    return pair[..., 0] - pair[..., 1]


def supervised_rank_loss(rank_logits: torch.Tensor, targets: torch.Tensor):
    """BCE over unique pairs; equal gold values have the paper's soft target 0.5."""
    if targets.ndim != 2 or targets.shape != (rank_logits.shape[0], 3):
        raise ValueError("targets must have shape [batch, 3]")
    pair_logit = _pair_binary_logit(rank_logits)
    delta = targets[:, None, :] - targets[None, :, :]
    gold = (delta > 0).to(pair_logit.dtype) + 0.5 * (delta == 0).to(pair_logit.dtype)
    mask = _unique_pair_mask(len(targets), targets.device)
    if not mask.any():
        return pair_logit.sum() * 0.0, torch.zeros(3, dtype=torch.long, device=targets.device)
    tie_counts = ((delta == 0) & mask).sum(dim=(0, 1))
    return F.binary_cross_entropy_with_logits(pair_logit[mask], gold[mask]), tie_counts


def pseudo_rank_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                     confidence_cutoff: float):
    """Hard pseudo-ranks from detached teacher, selected once per unordered pair."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student rank logits differ in shape")
    if not 0.5 <= confidence_cutoff <= 1.0:
        raise ValueError("confidence_cutoff must be in [0.5, 1]")
    student_pair_logit = _pair_binary_logit(student_logits)
    with torch.no_grad():
        teacher_probability = BertVADRank3D.pair_probability(teacher_logits.detach())
        confidence = torch.maximum(teacher_probability, 1.0 - teacher_probability)
        mask = _unique_pair_mask(len(teacher_logits), teacher_logits.device)
        accepted = mask & (confidence >= confidence_cutoff)
        hard_target = (teacher_probability >= 0.5).to(student_pair_logit.dtype)
        accepted_counts = accepted.sum(dim=(0, 1))
        possible_counts = mask.sum(dim=(0, 1))
    if not accepted.any():
        return student_pair_logit.sum() * 0.0, accepted_counts, possible_counts
    loss = F.binary_cross_entropy_with_logits(student_pair_logit[accepted], hard_target[accepted])
    return loss, accepted_counts, possible_counts
