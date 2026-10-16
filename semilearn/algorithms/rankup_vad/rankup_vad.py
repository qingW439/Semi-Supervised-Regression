"""Dimension-wise auxiliary ranking on the shared EmoBank VAD backbone.

The original scalar RankUp implementation is kept separate. This module
accepts normalized [batch, 3] labels and never uses dev or test labels as
training targets. An E2 trainer must supply genuinely distinct weak/strong
unlabeled views before enabling the unsupervised ranking term.
"""

import torch
from torch import nn
from torch.nn import functional as F


class RankUpVADNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.arc_classifier = nn.Linear(backbone.num_features, 3 * 2)

    def forward(self, **tokens):
        features = self.backbone.encode(**tokens)
        return {
            "logits": self.backbone.regress(features),
            "arc_logits": self.arc_classifier(features).reshape(-1, 3, 2),
        }

    @staticmethod
    def pair_logits(arc_logits):
        """For pair (i,j), class 0 means item i has larger V/A/D."""
        if arc_logits.ndim != 3 or arc_logits.shape[1:] != (3, 2):
            raise ValueError("arc_logits must have shape [batch, 3, 2]")
        return arc_logits[:, None] - arc_logits[None, :]


class RankUpVADLoss(nn.Module):
    def __init__(self, arc_loss_ratio=1.0, arc_ulb_loss_ratio=1.0,
                 p_cutoff=0.95, temperature=0.5):
        super().__init__()
        if not 0 <= p_cutoff <= 1 or temperature <= 0:
            raise ValueError("Invalid confidence threshold or temperature")
        self.arc_loss_ratio = arc_loss_ratio
        self.arc_ulb_loss_ratio = arc_ulb_loss_ratio
        self.p_cutoff = p_cutoff
        self.temperature = temperature

    def forward(self, labeled, targets, weak=None, strong=None):
        predictions = labeled["logits"]
        if predictions.shape != targets.shape or targets.ndim != 2 or targets.shape[1] != 3:
            raise ValueError("VAD predictions and targets must have shape [batch, 3]")
        regression = F.l1_loss(predictions, targets)
        pair_logits = RankUpVADNet.pair_logits(labeled["arc_logits"])
        n = targets.shape[0]
        difference = targets[:, None, :] - targets[None, :, :]
        valid = ~torch.eye(n, dtype=torch.bool, device=targets.device)[:, :, None]
        valid = valid & (difference != 0)
        classes = (difference < 0).long()
        if valid.any():
            supervised_rank = F.cross_entropy(pair_logits[valid], classes[valid])
        else:
            supervised_rank = pair_logits.sum() * 0

        unsupervised_rank = pair_logits.sum() * 0
        if (weak is None) != (strong is None):
            raise ValueError("Both weak and strong unlabeled views are required")
        if weak is not None:
            weak_pairs = RankUpVADNet.pair_logits(weak["arc_logits"])
            strong_pairs = RankUpVADNet.pair_logits(strong["arc_logits"])
            if weak_pairs.shape != strong_pairs.shape:
                raise ValueError("Unlabeled pair shapes differ")
            m = weak_pairs.shape[0]
            with torch.no_grad():
                # Match RankUp semantics: filter on raw probabilities, then
                # generate the hard pseudo-label from separately sharpened logits.
                confidence = weak_pairs.softmax(-1).max(-1).values
                pseudo = (weak_pairs / self.temperature).softmax(-1).argmax(-1)
                mask = confidence >= self.p_cutoff
                mask &= ~torch.eye(m, dtype=torch.bool, device=mask.device)[:, :, None]
            if mask.any():
                unsupervised_rank = F.cross_entropy(strong_pairs[mask], pseudo[mask])
        total = regression + self.arc_loss_ratio * (
            supervised_rank + self.arc_ulb_loss_ratio * unsupervised_rank)
        return {"loss": total, "regression": regression,
                "supervised_rank": supervised_rank,
                "unsupervised_rank": unsupervised_rank}
