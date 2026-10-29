"""Previous-batch MC uncertainty threshold, with explicit empty-mask behavior."""
from __future__ import annotations
import torch


def filter_mask(uncertainty: torch.Tensor, threshold: float) -> torch.Tensor:
    if uncertainty.ndim != 1 or not torch.isfinite(uncertainty).all() or not torch.isfinite(torch.tensor(threshold)):
        raise ValueError("Expected finite sample uncertainties and threshold")
    return uncertainty.detach() < threshold


def next_threshold(uncertainty: torch.Tensor, percentile: float = 95.0) -> float:
    if not 0 <= percentile <= 100 or uncertainty.ndim != 1 or uncertainty.numel() == 0 or not torch.isfinite(uncertainty).all():
        raise ValueError("Invalid uncertainty percentile input")
    return float(torch.quantile(uncertainty.detach().float(), percentile / 100.0).item())


def masked_mse(predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 3 or mask.shape != predictions.shape[:1]:
        raise ValueError("Expected [batch,3] predictions and targets with [batch] mask")
    # Official loss sums three dimensions per selected sample, then averages selected samples.
    per_sample = (predictions - targets.detach()).square().sum(dim=1)
    return (per_sample * mask.to(per_sample.dtype)).sum() / mask.sum().clamp_min(1)
