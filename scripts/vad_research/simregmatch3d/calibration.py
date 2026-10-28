"""SimRegMatch-3D batch-local similarity calibration on normalized VAD labels."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def mc_statistics(predictions: torch.Tensor):
    """predictions: [passes, unlabeled, 3]; sample variance follows torch.std default."""
    if predictions.ndim != 3 or predictions.shape[0] < 2 or predictions.shape[-1] != 3:
        raise ValueError("Expected at least two MC draws with shape [draws, batch, 3]")
    if not torch.isfinite(predictions).all():
        raise ValueError("Nonfinite MC predictions")
    mean = predictions.mean(0)
    variance = predictions.var(0, unbiased=True)
    return mean, variance, variance.sum(-1)


def similarity_labels(unlabeled_features, labeled_features, labeled_targets, temperature=1.0):
    if unlabeled_features.ndim != 2 or labeled_features.ndim != 2 or unlabeled_features.shape[1] != labeled_features.shape[1]:
        raise ValueError("Feature shapes are incompatible")
    if labeled_targets.shape != (labeled_features.shape[0], 3) or labeled_features.shape[0] == 0 or temperature <= 0:
        raise ValueError("Invalid labeled targets or temperature")
    u = F.normalize(unlabeled_features.detach(), dim=-1)
    l = F.normalize(labeled_features.detach(), dim=-1)
    weights = torch.softmax((u @ l.T) / temperature, dim=1)
    return weights @ labeled_targets.detach(), weights


def calibrated_targets(model_mean, neighbor_mean, beta=0.6):
    if model_mean.shape != neighbor_mean.shape or model_mean.ndim != 2 or model_mean.shape[1] != 3 or not 0 <= beta <= 1:
        raise ValueError("Expected matching [batch,3] targets and beta in [0,1]")
    return (beta * model_mean.detach() + (1-beta) * neighbor_mean.detach()).clamp(0.0, 1.0)
