"""HPL STS-B head-level mechanism adapted to three independent VAD targets.

Reference: sxq11/Heteroscedastic-Pseudo-Labels, sts/main_ours.py.
This module does not load EmoBank data or any Test split.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class VADUncertainty(nn.Module):
    """Shared 2->hidden->1 MLP applied separately to each V/A/D coordinate."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, difference: torch.Tensor, strong_prediction: torch.Tensor) -> torch.Tensor:
        if difference.ndim != 2 or difference.shape != strong_prediction.shape or difference.shape[-1] != 3:
            raise ValueError("Expected matching [batch, 3] difference and prediction")
        # The official STS-B uncertainty inputs do not backpropagate into the predictor.
        inputs = torch.stack((difference.detach(), strong_prediction.detach()), dim=-1)
        return self.net(inputs).squeeze(-1)


def weighted_pseudo_mse(strong: torch.Tensor, weak: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Official STS-B unlabeled term: mean(exp(-s)/2 * squared residual)."""
    if strong.ndim != 2 or strong.shape[-1] != 3 or weak.shape != strong.shape or log_variance.shape != strong.shape:
        raise ValueError("Expected three matching [batch, 3] tensors")
    return (0.5 * torch.exp(-log_variance) * (strong - weak.detach()).square()).mean()


def gradient_statistics(first, second) -> tuple[float, float, float]:
    """Norms and cosine for two gradient sequences; missing gradients count as zero."""
    if len(first) != len(second):
        raise ValueError("Gradient sequences must have matching lengths")
    first_sq = sum(float(grad.detach().square().sum()) for grad in first if grad is not None)
    second_sq = sum(float(grad.detach().square().sum()) for grad in second if grad is not None)
    dot = sum(float((left.detach() * right.detach()).sum())
              for left, right in zip(first, second) if left is not None and right is not None)
    if not all(math.isfinite(value) for value in (first_sq, second_sq, dot)):
        raise FloatingPointError("Nonfinite gradient diagnostic")
    first_norm, second_norm = math.sqrt(first_sq), math.sqrt(second_sq)
    cosine = dot / (first_norm * second_norm) if first_norm and second_norm else 0.0
    cosine = max(-1.0, min(1.0, cosine))
    if not all(math.isfinite(value) for value in (first_norm, second_norm, cosine)):
        raise FloatingPointError("Nonfinite gradient diagnostic")
    return first_norm, second_norm, cosine


def component_gradient_norm(component: torch.Tensor, parameters) -> float:
    """Measure a loss component without changing optimizer gradients."""
    params = tuple(parameters)
    if not component.requires_grad or not params:
        return 0.0
    grads = torch.autograd.grad(component, params, retain_graph=True, allow_unused=True)
    squared = sum(float(grad.detach().square().sum()) for grad in grads if grad is not None)
    norm = math.sqrt(squared)
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite component gradient diagnostic")
    return norm


def sample_statistics(values: torch.Tensor) -> dict[str, float]:
    """Population statistics for one sample-level diagnostic vector."""
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("Expected a nonempty finite 1D diagnostic tensor")
    values = values.detach()
    p10, p50, p90 = torch.quantile(values, values.new_tensor((0.1, 0.5, 0.9)))
    return {"mean": float(values.mean()), "std": float(values.std(unbiased=False)),
            "p10": float(p10), "p50": float(p50), "p90": float(p90)}


def save_last_checkpoint(model: nn.Module, uncertainty: nn.Module | None,
                         step: int, run_config: dict, path) -> None:
    """Save the completed run without affecting Dev-best selection."""
    torch.save({"model": model.state_dict(),
                "uncertainty": uncertainty.state_dict() if uncertainty is not None else None,
                "step": step, "run_config": run_config}, path)


def bilevel_head_step(
    head: nn.Module,
    head_optimizer: torch.optim.Optimizer,
    uncertainty: VADUncertainty,
    uncertainty_optimizer: torch.optim.Optimizer,
    labeled_features: torch.Tensor,
    labeled_targets: torch.Tensor,
    weak_features: torch.Tensor,
    strong_features: torch.Tensor,
    meta_features: torch.Tensor,
    meta_targets: torch.Tensor,
    lambda_u: float = 1.0,
    lambda2: float = 1.0,
) -> dict[str, float]:
    """One differentiable head-only inner update, then labeled meta update.

    The caller supplies pooled features after the encoder/dropout views. It must
    call this at the declared meta-update frequency and separately perform the
    main-model update. `higher` 0.2.1 is imported lazily so loss tests work
    without installing an additional package into the shared Python environment.
    """
    try:
        import higher
    except ImportError as exc:
        raise RuntimeError("HPL meta step requires higher==0.2.1") from exc
    if lambda_u < 0 or lambda2 < 0:
        raise ValueError("Loss weights must be nonnegative")
    if labeled_targets.ndim != 2 or labeled_targets.shape[-1] != 3 or meta_targets.ndim != 2 or meta_targets.shape[-1] != 3:
        raise ValueError("Labeled and meta targets must be [batch, 3]")
    # Detaching features restricts the virtual optimizer to the regression head.
    # The real model update must still use non-detached encoder features.
    with higher.innerloop_ctx(head, head_optimizer) as (fast_head, diff_optimizer):
        labeled_pred = fast_head(labeled_features.detach())
        supervised = F.mse_loss(labeled_pred, labeled_targets)
        with torch.no_grad():
            weak_pred = fast_head(weak_features.detach())
        strong_pred = fast_head(strong_features.detach())
        logvar = uncertainty(strong_pred - weak_pred, strong_pred)
        unsupervised = weighted_pseudo_mse(strong_pred, weak_pred, logvar)
        diff_optimizer.step(supervised + lambda_u * unsupervised)

        meta_pred = fast_head(meta_features.detach())
        meta_logvar = uncertainty(meta_pred - meta_targets, meta_pred)
        meta_mse = F.mse_loss(meta_pred, meta_targets)
        meta_logvar_mean = meta_logvar.mean()
        meta_regularization_term = -lambda2 * meta_logvar_mean
        meta_loss = meta_mse - lambda2 * meta_logvar_mean
        if not torch.isfinite(meta_loss):
            raise FloatingPointError("Nonfinite HPL meta loss")
        uncertainty_params = tuple(uncertainty.parameters())
        meta_mse_grad_norm = component_gradient_norm(meta_mse, uncertainty_params)
        meta_regularization_grad_norm = component_gradient_norm(meta_regularization_term, uncertainty_params)
        uncertainty_optimizer.zero_grad(set_to_none=True)
        meta_loss.backward()
        uncertainty_optimizer.step()
    return {
        "inner_supervised_mse": float(supervised.detach()),
        "inner_unlabeled_mse": float(unsupervised.detach()),
        "meta_loss": float(meta_loss.detach()),
        "meta_mse": float(meta_mse.detach()),
        "meta_logvar_mean": float(meta_logvar_mean.detach()),
        "meta_regularization_term": float(meta_regularization_term.detach()),
        "meta_mse_uncertainty_grad_norm": meta_mse_grad_norm,
        "meta_regularization_uncertainty_grad_norm": meta_regularization_grad_norm,
        "mean_log_variance": float(logvar.detach().mean()),
    }