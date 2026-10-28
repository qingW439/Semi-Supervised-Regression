"""TRAIN-only, head-only VAD virtual updates and 3 by 3 utility matrix."""
from __future__ import annotations

import torch
import torch.nn.functional as F
try:
    from torch.func import functional_call
except ImportError:  # PyTorch 1.x CPU compatibility
    from torch.nn.utils.stateless import functional_call


def disjoint_meta_batch(meta_iterator, supervised_ids, desired_size, max_tries=100):
    """Draw a full labeled TRAIN meta batch without current or repeated IDs."""
    if desired_size <= 0:
        raise ValueError("Meta batch size must be positive")
    excluded = set(supervised_ids)
    selected = set()
    chosen = []
    token_parts = {}
    target_parts = []
    for _ in range(max_tries):
        tokens, targets, ids = next(meta_iterator)
        for i, sample_id in enumerate(ids):
            if sample_id in excluded or sample_id in selected:
                continue
            selected.add(sample_id)
            chosen.append(sample_id)
            target_parts.append(targets[i:i + 1])
            for name, value in tokens.items():
                token_parts.setdefault(name, []).append(value[i:i + 1])
            if len(chosen) == desired_size:
                # The collator can pad different draws to different lengths.
                result = {}
                for name, parts in token_parts.items():
                    length = max(part.shape[-1] for part in parts)
                    result[name] = torch.cat([
                        F.pad(part, (0, length - part.shape[-1])) for part in parts
                    ], dim=0)
                if excluded.intersection(chosen) or len(selected) != desired_size:
                    raise AssertionError("Meta batch isolation failed")
                return result, torch.cat(target_parts, dim=0), chosen
    raise RuntimeError("Could not draw a full disjoint labeled TRAIN meta batch")


def per_dimension_unlabeled(strong, weak, detached_log_variance):
    if strong.ndim != 2 or strong.shape[1] != 3 or weak.shape != strong.shape or detached_log_variance.shape != strong.shape:
        raise ValueError("Expected three matching [batch, 3] tensors")
    # Mean over samples and then over dimensions reproduces weighted_pseudo_mse.
    return (0.5 * torch.exp(-detached_log_variance) * (strong - weak.detach()).square()).mean(dim=0)


def weighted_dimensional_pseudo_mse(strong, weak, detached_log_variance, weights):
    """The HPL elementwise loss with one nonnegative multiplier per dimension.

    Multiplication precedes the single batch-and-dimension reduction. This
    preserves the original HPL reduction order when all weights equal one.
    """
    if weights.shape != (3,) or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Expected three finite nonnegative weights")
    if strong.ndim != 2 or strong.shape[1] != 3 or weak.shape != strong.shape or detached_log_variance.shape != strong.shape:
        raise ValueError("Expected three matching [batch, 3] tensors")
    elements = 0.5 * torch.exp(-detached_log_variance) * (strong - weak.detach()).square()
    return (elements * weights.to(device=strong.device, dtype=strong.dtype).view(1, 3)).mean()


def _flat_double(grads, parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().double().reshape(-1)
                      for p, g in zip(parameters, grads)])


def virtual_utility_probe(head, unlabeled_losses, meta_features, meta_targets,
                          step_size, include_float32=False):
    """Probe M[unlabeled, meta] and scale-free first-order alignment.

    Uses stateless parameter substitution. No real parameter, optimizer state,
    encoder parameter or meta label is updated. The virtual step is SGD with
    the current head learning rate, used only to estimate directional utility.
    The controller receives gradient cosine, not a rescaled tiny finite
    difference. Positive M means the virtual unlabeled step improves meta MSE.
    """
    if unlabeled_losses.shape != (3,) or meta_targets.ndim != 2 or meta_targets.shape[1] != 3:
        raise ValueError("Expected three losses and [batch, 3] meta targets")
    if step_size <= 0:
        raise ValueError("Virtual step size must be positive")
    named = list(head.named_parameters())
    names = [name for name, _ in named]
    parameters = [param for _, param in named]
    base = {name: param.detach().double() for name, param in named}
    meta_x = meta_features.detach().double()
    meta_y = meta_targets.detach().double()
    meta_prediction = head(meta_features.detach())
    meta_losses = (meta_prediction - meta_targets).square().mean(dim=0)
    meta_grads = [_flat_double(torch.autograd.grad(meta_losses[k], parameters,
                                                   retain_graph=True, allow_unused=True), parameters)
                  for k in range(3)]
    with torch.no_grad():
        before64 = (functional_call(head, base, (meta_x,)) - meta_y).square().mean(dim=0)
        before32 = (head(meta_features.detach()) - meta_targets).square().mean(dim=0) if include_float32 else None
    parameter_norm = torch.linalg.vector_norm(torch.cat([p.reshape(-1) for p in base.values()]))
    rows64, rows32, rows_first, rows_cosine = [], [], [], []
    delta_norms, parameter_norms_after = [], []
    for unlabeled_dim in range(3):
        grads = torch.autograd.grad(unlabeled_losses[unlabeled_dim], parameters,
                                    retain_graph=True, allow_unused=True)
        gradient = _flat_double(grads, parameters)
        delta_norms.append(step_size * torch.linalg.vector_norm(gradient))
        virtual64 = {name: base[name] - step_size * (torch.zeros_like(param) if grad is None else grad).detach().double()
                     for (name, param), grad in zip(named, grads)}
        parameter_norms_after.append(torch.linalg.vector_norm(
            torch.cat([parameter.reshape(-1) for parameter in virtual64.values()])))
        with torch.no_grad():
            after64 = (functional_call(head, virtual64, (meta_x,)) - meta_y).square().mean(dim=0)
        rows64.append(before64 - after64)
        first = torch.stack([step_size * torch.dot(gradient, meta_gradient)
                             for meta_gradient in meta_grads])
        cosine = torch.stack([
            torch.dot(gradient, meta_gradient) /
            (torch.linalg.vector_norm(gradient) * torch.linalg.vector_norm(meta_gradient))
            if bool(torch.linalg.vector_norm(gradient) > 0) and bool(torch.linalg.vector_norm(meta_gradient) > 0)
            else gradient.new_zeros(())
            for meta_gradient in meta_grads
        ])
        rows_first.append(first)
        rows_cosine.append(cosine)
        if include_float32:
            virtual32 = {name: param.detach() - step_size * (torch.zeros_like(param) if grad is None else grad).detach()
                         for (name, param), grad in zip(named, grads)}
            with torch.no_grad():
                after32 = (functional_call(head, virtual32, (meta_features.detach(),)) - meta_targets).square().mean(dim=0)
            rows32.append((before32 - after32).double())
    raw = torch.stack(rows64)
    directional = torch.stack(rows_cosine)
    first_order = torch.stack(rows_first)
    delta_norm = torch.stack(delta_norms)
    tensors = (raw, directional, first_order, delta_norm)
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise FloatingPointError("Nonfinite virtual utility probe")
    return {"raw_matrix": raw.detach(), "normalized_matrix": directional.detach(),
            "first_order_matrix": first_order.detach(),
            "float32_matrix": torch.stack(rows32).detach() if include_float32 else None,
            "parameter_norm": float(parameter_norm),
            "virtual_parameter_norm_after": torch.stack(parameter_norms_after).detach(),
            "virtual_delta_norm": delta_norm.detach(),
            "relative_virtual_change": (delta_norm / parameter_norm.clamp_min(torch.finfo(torch.float64).tiny)).detach()}


def virtual_utility_matrix(head, unlabeled_losses, meta_features, meta_targets, step_size):
    """Compatibility wrapper: raw M[unlabeled dimension, meta dimension]."""
    return virtual_utility_probe(head, unlabeled_losses, meta_features, meta_targets,
                                 step_size)["raw_matrix"]
