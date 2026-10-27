"""TRAIN-only, head-only VAD virtual updates and 3 by 3 utility matrix."""
from __future__ import annotations

import torch
import torch.nn.functional as F
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


def virtual_utility_matrix(head, unlabeled_losses, meta_features, meta_targets, step_size):
    """M[d,k] = meta MSE_d before minus after a virtual step on unlabeled k.

    Uses stateless parameter substitution. No real parameter, optimizer state,
    encoder parameter or meta label is updated. The virtual step is SGD with
    the current head learning rate, used only to estimate directional utility.
    """
    if unlabeled_losses.shape != (3,) or meta_targets.ndim != 2 or meta_targets.shape[1] != 3:
        raise ValueError("Expected three losses and [batch, 3] meta targets")
    if step_size <= 0:
        raise ValueError("Virtual step size must be positive")
    named = list(head.named_parameters())
    names = [name for name, _ in named]
    parameters = [param for _, param in named]
    # Evaluate tiny 5e-6 virtual steps in float64 so the loss difference does
    # not disappear through float32 cancellation. This never casts the model.
    base = {name: param.detach().double() for name, param in named}
    meta_x = meta_features.detach().double()
    meta_y = meta_targets.detach().double()
    with torch.no_grad():
        before = (functional_call(head, base, (meta_x,)) - meta_y).square().mean(dim=0)
    columns = []
    for k in range(3):
        grads = torch.autograd.grad(unlabeled_losses[k], parameters, retain_graph=True, allow_unused=True)
        virtual = {
            name: (base[name] - step_size * grad.detach().double() if grad is not None else base[name])
            for name, param, grad in zip(names, parameters, grads)
        }
        with torch.no_grad():
            after = (functional_call(head, virtual, (meta_x,)) - meta_y).square().mean(dim=0)
        columns.append(before - after)
    matrix = torch.stack(columns, dim=1)
    if matrix.shape != (3, 3) or not torch.isfinite(matrix).all():
        raise FloatingPointError("Nonfinite 3x3 utility matrix")
    return matrix.detach()
