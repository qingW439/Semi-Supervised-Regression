"""Head-only GGR VLR gradient surgery.

The VLR rule is the ``vlr`` branch of the official
JiaheChen2002/GGR semilearn/algorithms/utils/ggr.py (main, 2026-10-27).
It leaves the supervised gradient intact and projects only a conflicting
auxiliary gradient onto the supervised gradient's half-space.
"""
from __future__ import annotations

import torch


def flatten_dense(parameters, gradients):
    """Flatten gradients in parameter order, replacing unused entries by zero."""
    if len(parameters) != len(gradients):
        raise ValueError("Parameter and gradient lengths differ")
    if not parameters:
        raise ValueError("At least one head parameter is required")
    return torch.cat([
        (torch.zeros_like(p) if g is None else g).contiguous().reshape(-1)
        for p, g in zip(parameters, gradients)
    ])


def split_like(vector, parameters):
    if vector.numel() != sum(p.numel() for p in parameters):
        raise ValueError("Gradient length does not match head parameters")
    return [part.reshape_as(p) for part, p in zip(vector.split([p.numel() for p in parameters]), parameters)]


@torch.no_grad()
def vlr_project(supervised, unlabeled, alpha=1.0):
    """Official asymmetric VLR rule; alpha=1 removes the conflicting component."""
    if supervised.shape != unlabeled.shape:
        raise ValueError("Gradient shapes differ")
    if alpha != 1.0:
        raise ValueError("M1 fixes GGR surgery_alpha to the official default 1.0")
    dot = torch.dot(unlabeled, supervised)
    if dot >= 0:
        return unlabeled.clone()
    denominator = torch.dot(supervised, supervised).clamp_min(1e-12)
    return unlabeled - (dot / denominator) * supervised


@torch.no_grad()
def merge_head_gradients(parameters, supervised_gradients, unlabeled_gradients, project=True):
    """Return one supervised plus one (possibly projected) unlabeled gradient.

    The caller assigns these tensors to head ``.grad`` after the full-model
    backward. Adding to ``.grad`` would double-count supervised MSE.
    """
    sup = flatten_dense(parameters, supervised_gradients)
    unsup = flatten_dense(parameters, unlabeled_gradients)
    projected = vlr_project(sup, unsup) if project else unsup
    return split_like(sup + projected, parameters), {
        "supervised_head_grad_norm": float(sup.norm()),
        "unlabeled_head_grad_norm": float(unsup.norm()),
        "projected_unlabeled_head_grad_norm": float(projected.norm()),
        "supervised_unlabeled_dot": float(torch.dot(sup, unsup)),
        "post_projection_dot": float(torch.dot(sup, projected)),
        "ggr_projection_applied": int(bool(torch.dot(sup, unsup) < 0)) if project else 0,
    }
