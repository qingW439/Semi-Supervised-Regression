"""Parameter-free LACT diagnostic primitives for the existing nonlinear P2 head."""
from __future__ import annotations

import torch


def mean_pool(model, tokens):
    states = model.encoder(**tokens, return_dict=True).last_hidden_state
    mask = tokens["attention_mask"].unsqueeze(-1).to(states.dtype)
    return (states * mask).sum(1) / mask.sum(1).clamp_min(1)


def label_subspace(z, y):
    """SVD of TRAIN-only cross covariance. Rank uses standard numerical tolerance."""
    if z.ndim != 2 or y.ndim != 2 or y.shape[1] != 3 or z.shape[0] != y.shape[0]:
        raise ValueError("Expected matched [n,d] and [n,3] matrices")
    with torch.no_grad():
        a = z.detach().to(torch.float64)
        b = y.detach().to(torch.float64)
        c = (a - a.mean(0)).T @ (b - b.mean(0)) / len(a)
        u, s, _ = torch.linalg.svd(c, full_matrices=False)
        tol = torch.finfo(s.dtype).eps * max(c.shape) * s.max()
        rank = int((s > tol).sum().item())
        return u[:, :rank].to(z.dtype).detach(), s.detach(), rank


def lact_loss(head, z_u, z_v, q, eps=1e-8):
    """J_h(z_u)(I-QQ') (z_v-z_u), keeping gradients to z and nonlinear head."""
    if z_u.shape != z_v.shape or z_u.ndim != 2:
        raise ValueError("z_u and z_v must have matching [batch,hidden] shape")
    if q.ndim != 2 or q.shape[0] != z_u.shape[1]:
        raise ValueError("Q has wrong shape")
    delta = z_v - z_u
    frozen_q = q.detach()
    r = delta - (delta @ frozen_q) @ frozen_q.T
    # The P2 head is Linear-GELU-Linear. It contains no dropout; call it directly.
    jvp = torch.func.jvp(lambda x: head(x), (z_u,), (r,))[1]
    denominator = delta.detach().square().sum(-1) + eps
    return (jvp.square().sum(-1) / denominator).mean()


def finite_difference_jvp(head, z, r, step=1e-3):
    return (head(z + step * r) - head(z - step * r)) / (2 * step)
