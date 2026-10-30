"""Objective modes for the JACF M1 ablation.

A0 multi : teacher multi-view only  (target=mu_T, reliability=1/(1+v_T))
A1 anchor: labeled anchor only      (target=mu_A, reliability=1/(1+v_A))
A2 seq   : original GCF + LA-PLC sequential combination
A3 jacf  : full joint anchor-consistency fusion
A4 no_rel: JACF target with reliability fixed to 1
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def evidence(mc, u_feat, l_feat, l_y, k=20, tau=0.1, eps_t=1e-4, eps_a=1e-4):
    mc = mc.detach()
    mu_t = mc.mean(0)
    v_t = mc.var(0, unbiased=False) + eps_t
    u = F.normalize(u_feat.detach().float(), dim=-1)
    l = F.normalize(l_feat.detach().float(), dim=-1)
    sim = u @ l.T
    kk = min(k, l.shape[0])
    topv, topi = sim.topk(kk, dim=1)
    w = torch.softmax(topv / tau, dim=1)
    nb = l_y.detach()[topi]
    mu_a = (w.unsqueeze(-1) * nb).sum(1)
    v_a = (w.unsqueeze(-1) * (nb - mu_a.unsqueeze(1)).square()).sum(1) + eps_a
    conf = topv[:, 0].clamp(0.0, 1.0)
    return mu_t, v_t, mu_a, v_a, conf


def build_objective(mode, ev, v0=1e-3, alpha=0.4, c0=0.3, gcf_tau=0.05,
                    alpha_fixed=0.75, shift=0.5):
    mu_t, v_t, mu_a, v_a, conf = ev
    if mode == 'multi':
        target = mu_t
        rel = v0 / (v0 + v_t)
    elif mode == 'anchor':
        target = mu_a
        rel = v0 / (v0 + v_a)
    elif mode == 'seq':
        gate = ((conf - c0) / max(1e-6, 1.0 - c0)).clamp(0.0, 1.0)
        a = (alpha * gate).unsqueeze(-1)
        target = (1.0 - a) * mu_t + a * mu_a
        sigma = v_t.clamp_min(0.0).sqrt()
        rel = torch.exp(-sigma / gcf_tau)
    elif mode in ('jacf', 'no_rel'):
        a = v_t / (v_t + v_a)
        target = (1.0 - a) * mu_t + a * mu_a
        diff = (mu_t - mu_a).square() / (v_t + v_a)
        c = torch.exp(-0.5 * diff)
        v_f = v_t * v_a / (v_t + v_a)
        r = c * (v0 / (v0 + v_f))
        rel = torch.ones_like(r) if mode == 'no_rel' else r
    elif mode == 'fixed':
        target = (1.0 - alpha_fixed) * mu_t + alpha_fixed * mu_a
        rel = torch.ones_like(v_t)
    elif mode == 'shift':
        a = (v_t / (v_t + v_a) + shift).clamp(0.0, 1.0)
        target = (1.0 - a) * mu_t + a * mu_a
        rel = torch.ones_like(v_t)
    else:
        raise ValueError(f'unknown mode {mode}')
    return target.detach(), rel.detach()


def fixed_alpha_objective(ev, alpha=0.75, rel_mode='none', v0=1e-3, dimnorm=False, filter_q=0.7):
    """Fixed anchor-prior target with a pluggable reliability weight.

    ``rel_mode`` replaces the original JACF reliability ``r``:

    none        : uniform weights (baseline)
    anchor_var  : v0 / (v0 + v_a)          -- anchor uncertainty only
    prop_var    : v0 / (v0 + a^2 v_a + (1-a)^2 v_t)
                  -- variance of the *actual* fused target, not the
                     precision-weighted harmonic variance v_f
    max_var     : v0 / (v0 + max(v_t, v_a))
    jacf        : original support * v0/(v0+v_f)
    anchor_var_conf : v0 / (v0 + v_a) * conf
    """
    mu_t, v_t, mu_a, v_a, conf = ev
    target = (1.0 - alpha) * mu_t + alpha * mu_a
    if rel_mode == 'none':
        rel = torch.ones_like(v_t)
    elif rel_mode == 'anchor_var':
        rel = v0 / (v0 + v_a)
    elif rel_mode == 'prop_var':
        rel = v0 / (v0 + alpha ** 2 * v_a + (1.0 - alpha) ** 2 * v_t)
    elif rel_mode == 'max_var':
        rel = v0 / (v0 + torch.maximum(v_t, v_a))
    elif rel_mode == 'anchor_var_conf':
        rel = (v0 / (v0 + v_a)) * conf.unsqueeze(-1).clamp(0.0, 1.0)
    elif rel_mode == 'anchor_filter':
        thr = torch.quantile(v_a, filter_q, dim=0, keepdim=True)
        rel = (v_a <= thr).to(v_t.dtype)
    elif rel_mode == 'jacf':
        diff = (mu_t - mu_a).square() / (v_t + v_a)
        c = torch.exp(-0.5 * diff)
        v_f = v_t * v_a / (v_t + v_a)
        rel = c * (v0 / (v0 + v_f))
    else:
        raise ValueError(f'unknown rel_mode {rel_mode}')
    if dimnorm:
        # keep each VAD dimension's mean weight at 1 so the ablation only
        # changes *within-dimension* sample ranking, not the VAD balance
        rel = rel / rel.mean(dim=0, keepdim=True).clamp_min(1e-8)
    return target.detach(), rel.detach()
