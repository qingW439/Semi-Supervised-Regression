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


def build_objective(mode, ev, v0=1e-3, alpha=0.4, c0=0.3, gcf_tau=0.05):
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
    else:
        raise ValueError(f'unknown mode {mode}')
    return target.detach(), rel.detach()
