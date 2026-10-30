#!/usr/bin/env python3
"""JACF gradient-reliability audit.

Fixed alpha=0.75 target, rel=1. Measures full-parameter gradient conflict
between supervised L1 and unsupervised L1, and compares one plain AdamW step
with one PCGrad-projected AdamW step on a held-out TRAIN split.

No DEV/TEST. No formal training.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS, read_gold, read_unlabeled  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402
from scripts.vad_research.jacf_modes import evidence, build_objective  # noqa: E402
from scripts.vad_research.train_acvr import mask_tokens  # noqa: E402


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--pretrain-path', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--tau', type=float, default=0.1)
    p.add_argument('--mc-passes', type=int, default=5)
    p.add_argument('--alpha-fixed', type=float, default=0.75)
    p.add_argument('--mask-prob', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


@torch.inference_mode()
def encode_eval(encoder, texts, tok, device, bs=32):
    encoder.eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True,
                max_length=128, return_tensors='pt')
        b = {k: v.to(device) for k, v in b.items()}
        st = encoder(**b, return_dict=True).last_hidden_state
        m = b['attention_mask'].unsqueeze(-1).to(dtype=st.dtype)
        out.append(F.normalize((st * m).sum(1) / m.sum(1).clamp_min(1), dim=-1).cpu())
    return torch.cat(out).to(device)


@torch.no_grad()
def mc_predict(model, tokens, passes):
    model.train()
    preds = [model(**tokens).detach() for _ in range(passes)]
    model.eval()
    return torch.stack(preds)


@torch.no_grad()
def eval_mae(model, tokens, y):
    model.eval()
    pred = model(**tokens)
    return float(F.l1_loss(pred, y))


def grads(loss, params):
    return torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)


def flat(gs):
    return torch.cat([g.reshape(-1) for g in gs if g is not None])


def project_conflict(g_sup, g_unsup, eps=1e-12):
    out = []
    for a, b in zip(g_sup, g_unsup):
        if a is None or b is None:
            out.append(b)
            continue
        dot = (a * b).sum()
        if dot < 0:
            b = b - dot / (a.square().sum() + eps) * a
        out.append(b)
    return out


def tensor_neg_fraction(gs, gu):
    neg = 0; total = 0
    for a, b in zip(gs, gu):
        if a is None or b is None:
            continue
        total += 1
        neg += int((a * b).sum() < 0)
    return neg / max(1, total)


def set_grads(params, gs):
    for p, g in zip(params, gs):
        p.grad = None if g is None else g.detach().clone()


def flat_params(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def main():
    a = cli()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    lb = read_gold(a.data_root / 'lb500_s0' / 'train_lb.parquet', expected_n=500).reset_index(drop=True)
    ulb = read_unlabeled(a.data_root / 'lb500_s0' / 'train_ulb.parquet',
                         labeled_ids=set(lb['id'])).reset_index(drop=True)
    perm = torch.randperm(500, generator=torch.Generator().manual_seed(7))
    tr_idx = perm[:400].tolist(); ev_idx = perm[400:].tolist()
    tr = lb.iloc[tr_idx].reset_index(drop=True); ev = lb.iloc[ev_idx].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    ckpt = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    base = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(dev)
    base.load_state_dict(ckpt['model'], strict=True)
    base.eval()
    del ckpt

    ev_tok = tok(ev['text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
    ev_tok = {k: v.to(dev) for k, v in ev_tok.items()}
    y_ev = torch.tensor(ev.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=dev)
    y_tr = torch.tensor(tr.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=dev)
    y_ulb = None
    f_bank = encode_eval(base.encoder, tr['text'].tolist(), tok, dev)
    f_ulb = encode_eval(base.encoder, ulb['text'].tolist(), tok, dev)

    trials = []
    for t in range(a.trials):
        torch.manual_seed(1000 + t)
        g = torch.Generator().manual_seed(2000 + t)
        bi = torch.randperm(len(tr), generator=g)[:16].tolist()
        ui = torch.randperm(len(ulb), generator=g)[:64].tolist()
        lb_tok = tok(tr.iloc[bi]['text'].tolist(), padding=True, truncation=True,
                     max_length=128, return_tensors='pt')
        lb_tok = {k: v.to(dev) for k, v in lb_tok.items()}
        y_lb = y_tr[bi]
        ulb_tok = tok(ulb.iloc[ui]['text'].tolist(), padding=True, truncation=True,
                      max_length=128, return_tensors='pt')
        ulb_tok = {k: v.to(dev) for k, v in ulb_tok.items()}
        ulb_strong = mask_tokens(ulb_tok, tok, a.mask_prob)

        model = copy.deepcopy(base).to(dev)
        model.train()
        teacher = copy.deepcopy(base).to(dev); teacher.eval()
        mc = mc_predict(teacher, ulb_tok, a.mc_passes)
        evd = evidence(mc, f_ulb[ui], f_bank, y_tr, k=a.k, tau=a.tau)
        target, rel = build_objective('fixed', evd, alpha_fixed=a.alpha_fixed)

        pred_lb = model(**lb_tok)
        pred_u = model(**ulb_strong)
        sup = F.l1_loss(pred_lb, y_lb)
        unsup = (rel * (pred_u - target).abs()).mean()

        params = [p for p in model.parameters() if p.requires_grad]
        g_sup = grads(sup, params)
        g_unsup = grads(unsup, params)
        fs, fu = flat(g_sup), flat(g_unsup)
        cos = float(F.cosine_similarity(fs, fu, dim=0))
        dot = float((fs * fu).sum())
        norm_s, norm_u = float(fs.norm()), float(fu.norm())
        neg_frac = tensor_neg_fraction(g_sup, g_unsup)

        per_dim = {}
        for d, name in enumerate(('V', 'A', 'D')):
            ls = F.l1_loss(pred_lb[:, d], y_lb[:, d])
            lu = (rel[:, d] * (pred_u[:, d] - target[:, d]).abs()).mean()
            gs = grads(ls, params); gu = grads(lu, params)
            per_dim[name] = {
                'cosine': float(F.cosine_similarity(flat(gs), flat(gu), dim=0)),
                'norm_ratio_sup_over_unsup': float(flat(gs).norm() / flat(gu).norm().clamp_min(1e-12)),
            }

        mae_before = eval_mae(model, ev_tok, y_ev)
        g_total = [None if x is None else x + y for x, y in zip(g_sup, g_unsup)]

        plain = copy.deepcopy(model).to(dev)
        opt_p = torch.optim.AdamW(plain.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        set_grads([p for p in plain.parameters() if p.requires_grad], g_total)
        p_before = flat_params(list(plain.parameters())).clone()
        opt_p.step(); opt_p.zero_grad(set_to_none=True)
        mae_plain = eval_mae(plain, ev_tok, y_ev)
        p_after_plain = flat_params(list(plain.parameters())).clone()

        pc = copy.deepcopy(model).to(dev)
        opt_c = torch.optim.AdamW(pc.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        g_proj = project_conflict(g_sup, g_unsup)
        g_total_c = [None if x is None else x + y for x, y in zip(g_sup, g_proj)]
        set_grads([p for p in pc.parameters() if p.requires_grad], g_total_c)
        opt_c.step(); opt_c.zero_grad(set_to_none=True)
        mae_pc = eval_mae(pc, ev_tok, y_ev)
        p_after_pc = flat_params(list(pc.parameters())).clone()

        up = p_after_plain - p_before
        uc = p_after_pc - p_before
        trials.append({
            'trial': t,
            'cosine_sup_unsup': cos, 'dot': dot,
            'neg_tensor_fraction': neg_frac,
            'norm_sup': norm_s, 'norm_unsup': norm_u,
            'norm_ratio': norm_s / max(norm_u, 1e-12),
            'per_dim': per_dim,
            'mae_before': mae_before, 'mae_plain': mae_plain, 'mae_pcgrad': mae_pc,
            'delta_plain': mae_plain - mae_before, 'delta_pcgrad': mae_pc - mae_before,
            'update_norm_plain': float(up.norm()), 'update_norm_pcgrad': float(uc.norm()),
            'update_cosine': float(F.cosine_similarity(up, uc, dim=0)),
        })
        print(json.dumps({k: v for k, v in trials[-1].items() if k != 'per_dim'}, indent=None), flush=True)

    def agg(key):
        vals = [t[key] for t in trials]
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                'values': [float(v) for v in vals]}
    summary = {k: agg(k) for k in ('cosine_sup_unsup', 'neg_tensor_fraction', 'norm_ratio',
                                   'delta_plain', 'delta_pcgrad', 'update_cosine')}
    out = {'protocol': 'TRAIN-only; fixed alpha=0.75; eval 100 held-out from this step',
           'alpha_fixed': a.alpha_fixed, 'trials': a.trials,
           'conflict': summary,
           'per_dim': {d: {'cosine_mean': float(np.mean([t['per_dim'][d]['cosine'] for t in trials])),
                           'norm_ratio_mean': float(np.mean([t['per_dim'][d]['norm_ratio_sup_over_unsup'] for t in trials]))}
                       for d in ('V', 'A', 'D')},
           'plain': {'mae_before_mean': float(np.mean([t['mae_before'] for t in trials])),
                     'mae_after_mean': float(np.mean([t['mae_plain'] for t in trials])),
                     'delta_mean': float(np.mean([t['delta_plain'] for t in trials]))},
           'pcgrad': {'mae_after_mean': float(np.mean([t['mae_pcgrad'] for t in trials])),
                      'delta_mean': float(np.mean([t['delta_pcgrad'] for t in trials]))},
           'pcgrad_wins': int(sum(t['delta_pcgrad'] < t['delta_plain'] for t in trials)),
           'raw': trials}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('conflict', 'per_dim', 'plain', 'pcgrad', 'pcgrad_wins')}, indent=2))


if __name__ == '__main__':
    main()
