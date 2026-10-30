#!/usr/bin/env python3
"""JACF dual-source attribution: when is teacher better than anchor, and does
the current fusion alpha systematically favour the wrong source?

Strict TRAIN-only 320/80/100 protocol; anchor bank = fit 320; no DEV/TEST.
Meta is used to describe/choose rules; probe is evaluated once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS, read_gold  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402
from scripts.vad_research.jacf_modes import evidence  # noqa: E402


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--pretrain-path', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--tau', type=float, default=0.1)
    p.add_argument('--mc-passes', type=int, default=5)
    p.add_argument('--bootstrap', type=int, default=1000)
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
def mc_predict(model, texts, tok, device, passes, bs=32):
    model.train()
    preds = []
    for _ in range(passes):
        chunks = []
        for i in range(0, len(texts), bs):
            b = tok(list(texts[i:i + bs]), padding=True, truncation=True,
                    max_length=128, return_tensors='pt')
            b = {k: v.to(device) for k, v in b.items()}
            chunks.append(model(**b).cpu())
        preds.append(torch.cat(chunks))
    model.eval()
    return torch.stack(preds).to(device)


def collect(ev, y):
    mu_t, v_t, mu_a, v_a, conf = ev
    alpha = v_t / (v_t + v_a)
    yt = (1.0 - alpha) * mu_t + alpha * mu_a
    e_t = (mu_t - y).abs()
    e_a = (mu_a - y).abs()
    e_f = (yt - y).abs()
    delta = e_t - e_a
    denom = (mu_a - mu_t)
    a_star = torch.where(denom.abs() > 1e-6,
                         ((y - mu_t) / denom).clamp(0.0, 1.0), alpha)
    conflict = (mu_t - mu_a).square() / (v_t + v_a)
    return {'mu_t': mu_t, 'mu_a': mu_a, 'alpha': alpha, 'y_tilde': yt,
            'e_t': e_t, 'e_a': e_a, 'e_f': e_f, 'delta': delta,
            'a_star': a_star, 'conflict': conflict, 'conf': conf,
            'v_t': v_t, 'v_a': v_a}


def auc(score, label):
    score = np.asarray(score).ravel(); label = np.asarray(label).ravel().astype(bool)
    n1 = label.sum(); n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def macro(x):
    return float(x.mean())


def summarize(name, s, y):
    out = {'mae_teacher': macro(s['e_t']), 'mae_anchor': macro(s['e_a']),
           'mae_fused': macro(s['e_f']),
           'mean_delta_teacher_minus_anchor': macro(s['delta']),
           'frac_anchor_better': float((s['delta'] > 0).float().mean()),
           'mean_alpha': macro(s['alpha']), 'mean_alpha_star': macro(s['a_star']),
           'mean_alpha_star_minus_alpha': macro(s['a_star'] - s['alpha']),
           'fusion_worse_than_best_source': float(
               (s['e_f'] > torch.minimum(s['e_t'], s['e_a']) + 1e-9).float().mean())}
    per = {}
    for i, d in enumerate(('V', 'A', 'D')):
        per[d] = {'mae_teacher': float(s['e_t'][:, i].mean()),
                  'mae_anchor': float(s['e_a'][:, i].mean()),
                  'mae_fused': float(s['e_f'][:, i].mean()),
                  'frac_anchor_better': float((s['delta'][:, i] > 0).float().mean()),
                  'mean_alpha': float(s['alpha'][:, i].mean()),
                  'mean_alpha_star': float(s['a_star'][:, i].mean())}
    out['per_dim'] = per
    return out


def quartile_table(s):
    a = s['alpha'].flatten()
    q = torch.quantile(a, torch.tensor([0.25, 0.5, 0.75], device=a.device))
    rows = []
    for lo, hi in ((-1e9, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 1e9)):
        sel = ((s['alpha'] > lo) & (s['alpha'] <= hi))
        rows.append({'alpha_range': [float(lo), float(hi)],
                     'count': int(sel.sum()),
                     'mean_delta': float(s['delta'][sel].mean()),
                     'frac_anchor_better': float((s['delta'][sel] > 0).float().mean()),
                     'mean_alpha': float(s['alpha'][sel].mean()),
                     'mean_alpha_star': float(s['a_star'][sel].mean())})
    return rows


def signals(s):
    return {'alpha': s['alpha'], 'conflict': s['conflict'],
            'conf': s['conf'].unsqueeze(-1).expand(-1, s['delta'].shape[1]),
            'v_t': s['v_t'], 'v_a': s['v_a']}


def signal_tables(s):
    out = {}
    for name, x in signals(s).items():
        out[name] = {
            'spearman_with_delta': float(spearmanr(x.flatten().cpu().numpy(),
                                                   s['delta'].flatten().cpu().numpy()).statistic),
            'auc_anchor_better': auc(x.cpu().numpy(), (s['delta'] > 0).cpu().numpy()),
        }
    return out


def best_fixed_alpha(meta, probe):
    best = (None, float('inf'))
    for a in np.arange(0.0, 1.0001, 0.05):
        yt = (1 - a) * meta['mu_t'] + a * meta['mu_a']
        mae = float((yt - meta['y']).abs().mean())
        if mae < best[1]:
            best = (float(a), mae)
    a = best[0]
    yt = (1 - a) * probe['mu_t'] + a * probe['mu_a']
    return {'meta_best_alpha': a, 'meta_best_mae': best[1],
            'probe_mae_at_meta_alpha': float((yt - probe['y']).abs().mean()),
            'meta_current_mae': float(meta['e_f'].mean()),
            'probe_current_mae': float(probe['e_f'].mean())}


def main():
    a = cli()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    man = json.loads(a.manifest.read_text())
    sp = man['split']
    lb = read_gold(a.data_root / 'lb500_s0' / 'train_lb.parquet', expected_n=500).set_index('id')
    fit = lb.loc[sp['fit_320_ids']].reset_index()
    meta = lb.loc[sp['meta_80_ids']].reset_index()
    probe = lb.loc[sp['probe_100_ids']].reset_index()

    d = torch.load(a.ckpt, map_location='cpu', weights_only=False, mmap=True)
    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(dev)
    model.load_state_dict(d['model'], strict=True)
    del d
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)

    f_fit = encode_eval(model.encoder, fit['text'].tolist(), tok, dev)
    f_meta = encode_eval(model.encoder, meta['text'].tolist(), tok, dev)
    f_probe = encode_eval(model.encoder, probe['text'].tolist(), tok, dev)
    y_fit = torch.tensor(fit.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=dev)
    y_meta = torch.tensor(meta.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=dev)
    y_probe = torch.tensor(probe.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=dev)

    mc_meta = mc_predict(model, meta['text'].tolist(), tok, dev, a.mc_passes)
    mc_probe = mc_predict(model, probe['text'].tolist(), tok, dev, a.mc_passes)
    s_meta = collect(evidence(mc_meta, f_meta, f_fit, y_fit, k=a.k, tau=a.tau), y_meta)
    s_probe = collect(evidence(mc_probe, f_probe, f_fit, y_fit, k=a.k, tau=a.tau), y_probe)
    s_meta['y'] = y_meta
    s_probe['y'] = y_probe

    meta_sum = summarize('meta', s_meta, y_meta)
    probe_sum = summarize('probe', s_probe, y_probe)
    fixed = best_fixed_alpha(s_meta, s_probe)
    sig_meta = signal_tables(s_meta)
    sig_probe = signal_tables(s_probe)
    q_meta = quartile_table(s_meta)
    q_probe = quartile_table(s_probe)

    # paired bootstrap on probe (by text) for delta and fusion gap
    rng = np.random.default_rng(a.seed)
    dlt, gap = [], []
    for _ in range(a.bootstrap):
        idx = rng.integers(0, len(probe), len(probe))
        dlt.append(float(s_probe['delta'][idx].mean()))
        best = torch.minimum(s_probe['e_t'][idx], s_probe['e_a'][idx])
        gap.append(float((s_probe['e_f'][idx] - best).mean()))
    boot = {'delta_mean_ci': [float(np.percentile(dlt, 2.5)), float(np.percentile(dlt, 97.5))],
            'fusion_gap_ci': [float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))]}

    # candidate rule: signal with AUC>=0.6 on both meta and probe (same direction)
    candidates = []
    for name in sig_meta:
        am, ap = sig_meta[name]['auc_anchor_better'], sig_probe[name]['auc_anchor_better']
        if am >= 0.6 and ap >= 0.6:
            candidates.append({'signal': name, 'auc_meta': am, 'auc_probe': ap})

    out = {'protocol': 'TRAIN-only 320/80/100; anchor bank=fit 320',
           'meta': meta_sum, 'probe': probe_sum,
           'meta_quartiles_by_alpha': q_meta, 'probe_quartiles_by_alpha': q_probe,
           'signals_meta': sig_meta, 'signals_probe': sig_probe,
           'best_fixed_alpha': fixed, 'bootstrap_probe': boot,
           'candidate_signals': candidates}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'meta': meta_sum, 'probe': probe_sum,
                      'best_fixed_alpha': fixed, 'bootstrap_probe': boot,
                      'candidates': candidates}, indent=2))
    print('== signals meta ==')
    for k, v in sig_meta.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()})
    print('== signals probe ==')
    for k, v in sig_probe.items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()})


if __name__ == '__main__':
    main()
