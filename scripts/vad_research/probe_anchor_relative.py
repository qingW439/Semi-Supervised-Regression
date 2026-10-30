#!/usr/bin/env python3
"""P1 ARCL admission probe: do local prediction residuals transfer?

Strict TRAIN-internal split from the archived 320/80/100 manifest:
  fit 320  -> reference model training (already done)
  meta 80  -> labeled anchor bank + teacher residuals
  probe100 -> pseudo-label quality evaluation (labels never used for fitting)

Compares on the 100 probe samples (no VRRC, no DEV/TEST):
  H0 teacher raw (deterministic eval)
  H1 JACF fusion (multi-view + anchor)
  H2 teacher + global mean residual
  H3 teacher + top-k local residual   (ARCL hypothesis)
  H4 teacher + permuted local residual (control)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import DIMENSIONS, NORM_COLUMNS, read_gold  # noqa: E402
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402
from scripts.vad_research.jacf import jacf_fuse  # noqa: E402


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


@torch.no_grad()
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
    model.train()  # dropout on
    preds = []
    for p in range(passes):
        chunks = []
        for i in range(0, len(texts), bs):
            b = tok(list(texts[i:i + bs]), padding=True, truncation=True,
                    max_length=128, return_tensors='pt')
            b = {k: v.to(device) for k, v in b.items()}
            chunks.append(model(**b).cpu())
        preds.append(torch.cat(chunks))
    model.eval()
    return torch.stack(preds).to(device)  # [K,N,3]


@torch.no_grad()
def predict_eval(model, texts, tok, device, bs=32):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True,
                max_length=128, return_tensors='pt')
        b = {k: v.to(device) for k, v in b.items()}
        out.append(model(**b).cpu())
    return torch.cat(out).to(device)


def report(y_norm, p_norm):
    m = metric_report(y_norm.cpu().numpy(), p_norm.cpu().numpy())
    macro = {k: float(np.mean([m[d][k] for d in DIMENSIONS]))
             for k in ('mae', 'rmse', 'pearson', 'srcc', 'ccc', 'r2')}
    return m, macro


def boot_macro(y, pa, pb, n, seed):
    """positive value => pb lower MAE than pa (pb better)."""
    rng = np.random.default_rng(seed)
    y = y.cpu().numpy(); pa = pa.cpu().numpy(); pb = pb.cpu().numpy()
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        ma = np.abs(y[idx] - pa[idx]).mean()
        mb = np.abs(y[idx] - pb[idx]).mean()
        out.append(ma - mb)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    a = cli()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    man = json.loads(a.manifest.read_text(encoding='utf-8'))
    split = man['split']
    lb = read_gold(a.data_root / 'lb500_s0' / 'train_lb.parquet', expected_n=500)
    by_id = lb.set_index('id')
    fit_ids = split['fit_320_ids']; meta_ids = split['meta_80_ids']; probe_ids = split['probe_100_ids']
    for name, ids in (('fit', fit_ids), ('meta', meta_ids), ('probe', probe_ids)):
        if len(set(ids) & set(lb['id'])) != len(ids):
            raise ValueError(f'{name} ids not in train_lb')
    meta = by_id.loc[meta_ids].reset_index()
    probe = by_id.loc[probe_ids].reset_index()

    d = torch.load(a.ckpt, map_location='cpu', weights_only=False, mmap=True)
    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(device)
    model.load_state_dict(d['model'], strict=True)
    meta_info = {'train_steps': d.get('train_steps'), 'git_commit': d.get('git_commit'),
                 'fit_id_sha256': d.get('fit_id_sha256')}
    del d
    gc.collect()
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)

    l_feat = encode_eval(model.encoder, meta['text'].tolist(), tok, device)
    p_feat = encode_eval(model.encoder, probe['text'].tolist(), tok, device)
    y_meta = torch.tensor(meta.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=device)
    y_probe = torch.tensor(probe.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=device)

    mc_meta = mc_predict(model, meta['text'].tolist(), tok, device, a.mc_passes)
    mc_probe = mc_predict(model, probe['text'].tolist(), tok, device, a.mc_passes)
    mu_meta = mc_meta.mean(0)
    mu_probe = mc_probe.mean(0)
    p0 = predict_eval(model, probe['text'].tolist(), tok, device)

    sim = F.normalize(p_feat, dim=-1) @ F.normalize(l_feat, dim=-1).T
    kk = min(a.k, l_feat.shape[0])
    topv, topi = sim.topk(kk, dim=1)
    w = torch.softmax(topv / a.tau, dim=1)

    res_meta = y_meta - mu_meta
    h2 = mu_probe + res_meta.mean(0, keepdim=True)
    h3 = mu_probe + (w.unsqueeze(-1) * res_meta[topi]).sum(1)
    g = torch.Generator(device=device).manual_seed(12345)
    perm = torch.randperm(len(meta), generator=g, device=device)
    h4 = mu_probe + (w.unsqueeze(-1) * res_meta[perm][topi]).sum(1)
    h1, _, _ = jacf_fuse(mc_probe, p_feat, l_feat, y_meta, k=a.k, tau=a.tau)

    preds = {'H0': p0, 'H1_JACF': h1, 'H2_global': h2, 'H3_local': h3, 'H4_perm': h4}
    results = {}
    for name, pred in preds.items():
        m, macro = report(y_probe, pred)
        results[name] = {'macro': macro, 'per_dim': m}
    pairs = [('H3_local', 'H0'), ('H3_local', 'H1_JACF'),
             ('H3_local', 'H2_global'), ('H3_local', 'H4_perm')]
    boot = {}
    ynp = y_probe.cpu().numpy()
    for x, y in pairs:
        lo, hi = boot_macro(y_probe, preds[y], preds[x], a.bootstrap, a.seed)
        gain_norm = float(np.abs(ynp - preds[y].cpu().numpy()).mean()
                          - np.abs(ynp - preds[x].cpu().numpy()).mean())
        boot[f'{x}_minus_{y}'] = {
            'mae_gain_normalized': gain_norm,
            'mae_gain_original': gain_norm * 4.0,
            'ci95_normalized': [lo, hi],
            'ci95_original': [lo * 4.0, hi * 4.0],
        }

    # residual-similarity structure on meta
    rng = np.random.default_rng(0)
    n = len(meta)
    ii = rng.integers(0, n, 3000); jj = rng.integers(0, n, 3000)
    keep = ii != jj
    fsim = (F.normalize(l_feat, dim=-1)[ii[keep]] * F.normalize(l_feat, dim=-1)[jj[keep]]).sum(-1)
    rsim = (F.normalize(res_meta, dim=-1)[ii[keep]] * F.normalize(res_meta, dim=-1)[jj[keep]]).sum(-1)
    rho = float(spearmanr(fsim.cpu().numpy(), rsim.cpu().numpy()).statistic)

    out = {'protocol': 'TRAIN-only 320/80/100; no DEV/TEST', 'meta': meta_info,
           'manifest_source_sha256': man.get('source_sha256'),
           'k': a.k, 'tau': a.tau, 'mc_passes': a.mc_passes,
           'results': results, 'bootstrap': boot,
           'meta_residual_similarity_spearman': rho}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    for name, r in results.items():
        print(f"{name}: MAE={r['macro']['mae']:.6f} CCC={r['macro']['ccc']:.4f} "
              f"R2={r['macro']['r2']:.4f} Pearson={r['macro']['pearson']:.4f} "
              f"SRCC={r['macro']['srcc']:.4f}")
    print('bootstrap:', json.dumps(boot, indent=2))
    print('residual similarity spearman:', rho)


if __name__ == '__main__':
    main()
