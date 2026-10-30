#!/usr/bin/env python3
"""TRAIN-internal reliability repair for JACF.

Fit 320 : reference model training + anchor bank
meta 80 : out-of-sample pseudo-labels + true errors -> calibrate error predictor
probe100: held-out evaluation of reliability scores Q0-Q3 (never used for fitting)

Pseudo-label fusion alpha/target are unchanged; only r is replaced by a
calibrated error estimate e_hat:
    r' = e0_d / (e0_d + e_hat),  e_hat = softplus(b_d + w^T [log vT, log vA, log(1+D)])
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS, read_gold  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402
from scripts.vad_research.jacf_modes import evidence, build_objective  # noqa: E402


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


def features(ev):
    mu_t, v_t, mu_a, v_a, _ = ev
    d = (mu_t - mu_a).square() / (v_t + v_a)
    return torch.stack([torch.log(v_t + 1e-8), torch.log(v_a + 1e-8),
                        torch.log1p(d)], dim=-1)  # [N,3,3]


def predictor(X, w, b):
    return F.softplus(X @ w + b)


def fit_predictor(X, e, lam, iters=800, lr=0.05):
    w = torch.zeros(3, device=X.device, requires_grad=True)
    b = torch.zeros(3, device=X.device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = F.mse_loss(predictor(X, w, b), e) + lam * (w.square().sum() + b.square().sum())
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def cv_lambda(X, e, grid, folds=5, seed=0):
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    best = (None, float('inf'))
    for lam in grid:
        errs = []
        for f in range(folds):
            val = perm[f::folds]
            tr = torch.tensor([i for i in range(n) if i not in set(val.tolist())], device=X.device)
            w, b = fit_predictor(X[tr], e[tr], lam)
            errs.append(float(F.mse_loss(predictor(X[val], w, b), e[val])))
        m = float(np.mean(errs))
        if m < best[1]:
            best = (lam, m)
    return best


def dim_metrics(r, e):
    if torch.is_tensor(r):
        r = r.detach().cpu().numpy()
    if torch.is_tensor(e):
        e = e.detach().cpu().numpy()
    rho = float(spearmanr(r, e).statistic) if np.std(r) > 1e-12 else float('nan')
    order = np.argsort(-r)
    se = e[order]
    cum = np.cumsum(se) / np.arange(1, len(se) + 1)
    aurc = float(cum.mean())
    q = np.quantile(r, [0.25, 0.5, 0.75])
    groups = []
    for lo, hi in ((-1e9, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 1e9)):
        sel = (r > lo) & (r <= hi)
        groups.append(float(e[sel].mean()) if sel.any() else float('nan'))
    hi_r = r >= q[2]; hi_e = e >= np.quantile(e, 0.75)
    frac = float((hi_r & hi_e).sum() / max(1, hi_e.sum()))
    return {'spearman': rho, 'aurc': aurc, 'quartile_mean_error': groups,
            'high_r_high_error_fraction': frac}


def _nanmean(values):
    vals = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float('nan')


def macro_metrics(scores, errors):
    per = {d: dim_metrics(scores[:, i], errors[:, i]) for i, d in enumerate(('V', 'A', 'D'))}
    macro = {'spearman': _nanmean([per[d]['spearman'] for d in per]),
             'aurc': float(np.mean([per[d]['aurc'] for d in per])),
             'quartile_mean_error': [_nanmean([per[d]['quartile_mean_error'][j] for d in per]) for j in range(4)],
             'high_r_high_error_fraction': float(np.mean([per[d]['high_r_high_error_fraction'] for d in per]))}
    return macro, per


def boot_compare(scores_a, scores_b, errors, n, seed):
    """a = Q3, b = Q1; positive Spearman diff / negative AURC diff means a better."""
    rng = np.random.default_rng(seed)
    scores_a = scores_a.detach().cpu().numpy() if torch.is_tensor(scores_a) else np.asarray(scores_a)
    scores_b = scores_b.detach().cpu().numpy() if torch.is_tensor(scores_b) else np.asarray(scores_b)
    errors = errors.detach().cpu().numpy() if torch.is_tensor(errors) else np.asarray(errors)
    rho_d, aurc_d = [], []
    for _ in range(n):
        idx = rng.integers(0, errors.shape[0], errors.shape[0])
        e = errors[idx]
        ra = scores_a[idx]; rb = scores_b[idx]
        rho_a = np.nanmean([spearmanr(ra[:, i], e[:, i]).statistic for i in range(3)])
        rho_b = np.nanmean([spearmanr(rb[:, i], e[:, i]).statistic for i in range(3)])
        rho_d.append(rho_a - rho_b)
        aa = np.mean([dim_metrics(ra[:, i], e[:, i])['aurc'] for i in range(3)])
        ab = np.mean([dim_metrics(rb[:, i], e[:, i])['aurc'] for i in range(3)])
        aurc_d.append(aa - ab)
    return {'spearman_diff_ci': [float(np.percentile(rho_d, 2.5)), float(np.percentile(rho_d, 97.5))],
            'aurc_diff_ci': [float(np.percentile(aurc_d, 2.5)), float(np.percentile(aurc_d, 97.5))]}


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
    ev_meta = evidence(mc_meta, f_meta, f_fit, y_fit, k=a.k, tau=a.tau)
    ev_probe = evidence(mc_probe, f_probe, f_fit, y_fit, k=a.k, tau=a.tau)

    tgt_meta, _ = build_objective('jacf', ev_meta)
    tgt_probe, r_orig = build_objective('jacf', ev_probe)
    e_meta = (tgt_meta - y_meta).abs()
    e_probe = (tgt_probe - y_probe).abs()

    Xm, Xp = features(ev_meta), features(ev_probe)
    grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    lam, cv_mse = cv_lambda(Xm, e_meta, grid, seed=a.seed)
    w, b = fit_predictor(Xm, e_meta, lam)
    e_hat = predictor(Xp, w, b)
    e0 = e_meta.median(0).values
    r_new = e0 / (e0 + e_hat)

    v0 = 1e-3
    _, v_t, _, v_a, _ = ev_probe
    scores = {
        'Q0_uniform': torch.ones_like(r_new),
        'Q1_original': r_orig,
        'Q2_teacher': v0 / (v0 + v_t),
        'Q2_anchor': v0 / (v0 + v_a),
        'Q3_calibrated': r_new,
    }
    results = {}
    for name, s in scores.items():
        macro, per = macro_metrics(s, e_probe)
        results[name] = {'macro': macro, 'per_dim': per}

    boot = {
        'Q3_vs_Q1': boot_compare(r_new, r_orig, e_probe, a.bootstrap, a.seed),
        'Q3_vs_Q2teacher': boot_compare(r_new, scores['Q2_teacher'], e_probe, a.bootstrap, a.seed),
    }
    out = {'protocol': 'TRAIN-only 320/80/100; anchor bank = fit 320; no DEV/TEST',
           'k': a.k, 'tau': a.tau, 'mc_passes': a.mc_passes,
           'calibration': {'lambda': lam, 'cv_mse': cv_mse,
                           'w': [float(x) for x in w], 'b': [float(x) for x in b],
                           'e0': [float(x) for x in e0]},
           'results': results, 'bootstrap': boot,
           'probe_error_mean': float(e_probe.mean()),
           'meta_error_mean': float(e_meta.mean())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    for name in scores:
        m = results[name]['macro']
        print(f"{name}: Spearman={m['spearman']:.4f} AURC={m['aurc']:.6f} "
              f"hiR-hiE={m['high_r_high_error_fraction']:.3f} "
              f"quartiles={[round(x,4) for x in m['quartile_mean_error']]}")
    print('calibration:', json.dumps(out['calibration']))
    print('bootstrap:', json.dumps(boot, indent=2))


if __name__ == '__main__':
    main()
