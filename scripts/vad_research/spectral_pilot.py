#!/usr/bin/env python3
"""S0/S1/S2 spectral-feature pilot (ICML 2026 proxy-covariate idea, simplified).

S0: raw frozen RoBERTa features + Ridge
S1: Nystroem RBF spectral features learned on TRAIN-400 only + Ridge
S2: Nystroem RBF spectral features learned on TRAIN-400 + ULB-7562 + Ridge

All three use the same folds, labels, base representation, and inner-CV Ridge
selection.  The only difference between S1 and S2 is whether the 7562 unlabeled
texts participate in learning the feature map.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS, read_gold  # noqa: E402
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_acvr import encode  # noqa: E402

DIMS = ("V", "A", "D")


def summary(y, p):
    m = metric_report(y, p)
    return {"mae": m["macro_mae_original_scale"],
            "ccc": float(np.mean([m[d]["ccc"] for d in DIMS])),
            "r2": float(np.mean([m[d]["r2"] for d in DIMS])),
            "per_dim_mae": {d: m[d]["mae"] for d in DIMS}}


def ridge_cv(X, y, seed):
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    kf = KFold(n_splits=4, shuffle=True, random_state=seed)
    best, best_a = np.inf, alphas[0]
    for alpha in alphas:
        errs = []
        for tr, va in kf.split(X):
            m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            m.fit(X[tr], y[tr])
            errs.append(float(np.mean((m.predict(X[va]) - y[va]) ** 2)))
        e = float(np.mean(errs))
        if e < best:
            best, best_a = e, alpha
    m = make_pipeline(StandardScaler(), Ridge(alpha=best_a))
    m.fit(X, y)
    return m, best_a


def median_gamma(X, rng):
    n = len(X)
    sub = X[rng.choice(n, size=min(1000, n), replace=False)]
    d2 = pairwise_distances(sub, sub, squared=True)
    med = np.median(d2[d2 > 1e-12])
    return float(1.0 / max(2.0 * med, 1e-12))


def nystrom(Xfit, Xquery_list, m, gamma, rng):
    if len(Xfit) > m:
        landmarks = Xfit[rng.choice(len(Xfit), size=m, replace=False)]
    else:
        landmarks = Xfit
    Kmm = rbf_kernel(landmarks, landmarks, gamma=gamma)
    w, V = eigh(Kmm)
    keep = w > max(1e-10, w.max() * 1e-10)
    w, V = w[keep], V[:, keep]
    scale = 1.0 / np.sqrt(w)
    Phi_fit = rbf_kernel(Xfit, landmarks, gamma=gamma) @ (V * scale)
    outs = [rbf_kernel(Xq, landmarks, gamma=gamma) @ (V * scale) for Xq in Xquery_list]
    return Phi_fit, outs, int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--oof-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--m", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    lb = read_gold(a.data_root / "lb500_s0" / "train_lb.parquet", expected_n=500)
    ulb = pd.read_parquet(a.data_root / "lb500_s0" / "train_ulb.parquet")
    y = lb.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32)
    man = json.loads(a.manifest.read_text())
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    enc = AutoModel.from_pretrained(str(a.pretrain_path), local_files_only=True).to(device).eval()
    feat_lb = encode(enc, lb["text"].tolist(), tok, device).cpu().numpy()
    feat_ulb = encode(enc, ulb["text"].tolist(), tok, device).cpu().numpy()
    del enc
    torch.cuda.empty_cache()
    # standardize base features once on TRAIN-500 (unsupervised scaling; no labels)
    sc = StandardScaler().fit(feat_lb)
    feat_lb = sc.transform(feat_lb).astype(np.float64)
    feat_ulb = sc.transform(feat_ulb).astype(np.float64)

    pred = {k: np.zeros((500, 3)) for k in ("s0", "s1", "s2")}
    info = {"s0": [], "s1": [], "s2": []}
    rng = np.random.default_rng(20261102)
    for f in range(5):
        hold = man["fold_ids"][str(f)]
        train = [i for i in lb["id"] if i not in set(hold)]
        ti = lb.index[lb["id"].isin(train)].to_numpy()
        hi = lb.index[lb["id"].isin(hold)].to_numpy()
        Xtr, Xho, ytr = feat_lb[ti], feat_lb[hi], y[ti]
        # S0
        m0, a0 = ridge_cv(Xtr, ytr, seed=300 + f)
        pred["s0"][hi] = m0.predict(Xho)
        info["s0"].append({"fold": f, "alpha": a0})
        # S1: spectral on TRAIN-400 only
        g1 = median_gamma(Xtr, rng)
        P1, (Q1,), r1 = nystrom(Xtr, [Xho], a.m, g1, rng)
        m1, a1 = ridge_cv(P1, ytr, seed=400 + f)
        pred["s1"][hi] = m1.predict(Q1)
        info["s1"].append({"fold": f, "alpha": a1, "gamma": g1, "rank": r1})
        # S2: spectral on TRAIN-400 + ULB-7562
        Xall = np.concatenate([Xtr, feat_ulb], axis=0)
        g2 = median_gamma(Xall, rng)
        _Pall, (Ptr, Q2), r2 = nystrom(Xall, [Xtr, Xho], a.m, g2, rng)
        m2, a2 = ridge_cv(Ptr, ytr, seed=500 + f)
        pred["s2"][hi] = m2.predict(Q2)
        info["s2"].append({"fold": f, "alpha": a2, "gamma": g2, "rank": r2})
        print(f"fold {f} done", flush=True)

    out = {
        "s0_raw_ridge": summary(y, pred["s0"]),
        "s1_spectral_labeled": summary(y, pred["s1"]),
        "s2_spectral_labeled_plus_ulb": summary(y, pred["s2"]),
        "delta_s2_minus_s1": {k: summary(y, pred["s2"])[k] - summary(y, pred["s1"])[k]
                              for k in ("mae", "ccc", "r2")},
        "per_dim_delta_mae_s2_minus_s1": {
            d: summary(y, pred["s2"])["per_dim_mae"][d] - summary(y, pred["s1"])["per_dim_mae"][d]
            for d in DIMS},
        "info": info,
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    np.savez(a.out.with_suffix(".npz"), y=y, s0=pred["s0"], s1=pred["s1"], s2=pred["s2"])
    for k in ("s0_raw_ridge", "s1_spectral_labeled", "s2_spectral_labeled_plus_ulb"):
        v = out[k]
        print(f'{k:34s} MAE={v["mae"]:.6f} CCC={v["ccc"]:.4f} R2={v["r2"]:.4f} '
              f'A/D MAE={v["per_dim_mae"]["A"]:.4f}/{v["per_dim_mae"]["D"]:.4f}')
    print("delta S2-S1", out["delta_s2_minus_s1"], out["per_dim_delta_mae_s2_minus_s1"])


if __name__ == "__main__":
    main()
