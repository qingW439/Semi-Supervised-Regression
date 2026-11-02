#!/usr/bin/env python3
"""H0/H1/H2/H3 spectral complementarity pilot with a fixed kernel bandwidth.

H0: raw frozen RoBERTa + Ridge
H1: Nystroem spectral features learned on TRAIN-400 + Ridge
H2: Nystroem spectral features learned on TRAIN-400 + ULB-7562 + Ridge
H3: RBF random features of the same dimension (400), fitted on TRAIN-400 + Ridge

The RBF bandwidth is estimated ONCE on the TRAIN-400 fold and reused for
H1/H2/H3, so H2's gain cannot come from a different gamma.  Ridge alpha is
selected inside the 400 by 4-fold CV; the held-out 100 labels are not used.
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
from sklearn.kernel_approximation import RBFSampler
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
            "per_dim_mae": {d: m[d]["mae"] for d in DIMS},
            "per_dim_ccc": {d: m[d]["ccc"] for d in DIMS}}


def ridge_cv(X, y, seed):
    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    kf = KFold(4, shuffle=True, random_state=seed)
    best, best_a = np.inf, alphas[0]
    for alpha in alphas:
        errs = []
        for tr, va in kf.split(X):
            m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[tr], y[tr])
            errs.append(float(np.mean((m.predict(X[va]) - y[va]) ** 2)))
        e = float(np.mean(errs))
        if e < best:
            best, best_a = e, alpha
    m = make_pipeline(StandardScaler(), Ridge(alpha=best_a)).fit(X, y)
    return m, best_a


def median_gamma(X, rng):
    sub = X[rng.choice(len(X), size=min(1000, len(X)), replace=False)]
    d2 = pairwise_distances(sub, sub, squared=True)
    med = np.median(d2[d2 > 1e-12])
    return float(1.0 / max(2.0 * med, 1e-12))


def nystrom(Xfit, Xq_list, m, gamma, rng):
    if len(Xfit) > m:
        L = Xfit[rng.choice(len(Xfit), size=m, replace=False)]
    else:
        L = Xfit
    K = rbf_kernel(L, L, gamma=gamma)
    w, V = eigh(K)
    keep = w > max(1e-10, w.max() * 1e-10)
    w, V = w[keep], V[:, keep]
    T = V / np.sqrt(w)
    return rbf_kernel(Xfit, L, gamma=gamma) @ T, [rbf_kernel(Xq, L, gamma=gamma) @ T for Xq in Xq_list]


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
    del enc; torch.cuda.empty_cache()
    sc = StandardScaler().fit(feat_lb)
    feat_lb = sc.transform(feat_lb); feat_ulb = sc.transform(feat_ulb)
    rng = np.random.default_rng(20261102)

    pred = {k: np.zeros((500, 3)) for k in ("h0", "h1", "h2", "h3")}
    info = {k: [] for k in pred}
    for f in range(5):
        hold = man["fold_ids"][str(f)]
        train = [i for i in lb["id"] if i not in set(hold)]
        ti = lb.index[lb["id"].isin(train)].to_numpy(); hi = lb.index[lb["id"].isin(hold)].to_numpy()
        Xtr, Xho, ytr = feat_lb[ti], feat_lb[hi], y[ti]
        gamma = median_gamma(Xtr, rng)  # fixed for H1/H2/H3
        # H0
        m0, a0 = ridge_cv(Xtr, ytr, seed=900 + f); pred["h0"][hi] = m0.predict(Xho)
        info["h0"].append({"fold": f, "alpha": a0})
        # H1
        _P1, (Q1,) = nystrom(Xtr, [Xho], a.m, gamma, rng)
        P1 = _P1
        m1, a1 = ridge_cv(P1, ytr, seed=910 + f); pred["h1"][hi] = m1.predict(Q1)
        info["h1"].append({"fold": f, "alpha": a1, "gamma": gamma})
        # H2
        _Pall, (Ptr, Q2) = nystrom(np.concatenate([Xtr, feat_ulb], 0), [Xtr, Xho], a.m, gamma, rng)
        m2, a2 = ridge_cv(Ptr, ytr, seed=920 + f); pred["h2"][hi] = m2.predict(Q2)
        info["h2"].append({"fold": f, "alpha": a2, "gamma": gamma})
        # H3 random RBF features
        rf = RBFSampler(gamma=gamma, n_components=a.m, random_state=930 + f).fit(Xtr)
        m3, a3 = ridge_cv(rf.transform(Xtr), ytr, seed=930 + f); pred["h3"][hi] = m3.predict(rf.transform(Xho))
        info["h3"].append({"fold": f, "alpha": a3, "gamma": gamma})
        print(f"fold {f} done", flush=True)

    out = {f"{k}_{n}": summary(y, pred[k]) for k, n in
           (("h0", "raw_ridge"), ("h1", "spectral_labeled"), ("h2", "spectral_labeled_ulb"), ("h3", "random_rbf"))}
    out["delta_h2_minus_h0"] = {k: out["h2_spectral_labeled_ulb"][k] - out["h0_raw_ridge"][k] for k in ("mae", "ccc", "r2")}
    out["delta_h2_minus_h1"] = {k: out["h2_spectral_labeled_ulb"][k] - out["h1_spectral_labeled"][k] for k in ("mae", "ccc", "r2")}
    out["delta_h2_minus_h3"] = {k: out["h2_spectral_labeled_ulb"][k] - out["h3_random_rbf"][k] for k in ("mae", "ccc", "r2")}
    out["per_dim_delta_h2_h1"] = {d: out["h2_spectral_labeled_ulb"]["per_dim_mae"][d] - out["h1_spectral_labeled"]["per_dim_mae"][d] for d in DIMS}
    out["info"] = info
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    np.savez(a.out.with_suffix(".npz"), y=y, **pred)
    for k in ("h0_raw_ridge", "h1_spectral_labeled", "h2_spectral_labeled_ulb", "h3_random_rbf"):
        v = out[k]
        print(f'{k:28s} MAE={v["mae"]:.6f} CCC={v["ccc"]:.4f} R2={v["r2"]:.4f}')
    print("H2-H0", out["delta_h2_minus_h0"], "H2-H1", out["delta_h2_minus_h1"], "H2-H3", out["delta_h2_minus_h3"])


if __name__ == "__main__":
    main()
