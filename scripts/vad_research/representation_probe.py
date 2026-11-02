#!/usr/bin/env python3
"""R0-R4 readout probes on the fixed 5-fold OOF split.

R0 frozen pretrained RoBERTa + Ridge
R1 fold encoder + Ridge
R2 fold encoder + small MLP
R3 model raw
R4 model + VRRC

Ridge/MLP hyper-parameters are selected *inside* each 400-item training fold by
an inner 4-fold CV; the held-out 100 labels are never used for selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
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


def fit_ridge_cv(X, y, seed):
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
    return m, best_a, best


def fit_mlp_cv(X, y, seed):
    alphas = [1e-4, 1e-2, 1.0]
    kf = KFold(n_splits=4, shuffle=True, random_state=seed)
    best, best_a = np.inf, alphas[0]
    for alpha in alphas:
        errs = []
        for tr, va in kf.split(X):
            m = make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(128,), alpha=alpha, max_iter=400,
                early_stopping=True, n_iter_no_change=20, random_state=seed))
            m.fit(X[tr], y[tr])
            errs.append(float(np.mean((m.predict(X[va]) - y[va]) ** 2)))
        e = float(np.mean(errs))
        if e < best:
            best, best_a = e, alpha
    m = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(128,), alpha=best_a, max_iter=400,
        early_stopping=True, n_iter_no_change=20, random_state=seed))
    m.fit(X, y)
    return m, best_a, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--oof-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    lb = read_gold(a.data_root / "lb500_s0" / "train_lb.parquet", expected_n=500)
    y = lb.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32)
    man = json.loads(a.manifest.read_text())
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    enc = AutoModel.from_pretrained(str(a.pretrain_path), local_files_only=True).to(device).eval()
    feat_frozen = encode(enc, lb["text"].tolist(), tok, device).cpu().numpy()
    del enc
    torch.cuda.empty_cache()

    pred = {k: np.zeros((500, 3)) for k in ("r0", "r1", "r2")}
    chosen = {"r0": [], "r1": [], "r2": []}
    for f in range(5):
        hold = man["fold_ids"][str(f)]
        fdir = a.oof_dir / f"fold{f}"
        Xb_fold = np.load(fdir / "features_labeled.npy")
        Xh_fold = np.load(fdir / "features_holdout.npy")
        train = [i for i in lb["id"] if i not in set(hold)]
        ti = lb.index[lb["id"].isin(train)].to_numpy()
        hi = lb.index[lb["id"].isin(hold)].to_numpy()
        yb = y[ti]
        for key, Xb, Xh in (("r0", feat_frozen[ti], feat_frozen[hi]),
                            ("r1", Xb_fold, Xh_fold)):
            m, alpha, cv = fit_ridge_cv(Xb, yb, seed=100 + f)
            pred[key][hi] = m.predict(Xh)
            chosen[key].append({"fold": f, "alpha": alpha, "inner_mse": cv})
        m, alpha, cv = fit_mlp_cv(Xb_fold, yb, seed=200 + f)
        pred["r2"][hi] = m.predict(Xh_fold)
        chosen["r2"].append({"fold": f, "alpha": alpha, "inner_mse": cv})
        print(f"fold {f} done", flush=True)

    oof = pd.read_parquet(a.oof_dir / "oof_all.parquet").set_index("id").loc[lb["id"].tolist()].reset_index()
    out = {
        "r0_frozen_ridge": summary(y, pred["r0"]),
        "r1_fold_ridge": summary(y, pred["r1"]),
        "r2_fold_mlp": summary(y, pred["r2"]),
        "r3_model_raw": summary(y, oof[[f"raw_{d}" for d in DIMS]].to_numpy()),
        "r4_model_vrr": summary(y, oof[[f"vrr_{d}" for d in DIMS]].to_numpy()),
        "chosen": chosen,
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out.items():
        if isinstance(v, dict) and "mae" in v:
            print(f'{k:20s} MAE={v["mae"]:.6f} CCC={v["ccc"]:.4f} R2={v["r2"]:.4f} '
                  f'A/D MAE={v["per_dim_mae"]["A"]:.4f}/{v["per_dim_mae"]["D"]:.4f}')


if __name__ == "__main__":
    main()
