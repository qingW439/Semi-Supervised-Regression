#!/usr/bin/env python3
"""Evaluate the nested OOF mechanism comparison (T0-T5) + LDE."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS, read_gold  # noqa: E402
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_acvr import encode  # noqa: E402

DIMS = ("V", "A", "D")
METHODS = ("T0", "T1", "T2", "T3", "T4", "T5")
LABEL = {"T0": "Teacher", "T1": "JACF", "T2": "BC", "T3": "KNN-R(tuned)",
         "T4": "GlobalResid", "T5": "PermResid"}


def metrics(y, p):
    m = metric_report(y, p)
    out = {"mae": m["macro_mae_original_scale"],
           "rmse": float(np.mean([m[d]["rmse"] for d in DIMS])),
           "ccc": float(np.mean([m[d]["ccc"] for d in DIMS])),
           "pearson": float(np.mean([m[d]["pearson"] for d in DIMS])),
           "r2": float(np.mean([m[d]["r2"] for d in DIMS]))}
    # Cb and s, delta per dim, then macro
    cbs, ss, ds = [], [], []
    for i, d in enumerate(DIMS):
        yy = y[:, i] * 4 + 1; pp = p[:, i] * 4 + 1
        my, mp = yy.mean(), pp.mean(); sy, sp = yy.std(), pp.std()
        s = sp / sy if sy > 1e-12 else np.nan
        delta = (mp - my) / sy if sy > 1e-12 else np.nan
        cb = 2 * s / (1 + s * s + delta * delta) if np.isfinite(s) else np.nan
        cbs.append(cb); ss.append(s); ds.append(delta)
    out["cb"] = float(np.mean(cbs)); out["s"] = float(np.mean(ss)); out["delta"] = float(np.mean(ds))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    d = pd.read_parquet(a.pred)
    y = d[[f"y_{k}" for k in DIMS]].to_numpy(np.float64)
    out = {"n": len(d), "methods": {}}
    for m in METHODS:
        p = d[[f"{m}_{k}" for k in DIMS]].to_numpy(np.float64)
        out["methods"][m] = metrics(y, p)
    # equivalence
    out["equivalence_T2_T3fix"] = float(np.max(np.abs(
        d[[f"T2_{k}" for k in DIMS]].to_numpy() - d[[f"T3fix_{k}" for k in DIMS]].to_numpy())))

    # LDE with fixed frozen-feature graph, per outer fold
    lb = read_gold(a.data_root / "lb500_s0" / "train_lb.parquet", expected_n=500)
    man = json.loads(a.manifest.read_text())
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    enc = AutoModel.from_pretrained(str(a.pretrain_path), local_files_only=True).to(device).eval()
    feat = encode(enc, lb["text"].tolist(), tok, device).cpu().numpy()
    del enc; torch.cuda.empty_cache()
    id2pos = {i: k for k, i in enumerate(lb["id"])}
    lde = {m: {k: {"lde": [], "zero": [], "rdelta": [], "slope": []} for k in DIMS} for m in METHODS}
    for f in range(5):
        hold = man["fold_ids"][str(f)]
        pos = np.array([id2pos[i] for i in hold])
        sub = d[d["id"].isin(hold)].set_index("id").loc[hold].reset_index()
        X = feat[pos]
        nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(X)
        _, idx = nn.kneighbors(X)
        pairs = set()
        for i in range(len(pos)):
            for j in idx[i, 1:]:
                if i != j:
                    pairs.add((min(i, j), max(i, j)))
        pairs = sorted(pairs)
        ii = np.array([p[0] for p in pairs]); jj = np.array([p[1] for p in pairs])
        yt = sub[[f"y_{k}" for k in DIMS]].to_numpy()
        td = yt[ii] - yt[jj]
        for m in METHODS:
            pp = sub[[f"{m}_{k}" for k in DIMS]].to_numpy()
            pd_ = pp[ii] - pp[jj]
            for di, k in enumerate(DIMS):
                diff = pd_[:, di] - td[:, di]
                lde[m][k]["lde"].append(float(np.mean(np.abs(diff))))
                lde[m][k]["zero"].append(float(np.mean(np.abs(td[:, di]))))
                lde[m][k]["rdelta"].append(float(pearsonr(pd_[:, di], td[:, di]).statistic))
                lde[m][k]["slope"].append(float(np.polyfit(td[:, di], pd_[:, di], 1)[0]))
    out["lde"] = {m: {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in md.items()}
                  for m, md in lde.items()}

    # paired bootstrap (sample-level) for MAE/CCC vs T1
    rng = np.random.default_rng(20261102)
    boot = {}
    for m in ("T2", "T3", "T4", "T5"):
        dm, dc = [], []
        for _ in range(a.boot):
            idx = rng.integers(0, len(d), len(d))
            yy = y[idx]
            pm = d[[f"{m}_{k}" for k in DIMS]].to_numpy()[idx]
            p1 = d[[f"T1_{k}" for k in DIMS]].to_numpy()[idx]
            dm.append(float(np.mean(np.abs(pm - yy)) - np.mean(np.abs(p1 - yy))) * 4.0)
            mc = lambda p: np.mean([metric_report(yy, p)[k]["ccc"] for k in DIMS])
            dc.append(float(mc(pm) - mc(p1)))
        boot[m] = {"dmae_vs_T1": float(np.mean(dm)), "dmae_ci": [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))],
                   "dccc_vs_T1": float(np.mean(dc)), "dccc_ci": [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))]}
    out["bootstrap_vs_T1"] = boot
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "nested_oof_eval.json").write_text(json.dumps(out, indent=2) + "\n")
    for m in METHODS:
        v = out["methods"][m]
        print(f'{m} {LABEL[m]:14s} MAE={v["mae"]:.6f} CCC={v["ccc"]:.4f} r={v["pearson"]:.4f} '
              f'Cb={v["cb"]:.4f} s={v["s"]:.3f} delta={v["delta"]:+.4f}')
    print("== LDE macro (mean over VAD) ==")
    for m in METHODS:
        l = out["lde"][m]
        print(f'{m} {LABEL[m]:14s} LDE={np.mean([l[k]["lde"] for k in DIMS]):.4f} '
              f'zero={np.mean([l[k]["zero"] for k in DIMS]):.4f} '
              f'rdelta={np.mean([l[k]["rdelta"] for k in DIMS]):.3f} '
              f'slope={np.mean([l[k]["slope"] for k in DIMS]):.3f}')
    print("equivalence", out["equivalence_T2_T3fix"])
    print("bootstrap vs T1", json.dumps(boot, indent=2))


if __name__ == "__main__":
    main()
