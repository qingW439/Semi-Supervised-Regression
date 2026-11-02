#!/usr/bin/env python3
"""D1: decompose the JACF CCC loss into Pearson r and bias factor C_b.

Uses the fixed step-96 alpha-sweep predictions: matched supervised vs JACF
alpha=0.75, three seeds, V/A/D.  Scale-invariant statistics are computed on the
original [1,5] scale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402

DIMS = ("V", "A", "D")


def decompose(y, p):
    m = metric_report(y, p)
    out = {}
    for d in DIMS:
        i = DIMS.index(d)
        yy = y[:, i] * 4 + 1
        pp = p[:, i] * 4 + 1
        my, mp = yy.mean(), pp.mean()
        sy, sp = yy.std(), pp.std()
        r = float(pearsonr(yy, pp).statistic)
        cb = float(2 * sy * sp / (sy ** 2 + sp ** 2 + (my - mp) ** 2))
        ccc = float(2 * np.mean((yy - my) * (pp - mp)) /
                    (sy ** 2 + sp ** 2 + (my - mp) ** 2))
        out[d] = {
            "mae": m[d]["mae"], "rmse": m[d]["rmse"], "r2": m[d]["r2"],
            "pearson": r, "srcc": float(spearmanr(yy, pp).statistic),
            "ccc": ccc, "cb": cb, "ccc_from_r_cb": r * cb,
            "true_mean": float(my), "pred_mean": float(mp),
            "mean_bias": float(mp - my), "true_sd": float(sy), "pred_sd": float(sp),
            "sd_ratio": float(sp / sy) if sy > 1e-12 else float("nan"),
        }
    # macro over dims
    out["macro"] = {k: float(np.mean([out[d][k] for d in DIMS])) for k in
                    ("mae", "rmse", "r2", "pearson", "srcc", "ccc", "cb", "sd_ratio", "mean_bias")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    df = pd.read_parquet(a.pred)
    results = {}
    deltas = {}
    for s in (0, 1, 2):
        srow = df[(df.arm == "sup") & (df.seed == s)].sort_values("id")
        jrow = df[(df.arm == "a0.75") & (df.seed == s)].sort_values("id")
        assert (srow.id.values == jrow.id.values).all()
        y = srow[[f"y_{d}" for d in DIMS]].to_numpy()
        # y is already normalized; decompose converts to [1,5]
        ps = srow[[f"raw_{d}" for d in DIMS]].to_numpy()
        pj = jrow[[f"raw_{d}" for d in DIMS]].to_numpy()
        ds, dj = decompose(y, ps), decompose(y, pj)
        results[f"seed{s}"] = {"sup": ds, "jacf": dj}
        deltas[f"seed{s}"] = {d: {k: dj[d][k] - ds[d][k] for k in
                                  ("mae", "rmse", "r2", "pearson", "srcc", "ccc", "cb",
                                   "mean_bias", "sd_ratio")} for d in list(DIMS) + ["macro"]}
    # three-seed mean
    def agg(section):
        return {d: {k: float(np.mean([results[f"seed{s}"][section][d][k] for s in (0, 1, 2)]))
                    for k in results["seed0"][section][d]} for d in list(DIMS) + ["macro"]}
    summary = {"sup_mean": agg("sup"), "jacf_mean": agg("jacf")}
    summary["delta_mean"] = {d: {k: summary["jacf_mean"][d][k] - summary["sup_mean"][d][k] for k in
                                 summary["sup_mean"][d]} for d in summary["sup_mean"]}
    a.out.write_text(json.dumps({"per_seed": results, "per_seed_delta": deltas, "summary": summary}, indent=2) + "\n")
    for d in list(DIMS) + ["macro"]:
        s, j = summary["sup_mean"][d], summary["jacf_mean"][d]
        print(f'{d}: sup r={s["pearson"]:.4f} Cb={s["cb"]:.4f} CCC={s["ccc"]:.4f} sd_ratio={s["sd_ratio"]:.3f} bias={s["mean_bias"]:+.4f} | '
              f'jacf r={j["pearson"]:.4f} Cb={j["cb"]:.4f} CCC={j["ccc"]:.4f} sd_ratio={j["sd_ratio"]:.3f} bias={j["mean_bias"]:+.4f} | '
              f'dCCC={j["ccc"]-s["ccc"]:+.4f} dr={j["pearson"]-s["pearson"]:+.4f} dCb={j["cb"]-s["cb"]:+.4f}')


if __name__ == "__main__":
    main()
