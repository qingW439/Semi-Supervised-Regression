#!/usr/bin/env python3
"""Aggregate the five OOF fold outputs into one 500-row table and report metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402

DIMS = ("V", "A", "D")


def metrics(y, p):
    m = metric_report(y, p)
    return {
        "macro_mae": m["macro_mae_original_scale"],
        "macro_ccc": float(np.mean([m[d]["ccc"] for d in DIMS])),
        "macro_r2": float(np.mean([m[d]["r2"] for d in DIMS])),
        "per_dim_mae": {d: m[d]["mae"] for d in DIMS},
        "per_dim_ccc": {d: m[d]["ccc"] for d in DIMS},
        "pred_std": {d: float(p[:, i].std()) for i, d in enumerate(DIMS)},
        "true_std": {d: float(y[:, i].std()) for i, d in enumerate(DIMS)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    frames = []
    for f in range(5):
        p = a.oof_dir / f"fold{f}" / "oof_predictions.parquet"
        d = pd.read_parquet(p)
        d["fold"] = f
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    if len(d) != 500 or d["id"].duplicated().any():
        raise ValueError(f"expected 500 unique OOF rows, got {len(d)}")
    y = d[[f"y_{k}" for k in DIMS]].to_numpy(np.float64)
    out = {"n": len(d), "overall": {}, "per_fold": {}, "std_ratio": {}}
    for name in ("raw", "vrr"):
        p = d[[f"{name}_{k}" for k in DIMS]].to_numpy(np.float64)
        out["overall"][name] = metrics(y, p)
        out["std_ratio"][name] = {f"{k}": out["overall"][name]["pred_std"][k] / out["overall"][name]["true_std"][k]
                                  for k in DIMS}
    for f in range(5):
        sub = d[d.fold == f]
        yy = sub[[f"y_{k}" for k in DIMS]].to_numpy(np.float64)
        out["per_fold"][str(f)] = {}
        for name in ("raw", "vrr"):
            pp = sub[[f"{name}_{k}" for k in DIMS]].to_numpy(np.float64)
            out["per_fold"][str(f)][name] = metrics(yy, pp)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    d.to_parquet(a.out_dir / "oof_all.parquet", index=False)
    (a.out_dir / "oof_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    for name in ("raw", "vrr"):
        m = out["overall"][name]
        print(f'{name}: MAE={m["macro_mae"]:.6f} CCC={m["macro_ccc"]:.4f} '
              f'R2={m["macro_r2"]:.4f} per-dim MAE={ {k: round(v,4) for k,v in m["per_dim_mae"].items()} }')
    print("per-fold raw MAE:", {k: round(v["raw"]["macro_mae"], 5) for k, v in out["per_fold"].items()})
    print("per-fold vrr MAE:", {k: round(v["vrr"]["macro_mae"], 5) for k, v in out["per_fold"].items()})


if __name__ == "__main__":
    main()
