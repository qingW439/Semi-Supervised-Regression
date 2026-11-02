#!/usr/bin/env python3
"""Paired bootstrap for the DEV P2 calibration comparison."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS  # noqa: E402
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402

DIMS = ("V", "A", "D")


def macro_ccc(y, p):
    m = metric_report(y, p)
    return float(np.mean([m[d]["ccc"] for d in DIMS]))


def macro_mae(y, p):
    return float(np.mean(np.abs(y - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    dev = pd.read_parquet(a.data_root / "dev_gold.parquet")
    d = pd.read_parquet(a.pred).set_index("id").loc[dev["id"]].reset_index()
    y = d[[f"y_{k}" for k in DIMS]].to_numpy() if "y_V" in d else dev.loc[:, list(NORM_COLUMNS)].to_numpy()
    methods = ["vrr", "global", "li", "la"]
    rng = np.random.default_rng(20261102)
    n = len(y)
    out = {}
    for m in methods:
        p = d[[f"{m}_{k}" for k in DIMS]].to_numpy()
        d_mae = np.abs(y - p).mean(1) - np.abs(y - d[[f"vrr_{k}" for k in DIMS]].to_numpy()).mean(1)
        mae_vals = []
        ccc_vals = []
        for _ in range(a.boot):
            idx = rng.integers(0, n, n)
            mae_vals.append(float(d_mae[idx].mean()))
            ccc_vals.append(macro_ccc(y[idx], p[idx]) - macro_ccc(y[idx], d[[f"vrr_{k}" for k in DIMS]].to_numpy()[idx]))
        out[m] = {
            "delta_mae_vs_vrr": float(d_mae.mean()),
            "delta_mae_ci95": [float(np.percentile(mae_vals, 2.5)), float(np.percentile(mae_vals, 97.5))],
            "delta_ccc_vs_vrr": float(macro_ccc(y, p) - macro_ccc(y, d[[f"vrr_{k}" for k in DIMS]].to_numpy())),
            "delta_ccc_ci95": [float(np.percentile(ccc_vals, 2.5)), float(np.percentile(ccc_vals, 97.5))],
        }
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out.items():
        print(f'{k:8s} dMAE={v["delta_mae_vs_vrr"]:+.6f} CI={v["delta_mae_ci95"]} '
              f'dCCC={v["delta_ccc_vs_vrr"]:+.4f} CI={v["delta_ccc_ci95"]}')


if __name__ == "__main__":
    main()
