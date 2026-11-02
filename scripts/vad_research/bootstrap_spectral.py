#!/usr/bin/env python3
"""Paired bootstrap for S2 - S1 (unlabeled spectral gain)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402

DIMS = ("V", "A", "D")


def macro_ccc(y, p):
    m = metric_report(y, p)
    return float(np.mean([m[d]["ccc"] for d in DIMS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=3000)
    a = ap.parse_args()
    z = np.load(a.npz)
    y, s1, s2 = z["y"], z["s1"], z["s2"]
    n = len(y)
    d_mae = (np.abs(y - s2).mean(1) - np.abs(y - s1).mean(1)) * 4.0
    rng = np.random.default_rng(20261102)
    mae, ccc = [], []
    for _ in range(a.boot):
        idx = rng.integers(0, n, n)
        mae.append(float(d_mae[idx].mean()))
        ccc.append(macro_ccc(y[idx], s2[idx]) - macro_ccc(y[idx], s1[idx]))
    out = {"delta_mae_s2_minus_s1": float(d_mae.mean()),
           "delta_mae_ci95": [float(np.percentile(mae, 2.5)), float(np.percentile(mae, 97.5))],
           "delta_ccc_s2_minus_s1": float(macro_ccc(y, s2) - macro_ccc(y, s1)),
           "delta_ccc_ci95": [float(np.percentile(ccc, 2.5)), float(np.percentile(ccc, 97.5))],
           "mae_scale": "original_[1,5]", "n": n, "boot": a.boot}
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
