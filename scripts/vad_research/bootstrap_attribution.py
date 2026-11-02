#!/usr/bin/env python3
"""Paired bootstrap for the A/B/C/D attribution deltas."""
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


def macro_ccc(y, p):
    m = metric_report(y, p)
    return float(np.mean([m[d]["ccc"] for d in DIMS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    d = pd.read_parquet(a.pred)
    rng = np.random.default_rng(20261102)
    out = {}
    for s in (0, 1, 2):
        a_ = d[(d.arm == "sup") & (d.seed == s)].sort_values("id")
        c_ = d[(d.arm == "jacf") & (d.seed == s)].sort_values("id")
        assert (a_.id.values == c_.id.values).all()
        y = a_[[f"y_{k}" for k in DIMS]].to_numpy()
        for mode in ("raw", "vrr"):
            pa = a_[[f"{mode}_{k}" for k in DIMS]].to_numpy()
            pc = c_[[f"{mode}_{k}" for k in DIMS]].to_numpy()
            d_mae = (np.abs(pa - y).mean(1) - np.abs(pc - y).mean(1)) * 4
            d_ccc = macro_ccc(y, pa) - macro_ccc(y, pc)
            bs = []
            for _ in range(a.boot):
                idx = rng.integers(0, len(y), len(y))
                bs.append(float(d_mae[idx].mean()))
            out[f"seed{s}_{mode}"] = {
                "delta_mae_A_minus_C": float(d_mae.mean()),
                "delta_mae_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "delta_ccc_A_minus_C": float(d_ccc),
            }
    # mean across seeds
    for mode in ("raw", "vrr"):
        vals = [out[f"seed{s}_{mode}"]["delta_mae_A_minus_C"] for s in (0, 1, 2)]
        out[f"mean_{mode}"] = {"delta_mae": float(np.mean(vals)), "per_seed": vals,
                               "sd": float(np.std(vals, ddof=1))}
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out.items():
        if "delta_mae_ci95" in v:
            print(f'{k}: dMAE={v["delta_mae_A_minus_C"]:+.6f} CI={v["delta_mae_ci95"]} dCCC={v["delta_ccc_A_minus_C"]:+.4f}')


if __name__ == "__main__":
    main()
