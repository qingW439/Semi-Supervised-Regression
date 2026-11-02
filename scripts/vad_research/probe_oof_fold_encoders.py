#!/usr/bin/env python3
"""Honest OOF probe on each fold model's own encoder features.

The fold encoder never saw its 100 held-out labels, so fitting a ridge on the
400 bank features and predicting the 100 held-out features is label-isolated.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS, read_gold  # noqa: E402
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402

DIMS = ("V", "A", "D")


def summary(y, p):
    m = metric_report(y, p)
    return {"mae": m["macro_mae_original_scale"],
            "ccc": float(np.mean([m[d]["ccc"] for d in DIMS])),
            "r2": float(np.mean([m[d]["r2"] for d in DIMS]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--oof-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    lb = read_gold(a.data_root / "lb500_s0" / "train_lb.parquet", expected_n=500)
    man = json.loads(a.manifest.read_text())
    out = {}
    for alpha in (100.0, 1000.0):
        pred = np.zeros((500, 3))
        for f in range(5):
            hold = man["fold_ids"][str(f)]
            fdir = a.oof_dir / f"fold{f}"
            Xb = np.load(fdir / "features_labeled.npy")
            Xh = np.load(fdir / "features_holdout.npy")
            train = [i for i in lb["id"] if i not in set(hold)]
            yb = lb[lb["id"].isin(train)].loc[:, list(NORM_COLUMNS)].to_numpy(np.float32)
            hi = lb.index[lb["id"].isin(hold)].to_numpy()
            m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            m.fit(Xb, yb)
            pred[hi] = m.predict(Xh)
        out[f"ridge{int(alpha)}"] = summary(lb.loc[:, list(NORM_COLUMNS)].to_numpy(), pred)
    oof = pd.read_parquet(a.oof_dir / "oof_all.parquet").set_index("id").loc[lb["id"].tolist()].reset_index()
    out["model_raw"] = summary(lb.loc[:, list(NORM_COLUMNS)].to_numpy(),
                               oof[[f"raw_{d}" for d in DIMS]].to_numpy())
    out["model_vrr"] = summary(lb.loc[:, list(NORM_COLUMNS)].to_numpy(),
                               oof[[f"vrr_{d}" for d in DIMS]].to_numpy())
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out.items():
        print(f'{k:12s} MAE={v["mae"]:.6f} CCC={v["ccc"]:.4f} R2={v["r2"]:.4f}')


if __name__ == "__main__":
    main()
