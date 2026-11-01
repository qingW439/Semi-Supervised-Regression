#!/usr/bin/env python3
"""Create a fixed, stratified 5-fold OOF manifest for the 500 labeled TRAIN items."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def sha_ids(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261101)
    a = ap.parse_args()
    lb = pd.read_parquet(a.data_root / "lb500_s0" / "train_lb.parquet").reset_index(drop=True)
    # stratify by V quartile so each fold covers the valence range
    vbin = pd.qcut(lb["V"], 4, labels=False, duplicates="drop").astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed)
    folds = [[] for _ in range(5)]
    for k, (_, hold) in enumerate(skf.split(np.zeros(len(lb)), vbin)):
        folds[k] = lb.iloc[hold]["id"].tolist()
    all_ids = [i for f in folds for i in f]
    assert len(all_ids) == len(set(all_ids)) == 500, (len(all_ids), len(set(all_ids)))
    assert all(len(f) == 100 for f in folds), [len(f) for f in folds]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": a.seed,
        "n": len(lb),
        "n_folds": 5,
        "stratify": "V quartile",
        "fold_ids": {str(k): folds[k] for k in range(5)},
        "fold_sha256": {str(k): sha_ids(folds[k]) for k in range(5)},
        "all_ids_sha256": sha_ids(sorted(all_ids)),
        "counts": [len(f) for f in folds],
        "vbin_counts_per_fold": [
            {str(b): int(sum(vbin[lb.index[lb.id.isin(f)]] == b)) for b in range(4)} for f in folds
        ],
    }
    (a.out_dir / "fold_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = [{"id": i, "fold": k} for k, f in enumerate(folds) for i in f]
    pd.DataFrame(rows).sort_values(["fold", "id"]).to_csv(a.out_dir / "fold_manifest.csv", index=False)
    print(json.dumps({k: manifest[k] for k in ("counts", "all_ids_sha256", "fold_sha256")}, indent=2))


if __name__ == "__main__":
    main()
