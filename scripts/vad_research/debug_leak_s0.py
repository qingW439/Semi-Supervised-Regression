#!/usr/bin/env python3
"""Leak forensics for the seed-0 OOF manifest (why is OOF R2 0.74 while dev R2 is 0.24?)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/root/autodl-tmp/rankup_project/data/emobank_processed")
MAN = Path("/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc/outputs/p0/fold_manifest.json")


def met(p, y):
    cov = ((p - p.mean()) * (y - y.mean())).mean()
    return dict(mae=float(np.abs(p - y).mean() * 4), r=float(cov / np.sqrt(p.var() * y.var())),
                r2=float(1 - ((p - y) ** 2).mean() / y.var()))


man = json.loads(MAN.read_text())
folds = {int(k): v for k, v in man["fold_ids"].items()}
lb = pd.read_parquet(DATA / "lb500_s0/train_lb.parquet")
print("manifest seed field:", man["seed"], "n:", man["n"], "n_folds:", man["n_folds"])
print("fold sizes:", {k: len(v) for k, v in folds.items()})
union = [i for f in sorted(folds) for i in folds[f]]
print("union", len(union), "unique", len(set(union)), "lb500_s0", len(lb))
print("union == lb id set:", set(union) == set(lb["id"]))
print("manifest sha256 of all ids:", man.get("all_ids_sha256"))

# duplicate texts inside the 500 labelled of seed 0
dup = lb[lb.duplicated("text", keep=False)].sort_values("text")
print("\nduplicate texts among the 500 labelled:", len(dup))
if len(dup):
    print(dup[["id", "text", "V", "A", "D"]].head(20).to_string())

# how much of the OOF performance could come from duplicate/near-duplicate leakage?
same_fold = 0
for f, ids in folds.items():
    s = lb[lb["id"].isin(ids)]
    if s["text"].duplicated().any():
        same_fold += 1
print("folds containing duplicate texts:", same_fold)

# read the produced OOF file and look at per-fold accuracy
oof = pd.read_parquet("/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-comp2/outputs/audit_e0/ridge_oof_s0_l2norm.parquet")
print("\nOOF rows", len(oof), "per-fold MAE / R2:")
for f in sorted(oof.fold.unique()):
    d = oof[oof.fold == f]
    print(f"  fold{f}: n={len(d)} MAE={np.abs(d.pred_V - d.y_V).mean()*4:.4f} "
          f"R2_V={met(d.pred_V.to_numpy(), d.y_V.to_numpy())['r2']:.4f}")

print("\nwhole OOF:", {k: round(v, 4) for k, v in met(oof[['pred_V','pred_A','pred_D']].to_numpy(),
                                                     oof[['y_V','y_A','y_D']].to_numpy()).items()})
# label statistics of the 500 labelled vs the dev set
dev = pd.read_parquet(DATA / "dev_gold.parquet")
for c in ("V_norm", "A_norm", "D_norm"):
    print(f"{c}: lb500_s0 var={lb[c].var():.5f} sd={lb[c].std():.4f} | dev var={dev[c].var():.5f} sd={dev[c].std():.4f}")
print("\nlb500_s0 label value counts (V_norm top):")
print(lb["V_norm"].value_counts().head(8).to_dict())
