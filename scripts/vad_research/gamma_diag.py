#!/usr/bin/env python3
"""Estimate the optimal anchor-teacher deviation coefficient gamma on labeled data.

r_i = y_i - a_i (anchor residual), d_i = t_i - a_i (teacher deviation).
JACF uses coefficient alpha=0.75 on a, i.e. 0.25 on (t-a).  If the OLS slope
of r on d is materially above 0.25, the target is under-expanding variation.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/autodl-tmp/rankup_project")
REPO = ROOT / "Semi-Supervised-Regression-avrrc"
lb = pd.read_parquet(ROOT / "data/emobank_processed/lb500_s0/train_lb.parquet")
oof = pd.read_parquet(REPO / "outputs/oof/oof_all.parquet")
feat = np.load(REPO / "outputs/m1_r0/features_f0_seed0.npz")["lb"]
feat = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
d = oof.set_index("id").loc[lb["id"]]
y = lb[["V_norm", "A_norm", "D_norm"]].to_numpy(np.float64)
t = d[["raw_V", "raw_A", "raw_D"]].to_numpy(np.float64)
S = feat @ feat.T
np.fill_diagonal(S, -np.inf)
k = 20
idx = np.argsort(-S, axis=1)[:, :k]
w = np.exp((np.take_along_axis(S, idx, 1) - 1.0) / 0.1)
w = w / w.sum(1, keepdims=True)
a = np.einsum("ik,ikj->ij", w, y[idx])
for j, dim in enumerate(("V", "A", "D")):
    dd = t[:, j] - a[:, j]
    rr = y[:, j] - a[:, j]
    slope = np.cov(dd, rr)[0, 1] / np.var(dd)
    corr = np.corrcoef(dd, rr)[0, 1]
    global_slope = np.cov(t[:, j], y[:, j])[0, 1] / np.var(t[:, j])
    print(f"{dim}: gamma(anchor-dev)={slope:.4f} corr={corr:.4f} "
          f"sd_d={dd.std():.4f} sd_r={rr.std():.4f} global_slope={global_slope:.4f}")
print("JACF variation coefficient = 0.25")
