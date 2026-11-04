#!/usr/bin/env python3
"""Aggregate the E0* candidate OOF results and freeze the strongest supervised baseline.

Pre-registered rule (docs/ADMISSION_V2_PREREGISTERED.md §3):
  * candidates: MLP-L1-160 (10 epochs), MLP-L1-640-bs64, MLP-L1-1600-bs64, Ridge
  * selection: lowest 3-seed mean OOF Macro-MAE (ties < 0.0005 break on CCC then R2)
  * constraint: no candidate may have a per-dimension OOF R2 below -0.10
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DIMS = ("V", "A", "D")
SCALE = 4.0


def metrics(p, y):
    mp, my = p.mean(), y.mean()
    vp, vy = p.var(), y.var()
    cov = ((p - mp) * (y - my)).mean()
    return dict(mae=float(np.abs(p - y).mean() * SCALE),
                pearson=float(cov / np.sqrt(vp * vy)),
                sd_ratio=float(np.sqrt(vp / vy)),
                ccc=float(2 * cov / (vp + vy + (mp - my) ** 2)),
                r2=float(1 - ((p - y) ** 2).mean() / vy))


def macro(p, y):
    m = {k: float(np.mean([metrics(p[:, i], y[:, i])[k] for i in range(3)]))
         for k in ("mae", "pearson", "sd_ratio", "ccc", "r2")}
    m["rho"] = m["pearson"] * m["sd_ratio"]
    return m


def boot_ci(d, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    return tuple(float(x) for x in np.percentile(d[idx].mean(1), [2.5, 97.5]))


def load_frames(root, tag):
    out = {}
    for s in (0, 1, 2):
        parts = []
        for f in range(5):
            p = root / f"outputs/oof_sup_{tag}" / f"seed{s}_fold{f}" / "sup_predictions.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p))
        if len(parts) != 5:
            print(f"  {tag} seed{s}: only {len(parts)}/5 folds")
            continue
        df = pd.concat(parts, ignore_index=True)
        df["seed"] = s
        out[s] = df
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("outputs/audit_e0/e0_freeze.json"))
    a = ap.parse_args()
    tags = ["640_lr1e4_bs64", "1600_lr1e4_bs64", "640_lr2e5_bs16"]
    res, frames = {}, {}
    for tag in tags:
        fr = load_frames(a.root, tag)
        frames[tag] = fr
        per = {}
        for s, df in fr.items():
            per[f"seed{s}"] = dict(
                oof=macro(df[["p_V", "p_A", "p_D"]].to_numpy(), df[["y_V", "y_A", "y_D"]].to_numpy()),
                per_dim={DIMS[i]: metrics(df[[f"p_{d}" for d in DIMS]].to_numpy()[:, i],
                                          df[[f"y_{d}" for d in DIMS]].to_numpy()[:, i])
                         for i in range(3)})
        mean = {k: float(np.mean([per[f"seed{s}"]["oof"][k] for s in (0, 1, 2)])) for k in
                ("mae", "pearson", "sd_ratio", "ccc", "r2")}
        sd = {k: float(np.std([per[f"seed{s}"]["oof"][k] for s in (0, 1, 2)])) for k in ("mae", "ccc", "r2")}
        res[tag] = dict(per_seed=per, mean=mean, sd=sd)
        print(f"{tag:20s} MAE={mean['mae']:.5f}±{sd['mae']:.5f} r={mean['pearson']:.4f} "
              f"sd={mean['sd_ratio']:.4f} CCC={mean['ccc']:.4f}±{sd['ccc']:.4f} R2={mean['r2']:.4f}±{sd['r2']:.4f}")

    # reference numbers already audited under the identical protocol
    ref = {"mlp_p1_10ep": dict(mae=0.20075, ccc=0.38064, r2=0.06137),
           "ridge_internal": dict(mae=0.18456, ccc=0.39644, r2=0.20014)}
    for k, v in ref.items():
        print(f"{k:20s} MAE={v['mae']:.5f} CCC={v['ccc']:.4f} R2={v['r2']:.4f}  (reference)")

    ranked = sorted(res.items(), key=lambda kv: kv[1]["mean"]["mae"])
    winner_tag, winner = ranked[0]
    print(f"\nE0* by OOF Macro-MAE: {winner_tag} (MAE={winner['mean']['mae']:.5f})")
    # per-dim sanity constraint
    bad = [d for d in DIMS if np.mean([winner["per_seed"][f"seed{s}"]["per_dim"][d]["r2"] for s in (0, 1, 2)]) < -0.10]
    print("per-dim R2 below -0.10:", bad or "none")

    # paired CI of the winner vs every other candidate (same folds, same samples)
    print("\n=== paired bootstrap vs the runner-up (MAE) ===")
    ci = {}
    if len(ranked) > 1:
        second = ranked[1][0]
        for s in (0, 1, 2):
            A = frames[winner_tag][s].set_index("id")
            B = frames[second][s].set_index("id")
            common = A.index.intersection(B.index)
            ea = np.abs(A.loc[common, [f"p_{d}" for d in DIMS]].to_numpy()
                        - A.loc[common, [f"y_{d}" for d in DIMS]].to_numpy()).mean(1) * SCALE
            eb = np.abs(B.loc[common, [f"p_{d}" for d in DIMS]].to_numpy()
                        - B.loc[common, [f"y_{d}" for d in DIMS]].to_numpy()).mean(1) * SCALE
            d = ea - eb
            lo, hi = boot_ci(d)
            ci[f"seed{s}"] = dict(mean=float(d.mean()), ci95=[lo, hi], n=len(d))
            print(f"seed{s}: {winner_tag} - {second} = {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")

    out = {"candidates": res, "reference": ref, "e0_star": winner_tag,
           "e0_star_mean": winner["mean"], "paired_vs_runner_up": ci,
           "rule": "lowest 3-seed mean OOF Macro-MAE; ties <0.0005 break on CCC then R2"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
