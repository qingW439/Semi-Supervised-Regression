#!/usr/bin/env python3
"""Evaluate M1-R0 relation audit (pair-level)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DIMS = ("V", "A", "D")


def acc(pred, true):
    return float(np.mean(np.sign(pred) == np.sign(true)))


def cr_ir(t, n, true):
    ts, ns, tr = np.sign(t), np.sign(n), np.sign(true)
    ok = np.abs(tr) > 1e-12
    wrong = ok & (ts != tr)
    corr = wrong & (ns == tr)
    intr = ok & (ts == tr) & (ns != tr)
    nw = int(wrong.sum()); nc = int(corr.sum()); ni = int(intr.sum())
    return {"teacher_wrong": nw, "corrected": nc, "cr": float(nc / nw) if nw else float("nan"),
            "introduced": ni, "ir": float(ni / max(1, ok.sum() - nw)),
            "net": int(nc - ni)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--graph-diag", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    d = pd.read_parquet(a.pairs)
    out = {"n_rows": len(d), "n_batches": int(d.groupby(["fold", "batch"]).ngroups),
           "methods": {}, "cr_ir": {}, "bootstrap": {}}
    for dim in DIMS:
        sub = d[d.dim == dim]
        out["methods"][dim] = {}
        for m in ("r0", "r1", "r2", "r3"):
            out["methods"][dim][m] = {
                "acc": acc(sub[m].to_numpy(), sub.true_diff.to_numpy()),
                "dist_rho": float(spearmanr(np.abs(sub[m]), np.abs(sub.true_diff)).statistic),
            }
        out["cr_ir"][dim] = {"r3_vs_r1": cr_ir(sub.r1.to_numpy(), sub.r3.to_numpy(), sub.true_diff.to_numpy())}
    # 3D pair table
    d["pair"] = d.fold.astype(str) + "|" + d.batch.astype(str) + "|" + d.i + "|" + d.j
    piv = d.pivot_table(index="pair", columns="dim", values=["true_diff", "r1", "r3"])
    tv = np.stack([piv[("true_diff", k)].to_numpy() for k in DIMS], axis=1)
    r1 = np.stack([piv[("r1", k)].to_numpy() for k in DIMS], axis=1)
    r3 = np.stack([piv[("r3", k)].to_numpy() for k in DIMS], axis=1)
    out["3d"] = {
        "direction_exact_r1": float(np.mean(np.all(np.sign(r1) == np.sign(tv), axis=1))),
        "direction_exact_r3": float(np.mean(np.all(np.sign(r3) == np.sign(tv), axis=1))),
        "distance_rho_r1": float(spearmanr(np.linalg.norm(r1, axis=1), np.linalg.norm(tv, axis=1)).statistic),
        "distance_rho_r3": float(spearmanr(np.linalg.norm(r3, axis=1), np.linalg.norm(tv, axis=1)).statistic),
    }
    # batch-level stats for fast cluster bootstrap
    batch_stats = {}
    for (f, b), g in d.groupby(["fold", "batch"]):
        stats = {}
        for dim in DIMS:
            s = g[g.dim == dim]
            stats[dim] = {
                "acc1": acc(s.r1.to_numpy(), s.true_diff.to_numpy()),
                "acc3": acc(s.r3.to_numpy(), s.true_diff.to_numpy()),
                "net": cr_ir(s.r1.to_numpy(), s.r3.to_numpy(), s.true_diff.to_numpy())["net"],
            }
        batch_stats[(f, b)] = stats
    batch_keys = list(batch_stats.keys())
    rng = np.random.default_rng(20261102)
    boot = {dim: {"dacc": [], "net": []} for dim in DIMS}
    for _ in range(a.boot):
        sel = rng.integers(0, len(batch_keys), len(batch_keys))
        for dim in DIMS:
            dacc = [batch_stats[batch_keys[k]][dim]["acc3"] - batch_stats[batch_keys[k]][dim]["acc1"] for k in sel]
            net = [batch_stats[batch_keys[k]][dim]["net"] for k in sel]
            boot[dim]["dacc"].append(float(np.mean(dacc)))
            boot[dim]["net"].append(float(np.mean(net)))
    for dim in DIMS:
        out["bootstrap"][dim] = {
            "dacc_mean": float(np.mean(boot[dim]["dacc"])),
            "dacc_ci95": [float(np.percentile(boot[dim]["dacc"], 2.5)), float(np.percentile(boot[dim]["dacc"], 97.5))],
            "net_mean": float(np.mean(boot[dim]["net"])),
            "net_ci95": [float(np.percentile(boot[dim]["net"], 2.5)), float(np.percentile(boot[dim]["net"], 97.5))],
        }
    out["graph_diag"] = {
        "negative_similarity_fraction_mean": float(np.mean([x["negative_similarity_fraction"] for x in json.load(open(a.graph_diag))])),
        "cond_L_mean": float(np.mean([x["cond_L"] for x in json.load(open(a.graph_diag))])),
        "eig_second_mean": float(np.mean([x["eig_second"] for x in json.load(open(a.graph_diag))])),
    }
    # equivalence
    out["r3_r3lp_max_abs"] = float(np.max(np.abs(d.r3.to_numpy() - d.r3lp.to_numpy())))
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    for dim in DIMS:
        print(f"== {dim} ==")
        for m in ("r0", "r1", "r2", "r3"):
            v = out["methods"][dim][m]
            print(f'{m}: acc={v["acc"]:.4f} dist_rho={v["dist_rho"]:.4f}')
        print("cr/ir r3 vs r1:", out["cr_ir"][dim]["r3_vs_r1"])
        print("bootstrap:", out["bootstrap"][dim])
    print("3D", out["3d"])
    print("equiv", out["r3_r3lp_max_abs"])


if __name__ == "__main__":
    main()
