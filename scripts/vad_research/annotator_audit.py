#!/usr/bin/env python3
"""Stage 2: EmoBank reader/writer annotation-disagreement audit.

Tests whether annotation uncertainty U_ij or cross-view difference disagreement
G_ij predicts M0 pairwise-difference error on TRAIN OOF pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DIMS = ("V", "A", "D")


def load_rw(path):
    d = pd.read_csv(path)
    d.columns = [c.strip().strip('"') for c in d.columns]
    for c in ("V", "A", "D", "stdV", "stdA", "stdD", "N"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.set_index("id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--audit-dir", type=Path, required=True)
    ap.add_argument("--manifest-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    rd = load_rw(a.corpus / "reader.csv")
    wr = load_rw(a.corpus / "writer.csv")
    out = {}
    for seed in (0, 1, 2):
        lb = pd.read_parquet(a.data_root / f"lb500_s{seed}" / "train_lb.parquet")
        z = np.load(a.audit_dir / f"audit_seed{seed}.npz")
        m0 = z["m0_mean"] * 4 + 1
        y = z["y"] * 4 + 1
        man = json.loads((a.manifest_root / (f"p0_s{seed}" if seed else "p0") / "fold_manifest.json").read_text())
        id2pos = {i: k for k, i in enumerate(lb["id"])}
        sub = lb[["id"]].copy()
        sub["r_V"] = sub.id.map(rd["V"]); sub["r_A"] = sub.id.map(rd["A"]); sub["r_D"] = sub.id.map(rd["D"])
        sub["w_V"] = sub.id.map(wr["V"]); sub["w_A"] = sub.id.map(wr["A"]); sub["w_D"] = sub.id.map(wr["D"])
        for d in DIMS:
            sub[f"rs_{d}"] = sub.id.map(rd[f"std{d}"]); sub[f"ws_{d}"] = sub.id.map(wr[f"std{d}"])
        sub["nr"] = sub.id.map(rd["N"]); sub["nw"] = sub.id.map(wr["N"])
        # verify combined mean vs labels
        diff = {}
        for d in DIMS:
            mc = (sub.nr * sub[f"r_{d}"] + sub.nw * sub[f"w_{d}"]) / (sub.nr + sub.nw)
            diff[d] = float(np.nanmax(np.abs(mc.to_numpy() - lb[d].to_numpy())))
            sub[f"c_{d}"] = mc
            sub[f"v_{d}"] = (sub.nr * sub[f"rs_{d}"].fillna(0) ** 2 +
                             sub.nw * sub[f"ws_{d}"].fillna(0) ** 2) / (sub.nr + sub.nw) ** 2
        per_dim = {}
        for d in DIMS:
            rows = []
            for f in range(5):
                hold = man["fold_ids"][str(f)]
                pos = np.array([id2pos[i] for i in hold])
                df = sub.iloc[pos].reset_index(drop=True)
                pp = m0[pos]
                yy = y[pos]
                iu = np.triu_indices(len(pos), 1)
                e = np.abs((pp[iu[0]] - pp[iu[1]])[:, DIMS.index(d)] - (yy[iu[0]] - yy[iu[1]])[:, DIMS.index(d)])
                U = df[f"v_{d}"].to_numpy()[iu[0]] + df[f"v_{d}"].to_numpy()[iu[1]]
                G = np.abs((df[f"r_{d}"].to_numpy()[iu[0]] - df[f"r_{d}"].to_numpy()[iu[1]]) -
                           (df[f"w_{d}"].to_numpy()[iu[0]] - df[f"w_{d}"].to_numpy()[iu[1]]))
                e_r = np.abs((pp[iu[0]] - pp[iu[1]])[:, DIMS.index(d)] -
                             (df[f"r_{d}"].to_numpy()[iu[0]] - df[f"r_{d}"].to_numpy()[iu[1]]))
                e_w = np.abs((pp[iu[0]] - pp[iu[1]])[:, DIMS.index(d)] -
                             (df[f"w_{d}"].to_numpy()[iu[0]] - df[f"w_{d}"].to_numpy()[iu[1]]))
                e_c = np.abs((pp[iu[0]] - pp[iu[1]])[:, DIMS.index(d)] -
                             (df[f"c_{d}"].to_numpy()[iu[0]] - df[f"c_{d}"].to_numpy()[iu[1]]))
                dp = (pp[iu[0]] - pp[iu[1]])[:, DIMS.index(d)]
                dy = (yy[iu[0]] - yy[iu[1]])[:, DIMS.index(d)]
                dacc = np.sign(dp) == np.sign(dy)
                for k in range(len(e)):
                    rows.append({"fold": f, "i": hold[iu[0][k]], "j": hold[iu[1][k]],
                                 "e": e[k], "U": U[k], "G": G[k], "d": bool(dacc[k]),
                                 "e_r": e_r[k], "e_w": e_w[k], "e_c": e_c[k]})
            # cluster bootstrap by text endpoint
            rng = np.random.default_rng(1000 + seed)
            rho_u, rho_g = [], []
            for _ in range(a.boot):
                sel = set(rng.choice(lb["id"].to_numpy(), size=100, replace=True))
                rr = [r for r in rows if r["i"] in sel or r["j"] in sel]
                if len(rr) < 30:
                    continue
                e = np.array([r["e"] for r in rr]); U = np.array([r["U"] for r in rr]); G = np.array([r["G"] for r in rr])
                rho_u.append(spearmanr(e, U).statistic)
                rho_g.append(spearmanr(e, G).statistic)
            e = np.array([r["e"] for r in rows]); U = np.array([r["U"] for r in rows]); G = np.array([r["G"] for r in rows])
            per_dim[d] = {
                "n_pairs": len(rows),
                "mae_diff": float(e.mean()),
                "dir_acc": float(np.mean([r["d"] for r in rows])),
                "rho_U": float(spearmanr(e, U).statistic),
                "rho_G": float(spearmanr(e, G).statistic),
                "rho_U_ci": [float(np.percentile(rho_u, 2.5)), float(np.percentile(rho_u, 97.5))],
                "rho_G_ci": [float(np.percentile(rho_g, 2.5)), float(np.percentile(rho_g, 97.5))],
                "mae_reader": float(np.mean([r["e_r"] for r in rows])),
                "mae_writer": float(np.mean([r["e_w"] for r in rows])),
                "mae_combined": float(np.mean([r["e_c"] for r in rows])),
                "U_quintile_err": [float(e[(U >= lo) & (U <= hi)].mean())
                                   for lo, hi in zip(np.quantile(U, [0, .2, .4, .6, .8]),
                                                     np.quantile(U, [.2, .4, .6, .8, 1.0]))],
                "G_quintile_err": [float(e[(G >= lo) & (G <= hi)].mean())
                                   for lo, hi in zip(np.quantile(G, [0, .2, .4, .6, .8]),
                                                     np.quantile(G, [.2, .4, .6, .8, 1.0]))],
            }
        out[f"seed{seed}"] = {"combined_mean_maxdiff": diff, "per_dim": per_dim}
        print("seed", seed, {d: (round(per_dim[d]["rho_U"], 3), round(per_dim[d]["rho_G"], 3)) for d in DIMS})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
