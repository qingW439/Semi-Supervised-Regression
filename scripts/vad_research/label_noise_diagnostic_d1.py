#!/usr/bin/env python3
"""D1: annotation-consistency diagnostic for EmoBank TRAIN-500.

Adds (i) proper signed-residual / abs-error / RMSE / bias metrics, (ii) separate
Reader / Writer / Combined reliability, and (iii) a paired Reader-vs-Writer
comparison.  CPU only; hidden ULB labels are not read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel

DIMS = ("V", "A", "D")


def icc_oneway(groups):
    groups = [g for g in groups if len(g) >= 2]
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n_total - k)
    n0 = (n_total - sum(len(g) ** 2 for g in groups) / n_total) / (k - 1)
    return {
        "icc_1_1": float((msb - msw) / (msb + (n0 - 1) * msw)) if msb + (n0 - 1) * msw > 0 else float("nan"),
        "icc_1_k": float((msb - msw) / msb) if msb > 0 else float("nan"),
    }


def split_half(groups, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    ids = [i for i, g in groups.items() if len(g) >= 4]
    rs = []
    for _ in range(reps):
        a, b = [], []
        for i in ids:
            g = groups[i].copy()
            rng.shuffle(g)
            h = len(g) // 2
            a.append(g[:h].mean())
            b.append(g[h:2 * h].mean())
        r = pearsonr(a, b).statistic if np.std(a) > 1e-12 and np.std(b) > 1e-12 else np.nan
        if np.isfinite(r):
            rs.append(r)
    r = float(np.nanmean(rs)) if rs else float("nan")
    return {"split_half_r": r,
            "spearman_brown": float(2 * r / (1 + r)) if np.isfinite(r) and r > -1 else float("nan"),
            "n_items": len(ids), "reps": reps}


def source_stats(frame, ids, gold_norm):
    out = {}
    for di, d in enumerate(DIMS):
        groups = {}
        means = []
        for i in ids:
            g = frame.loc[frame["id"] == i, d].to_numpy(np.float64)
            groups[i] = g
            means.append(g.mean() if len(g) else np.nan)
        gs = [g for g in groups.values() if len(g) >= 2]
        out[d] = {
            "n_mean": float(np.mean([len(g) for g in groups.values()])),
            "n_min": int(min(len(g) for g in groups.values())),
            "within_sd_mean": float(np.mean([g.std(ddof=1) for g in gs])),
            "icc": icc_oneway(gs),
            "split_half": split_half(groups),
            "mean_abs_diff_to_gold": float(np.mean(np.abs(np.array(means) - (gold_norm[:, di] * 4.0 + 1.0)))),
        }
    return out


def paired_reader_writer(r, w, ids, gold_norm):
    out = {}
    for di, d in enumerate(DIMS):
        rm, wm, kept = [], [], []
        for i in ids:
            rg = r.loc[r["id"] == i, d].to_numpy(np.float64)
            wg = w.loc[w["id"] == i, d].to_numpy(np.float64)
            if len(rg) and len(wg):
                rm.append(rg.mean()); wm.append(wg.mean()); kept.append(i)
        rm, wm = np.array(rm), np.array(wm)
        diff = wm - rm
        t = ttest_rel(wm, rm) if len(diff) > 2 else None
        out[d] = {
            "n_paired": len(diff),
            "reader_mean": float(rm.mean()), "writer_mean": float(wm.mean()),
            "writer_minus_reader": float(diff.mean()),
            "diff_sd": float(diff.std(ddof=1)),
            "paired_t_p": float(t.pvalue) if t is not None else float("nan"),
            "pearson_reader_writer": float(pearsonr(rm, wm).statistic) if len(diff) > 2 else float("nan"),
            "mean_abs_gold_reader": float(np.mean(np.abs(rm - (gold_norm[[ids.index(i) for i in kept], di] * 4.0 + 1.0)))),
            "mean_abs_gold_writer": float(np.mean(np.abs(wm - (gold_norm[[ids.index(i) for i in kept], di] * 4.0 + 1.0)))),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--emobank-root", type=Path, required=True)
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    lb = pd.read_parquet(a.data_root / "lb500_s0" / "train_lb.parquet")
    ids = lb["id"].tolist()
    gold_norm = lb.loc[:, [f"{d}_norm" for d in DIMS]].to_numpy(np.float64)
    r_full = pd.read_csv(a.emobank_root / "corpus" / "individual_reader_ratings.csv")
    w_full = pd.read_csv(a.emobank_root / "corpus" / "individual_writer_ratings.csv")
    r = r_full[~((r_full["V"] == 1) & (r_full["A"] == 1) & (r_full["D"] == 1))].copy()
    w = w_full[~((w_full["V"] == 1) & (w_full["A"] == 1) & (w_full["D"] == 1))].copy()
    filtered_rows = (len(r_full) - len(r)) + (len(w_full) - len(w))
    r = r[r["id"].isin(ids)].copy(); w = w[w["id"].isin(ids)].copy()
    combined = pd.concat([r.assign(source="reader"), w.assign(source="writer")], ignore_index=True)

    oof = pd.read_parquet(a.oof).set_index("id").loc[ids].reset_index()
    pred = oof[[f"raw_{d}" for d in DIMS]].to_numpy(np.float64)
    residual = (pred - gold_norm) * 4.0
    abs_error = np.abs(residual)

    report = {
        "protocol": {
            "n_train": len(ids),
            "all_one_filtered_rows": int(filtered_rows),
            "reader_n_mean": float(r.groupby("id").size().mean()),
            "writer_n_mean": float(w.groupby("id").size().mean()),
            "note": "Reader/Writer are separate annotation perspectives; Combined is their pooled ratings.",
        },
        "sources": {
            "reader": source_stats(r, ids, gold_norm),
            "writer": source_stats(w, ids, gold_norm),
            "combined": source_stats(combined, ids, gold_norm),
        },
        "reader_vs_writer": paired_reader_writer(r, w, ids, gold_norm),
        "residuals_oof": {},
        "noise_share_exploratory": {},
    }
    for di, d in enumerate(DIMS):
        report["residuals_oof"][d] = {
            "signed_residual_sd": float(np.std(residual[:, di], ddof=1)),
            "abs_error_sd": float(np.std(abs_error[:, di], ddof=1)),
            "rmse": float(np.sqrt(np.mean(residual[:, di] ** 2))),
            "bias": float(np.mean(residual[:, di])),
            "mae": float(np.mean(abs_error[:, di])),
        }
        # exploratory only; iid-rater assumption
        g = combined.loc[combined["id"].isin(ids)]
        within = np.array([np.var(g.loc[g["id"] == i, d].to_numpy(), ddof=1) for i in ids])
        n = np.array([len(g.loc[g["id"] == i, d]) for i in ids])
        noise_sd = float(np.sqrt(np.mean(within / np.maximum(n, 1))))
        report["noise_share_exploratory"][d] = {
            "noise_sd_of_mean": noise_sd,
            "model_signed_residual_sd": report["residuals_oof"][d]["signed_residual_sd"],
            "ratio_noise_var_to_resid_var": float(noise_sd ** 2 / max(report["residuals_oof"][d]["signed_residual_sd"] ** 2, 1e-12)),
            "caveat": "iid raters; not a strict MAE bound",
        }

    (a.out_dir / "label_noise.json").write_text(json.dumps(report, indent=2) + "\n")
    for src in ("reader", "writer", "combined"):
        print(f"== {src} ==")
        for d in DIMS:
            s = report["sources"][src][d]
            print(f'{d}: n={s["n_mean"]:.2f} SD={s["within_sd_mean"]:.3f} '
                  f'ICC11={s["icc"]["icc_1_1"]:.3f} split={s["split_half"]["split_half_r"]:.3f} '
                  f'|mean-gold|={s["mean_abs_diff_to_gold"]:.3f}')
    print("== reader vs writer ==")
    for d in DIMS:
        x = report["reader_vs_writer"][d]
        print(f'{d}: writer-reader={x["writer_minus_reader"]:+.3f} p={x["paired_t_p"]:.3g} '
              f'r={x["pearson_reader_writer"]:.3f} n={x["n_paired"]}')
    print("== OOF residuals ==")
    for d in DIMS:
        x = report["residuals_oof"][d]
        print(f'{d}: signedSD={x["signed_residual_sd"]:.3f} absSD={x["abs_error_sd"]:.3f} '
              f'RMSE={x["rmse"]:.3f} bias={x["bias"]:+.3f}')
    print("WROTE", a.out_dir / "label_noise.json")


if __name__ == "__main__":
    main()
