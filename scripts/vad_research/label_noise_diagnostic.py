#!/usr/bin/env python3
"""Label-noise / annotation-consistency diagnostic for EmoBank TRAIN-500.

Uses the released individual reader/writer ratings.  Verifies the official
label aggregation, computes within-item dispersion, one-way ICC, split-half
reliability, and relates dispersion to model error.  CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DIMS = ("V", "A", "D")


def icc_oneway(groups: list[np.ndarray]) -> dict:
    groups = [g for g in groups if len(g) >= 2]
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n_total - k)
    n0 = (n_total - sum(len(g) ** 2 for g in groups) / n_total) / (k - 1)
    icc11 = (msb - msw) / (msb + (n0 - 1) * msw) if msb + (n0 - 1) * msw > 0 else np.nan
    icc1k = (msb - msw) / msb if msb > 0 else np.nan
    return {"icc_1_1": float(icc11), "icc_1_k": float(icc1k), "msb": float(msb), "msw": float(msw), "n0": float(n0)}


def split_half(groups: dict[str, np.ndarray], reps=200, seed=0) -> dict:
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
    r = float(np.nanmean(rs)) if rs else np.nan
    sb = float(2 * r / (1 + r)) if np.isfinite(r) and r > -1 else np.nan
    return {"mean_split_half_r": r, "spearman_brown_full": sb, "n_items": len(ids), "reps": reps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--emobank-root", type=Path, required=True)
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--dev-pred", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    lb = pd.read_parquet(a.data_root / "lb500_s0" / "train_lb.parquet")
    r = pd.read_csv(a.emobank_root / "corpus" / "individual_reader_ratings.csv")
    w = pd.read_csv(a.emobank_root / "corpus" / "individual_writer_ratings.csv")
    r["source"] = "reader"; w["source"] = "writer"
    allr = pd.concat([r, w], ignore_index=True)
    # official EmoBank aggregation filters out all-1 ratings before averaging
    before = len(allr)
    allr = allr[~((allr["V"] == 1) & (allr["A"] == 1) & (allr["D"] == 1))].copy()
    filtered = before - len(allr)
    gold = pd.read_csv(a.emobank_root / "corpus" / "emobank.csv", dtype={"id": str}).set_index("id")

    # verify gold = weighted mean of all individual ratings
    gmean = allr.groupby("id")[list(DIMS)].mean()
    check = gold.loc[lb["id"], list(DIMS)].to_numpy()
    recomputed = gmean.loc[lb["id"], list(DIMS)].to_numpy()
    max_diff = float(np.max(np.abs(check - recomputed)))
    agg_ok = max_diff < 0.02

    oof = pd.read_parquet(a.oof).set_index("id").loc[lb["id"].tolist()].reset_index()
    oof_err = np.abs(oof[[f"raw_{d}" for d in DIMS]].to_numpy() - lb.loc[:, [f"{d}_norm" for d in DIMS]].to_numpy()) * 4

    report = {
        "protocol": {
            "note": "individual ratings used only for TRAIN-500 and DEV diagnostics; no ulb hidden labels",
            "gold_aggregation": "weighted mean of all reader+writer individual ratings",
            "max_abs_gold_recompute_diff": max_diff,
            "aggregation_verified": agg_ok,
            "all_one_rows_filtered": filtered,
            "n_train": len(lb),
        },
        "dims": {},
    }
    sample_rows = []
    for di, d in enumerate(DIMS):
        # groups for TRAIN-500
        groups = []
        sds, ranges, ns = [], [], []
        for i in lb["id"]:
            g = allr.loc[allr["id"] == i, d].to_numpy(np.float64)
            groups.append(g)
            sds.append(float(g.std(ddof=1)) if len(g) > 1 else 0.0)
            ranges.append(float(g.max() - g.min()) if len(g) else 0.0)
            ns.append(len(g))
            sample_rows.append({"id": i, "dim": d, "n": len(g), "mean": float(g.mean()),
                                "sd": float(g.std(ddof=1)) if len(g) > 1 else 0.0,
                                "range": float(g.max() - g.min()) if len(g) else 0.0})
        sds = np.array(sds); ranges = np.array(ranges); ns = np.array(ns)
        icc = icc_oneway(groups)
        sh = split_half({i: allr.loc[allr["id"] == i, d].to_numpy(np.float64) for i in lb["id"]})
        # dispersion vs model error
        oof_s = spearmanr(sds, oof_err[:, di]).statistic
        # extreme split on gold intensity
        inten = np.abs(lb[d].to_numpy() - 3.0)
        hi = inten >= np.quantile(inten, 0.8)
        lo = inten <= np.quantile(inten, 0.2)
        # noise variance of the aggregated mean
        within_var = np.array([np.var(g, ddof=1) for g in groups])
        noise_sd = float(np.sqrt(np.mean(within_var / np.maximum(ns, 1))))
        # model residual sd
        model_resid_sd = float(np.std(oof_err[:, di]))
        report["dims"][d] = {
            "n_mean": float(ns.mean()), "n_min": int(ns.min()), "n_max": int(ns.max()),
            "within_sd_mean": float(sds.mean()), "within_sd_median": float(np.median(sds)),
            "within_range_mean": float(ranges.mean()),
            "icc": icc, "split_half": sh,
            "spearman_sd_vs_oof_error": float(oof_s),
            "high_intensity_sd": float(sds[hi].mean()), "low_intensity_sd": float(sds[lo].mean()),
            "estimated_noise_sd_of_mean": noise_sd,
            "model_oof_residual_sd": model_resid_sd,
            "noise_share_of_residual_var": float(noise_sd ** 2 / max(model_resid_sd ** 2, 1e-12)),
        }
        print(f'{d}: ICC11={icc["icc_1_1"]:.3f} split-half={sh["mean_split_half_r"]:.3f} '
              f'SD={sds.mean():.3f} rho(SD,err)={oof_s:.3f} noiseSD={noise_sd:.3f} residSD={model_resid_sd:.3f}')
    pd.DataFrame(sample_rows).to_csv(a.out_dir / "label_noise_samples.csv", index=False)
    (a.out_dir / "label_noise.json").write_text(json.dumps(report, indent=2) + "\n")
    print("WROTE", a.out_dir / "label_noise.json")


if __name__ == "__main__":
    main()
