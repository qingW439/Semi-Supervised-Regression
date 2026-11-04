#!/usr/bin/env python3
"""Ridge alpha / preprocessing stability diagnostic (E0* audit phase 1, item 3).

Questions
  Q1 how much does the *choice* of (prep, alpha) matter for the OOF score?
  Q2 is the inner-CV choice stable across bootstrap resamples of the 400 training samples?
  Q3 does a single global (prep, alpha) beat per-fold selection on OOF?

Everything runs on the per-fold `features_labeled.npy` / `features_holdout.npy` produced by
the OOF trainer, so no encoder is needed (CPU only).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DIMS = ("V", "A", "D")
SCALE = 4.0
ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
PREPS = ("center", "zscore")


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
    return {k: float(np.mean([metrics(p[:, i], y[:, i])[k] for i in range(3)]))
            for k in ("mae", "pearson", "sd_ratio", "ccc", "r2")}


def fit(Xtr, Ytr, alpha, prep):
    mu = Xtr.mean(0, keepdims=True)
    Xc = Xtr - mu
    sd = None
    if prep == "zscore":
        sd = Xc.std(0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        Xc = Xc / sd
    ym = Ytr.mean(0, keepdims=True)
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ (Ytr - ym))

    def pred(Xq):
        Z = Xq - mu
        if sd is not None:
            Z = Z / sd
        return Z @ W + ym
    return pred


def inner_pick(X, Y, idx, k=5, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(idx)
    best, be = (PREPS[0], ALPHAS[0]), np.inf
    for prep in PREPS:
        for al in ALPHAS:
            errs = []
            for f in np.array_split(order, k):
                tr = np.setdiff1d(order, f)
                errs.append(np.abs(fit(X[tr], Y[tr], al, prep)(X[f]) - Y[f]).mean())
            e = float(np.mean(errs))
            if e < be:
                best, be = (prep, al), e
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/rankup_project/data/emobank_processed"))
    ap.add_argument("--mainline", type=Path, default=Path("/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc"))
    ap.add_argument("--out", type=Path, default=Path("outputs/audit_e0/ridge_stability.json"))
    ap.add_argument("--boot", type=int, default=30)
    a = ap.parse_args()

    grid_mae = {(prep, al): [] for prep in PREPS for al in ALPHAS}
    chosen, boot_freq = Counter(), Counter()
    oof_by_choice = {"inner": [], "fixed_center_0.1": [], "fixed_zscore_10": []}
    per_fold = []

    for s in (0, 1, 2):
        lb = pd.read_parquet(a.data_root / f"lb500_s{s}" / "train_lb.parquet")
        for f in range(5):
            d = (a.mainline / "outputs" / "oof" / f"fold{f}") if s == 0 else \
                (Path(".") / "outputs" / f"oof_seed{s}" / f"fold{f}")
            par = pd.read_parquet(d / "oof_predictions.parquet")
            hold = par["id"].tolist()
            Yh = lb.set_index("id").loc[hold][["V_norm", "A_norm", "D_norm"]].to_numpy(float)
            Yt = lb[~lb["id"].isin(set(hold))][["V_norm", "A_norm", "D_norm"]].to_numpy(float)
            Fl, Fh = np.load(d / "features_labeled.npy"), np.load(d / "features_holdout.npy")
            idx = np.arange(len(Fl))

            for (prep, al) in grid_mae:
                grid_mae[(prep, al)].append(metrics(fit(Fl, Yt, al, prep)(Fh), Yh)["mae"])

            pick = inner_pick(Fl, Yt, idx, seed=f)
            chosen[pick] += 1
            oof_by_choice["inner"].append(metrics(fit(Fl, Yt, pick[1], pick[0])(Fh), Yh))
            oof_by_choice["fixed_center_0.1"].append(metrics(fit(Fl, Yt, 0.1, "center")(Fh), Yh))
            oof_by_choice["fixed_zscore_10"].append(metrics(fit(Fl, Yt, 10.0, "zscore")(Fh), Yh))

            for b in range(a.boot):
                boot_idx = np.random.default_rng(1000 * s + f * 100 + b).choice(idx, len(idx), replace=True)
                bp = inner_pick(Fl[boot_idx], Yt[boot_idx], np.arange(len(boot_idx)), seed=b)
                boot_freq[bp] += 1
            per_fold.append({"seed": s, "fold": f, "pick": list(pick),
                             "mae_at_pick": grid_mae[(pick[0], pick[1])][-1]})
            print(f"seed{s} fold{f}: pick={pick} holdoutMAE={grid_mae[(pick[0], pick[1])][-1]:.5f}",
                  flush=True)

    out = {"mae_grid_mean": {f"{p}_a{al}": float(np.mean(v)) for (p, al), v in grid_mae.items()},
           "inner_choice_counts": {f"{p}_a{al}": c for (p, al), c in chosen.items()},
           "bootstrap_choice_freq": {f"{p}_a{al}": c for (p, al), c in boot_freq.most_common()},
           "oof_macro_by_strategy": {}, "per_fold": per_fold}
    for k, lst in oof_by_choice.items():
        out["oof_macro_by_strategy"][k] = {m: float(np.mean([x[m] for x in lst])) for m in
                                           ("mae", "pearson", "sd_ratio", "ccc", "r2")}

    print("\n=== holdout MAE averaged over 15 folds, by (prep, alpha) ===")
    for key, v in sorted(out["mae_grid_mean"].items(), key=lambda kv: kv[1]):
        print(f"  {key:16s} {v:.5f}")
    print("\n=== inner-CV choices ===", out["inner_choice_counts"])
    print("=== bootstrap choice frequency (top 6) ===",
          dict(list(out["bootstrap_choice_freq"].items())[:6]))
    print("\n=== OOF macro by strategy (pooled folds) ===")
    for k, v in out["oof_macro_by_strategy"].items():
        print(f"  {k:20s} MAE={v['mae']:.5f} r={v['pearson']:.4f} sd={v['sd_ratio']:.4f} "
              f"CCC={v['ccc']:.4f} R2={v['r2']:.4f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
