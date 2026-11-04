#!/usr/bin/env python3
"""Closed-form screening of information sources that could raise rho = r * gamma
(the quantity the M1 admission line is actually constrained by).

All variants share: frozen features, exactly the same 500 labels (lb500_s0), dev n=1000,
alpha chosen by 5-fold CV inside the labelled subset.  What differs is the information
each variant is allowed to use:

  R0   ridge                                   (baseline)
  IV   ridge weighted by 1 / annotator variance (reader/writer SD, N)  <- labelled side, new info
  ST   ridge + pseudo-labelled unlabelled pool, weight w               <- unlabelled side, self-training
  ST-IV same, with annotator variance weights on the real labels
  SPEC fixed-scale Nystrom eigenfeatures on all proxy covariates (ICML26-style), and [raw; spec]

rho = 1 - sigma2_noise(labels) / Var(y) is what a perfect model would reach (0.661 macro),
so this table shows how much of the remaining gap each information source closes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

S = 4.0
ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
DATA = Path("/root/autodl-tmp/rankup_project/data")


def metrics(p, y):
    mp, my = p.mean(), y.mean()
    vp, vy = p.var(), y.var()
    cov = ((p - mp) * (y - my)).mean()
    return dict(mae=float(np.abs(p - y).mean() * S), pearson=float(cov / np.sqrt(vp * vy)),
                sd_ratio=float(np.sqrt(vp / vy)), ccc=float(2 * cov / (vp + vy + (mp - my) ** 2)),
                r2=float(1 - ((p - y) ** 2).mean() / vy))


def macro(pred, y):
    return {k: float(np.mean([metrics(pred[:, i], y[:, i])[k] for i in range(3)]))
            for k in ("mae", "pearson", "sd_ratio", "ccc", "r2")}


class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, Y, sw=None):
        sw = np.ones(len(X)) if sw is None else np.asarray(sw, float)
        sw = sw / sw.mean()
        self.xm = np.average(X, axis=0, weights=sw)[None, :]
        self.ym = np.average(Y, axis=0, weights=sw)[None, :]
        Xc, Yc = X - self.xm, Y - self.ym
        G = (Xc * sw[:, None]).T @ Xc + self.alpha * np.eye(X.shape[1])
        self.W = np.linalg.solve(G, (Xc * sw[:, None]).T @ Yc)
        return self

    def predict(self, X):
        return (X - self.xm) @ self.W + self.ym


def cv_alpha(X, Y, sw=None):
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    best, best_err = ALPHAS[0], np.inf
    for a in ALPHAS:
        errs = []
        for f in np.array_split(idx, 5):
            tr = np.setdiff1d(idx, f)
            m = Ridge(a).fit(X[tr], Y[tr], None if sw is None else sw[tr])
            errs.append(np.abs(m.predict(X[f]) - Y[f]).mean())
        e = float(np.mean(errs))
        if e < best_err:
            best, best_err = a, e
    return best


def annotator_noise(ids):
    """Per-sample variance of the published label estimate (reader+writer)."""
    rd = pd.read_csv(DATA / "emobank_source/EmoBank/corpus/reader.csv")
    wr = pd.read_csv(DATA / "emobank_source/EmoBank/corpus/writer.csv")
    rd = rd.set_index("id")
    wr = wr.set_index("id")
    out = np.full((len(ids), 3), np.nan)
    for i, k in enumerate(ids):
        nr = float(rd.at[k, "N"]) if k in rd.index else 0.0
        nw = float(wr.at[k, "N"]) if k in wr.index else 0.0
        if nr + nw == 0:
            continue
        for j, d in enumerate("VAD"):
            sr = float(rd.at[k, "std" + d]) if k in rd.index else 0.0
            sw_ = float(wr.at[k, "std" + d]) if k in wr.index else 0.0
            out[i, j] = (nr * sr ** 2 + nw * sw_ ** 2) / (nr + nw) ** 2
    return out


def median_gamma(X, seed=0):
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), size=min(800, len(X)), replace=False)]
    d2 = np.maximum(2 - 2 * (sub @ sub.T), 0.0)
    return 1.0 / (2.0 * np.median(d2[d2 > 0]))


def nystrom_std(Xall, n_land=2000, n_eig=512, seed=0):
    """Nystrom eigenfeatures, each column standardised (fixes the scale bug of v1)."""
    rng = np.random.default_rng(seed)
    g = median_gamma(Xall, seed)
    li = rng.choice(len(Xall), size=min(n_land, len(Xall)), replace=False)
    L = Xall[li]
    d2 = (Xall * Xall).sum(1)[:, None] + (L * L).sum(1)[None, :] - 2 * Xall @ L.T
    K = np.exp(-g * np.clip(d2, 0, None))
    dLL = np.maximum(2 - 2 * (L @ L.T), 0.0)
    W = np.exp(-g * dLL)
    lam, U = np.linalg.eigh(W)
    o = np.argsort(lam)[::-1]
    lam = np.clip(lam[o][:n_eig], 1e-6, None)
    U = U[:, o][:, :n_eig]
    Z = K @ (U / np.sqrt(lam))
    Z = Z - Z.mean(0, keepdims=True)
    Z = Z / Z.std(0, keepdims=True).clip(1e-8)
    return Z, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/auto_search/diag_info_sources.json"))
    a = ap.parse_args()
    z = np.load(a.cache, allow_pickle=True)
    X, Xd, order = z["Ftr"], z["Fdev"], [str(x) for x in z["order"]]
    lb = pd.read_parquet(DATA / "emobank_processed/lb500_s0/train_lb.parquet")
    dv = pd.read_parquet(DATA / "emobank_processed/dev_gold.parquet")
    ydev = dv[["V_norm", "A_norm", "D_norm"]].to_numpy(float)
    pos = {k: i for i, k in enumerate(order)}
    lab = np.array([pos[i] for i in lb["id"]])
    tr, ulb = lab[:500], lab[500:]
    Y = lb[["V_norm", "A_norm", "D_norm"]].to_numpy(float)
    res = {}

    def add(name, pred, extra=None):
        res[name] = macro(pred, ydev)
        if extra:
            res[name].update(extra)
        print(f"{name:14s} MAE={res[name]['mae']:.5f} r={res[name]['pearson']:.4f} "
              f"sd={res[name]['sd_ratio']:.4f} CCC={res[name]['ccc']:.4f} "
              f"R2={res[name]['r2']:.4f} rho={res[name]['pearson']*res[name]['sd_ratio']:.4f}",
              flush=True)

    # R0
    al = cv_alpha(X[tr], Y)
    add("R0_raw", Ridge(al).fit(X[tr], Y).predict(Xd), {"alpha": al})

    # IV: annotator-variance weighting
    noise = annotator_noise(lb["id"].tolist())
    inv = 1.0 / (noise + 1e-4)
    inv = np.nan_to_num(inv, nan=np.nanmedian(inv))
    for name, sw in (("IV_perdim", inv.mean(1)), ("IV_scalar", 1.0 / (noise.mean(1) + 1e-4))):
        for beta in (0.5, 1.0):
            w = sw ** beta
            al = cv_alpha(X[tr], Y, w)
            add(f"{name}_b{beta}", Ridge(al).fit(X[tr], Y, w).predict(Xd), {"alpha": al})

    # ST: closed-form self-training with correct scaling
    base = Ridge(cv_alpha(X[tr], Y)).fit(X[tr], Y)
    p_ulb = base.predict(X[ulb])
    for w in (0.02, 0.05, 0.1, 0.3):
        Xa, Ya = np.vstack([X[tr], X[ulb]]), np.vstack([Y, p_ulb])
        sweight = np.concatenate([np.ones(len(tr)), w * np.ones(len(ulb))])
        al = cv_alpha(X[tr], Y)
        add(f"ST_w{w}", Ridge(al).fit(Xa, Ya, sweight).predict(Xd), {"alpha": al})
    # ST-IV
    for w in (0.05, 0.1):
        Xa, Ya = np.vstack([X[tr], X[ulb]]), np.vstack([Y, p_ulb])
        sweight = np.concatenate([inv.mean(1) / np.mean(inv.mean(1)), w * np.ones(len(ulb))])
        al = cv_alpha(X[tr], Y, inv.mean(1))
        add(f"STIV_w{w}", Ridge(al).fit(Xa, Ya, sweight).predict(Xd), {"alpha": al})

    # SPEC (fixed scale) and [raw; spec]
    Z, g = nystrom_std(np.vstack([X, Xd]))
    Zt, Zd = Z[:len(X)], Z[len(X):]
    Zl = Zt[tr]
    al = cv_alpha(Zl, Y)
    add("SPEC", Ridge(al).fit(Zl, Y).predict(Zd), {"alpha": al, "gamma": g})
    for lam in (1.0, 3.0, 10.0):
        Atr, Ad = np.hstack([X[tr], lam * Zl]), np.hstack([Xd, lam * Zd])
        al = cv_alpha(Atr, Y)
        add(f"RAWSPEC_l{lam}", Ridge(al).fit(Atr, Y).predict(Ad), {"alpha": al, "lam": lam})

    print()
    print("rho = r * sd_ratio  (perfect model -> 0.661 macro)")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["ccc"]):
        print(f"{k:16s} rho={v['pearson']*v['sd_ratio']:.4f} r={v['pearson']:.4f} "
              f"sd={v['sd_ratio']:.4f} MAE={v['mae']:.5f} CCC={v['ccc']:.4f} R2={v['r2']:.4f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
