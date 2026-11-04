#!/usr/bin/env python3
"""DIAGNOSTIC: can unlabeled proxy covariates improve the closed-form readout?

Reference mechanism (verified): Kim & Kim, "Semi-Supervised Learning with Noisy Proxy
Covariates: Generalization Bounds and Distribution Regression", ICML 2026
(arXiv:2606.00512): learn kernel eigenfeatures from ALL proxy covariates (labelled +
unlabelled) and fit a ridge predictor on the labelled part.

Adapted to frozen RoBERTa sentence embeddings (768-d, L2-normalised), lb500 seed 0,
dev n=1000.  Every variant uses exactly the same 500 labels; only what the unlabelled
7562 texts may contribute differs.  Alpha / lambda are chosen by 5-fold CV inside the
labelled subset only.

Variants
  raw        : ridge on the 768-d frozen features                (baseline S0)
  krr        : exact kernel ridge with the same RBF kernel
  spec       : Nystrom RBF eigenfeatures only                    (S2, previously NO-GO)
  raw+spec   : ridge on [raw ; lambda * spec]
  selftrain  : ridge on the 500 labels + pseudo-labelled unlabelled pool, weight w
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

S = 4.0
CACHE = Path("outputs/auto_search/features_frozen.npz")
ROOT = Path("/root/autodl-tmp/rankup_project/data")
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]


def metrics(p, y):
    mp, my = p.mean(), y.mean()
    vp, vy = p.var(), y.var()
    cov = ((p - mp) * (y - my)).mean()
    return dict(mae=float(np.abs(p - y).mean() * S),
                pearson=float(cov / np.sqrt(vp * vy)),
                sd_ratio=float(np.sqrt(vp / vy)),
                ccc=float(2 * cov / (vp + vy + (mp - my) ** 2)),
                r2=float(1 - ((p - y) ** 2).mean() / vy))


def macro_metrics(pred, y):
    return {k: float(np.mean([metrics(pred[:, i], y[:, i])[k] for i in range(3)]))
            for k in ("mae", "pearson", "sd_ratio", "ccc", "r2")}


class Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, Y, sw=None):
        if sw is None:
            sw = np.ones(len(X))
        sw = np.asarray(sw, float)
        sw = sw / sw.mean()
        self.xm = np.average(X, axis=0, weights=sw)[None, :]
        self.ym = np.average(Y, axis=0, weights=sw)[None, :]
        Xc = X - self.xm
        Yc = Y - self.ym
        G = (Xc * sw[:, None]).T @ Xc + self.alpha * np.eye(X.shape[1])
        self.W = np.linalg.solve(G, (Xc * sw[:, None]).T @ Yc)
        return self

    def predict(self, X):
        return (X - self.xm) @ self.W + self.ym


def cv_alpha(X, Y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    best, best_err = ALPHAS[0], np.inf
    for a in ALPHAS:
        errs = []
        for f in np.array_split(idx, k):
            tr = np.setdiff1d(idx, f)
            m = Ridge(a).fit(X[tr], Y[tr])
            errs.append(np.abs(m.predict(X[f]) - Y[f]).mean())
        e = float(np.mean(errs))
        if e < best_err:
            best, best_err = a, e
    return best


def median_gamma(X, seed=0):
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), size=min(800, len(X)), replace=False)]
    d2 = np.maximum(2 - 2 * (sub @ sub.T), 0.0)
    return float(1.0 / (2.0 * np.median(d2[d2 > 0])))


def rbf(A, B, gamma):
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * (A @ B.T)
    return np.exp(-gamma * np.clip(d2, 0, None))


def nystrom(X_all, n_land=2000, n_eig=512, gamma=None, seed=0):
    rng = np.random.default_rng(seed)
    gamma = median_gamma(X_all, seed) if gamma is None else gamma
    li = rng.choice(len(X_all), size=min(n_land, len(X_all)), replace=False)
    L = X_all[li]
    lam, U = np.linalg.eigh(rbf(L, L, gamma))
    o = np.argsort(lam)[::-1]
    lam = np.clip(lam[o][:n_eig], 1e-8, None)
    U = U[:, o][:, :n_eig]
    Z = rbf(X_all, L, gamma) @ (U / np.sqrt(lam))
    Z = Z / np.sqrt(len(X_all))
    return Z, gamma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--n-labeled", type=int, default=500)
    ap.add_argument("--n-eig", type=int, default=512)
    ap.add_argument("--out", type=Path, default=Path("outputs/auto_search/diag_proxy_covariates.json"))
    a = ap.parse_args()

    z = np.load(a.cache, allow_pickle=True)
    X, Xd, order = z["Ftr"], z["Fdev"], [str(x) for x in z["order"]]
    lb = pd.read_parquet(ROOT / "emobank_processed/lb500_s0/train_lb.parquet")
    dv = pd.read_parquet(ROOT / "emobank_processed/dev_gold.parquet")
    ydev = dv[["V_norm", "A_norm", "D_norm"]].to_numpy(float)
    pos = {k: i for i, k in enumerate(order)}
    lab = np.array([pos[i] for i in lb["id"]])
    n = a.n_labeled
    tr, ulb = lab[:n], lab[n:]
    Y = lb[["V_norm", "A_norm", "D_norm"]].to_numpy(float)[:n]
    res = {}

    # 1. raw features
    al = cv_alpha(X[tr], Y)
    res["raw"] = dict(alpha=al, **macro_metrics(Ridge(al).fit(X[tr], Y).predict(Xd), ydev))
    print("raw      ", {k: round(v, 4) for k, v in res["raw"].items() if k != "alpha"}, flush=True)

    # 2. exact kernel ridge
    gam = median_gamma(X)
    K, Kd = rbf(X[tr], X[tr], gam), rbf(Xd, X[tr], gam)
    best = (None, np.inf)
    for al2 in (1e-3, 1e-2, 1e-1, 1.0):
        m = K @ np.linalg.solve(K + al2 * np.eye(len(K)), Y - Y.mean(0)) + Y.mean(0)
        e = float(np.abs(m - Y).mean())
        if e < best[1]:
            best = (al2, e)
    al2 = best[0]
    pred = Kd @ np.linalg.solve(K + al2 * np.eye(len(K)), Y - Y.mean(0)) + Y.mean(0)
    res["krr"] = dict(alpha=al2, gamma=gam, **macro_metrics(pred, ydev))
    print("krr      ", {k: round(v, 4) for k, v in res["krr"].items() if k not in ("alpha", "gamma")}, flush=True)

    # 3/4. Nystrom spectral features on ALL (labelled + unlabelled) proxy covariates
    Z, gam2 = nystrom(np.vstack([X, Xd]), n_eig=a.n_eig)
    Ztr_all, Zdev = Z[:len(X)], Z[len(X):]
    Zl = Ztr_all[tr]
    al = cv_alpha(Zl, Y)
    res["spec"] = dict(alpha=al, gamma=gam2, n_eig=a.n_eig,
                       **macro_metrics(Ridge(al).fit(Zl, Y).predict(Zdev), ydev))
    print("spec     ", {k: round(v, 4) for k, v in res["spec"].items()
                        if k not in ("alpha", "gamma", "n_eig")}, flush=True)

    best = None
    for lam in (0.1, 0.3, 1.0, 3.0):
        Atr, Ad = np.hstack([X[tr], lam * Zl]), np.hstack([Xd, lam * Zdev])
        al = cv_alpha(Atr, Y)
        m = macro_metrics(Ridge(al).fit(Atr, Y).predict(Ad), ydev)
        print(f"raw+spec lam={lam}", {k: round(v, 4) for k, v in m.items()}, flush=True)
        if best is None or m["mae"] < best[2]["mae"]:
            best = (lam, al, m)
    res["raw+spec"] = dict(lam=best[0], alpha=best[1], **best[2])

    # 5. closed-form pseudo-label self-training (weighted)
    base = Ridge(cv_alpha(X[tr], Y)).fit(X[tr], Y)
    p_ulb = base.predict(X[ulb])
    for w in (0.05, 0.1, 0.3, 1.0):
        Xa, Ya = np.vstack([X[tr], X[ulb]]), np.vstack([Y, p_ulb])
        sw = np.concatenate([np.ones(len(tr)), w * np.ones(len(ulb))])
        al = cv_alpha(X[tr], Y)
        r = Ridge(al).fit(Xa, Ya, sw)
        res[f"selftrain_w{w}"] = dict(alpha=al, w=w, **macro_metrics(r.predict(Xd), ydev))
        print(f"selftrain w={w}", {k: round(v, 4) for k, v in res[f"selftrain_w{w}"].items()
                                   if k not in ("alpha", "w")}, flush=True)

    print()
    print(f"{'variant':20s} {'MAE':>8s} {'r':>7s} {'sd':>7s} {'CCC':>7s} {'R2':>7s}")
    for k, v in res.items():
        print(f"{k:20s} {v['mae']:8.5f} {v['pearson']:7.4f} {v['sd_ratio']:7.4f} "
              f"{v['ccc']:7.4f} {v['r2']:7.4f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
