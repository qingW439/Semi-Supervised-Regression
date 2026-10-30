#!/usr/bin/env python3
"""Evaluate candidate reliability signals on the saved ceiling arrays.

All metrics are per *element* (sample x dimension), because the JACF loss
normalises over the B x 3 elements of a batch.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def safe_spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def within_batch(x, y, bs=64, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y) // 3
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, bs)
        e = y.reshape(n, 3)[idx].reshape(-1)
        s = x.reshape(n, 3)[idx].reshape(-1)
        v = safe_spearman(s, e)
        if np.isfinite(v):
            vals.append(-v)
    return float(np.mean(vals)) if vals else np.nan


def capture(rel, err, bs=64, tau=8.0):
    n = len(err) // 3
    nb = n // bs
    base, orc, sig = [], [], []
    for b in range(nb):
        sl = slice(b * bs * 3, (b + 1) * bs * 3)
        e = err[sl]
        r = rel[sl]
        k = len(e)
        rk_o = np.empty(k)
        rk_o[np.argsort(e)] = np.arange(k)
        w_o = np.exp(-rk_o / tau)
        orc.append((w_o * e).sum() / w_o.sum())
        rk_s = np.empty(k)
        order = np.argsort(-r)
        rk_s[order] = np.arange(k)
        w_s = np.exp(-rk_s / tau)
        sig.append((w_s * e).sum() / w_s.sum())
        base.append(e.mean())
    base, orc, sig = np.mean(base), np.mean(orc), np.mean(sig)
    return {
        "uniform": float(base), "oracle": float(orc), "signal": float(sig),
        "gain_pct": float(100 * (base - sig) / base),
        "capture_pct": float(100 * (base - sig) / (base - orc)) if abs(base - orc) > 1e-12 else np.nan,
    }


def filter_capture(rel, err, q=0.7):
    n = len(err)
    k = max(8, int(round(q * n)))
    base = err.mean()
    sig = err[np.argsort(-rel)[:k]].mean()
    orc = err[np.argsort(err)[:k]].mean()
    return {
        "gain_pct": float(100 * (base - sig) / base),
        "capture_pct": float(100 * (base - sig) / (base - orc)) if abs(base - orc) > 1e-12 else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    y = z["y"]; mu_t = z["mu_t"]; mu_a = z["mu_a"]; v_t = z["v_t"]; v_a = z["v_a"]
    target = z["target"]; alpha_j = z["alpha_j"]; conf = z["conf"]; density = z["density"]
    nb_range = z["nb_range"]; mc_range = z["mc_range"]; r_jacf = z["r_jacf"]
    conflict = z["conflict"]; support = z["support"]; e_f = z["e_f"]
    n = len(y)
    alpha = 0.75
    err = e_f.reshape(-1)  # sample-major: V,A,D per sample
    signals = {
        "v_a": -v_a,
        "v_prop": -(alpha ** 2 * v_a + (1 - alpha) ** 2 * v_t),
        "v_sum": -(v_a + v_t),
        "v_max": -np.maximum(v_a, v_t),
        "v_f": -z["v_f"],
        "r_jacf": r_jacf,
        "conf": np.broadcast_to(conf[:, None], (n, 3)),
        "density": np.broadcast_to(density[:, None], (n, 3)),
        "neighbor_range": -nb_range,
        "v_a_over_conf": -v_a / (conf[:, None] + 1e-3),
        "rank_v_a_plus_range": None,
        "rank_v_prop_plus_conf": None,
    }
    # rank combinations
    def rank(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = np.broadcast_to(x[:, None], (n, 3))
        x = x.reshape(-1)
        r = np.empty(len(x))
        r[np.argsort(x)] = np.arange(len(x))
        return (r / max(1, len(x) - 1)).reshape(n, 3)
    signals["rank_v_a_plus_range"] = -(rank(v_a) + rank(nb_range))
    signals["rank_v_prop_plus_conf"] = -(rank(alpha ** 2 * v_a + (1 - alpha) ** 2 * v_t) - rank(conf))
    rows = []
    for name, s in signals.items():
        s = s.reshape(-1)
        rows.append({
            "signal": name,
            "spearman_global": safe_spearman(-s, err),  # reliability should anticorrelate
            "auc_high_error": float(roc_auc_score((err > np.median(err)).astype(int), -s)),
            "within_batch": within_batch(s, err),
            "capture": capture(s, err, tau=8.0),
            "filter70": filter_capture(s, err, 0.7),
        })
    rows.sort(key=lambda r: -r["capture"]["capture_pct"])
    out = {"n": n, "rows": rows}
    a.out.write_text(json.dumps(out, indent=2))
    for r in rows:
        c = r["capture"]
        print(f'{r["signal"]:22s} rho={r["spearman_global"]:+.3f} auc={r["auc_high_error"]:.3f} '
              f'wb={r["within_batch"]:+.3f} gain={c["gain_pct"]:+.1f}% cap={c["capture_pct"]:+.1f}%')


if __name__ == "__main__":
    main()
