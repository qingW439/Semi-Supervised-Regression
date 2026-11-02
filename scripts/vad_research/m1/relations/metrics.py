"""Pairwise relation metrics for M1-R0."""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr


def pairwise_metrics(pred: np.ndarray, true: np.ndarray, direction_invariant: bool = False):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    n = len(true)
    tau = kendalltau(pred, true).statistic if n > 2 and np.std(pred) > 1e-12 else np.nan
    if direction_invariant and np.isfinite(tau):
        tau = abs(tau)
    ii, jj = np.triu_indices(n, 1)
    dp = pred[ii] - pred[jj]
    dt = true[ii] - true[jj]
    ok = np.abs(dt) > 1e-12
    acc = float(np.mean(np.sign(dp[ok]) == np.sign(dt[ok]))) if ok.any() else np.nan
    if direction_invariant and np.isfinite(acc):
        acc = max(acc, 1.0 - acc)
    rho = spearmanr(np.abs(dp[ok]), np.abs(dt[ok])).statistic if ok.sum() > 2 else np.nan
    return {"tau": float(tau) if np.isfinite(tau) else np.nan,
            "acc": float(acc) if np.isfinite(acc) else np.nan,
            "dist_rho": float(rho) if np.isfinite(rho) else np.nan,
            "n_pairs": int(ok.sum())}


def cr_ir(pred_teacher, pred_new, true):
    pred_teacher = np.asarray(pred_teacher); pred_new = np.asarray(pred_new); true = np.asarray(true)
    ii, jj = np.triu_indices(len(true), 1)
    dt = np.sign(true[ii] - true[jj])
    ok = np.abs(dt) > 1e-12
    t_ok = np.sign(pred_teacher[ii] - pred_teacher[jj]) == dt
    n_ok = np.sign(pred_new[ii] - pred_new[jj]) == dt
    wrong_t = np.sum(ok & ~t_ok)
    corrected = np.sum(ok & ~t_ok & n_ok)
    correct_t = np.sum(ok & t_ok)
    introduced = np.sum(ok & t_ok & ~n_ok)
    return {
        "n_teacher_wrong": int(wrong_t),
        "n_corrected": int(corrected),
        "cr": float(corrected / wrong_t) if wrong_t else float("nan"),
        "n_teacher_correct": int(correct_t),
        "n_introduced": int(introduced),
        "ir": float(introduced / correct_t) if correct_t else float("nan"),
        "net": int(corrected - introduced),
    }
