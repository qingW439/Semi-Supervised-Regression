#!/usr/bin/env python3
"""Reliability ceiling diagnostic for JACF on EmoBank.

This script uses the *hidden* gold labels of the official EmoBank train split
ONLY for analysis.  Those labels are never used to train any model here; they
are used to measure how much of the pseudo-label error is explainable and how
much gain an oracle reliability mechanism could possibly deliver.

Outputs
-------
* ``reliability_ceiling_*.npz``: per-sample arrays (predictions, signals, errors)
* ``reliability_ceiling_*.json``: error structure, oracle ceilings, signal
  screening table, and supervised learnability ceiling.

The script is deliberately self-contained so it can be re-run on any fixed
JACF checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402

DIMS = ("V", "A", "D")


# --------------------------------------------------------------------------- #
# data / model
# --------------------------------------------------------------------------- #
def load_data(args):
    lb = pd.read_parquet(args.data_root / f"lb{args.budget}_s{args.seed}" / "train_lb.parquet")
    ulb = pd.read_parquet(args.data_root / f"lb{args.budget}_s{args.seed}" / "train_ulb.parquet")
    raw = pd.read_csv(args.raw_csv, dtype={"id": str}, keep_default_na=False)
    raw = raw.set_index("id")
    missing = [i for i in ulb["id"] if i not in raw.index]
    if missing:
        raise RuntimeError(f"{len(missing)} unlabeled ids missing from raw EmoBank csv")
    hidden = raw.loc[ulb["id"].tolist(), ["V", "A", "D"]].to_numpy(dtype=np.float64)
    if not np.isfinite(hidden).all():
        raise RuntimeError("non-finite hidden labels")
    ulb = ulb.copy()
    for j, d in enumerate(DIMS):
        ulb[d] = hidden[:, j]
        ulb[f"{d}_norm"] = (hidden[:, j] - 1.0) / 4.0
    return lb.reset_index(drop=True), ulb.reset_index(drop=True)


class Encoder:
    def __init__(self, model, tokenizer, device, max_length=128, bs=64):
        self.model, self.tok, self.device, self.max_length, self.bs = model, tokenizer, device, max_length, bs

    @torch.inference_mode()
    def features(self, texts):
        self.model.encoder.eval()
        feats, norms = [], []
        for i in range(0, len(texts), self.bs):
            tok = self.tok(list(texts[i:i + self.bs]), padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt")
            tok = {k: v.to(self.device) for k, v in tok.items()}
            st = self.model.encoder(**tok, return_dict=True).last_hidden_state
            m = tok["attention_mask"].unsqueeze(-1).to(dtype=st.dtype)
            pooled = (st * m).sum(1) / m.sum(1).clamp_min(1)
            norms.append(pooled.float().norm(dim=-1).cpu())
            feats.append(F.normalize(pooled.float(), dim=-1).cpu())
        return torch.cat(feats), torch.cat(norms)

    @torch.inference_mode()
    def predict_det(self, texts):
        self.model.eval()
        out = []
        for i in range(0, len(texts), self.bs):
            tok = self.tok(list(texts[i:i + self.bs]), padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt")
            tok = {k: v.to(self.device) for k, v in tok.items()}
            out.append(self.model(**tok).float().cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def predict_mc(self, texts, passes):
        was = self.model.training
        self.model.train()
        chunks = []
        for i in range(0, len(texts), self.bs):
            tok = self.tok(list(texts[i:i + self.bs]), padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt")
            tok = {k: v.to(self.device) for k, v in tok.items()}
            ps = [self.model(**tok).float().cpu() for _ in range(passes)]
            chunks.append(torch.stack(ps, dim=0))
        self.model.train(was)
        return torch.cat(chunks, dim=1).numpy()  # [K,N,3]


def anchor_evidence(u_feat, l_feat, l_y, k, tau, eps_a=1e-4):
    sim = u_feat @ l_feat.T
    kk = min(k, l_feat.shape[0])
    topv, topi = sim.topk(kk, dim=1)
    w = torch.softmax(topv / tau, dim=1)
    nb = l_y[topi]
    mu_a = (w.unsqueeze(-1) * nb).sum(1)
    v_a = (w.unsqueeze(-1) * (nb - mu_a.unsqueeze(1)).square()).sum(1) + eps_a
    conf = topv[:, 0].clamp(0.0, 1.0)
    density = topv.mean(1)
    knn_dist = 1.0 - density
    nb_range = (nb.max(1).values - nb.min(1).values)
    return mu_a, v_a, conf, density, knn_dist, nb_range


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def bootstrap_spearman(x, y, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = safe_spearman(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def aurc(risk, error):
    """Area under risk-coverage curve (lower is better, random ~ mean error)."""
    risk = np.asarray(risk, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    ok = np.isfinite(risk) & np.isfinite(error)
    risk, error = risk[ok], error[ok]
    if len(error) == 0:
        return float("nan")
    order = np.argsort(-risk)  # highest risk first
    e = error[order]
    cum = np.cumsum(e) / (np.arange(len(e)) + 1)
    return float(np.mean(cum))


def capture_rank_weights(reliability, error, batch_size=64, taus=(2.0, 4.0, 8.0, 16.0)):
    """Fraction of oracle rank-weighting gain captured by a reliability score.

    Within each batch (the unit that matters for the normalised JACF loss), the
    most reliable sample gets weight 1, the next exp(-1/tau), ...  The oracle
    ranks by true pseudo-label error.  Returns capture in [ -inf, 1 ]; 0 means
    no better than uniform, 1 equals the oracle.
    """
    rel = np.asarray(reliability, dtype=np.float64)
    err = np.asarray(error, dtype=np.float64)
    ok = np.isfinite(rel) & np.isfinite(err)
    rel, err = rel[ok], err[ok]
    n = len(err)
    if n < batch_size:
        return {str(t): float("nan") for t in taus}
    out = {}
    nb = n // batch_size
    for tau in taus:
        base, oracle_v, sig_v = [], [], []
        for b in range(nb):
            s = slice(b * batch_size, (b + 1) * batch_size)
            e = err[s]
            # oracle: rank by ascending error
            rk_o = np.empty(batch_size)
            rk_o[np.argsort(e)] = np.arange(batch_size)
            w_o = np.exp(-rk_o / tau)
            oracle_v.append(float((w_o * e).sum() / w_o.sum()))
            # signal: higher reliability first
            rk_s = np.empty(batch_size)
            order = np.argsort(-rel[s])
            rk_s[order] = np.arange(batch_size)
            w_s = np.exp(-rk_s / tau)
            sig_v.append(float((w_s * e).sum() / w_s.sum()))
            base.append(float(e.mean()))
        base_m, oracle_m, sig_m = np.mean(base), np.mean(oracle_v), np.mean(sig_v)
        denom = base_m - oracle_m
        out[str(tau)] = {
            "uniform": float(base_m),
            "oracle": float(oracle_m),
            "signal": float(sig_m),
            "gain_pct": float(100.0 * (base_m - sig_m) / base_m),
            "capture_pct": float(100.0 * (base_m - sig_m) / denom) if abs(denom) > 1e-12 else float("nan"),
        }
    return out


def capture_filter(reliability, error, q=0.7):
    rel = np.asarray(reliability, dtype=np.float64)
    err = np.asarray(error, dtype=np.float64)
    ok = np.isfinite(rel) & np.isfinite(err)
    rel, err = rel[ok], err[ok]
    n = len(err)
    k = max(8, int(round(q * n)))
    keep_sig = np.argsort(-rel)[:k]
    keep_or = np.argsort(err)[:k]
    base = float(err.mean())
    sig = float(err[keep_sig].mean())
    oracle = float(err[keep_or].mean())
    return {
        "q": q,
        "uniform": base,
        "signal": sig,
        "oracle": oracle,
        "gain_pct": float(100.0 * (base - sig) / base),
        "capture_pct": float(100.0 * (base - sig) / (base - oracle)) if abs(base - oracle) > 1e-12 else float("nan"),
    }


def within_batch_spearman(reliability, error, batch_size=64, reps=100, seed=0):
    rng = np.random.default_rng(seed)
    rel = np.asarray(reliability, dtype=np.float64)
    err = np.asarray(error, dtype=np.float64)
    n = len(err)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, batch_size)
        v = safe_spearman(rel[idx], err[idx])
        if np.isfinite(v):
            # reliability should be negatively correlated with error
            vals.append(-v)
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# supervised learnability ceiling
# --------------------------------------------------------------------------- #
def supervised_ceiling(feats, target_err, target_alpha, target_anchor_better, n_splits=5, seed=0):
    """Train reliability predictors on half the unlabeled data (hidden labels)
    and evaluate on the other half.  This is an upper-bound analysis, not a
    deployable method."""
    rng = np.random.default_rng(seed)
    n = len(target_err)
    results = {}
    for name in ("ridge", "hgb"):
        rhos, r2s, maes, alphas_mae, aucs = [], [], [], [], []
        for split in range(n_splits):
            perm = rng.permutation(n)
            tr, te = perm[: n // 2], perm[n // 2:]
            if name == "ridge":
                sc = StandardScaler().fit(feats[tr])
                model = Ridge(alpha=10.0).fit(sc.transform(feats[tr]), target_err[tr])
                pred = model.predict(sc.transform(feats[te]))
            else:
                model = HistGradientBoostingRegressor(
                    max_iter=300, max_depth=6, learning_rate=0.05,
                    min_samples_leaf=40, l2_regularization=1.0,
                    random_state=seed + split,
                ).fit(feats[tr], target_err[tr])
                pred = model.predict(feats[te])
            rhos.append(safe_spearman(pred, target_err[te]))
            ss_res = np.sum((target_err[te] - pred) ** 2)
            ss_tot = np.sum((target_err[te] - target_err[te].mean()) ** 2)
            r2s.append(float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"))
            mae = float(np.mean(np.abs(pred - target_err[te])))
            maes.append(mae)
            # alpha* predictor: train a separate model on the same features
            if name == "ridge":
                am = Ridge(alpha=10.0).fit(sc.transform(feats[tr]), target_alpha[tr])
                ap = am.predict(sc.transform(feats[te]))
            else:
                am = HistGradientBoostingRegressor(
                    max_iter=300, max_depth=6, learning_rate=0.05,
                    min_samples_leaf=40, l2_regularization=1.0,
                    random_state=seed + split + 100,
                ).fit(feats[tr], target_alpha[tr])
                ap = am.predict(feats[te])
            ap = np.clip(ap, 0.0, 1.0)
            alphas_mae.append(float(np.mean(np.abs(ap - target_alpha[te]))))
        results[name] = {
            "spearman_pred_error": float(np.nanmean(rhos)),
            "r2_pred_error": float(np.nanmean(r2s)),
            "mae_pred_error": float(np.nanmean(maes)),
            "mae_alpha_star": float(np.nanmean(alphas_mae)),
        }
    # classifier: anchor better than teacher
    aucs = []
    for split in range(n_splits):
        perm = rng.permutation(n)
        tr, te = perm[: n // 2], perm[n // 2:]
        sc = StandardScaler().fit(feats[tr])
        clf = HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=40, l2_regularization=1.0,
            random_state=seed + split + 7,
        ).fit(sc.transform(feats[tr]), target_anchor_better[tr])
        p = clf.predict_proba(sc.transform(feats[te]))[:, 1]
        if len(np.unique(target_anchor_better[te])) > 1:
            aucs.append(float(roc_auc_score(target_anchor_better[te], p)))
    results["anchor_better_clf"] = {"auc": float(np.nanmean(aucs))}
    return results


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--raw-csv", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--mc-passes", type=int, default=5)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--v0", type=float, default=1e-3)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    t0 = time.time()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    lb, ulb = load_data(a)
    y = ulb.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float32)
    print(f"labeled={len(lb)} unlabeled={len(ulb)} hidden labels recovered", flush=True)

    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(device)
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    del ckpt
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    enc = Encoder(model, tok, device)

    l_feat, _ = enc.features(lb["text"].tolist())
    u_feat, u_norm = enc.features(ulb["text"].tolist())
    l_feat = l_feat.to(device)
    u_feat = u_feat.to(device)
    l_y = torch.tensor(lb.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float32), device=device)
    print(f"features done {time.time()-t0:.1f}s", flush=True)

    det = enc.predict_det(ulb["text"].tolist())
    mc = enc.predict_mc(ulb["text"].tolist(), a.mc_passes)
    mu_t = mc.mean(0)
    v_t = mc.var(0, ddof=0) + 1e-4
    mc_range = mc.max(0) - mc.min(0)
    mc_mad = np.abs(mc - mu_t[None]).mean(0)
    print(f"mc done {time.time()-t0:.1f}s", flush=True)

    mu_a, v_a, conf, density, knn_dist, nb_range = anchor_evidence(
        u_feat, l_feat, l_y, a.k, a.tau)
    mu_a, v_a = mu_a.cpu().numpy(), v_a.cpu().numpy()
    conf, density, knn_dist, nb_range = (x.cpu().numpy() for x in (conf, density, knn_dist, nb_range))

    alpha_fixed = a.alpha
    target = (1 - alpha_fixed) * mu_t + alpha_fixed * mu_a
    alpha_j = v_t / (v_t + v_a)
    target_j = (1 - alpha_j) * mu_t + alpha_j * mu_a
    v_f = v_t * v_a / (v_t + v_a)
    conflict = (mu_t - mu_a) ** 2 / (v_t + v_a)
    support = np.exp(-0.5 * conflict)
    r_jacf = support * (a.v0 / (a.v0 + v_f))
    diff_abs = np.abs(mu_t - mu_a)

    e_t = np.abs(mu_t - y)
    e_a = np.abs(mu_a - y)
    e_f = np.abs(target - y)
    e_j = np.abs(target_j - y)
    # oracle per-dim alpha (minimises |(1-a)mu_t + a mu_a - y|)
    denom = np.where(np.abs(mu_a - mu_t) < 1e-8, np.inf, mu_a - mu_t)
    alpha_star = np.clip((y - mu_t) / denom, 0.0, 1.0)
    alpha_star = np.where(np.isfinite(alpha_star), alpha_star, 0.0)
    target_star = (1 - alpha_star) * mu_t + alpha_star * mu_a
    e_star = np.abs(target_star - y)
    e_best_src = np.minimum(e_t, e_a)
    anchor_better = (e_a < e_t).astype(np.int32)

    # original [1,5] scale factor
    scale = 4.0
    err_stats = {}
    for name, e in (("teacher", e_t), ("anchor", e_a), ("fixed_alpha", e_f),
                    ("jacf_alpha", e_j), ("oracle_alpha", e_star), ("best_source", e_best_src)):
        err_stats[name] = {
            "macro_mae_norm": float(e.mean()),
            "macro_mae_orig": float(e.mean() * scale),
            "per_dim_mae_orig": [float(e[:, d].mean() * scale) for d in range(3)],
            "per_dim_std_orig": [float(e[:, d].std() * scale) for d in range(3)],
        }
    err_stats["corr_teacher_anchor_error"] = [
        float(np.corrcoef(e_t[:, d], e_a[:, d])[0, 1]) for d in range(3)]
    err_stats["anchor_better_frac"] = [float(anchor_better[:, d].mean()) for d in range(3)]
    err_stats["error_cv"] = [float(e_f[:, d].std() / e_f[:, d].mean()) for d in range(3)]
    err_stats["oracle_alpha_gain_pct"] = float(100 * (e_f.mean() - e_star.mean()) / e_f.mean())
    err_stats["best_source_gain_pct"] = float(100 * (e_f.mean() - e_best_src.mean()) / e_f.mean())

    # ------------------ oracle weighting / filtering ceilings ------------------
    macro_err = e_f.mean(1)
    oracle_weights = capture_rank_weights(1.0 / (macro_err + 1e-4), macro_err, taus=(2.0, 4.0, 8.0, 16.0))
    # true oracle: weights by inverse error (not rank); report both
    inv = 1.0 / (macro_err + 1e-4)
    uniform = float(macro_err.mean())
    oracle_inv = float((inv * macro_err).sum() / inv.sum())
    oracle_ceilings = {
        "uniform_macro_mae_orig": uniform * scale,
        "oracle_inverse_error_weight_orig": oracle_inv * scale,
        "oracle_inverse_error_gain_pct": float(100 * (uniform - oracle_inv) / uniform),
        "rank_weight": oracle_weights,
        "filter_q70": capture_filter(-macro_err, macro_err, q=0.7),
        "filter_q50": capture_filter(-macro_err, macro_err, q=0.5),
    }

    # ------------------ signal screening ------------------
    per_sample = {
        "conf": conf, "density": density, "knn_dist": knn_dist,
        "text_len": ulb["text"].str.split().str.len().to_numpy(dtype=np.float64),
        "feat_norm": u_norm.numpy().astype(np.float64),
    }
    per_dim = {
        "v_t": v_t, "v_a": v_a, "v_f": v_f, "conflict": conflict, "support": support,
        "r_jacf": r_jacf, "alpha_jacf": alpha_j, "diff_abs": diff_abs,
        "mu_t": mu_t, "mu_a": mu_a, "target": target,
        "pred_extremity": np.abs(mu_t - 0.5), "target_extremity": np.abs(target - 0.5),
        "neighbor_range": nb_range, "mc_range": mc_range, "mc_mad": mc_mad,
    }
    # orientation: +1 means higher signal is more reliable; -1 means uncertainty
    orientation = {
        "v_t": -1, "v_a": -1, "v_f": -1, "conflict": -1, "support": 1,
        "r_jacf": 1, "alpha_jacf": 1, "diff_abs": -1, "conf": 1, "density": 1,
        "knn_dist": -1, "neighbor_range": -1, "mc_range": -1, "mc_mad": -1,
    }
    signals = {}
    for k, v in per_sample.items():
        signals[k] = np.broadcast_to(v[:, None], (len(v), 3))
    for k, v in per_dim.items():
        signals[k] = v

    table = []
    for name, sig in signals.items():
        rel = orientation.get(name, 0) * sig
        row = {"signal": name, "orientation": orientation.get(name, 0)}
        rhos, cis = [], []
        for d in range(3):
            rho = safe_spearman(sig[:, d], e_f[:, d])
            rhos.append(rho)
        row["spearman_macro"] = float(np.nanmean(rhos))
        row["spearman_per_dim"] = rhos
        ci = bootstrap_spearman(sig.mean(1), e_f.mean(1), a.bootstrap, 1234 + len(table))
        row["spearman_ci95"] = ci
        row["auc_high_error"] = float(roc_auc_score((e_f.mean(1) > np.median(e_f.mean(1))).astype(int), sig.mean(1)))
        row["auc_anchor_better"] = float(roc_auc_score(anchor_better.mean(1) > 0.5, sig.mean(1))) if len(np.unique(anchor_better.mean(1) > 0.5)) > 1 else float("nan")
        # AURC with uncertainty orientation (higher sig = riskier for -1; invert for +1)
        risk = -rel.mean(1) if orientation.get(name, 0) >= 0 else rel.mean(1)
        # rel = orientation*sig; for confidence +1, risk should be -rel = -sig
        risk = -rel.mean(1)
        row["aurc"] = aurc(risk, e_f.mean(1))
        row["within_batch_spearman"] = within_batch_spearman(rel.mean(1), e_f.mean(1), batch_size=64, reps=100, seed=7)
        cap = capture_rank_weights(rel.mean(1), e_f.mean(1), batch_size=64, taus=(4.0, 8.0))
        row["capture_tau4"] = cap["4.0"]["capture_pct"]
        row["capture_tau8"] = cap["8.0"]["capture_pct"]
        row["gain_tau8_pct"] = cap["8.0"]["gain_pct"]
        table.append(row)
    table.sort(key=lambda r: -abs(r["spearman_macro"]))

    # ------------------ supervised learnability ceiling ------------------
    # feature matrix per sample x dim
    feat_names = list(per_dim.keys()) + list(per_sample.keys()) + ["dim_V", "dim_A", "dim_D"]
    rows = []
    for d in range(3):
        cols = [per_dim[k][:, d] for k in per_dim] + [per_sample[k] for k in per_sample]
        cols += [np.full(len(y), 1.0 if j == d else 0.0) for j in range(3)]
        rows.append(np.stack(cols, axis=1))
    X = np.stack(rows, axis=1).reshape(-1, len(feat_names))
    y_err = e_f.reshape(-1)
    y_alpha = alpha_star.reshape(-1)
    y_anchor = anchor_better.reshape(-1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # alignment sanity: element 0/1/2 must be sample 0 dim V/A/D
    j = feat_names.index("mu_t")
    assert abs(X[0, j] - mu_t[0, 0]) < 1e-5 and abs(X[1, j] - mu_t[0, 1]) < 1e-5
    assert abs(y_err[0] - e_f[0, 0]) < 1e-6 and abs(y_err[1] - e_f[0, 1]) < 1e-6
    ceiling = supervised_ceiling(X, y_err, y_alpha, y_anchor, n_splits=5, seed=11)
    # target quality if alpha* is predicted by the supervised ceiling (rough:
    # replace alpha with predicted alpha from the same fold is done inside; here
    # report the oracle upper bound as reference)
    ceiling["oracle_alpha_star_mae_orig"] = float(e_star.mean() * scale)
    ceiling["fixed_alpha_mae_orig"] = float(e_f.mean() * scale)
    ceiling["feature_names"] = feat_names

    out_dir = a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"reliability_ceiling_lb{a.budget}_s{a.seed}"
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        ids=ulb["id"].to_numpy(),
        y=y, mu_t=mu_t, v_t=v_t, mu_a=mu_a, v_a=v_a,
        target=target, target_j=target_j, alpha_j=alpha_j,
        conf=conf, density=density, knn_dist=knn_dist, feat_norm=u_norm.numpy(),
        nb_range=nb_range, mc_range=mc_range, mc_mad=mc_mad,
        alpha_star=alpha_star, e_t=e_t, e_a=e_a, e_f=e_f, e_j=e_j,
        e_star=e_star, e_best_src=e_best_src, anchor_better=anchor_better,
        v_f=v_f, conflict=conflict, support=support, r_jacf=r_jacf,
    )
    report = {
        "protocol": {
            "budget": a.budget, "seed": a.seed, "mc_passes": a.mc_passes,
            "k": a.k, "tau": a.tau, "alpha": a.alpha,
            "note": "hidden EmoBank labels used for analysis only; never for training",
        },
        "errors": err_stats,
        "oracle_weighting": oracle_ceilings,
        "signal_table": table,
        "supervised_ceiling": ceiling,
        "runtime_sec": round(time.time() - t0, 1),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("errors", "oracle_weighting")}, indent=2)[:4000])
    print("SIGNALS")
    for r in table:
        print(f'{r["signal"]:16s} rho={r["spearman_macro"]:+.3f} '
              f'auc={r["auc_high_error"]:.3f} cap8={r["capture_tau8"]:+.1f}% '
              f'wb={r["within_batch_spearman"]:+.3f}')
    print("CEILING", json.dumps(ceiling, indent=2))
    print("OUT", out_dir / f"{stem}.json")


if __name__ == "__main__":
    main()
