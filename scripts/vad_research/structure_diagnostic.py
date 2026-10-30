#!/usr/bin/env python3
"""M2 structural-compression diagnostic.

Question: does the JACF anchor correction / VRRC calibration compress *real*
local VAD variation, or only remove noise?  Uses hidden EmoBank labels for the
unlabeled pool (analysis only; never for training).

Outputs a JSON with, per VAD dimension:
  * variance ratio  Var(pred) / Var(true)
  * local-pair attenuation slope  (pred_diff regressed on true_diff; 1 = ideal)
  * MAE in low / middle / high true-intensity bins
  * calibration gain diagnostics for JACF
and the same for raw vs VRRC-calibrated DEV predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402

DIMS = ("V", "A", "D")


def encode(encoder, texts, tok, device, bs=64):
    encoder.eval()
    feats = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            b = tok(list(texts[i:i + bs]), padding=True, truncation=True,
                    max_length=128, return_tensors="pt")
            b = {k: v.to(device) for k, v in b.items()}
            st = encoder(**b, return_dict=True).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(dtype=st.dtype)
            pooled = (st * m).sum(1) / m.sum(1).clamp_min(1)
            feats.append(F.normalize(pooled.float(), dim=-1))
    return torch.cat(feats)


@torch.no_grad()
def predict(model, texts, tok, device, bs=64):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True,
                max_length=128, return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        out.append(model(**b).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def mc_predict(model, texts, tok, device, passes=5, bs=64):
    model.train()
    chunks = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True,
                max_length=128, return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        chunks.append(torch.stack([model(**b).float().cpu() for _ in range(passes)]))
    return torch.cat(chunks, dim=1).numpy()


def knn_calibrate(pred, query_feat, l_feat, l_y, k, temp, alpha, c0):
    sim = query_feat @ l_feat.T
    kk = min(k, l_feat.shape[0])
    topv, topi = sim.topk(kk, dim=1)
    w = torch.softmax(topv / temp, dim=1)
    knn = (w.unsqueeze(-1) * l_y[topi]).sum(1)
    conf = topv[:, 0].clamp(0.0, 1.0)
    gate = torch.clamp((conf - c0) / max(1e-6, 1.0 - c0), 0.0, 1.0)
    a = (alpha * gate).unsqueeze(-1)
    return (1.0 - a) * pred + a * knn


def local_pairs(feat, n_anchor=2000, k=5, seed=0):
    rng = np.random.default_rng(seed)
    n = feat.shape[0]
    idx = rng.choice(n, size=min(n_anchor, n), replace=False)
    sim = feat[idx] @ feat.T
    sim[np.arange(len(idx)), idx] = -2.0
    nbr = np.argsort(-sim, axis=1)[:, :k]
    rows = np.repeat(idx, k)
    cols = nbr.reshape(-1)
    keep = rows != cols
    return rows[keep], cols[keep]


def structure(pred, y, pairs, dim):
    i, j = pairs
    td = (y[i, dim] - y[j, dim]).astype(np.float64)
    pd_ = (pred[i, dim] - pred[j, dim]).astype(np.float64)
    ok = np.isfinite(td) & np.isfinite(pd_)
    td, pd_ = td[ok], pd_[ok]
    if len(td) < 20 or np.std(td) < 1e-9:
        return {}
    slope = float(np.polyfit(td, pd_, 1)[0])
    corr = float(np.corrcoef(td, pd_)[0, 1])
    return {
        "slope": slope,
        "corr": corr,
        "variance_ratio": float(np.var(pd_) / max(1e-12, np.var(td))),
        "mean_abs_ratio": float(np.mean(np.abs(pd_)) / max(1e-12, np.mean(np.abs(td)))),
    }


def intensity_mae(pred, y, dim, q=0.2):
    # deviation from the neutral point (original scale 3.0 -> norm 0.5)
    s = np.abs(y[:, dim] - 0.5)
    lo, hi = np.quantile(s, [q, 1 - q])
    groups = {"low": s <= lo, "mid": (s > lo) & (s < hi), "high": s >= hi}
    return {g: float(np.abs(pred[m, dim] - y[m, dim]).mean()) for g, m in groups.items() if m.sum() > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--raw-csv", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    lb = pd.read_parquet(a.data_root / f"lb{a.budget}_s{a.seed}" / "train_lb.parquet")
    ulb = pd.read_parquet(a.data_root / f"lb{a.budget}_s{a.seed}" / "train_ulb.parquet")
    dev = pd.read_parquet(a.data_root / "dev_gold.parquet")
    raw = pd.read_csv(a.raw_csv, dtype={"id": str}, keep_default_na=False).set_index("id")
    hidden = raw.loc[ulb["id"].tolist(), ["V", "A", "D"]].to_numpy(np.float64)
    y = ((hidden - 1.0) / 4.0).astype(np.float32)
    y_dev = dev.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32)

    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    del ck
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)

    l_feat = encode(model.encoder, lb["text"].tolist(), tok, device)
    u_feat = encode(model.encoder, ulb["text"].tolist(), tok, device)
    d_feat = encode(model.encoder, dev["text"].tolist(), tok, device)
    l_y = torch.tensor(lb.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=device)
    d_y = torch.tensor(y_dev, device=device)

    mu_t = mc_predict(model, ulb["text"].tolist(), tok, device, passes=5)
    mu_t = mu_t.mean(0)
    mu_a, v_a, conf = _anchor(u_feat, l_feat, l_y, a.k, a.tau)
    mu_a, v_a, conf = mu_a.cpu().numpy(), v_a.cpu().numpy(), conf.cpu().numpy()
    target = (1 - a.alpha) * mu_t + a.alpha * mu_a

    pairs = local_pairs(u_feat.cpu().numpy())
    methods = {"teacher": mu_t, "anchor": mu_a, "fixed_alpha": target}
    struct = {name: {d: structure(p, y, pairs, j) for j, d in enumerate(DIMS)} for name, p in methods.items()}
    extrema = {name: {d: intensity_mae(p, y, j) for j, d in enumerate(DIMS)} for name, p in methods.items()}

    # calibration gain of the fixed-alpha correction
    delta = target - mu_t
    gain = np.abs(mu_t - y) - np.abs(target - y)
    calib = {}
    for j, d in enumerate(DIMS):
        calib[d] = {
            "frac_improved": float((gain[:, j] > 0).mean()),
            "mean_gain": float(gain[:, j].mean()),
            "mean_abs_delta": float(np.abs(delta[:, j]).mean()),
            "spearman_delta_gain": float(spearmanr(np.abs(delta[:, j]), gain[:, j]).statistic),
        }

    # DEV raw vs VRRC
    raw_dev = predict(model, dev["text"].tolist(), tok, device)
    raw_np = raw_dev.numpy()
    pt = knn_calibrate(raw_dev.to(device), d_feat, l_feat, l_y, a.k, a.tau, 0.4, 0.3)
    vrr0 = pt.cpu().numpy()
    vrr = vrr0.mean(0) + (vrr0 - vrr0.mean(0)) * 1.2
    dev_metrics = {"raw": metric_report(y_dev, raw_np), "vrr": metric_report(y_dev, vrr)}
    scale_sweep = {}
    for s in (1.0, 1.2, 1.5, 2.0, 3.0, 4.0):
        q = vrr0.mean(0) + (vrr0 - vrr0.mean(0)) * s
        m = metric_report(y_dev, q)
        scale_sweep[str(s)] = {
            "macro_mae": m["macro_mae_original_scale"],
            "macro_ccc": float(np.mean([m[d]["ccc"] for d in DIMS])),
            "per_dim_ccc": {d: m[d]["ccc"] for d in DIMS},
            "per_dim_mae": {d: m[d]["mae"] for d in DIMS},
        }
    # oracle global linear map pred -> y, fitted on DEV (diagnostic only)
    lin = raw_np.copy()
    for j in range(3):
        coef = np.polyfit(raw_np[:, j], y_dev[:, j], 1)
        lin[:, j] = coef[0] * raw_np[:, j] + coef[1]
    ml = metric_report(y_dev, lin)
    linear_oracle = {
        "macro_mae": ml["macro_mae_original_scale"],
        "macro_ccc": float(np.mean([ml[d]["ccc"] for d in DIMS])),
        "per_dim_ccc": {d: ml[d]["ccc"] for d in DIMS},
        "per_dim_mae": {d: ml[d]["mae"] for d in DIMS},
    }
    d_pairs = local_pairs(d_feat.cpu().numpy(), n_anchor=500, k=5)
    dev_struct = {
        "raw": {d: structure(raw_np, y_dev, d_pairs, j) for j, d in enumerate(DIMS)},
        "vrr": {d: structure(vrr, y_dev, d_pairs, j) for j, d in enumerate(DIMS)},
    }

    out = {
        "protocol": {"budget": a.budget, "seed": a.seed, "alpha": a.alpha,
                     "note": "hidden labels for analysis only"},
        "unlabeled": {"n": len(ulb), "struct": struct, "extrema": extrema, "calibration": calib},
        "dev": {"metrics": dev_metrics, "struct": dev_struct,
                "scale_sweep": scale_sweep, "linear_oracle": linear_oracle},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:6000])


def _anchor(u_feat, l_feat, l_y, k, tau, eps=1e-4):
    sim = u_feat @ l_feat.T
    kk = min(k, l_feat.shape[0])
    topv, topi = sim.topk(kk, dim=1)
    w = torch.softmax(topv / tau, dim=1)
    nb = l_y[topi]
    mu_a = (w.unsqueeze(-1) * nb).sum(1)
    v_a = (w.unsqueeze(-1) * (nb - mu_a.unsqueeze(1)).square()).sum(1) + eps
    return mu_a, v_a, topv[:, 0].clamp(0, 1)


if __name__ == "__main__":
    main()
