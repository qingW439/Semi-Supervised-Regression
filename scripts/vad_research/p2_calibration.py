#!/usr/bin/env python3
"""P2: deployable calibration comparison on TRAIN-only OOF predictions.

Base predictor: the 500-label fixed-alpha JACF checkpoint.
Calibration data: 500 OOF rows produced by the five 400-label fold models.
Retrieval geometry: a single frozen pretrained RoBERTa encoder for *all*
train/dev/oof queries (so folds share one feature space).

Methods compared on DEV:
  raw, VRRC-fixed, Global Affine, Local Intercept, Local Affine.
Also reports OOF cross-fitted metrics (fit on 4 folds, evaluate on the 5th).
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
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS, read_gold  # noqa: E402
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402
from scripts.vad_research.train_acvr import encode, knn_calibrate, predict  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator  # noqa: E402

DIMS = ("V", "A", "D")


def summary(y, p):
    m = metric_report(y, p)
    return {
        "macro_mae": m["macro_mae_original_scale"],
        "macro_ccc": float(np.mean([m[d]["ccc"] for d in DIMS])),
        "macro_r2": float(np.mean([m[d]["r2"] for d in DIMS])),
        "per_dim_mae": {d: m[d]["mae"] for d in DIMS},
        "per_dim_ccc": {d: m[d]["ccc"] for d in DIMS},
        "out_of_range": int(np.sum((p < 0) | (p > 1))),
        "pred_std": {d: float(p[:, i].std()) for i, d in enumerate(DIMS)},
    }


@torch.no_grad()
def predict_texts(model, texts, tok, device, bs=64):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=128, return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        out.append(model(**b).float().cpu())
    return torch.cat(out)


def global_affine(q_fit, y_fit, q_apply):
    out = np.zeros_like(q_apply)
    coefs = {}
    for d in range(3):
        a, b = np.polyfit(q_fit[:, d], y_fit[:, d], 1)
        out[:, d] = a * q_apply[:, d] + b
        coefs[DIMS[d]] = {"slope": float(a), "intercept": float(b)}
    return out, coefs


def local_fit(q_dev, feat_dev, q_bank, y_bank, feat_bank, k=50, tau=0.1,
              ridge=1.0, shrink=0.5, affine=True):
    """Weighted local residual regression. Returns predictions and diagnostics."""
    sim = feat_dev @ feat_bank.T
    kk = min(k, feat_bank.shape[0])
    topv, topi = sim.topk(kk, dim=1)
    w = torch.softmax(topv / tau, dim=1).cpu().numpy()
    qi = q_dev.cpu().numpy()
    qb = q_bank.cpu().numpy()
    yb = y_bank.cpu().numpy()
    out = np.zeros_like(qi)
    coefs = np.zeros((len(qi), 3, 2 if affine else 1))
    for i in range(len(qi)):
        idx = topi[i].cpu().numpy()
        wi = w[i]
        for d in range(3):
            qj = qb[idx, d]
            rj = yb[idx, d] - qj
            if affine:
                X = np.stack([np.ones_like(qj), qj], axis=1)
                W = wi[:, None]
                A = X.T @ (W * X) + ridge * np.eye(2)
                b = X.T @ (W[:, 0] * rj)
                beta = np.linalg.solve(A, b)
                out[i, d] = qi[i, d] + shrink * (beta[0] + beta[1] * qi[i, d])
                coefs[i, d] = beta
            else:
                c = float(np.sum(wi * rj) / max(np.sum(wi), 1e-8))
                out[i, d] = qi[i, d] + shrink * c
                coefs[i, d, 0] = c
    return out, coefs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--fold-manifest", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--base-feature-source", choices=("pretrained", "model"), default="model",
                    help="encoder used for the VRRC anchor bank / query geometry")
    ap.add_argument("--cal-feature-source", choices=("pretrained", "model"), default="pretrained",
                    help="shared encoder used only for local-calibration neighbours")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    lb = read_gold(a.data_root / "lb500_s0" / "train_lb.parquet", expected_n=500)
    dev = read_gold(a.data_root / "dev_gold.parquet", expected_n=1000)
    ulb = pd.read_parquet(a.data_root / "lb500_s0" / "train_ulb.parquet")
    oof = pd.read_parquet(a.oof)
    if len(oof) != 500 or oof["id"].duplicated().any():
        raise ValueError("OOF table must contain 500 unique TRAIN ids")
    oof = oof.set_index("id").loc[lb["id"].tolist()].reset_index()
    y_oof = oof[[f"y_{d}" for d in DIMS]].to_numpy(np.float64)
    y_dev = dev.loc[:, list(NORM_COLUMNS)].to_numpy(np.float64)
    man = json.loads(a.fold_manifest.read_text())

    # frozen retrieval encoder (or the fine-tuned 500-label model encoder)
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    del ck
    if a.base_feature_source == "model":
        feat_base_train = encode(model.encoder, lb["text"].tolist(), tok, device).to(device)
        feat_base_dev = encode(model.encoder, dev["text"].tolist(), tok, device).to(device)
    else:
        enc = AutoModel.from_pretrained(str(a.pretrain_path), local_files_only=True).to(device).eval()
        feat_base_train = encode(enc, lb["text"].tolist(), tok, device).to(device)
        feat_base_dev = encode(enc, dev["text"].tolist(), tok, device).to(device)
        del enc
    if a.cal_feature_source == "pretrained":
        enc = AutoModel.from_pretrained(str(a.pretrain_path), local_files_only=True).to(device).eval()
        feat_cal_train = encode(enc, lb["text"].tolist(), tok, device).to(device)
        feat_cal_dev = encode(enc, dev["text"].tolist(), tok, device).to(device)
        del enc
    else:
        feat_cal_train = encode(model.encoder, lb["text"].tolist(), tok, device).to(device)
        feat_cal_dev = encode(model.encoder, dev["text"].tolist(), tok, device).to(device)
    torch.cuda.empty_cache()

    dev_dl = DataLoader(GoldData(dev), batch_size=32, shuffle=False,
                        collate_fn=TextCollator(tok, 128))
    _, raw_dev, _ = predict(model, dev_dl, device)
    raw_ulb = predict_texts(model, ulb["text"].tolist(), tok, device, bs=64)
    mu_ulb = raw_ulb.mean(0).to(device)
    l_y = torch.tensor(lb.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=device)

    # DEV base VRRC-fixed using the base feature space
    raw_dev_t = torch.tensor(raw_dev, device=device)
    vrr_dev_t = knn_calibrate(raw_dev_t, feat_base_dev, feat_base_train, l_y, 20, 0.1, 0.4, 0.3)
    vrr_dev_t = mu_ulb + (vrr_dev_t - mu_ulb) * 1.2
    vrr_dev = vrr_dev_t.cpu().numpy()

    # OOF base VRRC-fixed per fold: use the *fold model's own* encoder features
    # (features_*.npy saved by train_oof_fold.py).  This avoids using the final
    # 500-label encoder on held-out TRAIN labels.
    vrr_oof = np.zeros((500, 3))
    for f in range(5):
        hold = man["fold_ids"][str(f)]
        train = [i for i in lb["id"] if i not in set(hold)]
        ti = lb.index[lb["id"].isin(train)].to_numpy()
        hi = lb.index[lb["id"].isin(hold)].to_numpy()
        fdir = a.oof.parent / f"fold{f}"
        bank_feat = torch.tensor(np.load(fdir / "features_labeled.npy"), device=device)
        query_feat = torch.tensor(np.load(fdir / "features_holdout.npy"), device=device)
        bank_y = torch.tensor(lb.iloc[ti][list(NORM_COLUMNS)].to_numpy(np.float32), device=device)
        qraw = torch.tensor(oof.iloc[hi][[f"raw_{d}" for d in DIMS]].to_numpy(np.float32), device=device)
        q = knn_calibrate(qraw, query_feat, bank_feat, bank_y, 20, 0.1, 0.4, 0.3)
        cs = json.loads((fdir / "centering_stats.json").read_text())
        mu = torch.tensor(cs["mu_ulb_norm"], device=device, dtype=torch.float32)
        q = mu + (q - mu) * 1.2
        vrr_oof[hi] = q.cpu().numpy()

    # distribution audit
    dist = {}
    for name, arr in (("oof_raw", oof[[f"raw_{d}" for d in DIMS]].to_numpy()),
                      ("oof_vrr", vrr_oof), ("dev_raw", raw_dev), ("dev_vrr", vrr_dev)):
        dist[name] = {d: {"mean": float(arr[:, i].mean()), "std": float(arr[:, i].std()),
                          "q10": float(np.quantile(arr[:, i], 0.1)),
                          "q50": float(np.quantile(arr[:, i], 0.5)),
                          "q90": float(np.quantile(arr[:, i], 0.9))}
                      for i, d in enumerate(DIMS)}

    res = {
        "dev_raw": summary(y_dev, raw_dev),
        "dev_vrr_fixed": summary(y_dev, vrr_dev),
    }
    # global affine
    g_dev, coefs = global_affine(vrr_oof, y_oof, vrr_dev)
    res["dev_global_affine"] = summary(y_dev, g_dev)
    res["global_coefs"] = coefs
    # local intercept / affine on DEV
    q_oof_t = torch.tensor(vrr_oof, dtype=torch.float32)
    y_oof_t = torch.tensor(y_oof, dtype=torch.float32)
    li_dev, _ = local_fit(torch.tensor(vrr_dev, dtype=torch.float32), feat_cal_dev,
                          q_oof_t, y_oof_t, feat_cal_train, k=a.k, tau=a.tau,
                          ridge=a.ridge, shrink=a.shrink, affine=False)
    la_dev, la_coefs = local_fit(torch.tensor(vrr_dev, dtype=torch.float32), feat_cal_dev,
                                 q_oof_t, y_oof_t, feat_cal_train, k=a.k, tau=a.tau,
                                 ridge=a.ridge, shrink=a.shrink, affine=True)
    res["dev_local_intercept"] = summary(y_dev, li_dev)
    res["dev_local_affine"] = summary(y_dev, la_dev)
    res["local_affine_coef_mean"] = {d: {"c": float(la_coefs[:, i, 0].mean()),
                                          "gamma": float(la_coefs[:, i, 1].mean())}
                                      for i, d in enumerate(DIMS)}

    # OOF cross-fitted global/local metrics
    cross = {"global_affine": np.zeros((500, 3)), "local_intercept": np.zeros((500, 3)),
             "local_affine": np.zeros((500, 3))}
    for f in range(5):
        hold = man["fold_ids"][str(f)]
        hi = lb.index[lb["id"].isin(hold)].to_numpy()
        ti = lb.index[~lb["id"].isin(hold)].to_numpy()
        g, _ = global_affine(vrr_oof[ti], y_oof[ti], vrr_oof[hi])
        cross["global_affine"][hi] = g
        li, _ = local_fit(torch.tensor(vrr_oof[hi], dtype=torch.float32), feat_cal_train[hi],
                          torch.tensor(vrr_oof[ti], dtype=torch.float32),
                          torch.tensor(y_oof[ti], dtype=torch.float32), feat_cal_train[ti],
                          k=a.k, tau=a.tau, ridge=a.ridge, shrink=a.shrink, affine=False)
        la, _ = local_fit(torch.tensor(vrr_oof[hi], dtype=torch.float32), feat_cal_train[hi],
                          torch.tensor(vrr_oof[ti], dtype=torch.float32),
                          torch.tensor(y_oof[ti], dtype=torch.float32), feat_cal_train[ti],
                          k=a.k, tau=a.tau, ridge=a.ridge, shrink=a.shrink, affine=True)
        cross["local_intercept"][hi] = li
        cross["local_affine"][hi] = la
    res["oof_cross"] = {"raw": summary(y_oof, oof[[f"raw_{d}" for d in DIMS]].to_numpy()),
                        "vrr": summary(y_oof, vrr_oof)}
    for name, arr in cross.items():
        res["oof_cross"][name] = summary(y_oof, arr)

    (a.out_dir / "p2_calibration.json").write_text(json.dumps({"config": vars(a), "distribution": dist,
                                                               "results": res}, indent=2, default=str) + "\n")
    pd.DataFrame({"id": dev["id"], **{f"raw_{d}": raw_dev[:, i] for i, d in enumerate(DIMS)},
                  **{f"vrr_{d}": vrr_dev[:, i] for i, d in enumerate(DIMS)},
                  **{f"global_{d}": g_dev[:, i] for i, d in enumerate(DIMS)},
                  **{f"li_{d}": li_dev[:, i] for i, d in enumerate(DIMS)},
                  **{f"la_{d}": la_dev[:, i] for i, d in enumerate(DIMS)}}).to_parquet(
        a.out_dir / "p2_dev_predictions.parquet", index=False)
    for k, v in res.items():
        if isinstance(v, dict) and "macro_mae" in v:
            print(f'{k:24s} MAE={v["macro_mae"]:.6f} CCC={v["macro_ccc"]:.4f} R2={v["macro_r2"]:.4f}')
    print("OOF CROSS")
    for k, v in res["oof_cross"].items():
        print(f'{k:24s} MAE={v["macro_mae"]:.6f} CCC={v["macro_ccc"]:.4f}')
    print("WROTE", a.out_dir / "p2_calibration.json")


if __name__ == "__main__":
    main()
