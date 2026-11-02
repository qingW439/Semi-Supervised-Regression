#!/usr/bin/env python3
"""Evaluate the pre-registered alpha sweep at fixed step 96.

Raw is the primary result; VRRC-fixed is secondary.  The supervised reference is
the matched lambda_u=0 run at the same step.  All VRRC centering uses the
model's own ULB prediction mean, never the DEV batch mean.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    GoldData, NORM_COLUMNS, TextCollator, read_gold,
)
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402
from scripts.vad_research.train_acvr import encode, knn_calibrate, predict  # noqa: E402

DIMS = ("V", "A", "D")
ALPHAS = ("0.55", "0.60", "0.65", "0.70", "0.75")


@torch.no_grad()
def predict_texts(model, texts, tok, device, bs=64):
    model.eval(); out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=128, return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        out.append(model(**b).float().cpu())
    return torch.cat(out)


def summary(y, p):
    m = metric_report(y, p)
    return {"mae": m["macro_mae_original_scale"],
            "ccc": float(np.mean([m[d]["ccc"] for d in DIMS])),
            "r2": float(np.mean([m[d]["r2"] for d in DIMS])),
            "per_dim_mae": {d: m[d]["mae"] for d in DIMS},
            "per_dim_ccc": {d: m[d]["ccc"] for d in DIMS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    lb = read_gold(a.data_root / "lb500_s0" / "train_lb.parquet", expected_n=500)
    dev = read_gold(a.data_root / "dev_gold.parquet", expected_n=1000)
    ulb = pd.read_parquet(a.data_root / "lb500_s0" / "train_ulb.parquet")
    y = dev.loc[:, list(NORM_COLUMNS)].to_numpy()
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    dev_dl = DataLoader(GoldData(dev), batch_size=32, shuffle=False, collate_fn=TextCollator(tok, 128))
    rows = []
    metrics = {}
    for arm in [f"a{alpha}" for alpha in ALPHAS] + ["sup"]:
        for s in (0, 1, 2):
            ck = a.ckpt_root / f"{arm}_s{s}" / "model_step96.pth"
            if not ck.is_file():
                raise FileNotFoundError(ck)
            model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(device)
            d = torch.load(ck, map_location="cpu", weights_only=False)
            model.load_state_dict(d["model"], strict=True)
            del d
            l_feat = encode(model.encoder, lb["text"].tolist(), tok, device).to(device)
            d_feat = encode(model.encoder, dev["text"].tolist(), tok, device).to(device)
            l_y = torch.tensor(lb.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=device)
            _, raw, _ = predict(model, dev_dl, device)
            raw_ulb = predict_texts(model, ulb["text"].tolist(), tok, device)
            mu = raw_ulb.mean(0).to(device)
            q = knn_calibrate(torch.tensor(raw, device=device), d_feat, l_feat, l_y, 20, 0.1, 0.4, 0.3)
            q = mu + (q - mu) * 1.2
            vrr = q.cpu().numpy()
            metrics[f"{arm}_s{s}"] = {"raw": summary(y, raw), "vrr": summary(y, vrr)}
            for i, idx in enumerate(dev["id"]):
                rows.append({"id": idx, "arm": arm, "seed": s,
                             **{f"y_{k}": y[i, j] for j, k in enumerate(DIMS)},
                             **{f"raw_{k}": raw[i, j] for j, k in enumerate(DIMS)},
                             **{f"vrr_{k}": vrr[i, j] for j, k in enumerate(DIMS)}})
            print(f"{arm} s{s} raw={metrics[f'{arm}_s{s}']['raw']['mae']:.6f}/"
                  f"{metrics[f'{arm}_s{s}']['raw']['ccc']:.4f} "
                  f"vrr={metrics[f'{arm}_s{s}']['vrr']['mae']:.6f}/"
                  f"{metrics[f'{arm}_s{s}']['vrr']['ccc']:.4f}", flush=True)
            del model, l_feat, d_feat
            torch.cuda.empty_cache()
    agg = {}
    for arm in [f"a{alpha}" for alpha in ALPHAS] + ["sup"]:
        for mode in ("raw", "vrr"):
            vals = [metrics[f"{arm}_s{s}"][mode] for s in (0, 1, 2)]
            agg[f"{arm}_{mode}"] = {
                "mae_mean": float(np.mean([v["mae"] for v in vals])),
                "mae_sd": float(np.std([v["mae"] for v in vals], ddof=1)),
                "ccc_mean": float(np.mean([v["ccc"] for v in vals])),
                "ccc_sd": float(np.std([v["ccc"] for v in vals], ddof=1)),
                "r2_mean": float(np.mean([v["r2"] for v in vals])),
                "per_dim_mae_mean": {d: float(np.mean([v["per_dim_mae"][d] for v in vals])) for d in DIMS},
                "per_dim_ccc_mean": {d: float(np.mean([v["per_dim_ccc"][d] for v in vals])) for d in DIMS},
            }
    go = {}
    for alpha in ALPHAS:
        x = agg[f"a{alpha}_raw"]
        go[alpha] = {"mae_ok": x["mae_mean"] <= 0.1710, "ccc_ok": x["ccc_mean"] >= 0.4400,
                     "pass": x["mae_mean"] <= 0.1710 and x["ccc_mean"] >= 0.4400,
                     "mae": x["mae_mean"], "ccc": x["ccc_mean"]}
    out = {"metrics": metrics, "aggregate": agg, "go_raw": go,
           "protocol": {"step": 96, "primary": "raw", "secondary": "vrr",
                        "go": {"raw_mae_max": 0.1710, "raw_ccc_min": 0.4400},
                        "checkpoint_selection": "fixed step 96; no DEV best selection"}}
    (a.out_dir / "alpha_sweep.json").write_text(json.dumps(out, indent=2) + "\n")
    pd.DataFrame(rows).to_parquet(a.out_dir / "alpha_sweep_predictions.parquet", index=False)

    # Pareto plots
    for mode in ("raw", "vrr"):
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        sup = agg[f"sup_{mode}"]
        ax.scatter(sup["mae_mean"], sup["ccc_mean"], marker="*", s=180, label="matched supervised", zorder=5)
        for alpha in ALPHAS:
            v = agg[f"a{alpha}_{mode}"]
            ax.scatter(v["mae_mean"], v["ccc_mean"], label=f"JACF α={alpha}", s=45)
            ax.annotate(alpha, (v["mae_mean"], v["ccc_mean"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel("DEV Macro-MAE (lower is better)")
        ax.set_ylabel("DEV Macro-CCC (higher is better)")
        ax.set_title(f"α sweep: {mode}")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(a.out_dir / f"pareto_{mode}.png", dpi=180); plt.close(fig)
    for mode in ("raw", "vrr"):
        print(f"== {mode} ==")
        for alpha in ALPHAS:
            v = agg[f"a{alpha}_{mode}"]
            print(f"α={alpha}: MAE={v['mae_mean']:.6f}±{v['mae_sd']:.6f} CCC={v['ccc_mean']:.4f}±{v['ccc_sd']:.4f} R2={v['r2_mean']:.4f}")
        sup = agg[f"sup_{mode}"]
        print(f"sup: MAE={sup['mae_mean']:.6f} CCC={sup['ccc_mean']:.4f} R2={sup['r2_mean']:.4f}")
    print("GO raw", json.dumps(go, indent=2))


if __name__ == "__main__":
    main()
