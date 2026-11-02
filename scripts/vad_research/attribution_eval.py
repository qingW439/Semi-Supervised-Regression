#!/usr/bin/env python3
"""Evaluate the A/B/C/D JACF attribution matrix with a frozen VRRC operator.

A: matched supervised, raw
B: matched supervised, VRRC-fixed
C: JACF fixed-alpha, raw
D: JACF fixed-alpha, VRRC-fixed

VRRC-fixed uses the model's own ULB prediction mean as the centering statistic
(no DEV batch mean).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


@torch.no_grad()
def predict_texts(model, texts, tok, device, bs=64):
    model.eval()
    out = []
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
    y_dev = dev.loc[:, list(NORM_COLUMNS)].to_numpy()
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    dev_dl = DataLoader(GoldData(dev), batch_size=32, shuffle=False, collate_fn=TextCollator(tok, 128))
    rows = []
    metrics = {}
    for arm in ("sup", "jacf"):
        for s in (0, 1, 2):
            ck = a.ckpt_root / f"attr_{arm}_s{s}" / "model_best.pth"
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
            q = knn_calibrate(torch.tensor(raw, device=device), d_feat, l_feat, l_y,
                              20, 0.1, 0.4, 0.3)
            q = mu + (q - mu) * 1.2
            vrr = q.cpu().numpy()
            metrics[f"{arm}_s{s}"] = {"raw": summary(y_dev, raw), "vrr": summary(y_dev, vrr),
                                      "centering_mu": mu.cpu().tolist(),
                                      "checkpoint": str(ck)}
            for i, idx in enumerate(dev["id"]):
                rows.append({"id": idx, "arm": arm, "seed": s,
                             **{f"y_{d}": y_dev[i, j] for j, d in enumerate(DIMS)},
                             **{f"raw_{d}": raw[i, j] for j, d in enumerate(DIMS)},
                             **{f"vrr_{d}": vrr[i, j] for j, d in enumerate(DIMS)}})
            print(f"{arm} s{s} raw={metrics[f'{arm}_s{s}']['raw']['mae']:.6f} "
                  f"vrr={metrics[f'{arm}_s{s}']['vrr']['mae']:.6f}", flush=True)
            del model, l_feat, d_feat
            torch.cuda.empty_cache()
    # deltas per seed
    deltas = {}
    for s in (0, 1, 2):
        A = metrics[f"sup_s{s}"]["raw"]["mae"]; B = metrics[f"sup_s{s}"]["vrr"]["mae"]
        C = metrics[f"jacf_s{s}"]["raw"]["mae"]; D = metrics[f"jacf_s{s}"]["vrr"]["mae"]
        deltas[str(s)] = {
            "train_raw_A_minus_C": A - C,
            "train_cal_B_minus_D": B - D,
            "infer_sup_A_minus_B": A - B,
            "infer_jacf_C_minus_D": C - D,
            "interaction": (B - A) - (D - C),
            "A": A, "B": B, "C": C, "D": D,
        }
    mean = {k: float(np.mean([deltas[str(s)][k] for s in range(3)])) for k in deltas["0"]}
    (a.out_dir / "attribution_metrics.json").write_text(json.dumps({"metrics": metrics, "deltas": deltas, "mean": mean}, indent=2) + "\n")
    pd.DataFrame(rows).to_parquet(a.out_dir / "attribution_predictions.parquet", index=False)
    for s in (0, 1, 2):
        d = deltas[str(s)]
        print(f"s{s}: Δtrain_raw={d['train_raw_A_minus_C']:+.6f} Δtrain_cal={d['train_cal_B_minus_D']:+.6f} "
              f"Δinfer_sup={d['infer_sup_A_minus_B']:+.6f} Δinfer_jacf={d['infer_jacf_C_minus_D']:+.6f}")
    print("MEAN", {k: round(v, 6) if isinstance(v, float) else v for k, v in mean.items() if k in
                   ("train_raw_A_minus_C", "train_cal_B_minus_D", "infer_sup_A_minus_B", "infer_jacf_C_minus_D")})


if __name__ == "__main__":
    main()
