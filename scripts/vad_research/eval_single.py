#!/usr/bin/env python3
"""Evaluate one checkpoint: raw and frozen-center VRRC metrics + CCC decomposition."""
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
from semilearn.datasets.nlp_datasets.emobank import GoldData, NORM_COLUMNS, TextCollator, read_gold  # noqa
from scripts.emobank.train_emobank_supervised import metric_report  # noqa
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa
from scripts.vad_research.train_acvr import encode, knn_calibrate, predict  # noqa
from scripts.vad_research.ccc_decomposition import decompose  # noqa

DIMS = ("V", "A", "D")


@torch.no_grad()
def predict_texts(model, texts, tok, device, bs=64):
    model.eval(); out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=128, return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        out.append(model(**b).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    lb = read_gold(a.data_root / f"lb500_s{a.data_seed}" / "train_lb.parquet", expected_n=500)
    dev = read_gold(a.data_root / "dev_gold.parquet", expected_n=1000)
    ulb = pd.read_parquet(a.data_root / f"lb500_s{a.data_seed}" / "train_ulb.parquet")
    y = dev.loc[:, list(NORM_COLUMNS)].to_numpy()
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(device)
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(d["model"], strict=True); del d
    lb_feat = encode(model.encoder, lb["text"].tolist(), tok, device).to(device)
    dev_feat = encode(model.encoder, dev["text"].tolist(), tok, device).to(device)
    l_y = torch.tensor(lb.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=device)
    dl = DataLoader(GoldData(dev), batch_size=32, shuffle=False, collate_fn=TextCollator(tok, 128))
    _, raw, _ = predict(model, dl, device)
    raw_ulb = predict_texts(model, ulb["text"].tolist(), tok, device)
    mu = raw_ulb.mean(0).to(device)
    q = knn_calibrate(torch.tensor(raw, device=device), dev_feat, lb_feat, l_y, 20, 0.1, 0.4, 0.3)
    vrr = (mu + (q - mu) * 1.2).cpu().numpy()
    dec = decompose(y, raw)
    out = {"raw": dec["macro"], "raw_per_dim": {k: dec[k] for k in DIMS},
           "vrr": {"mae": metric_report(y, vrr)["macro_mae_original_scale"],
                   "ccc": float(np.mean([metric_report(y, vrr)[k]["ccc"] for k in DIMS]))}}
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f'raw MAE={dec["macro"]["mae"]:.6f} CCC={dec["macro"]["ccc"]:.4f} '
          f'r={dec["macro"]["pearson"]:.4f} Cb={dec["macro"]["cb"]:.4f} '
          f's={dec["macro"]["sd_ratio"]:.3f} | vrr MAE={out["vrr"]["mae"]:.6f} CCC={out["vrr"]["ccc"]:.4f}')


if __name__ == "__main__":
    main()
