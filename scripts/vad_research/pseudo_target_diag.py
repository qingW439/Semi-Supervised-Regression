#!/usr/bin/env python3
"""D4: compare Teacher / Anchor / JACF target / Student on held-out DEV.

All signals are evaluated against DEV gold; DEV is used only for mechanism
diagnosis, not for training or checkpoint selection.
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
from semilearn.datasets.nlp_datasets.emobank import GoldData, NORM_COLUMNS, TextCollator, read_gold  # noqa
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa
from scripts.vad_research.train_acvr import encode, mc_predict, predict  # noqa
from scripts.vad_research.jacf_modes import evidence  # noqa
from scripts.vad_research.ccc_decomposition import decompose  # noqa

DIMS = ("V", "A", "D")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    lb = read_gold(a.data_root / "lb500_s0" / "train_lb.parquet", expected_n=500)
    dev = read_gold(a.data_root / "dev_gold.parquet", expected_n=1000)
    y = dev.loc[:, list(NORM_COLUMNS)].to_numpy()
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(device)
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(d["model"], strict=True); del d
    lb_feat = encode(model.encoder, lb["text"].tolist(), tok, device).to(device)
    dev_feat = encode(model.encoder, dev["text"].tolist(), tok, device).to(device)
    l_y = torch.tensor(lb.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=device)
    dev_tok = tok(dev["text"].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")
    dev_tok = {k: v.to(device) for k, v in dev_tok.items()}
    mc = mc_predict(model, dev_tok, 5)
    mu_t, v_t, mu_a, v_a, conf = evidence(mc, dev_feat, lb_feat, l_y, k=20, tau=0.1)
    target = 0.25 * mu_t + 0.75 * mu_a
    dev_dl = DataLoader(GoldData(dev), batch_size=32, shuffle=False, collate_fn=TextCollator(tok, 128))
    _, student, _ = predict(model, dev_dl, device)
    signals = {"teacher": mu_t.cpu().numpy(), "anchor": mu_a.cpu().numpy(),
               "jacf_target": target.cpu().numpy(), "student": student}
    out = {}
    for name, p in signals.items():
        dec = decompose(y, p)
        out[name] = dec
        m = dec["macro"]
        print(f'{name:12s} MAE={m["mae"]:.6f} CCC={m["ccc"]:.4f} r={m["pearson"]:.4f} '
              f'Cb={m["cb"]:.4f} sd_ratio={m["sd_ratio"]:.3f} bias={m["mean_bias"]:+.4f}')
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
