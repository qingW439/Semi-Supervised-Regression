#!/usr/bin/env python3
"""D2: matched Supervised / Mean Teacher / JACF comparison at fixed step 96."""
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
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    lb = read_gold(a.data_root / "lb500_s0" / "train_lb.parquet", expected_n=500)
    dev = read_gold(a.data_root / "dev_gold.parquet", expected_n=1000)
    ulb = pd.read_parquet(a.data_root / "lb500_s0" / "train_ulb.parquet")
    y = dev.loc[:, list(NORM_COLUMNS)].to_numpy()
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    dev_dl = DataLoader(GoldData(dev), batch_size=32, shuffle=False, collate_fn=TextCollator(tok, 128))
    rows = []
    per_seed = {}
    for arm in ("sup", "mt", "a0.75"):
        for s in (0, 1, 2):
            ck = a.ckpt_root / f"{arm}_s{s}" / "model_step96.pth"
            if not ck.is_file():
                raise FileNotFoundError(ck)
            model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(device)
            d = torch.load(ck, map_location="cpu", weights_only=False)
            model.load_state_dict(d["model"], strict=True); del d
            l_feat = encode(model.encoder, lb["text"].tolist(), tok, device).to(device)
            d_feat = encode(model.encoder, dev["text"].tolist(), tok, device).to(device)
            l_y = torch.tensor(lb.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=device)
            _, raw, _ = predict(model, dev_dl, device)
            raw_ulb = predict_texts(model, ulb["text"].tolist(), tok, device)
            mu = raw_ulb.mean(0).to(device)
            q = knn_calibrate(torch.tensor(raw, device=device), d_feat, l_feat, l_y, 20, 0.1, 0.4, 0.3)
            vrr = (mu + (q - mu) * 1.2).cpu().numpy()
            dec = decompose(y, raw)
            per_seed[f"{arm}_s{s}"] = {
                "raw": {"mae": dec["macro"]["mae"], "ccc": dec["macro"]["ccc"],
                        "pearson": dec["macro"]["pearson"], "cb": dec["macro"]["cb"],
                        "sd_ratio": dec["macro"]["sd_ratio"], "bias": dec["macro"]["mean_bias"]},
                "vrr": {"mae": metric_report(y, vrr)["macro_mae_original_scale"],
                        "ccc": float(np.mean([metric_report(y, vrr)[d]["ccc"] for d in DIMS]))},
                "per_dim": {d: dec[d] for d in DIMS},
            }
            for i, idx in enumerate(dev["id"]):
                rows.append({"id": idx, "arm": arm, "seed": s,
                             **{f"y_{k}": y[i, j] for j, k in enumerate(DIMS)},
                             **{f"raw_{k}": raw[i, j] for j, k in enumerate(DIMS)},
                             **{f"vrr_{k}": vrr[i, j] for j, k in enumerate(DIMS)}})
            print(f"{arm} s{s}: raw MAE={dec['macro']['mae']:.6f} CCC={dec['macro']['ccc']:.4f} "
                  f"r={dec['macro']['pearson']:.4f} Cb={dec['macro']['cb']:.4f} "
                  f"sd_ratio={dec['macro']['sd_ratio']:.3f}", flush=True)
            del model, l_feat, d_feat
            torch.cuda.empty_cache()
    agg = {}
    for arm in ("sup", "mt", "a0.75"):
        for mode in ("raw", "vrr"):
            vals = [per_seed[f"{arm}_s{s}"][mode] for s in (0, 1, 2)]
            agg[f"{arm}_{mode}"] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "mt_comparison.json").write_text(json.dumps({"per_seed": per_seed, "aggregate": agg}, indent=2) + "\n")
    pd.DataFrame(rows).to_parquet(a.out_dir / "mt_predictions.parquet", index=False)
    print("== aggregate ==")
    for arm in ("sup", "mt", "a0.75"):
        r, v = agg[f"{arm}_raw"], agg[f"{arm}_vrr"]
        print(f"{arm}: raw MAE={r['mae']:.6f} CCC={r['ccc']:.4f} r={r['pearson']:.4f} "
              f"Cb={r['cb']:.4f} sd_ratio={r['sd_ratio']:.3f} | vrr MAE={v['mae']:.6f} CCC={v['ccc']:.4f}")


if __name__ == "__main__":
    main()
