#!/usr/bin/env python3
"""Quick representation-bottleneck probe.

Question: does the fine-tuned encoder contain predictable VAD variance that the
current 3-dim head does not extract?  We fit ridge / MLP probes on the frozen
encoder features (500 TRAIN) and evaluate DEV, with and without the model's own
raw prediction as an extra input.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS, read_gold  # noqa: E402
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402
from scripts.vad_research.train_acvr import encode  # noqa: E402

DIMS = ("V", "A", "D")


def summary(y, p):
    m = metric_report(y, p)
    return {"mae": m["macro_mae_original_scale"],
            "ccc": float(np.mean([m[d]["ccc"] for d in DIMS])),
            "r2": float(np.mean([m[d]["r2"] for d in DIMS])),
            "per_dim_mae": {d: m[d]["mae"] for d in DIMS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    lb = read_gold(a.data_root / "lb500_s0" / "train_lb.parquet", expected_n=500)
    dev = read_gold(a.data_root / "dev_gold.parquet", expected_n=1000)
    y_lb = lb.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32)
    y_dev = dev.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32)
    oof = pd.read_parquet(a.oof).set_index("id").loc[lb["id"].tolist()].reset_index()
    raw_oof = oof[[f"raw_{d}" for d in DIMS]].to_numpy(np.float32)
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    del ck
    feat_lb = encode(model.encoder, lb["text"].tolist(), tok, device).cpu().numpy()
    feat_dev = encode(model.encoder, dev["text"].tolist(), tok, device).cpu().numpy()

    # OOF ridge probe (fit on 4 folds, predict 1) for a deployable estimate
    oof_pred = {}
    for name, alpha in (("ridge10", 10.0), ("ridge100", 100.0), ("ridge1000", 1000.0)):
        p = np.zeros_like(y_lb)
        for f in range(5):
            hold = oof["fold"].to_numpy() == f
            m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            m.fit(feat_lb[~hold], y_lb[~hold])
            p[hold] = m.predict(feat_lb[hold])
        oof_pred[name] = p
    out = {"oof": {k: summary(y_lb, v) for k, v in oof_pred.items()},
           "model_oof_raw": summary(y_lb, raw_oof)}
    # DEV probe fit on all 500
    dev_pred = {}
    for name, alpha in (("ridge10", 10.0), ("ridge100", 100.0), ("ridge1000", 1000.0)):
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        m.fit(feat_lb, y_lb)
        dev_pred[name] = m.predict(feat_dev)
    # with the model's own raw prediction as an extra input (fit on OOF, evaluate on OOF)
    m = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    m.fit(np.concatenate([feat_lb, raw_oof], axis=1), y_lb)
    plus_oof = np.zeros_like(y_lb)
    for f in range(5):
        hold = oof["fold"].to_numpy() == f
        mf = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        mf.fit(np.concatenate([feat_lb[~hold], raw_oof[~hold]], axis=1), y_lb[~hold])
        plus_oof[hold] = mf.predict(np.concatenate([feat_lb[hold], raw_oof[hold]], axis=1))
    out["oof"]["ridge_feat_plus_raw"] = summary(y_lb, plus_oof)
    out["dev"] = {k: summary(y_dev, v) for k, v in dev_pred.items()}
    out["note"] = "OOF probe is cross-fitted by fold; DEV probe is fit on all 500 TRAIN."
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    for split in ("oof", "dev"):
        for k, v in out[split].items():
            print(f'{split:4s} {k:16s} MAE={v["mae"]:.6f} CCC={v["ccc"]:.4f} R2={v["r2"]:.4f}')


if __name__ == "__main__":
    main()
