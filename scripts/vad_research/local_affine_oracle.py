#!/usr/bin/env python3
"""Oracle local-affine calibration ceiling on DEV.

For each DEV sample, use its k nearest DEV neighbours' *true* labels and raw
predictions to fit a local affine map (or local variance match), then apply it
to the sample itself.  The sample's own label is never used.  This is an upper
bound for a deployable local calibrator, not a method.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
            b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=128, return_tensors="pt")
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
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=128, return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        out.append(model(**b).float().cpu())
    return torch.cat(out).numpy()


def knn_calibrate(pred, qf, lf, ly, k, temp, alpha, c0):
    sim = qf @ lf.T
    kk = min(k, lf.shape[0])
    topv, topi = sim.topk(kk, dim=1)
    w = torch.softmax(topv / temp, dim=1)
    knn = (w.unsqueeze(-1) * ly[topi]).sum(1)
    conf = topv[:, 0].clamp(0, 1)
    gate = ((conf - c0) / max(1e-6, 1 - c0)).clamp(0, 1)
    a = (alpha * gate).unsqueeze(-1)
    return ((1 - a) * pred + a * knn).cpu().numpy()


def local_maps(feat, pred, y, k=50):
    sim = feat @ feat.T
    np.fill_diagonal(sim, -2.0)
    nbr = np.argsort(-sim, axis=1)[:, :k]
    q_lin = np.zeros_like(pred)
    q_var = np.zeros_like(pred)
    for i in range(len(pred)):
        j = nbr[i]
        for d in range(3):
            yj, pj = y[j, d], pred[j, d]
            if np.std(pj) > 1e-9:
                a, b = np.polyfit(pj, yj, 1)
                q_lin[i, d] = a * pred[i, d] + b
            else:
                q_lin[i, d] = yj.mean()
            sy, sp = np.std(yj), np.std(pj)
            mu_y, mu_p = yj.mean(), pj.mean()
            ratio = sy / sp if sp > 1e-9 else 1.0
            ratio = np.clip(ratio, 0.0, 4.0)
            q_var[i, d] = mu_y + (pred[i, d] - mu_p) * ratio
    return q_lin, q_var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    dev = pd.read_parquet(a.data_root / "dev_gold.parquet")
    y = dev.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32)
    lb = pd.read_parquet(a.data_root / "lb500_s0" / "train_lb.parquet")
    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    del ck
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    d_feat = encode(model.encoder, dev["text"].tolist(), tok, device)
    l_feat = encode(model.encoder, lb["text"].tolist(), tok, device)
    l_y = torch.tensor(lb.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=device)
    p = predict(model, dev["text"].tolist(), tok, device)
    pp = torch.tensor(p, device=device)
    vrr = knn_calibrate(pp, d_feat, l_feat, l_y, 20, 0.1, 0.4, 0.3)
    vrr = vrr.mean(0) + (vrr - vrr.mean(0)) * 1.2
    out = {"raw": metric_report(y, p), "vrr": metric_report(y, vrr)}
    for k in (20, 50, 100):
        ql, qv = local_maps(d_feat.cpu().numpy(), p, y, k=k)
        out[f"local_lin_k{k}"] = metric_report(y, ql)
        out[f"local_var_k{k}"] = metric_report(y, qv)
        for w in (0.25, 0.5, 0.75):
            out[f"mix_lin_k{k}_w{w}"] = metric_report(y, (1 - w) * vrr + w * ql)
            out[f"mix_var_k{k}_w{w}"] = metric_report(y, (1 - w) * vrr + w * qv)
    # 2-D oracle mix around the existing VRRC (DEV-tuned upper bound)
    ql20, qv20 = local_maps(d_feat.cpu().numpy(), p, y, k=20)
    pareto = []
    for aa in (0.0, 0.25, 0.5, 0.75, 1.0):
        for bb in (0.0, 0.25, 0.5, 0.75, 1.0):
            q = vrr + aa * (ql20 - vrr) + bb * (qv20 - vrr)
            m = metric_report(y, q)
            pareto.append({
                "a_lin": aa, "b_var": bb,
                "mae": m["macro_mae_original_scale"],
                "ccc": float(np.mean([m[d]["ccc"] for d in DIMS])),
            })
    out["pareto"] = pareto
    a.out.write_text(json.dumps(out, indent=2))
    for name, m in out.items():
        if name == "pareto":
            continue
        print(f'{name:16s} MAE={m["macro_mae_original_scale"]:.6f} '
              f'CCC={np.mean([m[d]["ccc"] for d in DIMS]):.4f} '
              f'CCC_V/A/D={m["V"]["ccc"]:.3f}/{m["A"]["ccc"]:.3f}/{m["D"]["ccc"]:.3f}')


if __name__ == "__main__":
    main()
