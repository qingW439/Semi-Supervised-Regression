#!/usr/bin/env python3
"""Offline diagnosis: does label-anchored kNN calibration / DA improve pseudo-label quality?

Uses the frozen P1 (or P2 step-128) model. DEV labels are used for diagnosis only,
never for training. Prints MAE/CCC before and after calibration.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    DIMENSIONS, NORM_COLUMNS, GoldData, TextCollator, read_gold,
)
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import (  # noqa: E402
    RobertaVADRegressor, set_seed,
)
from transformers import AutoTokenizer


def encode(encoder, tokenizer, texts, device, max_length=128, bs=64):
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            tok = tokenizer(list(texts[i:i + bs]), padding=True, truncation=True,
                            max_length=max_length, return_tensors='pt')
            tok = {k: v.to(device) for k, v in tok.items()}
            st = encoder(**tok, return_dict=True).last_hidden_state
            m = tok['attention_mask'].unsqueeze(-1).to(dtype=st.dtype)
            out.append(F.normalize((st * m).sum(1) / m.sum(1).clamp_min(1), dim=-1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--pretrain-path', type=Path, required=True)
    ap.add_argument('--budget', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-length', type=int, default=128)
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    set_seed(a.seed)
    lb = read_gold(a.data_root / f'lb{a.budget}_s{a.seed}' / 'train_lb.parquet', expected_n=a.budget)
    dev = read_gold(a.data_root / 'dev_gold.parquet', expected_n=1000)
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)

    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(device)
    ckpt = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    dev_dl = DataLoader(GoldData(dev), batch_size=32, shuffle=False,
                        collate_fn=TextCollator(tok, a.max_length))
    ys, ps = [], []
    with torch.inference_mode():
        for t, y, _ in dev_dl:
            t = {k: v.to(device) for k, v in t.items()}
            ps.append(model(**t).cpu().numpy())
            ys.append(y.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    base = metric_report(y, p)
    print(f"RAW         MAE={base['macro_mae_original_scale']:.6f} "
          f"CCC={np.mean([base[d]['ccc'] for d in DIMENSIONS]):.5f}")

    lf = encode(model.encoder, tok, lb['text'].tolist(), device, a.max_length)
    df = encode(model.encoder, tok, dev['text'].tolist(), device, a.max_length)
    ly = torch.tensor(lb.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float32))
    for k in (10, 20, 50):
        sim = df @ lf.T
        topv, topi = sim.topk(k, dim=1)
        w = torch.softmax(topv / 0.1, dim=1)
        knn = (w.unsqueeze(-1) * ly[topi]).sum(1).numpy()
        conf = topv[:, 0].numpy()
        gate = np.clip((conf - 0.3) / 0.7, 0, 1)
        for alpha in (0.2, 0.3, 0.4, 0.6):
            a_ = (alpha * gate)[:, None]
            pc = (1 - a_) * p + a_ * knn
            for s in (1.0, 1.1, 1.2, 1.3, 1.5):
                ps = pc.mean(0) + (pc - pc.mean(0)) * s
                m = metric_report(y, ps)
                print(f"k={k:2d} a={alpha:.1f} s={s:.1f} "
                      f"MAE={m['macro_mae_original_scale']:.6f} "
                      f"CCC={np.mean([m[d]['ccc'] for d in DIMENSIONS]):.5f} "
                      f"R2={np.mean([m[d]['r2'] for d in DIMENSIONS]):.4f}")

    # distribution alignment of dev predictions to labelled stats
    mu_t = ly.mean(0).numpy(); sd_t = ly.std(0).numpy()
    mu_p = p.mean(0); sd_p = p.std(0)
    pa = (p - mu_p) / np.maximum(sd_p, 1e-6) * sd_t + mu_t
    m = metric_report(y, pa)
    print(f"DA          MAE={m['macro_mae_original_scale']:.6f} "
          f"CCC={np.mean([m[d]['ccc'] for d in DIMENSIONS]):.5f}")

    # mean prediction (constant) reference
    const = np.broadcast_to(ly.mean(0).numpy(), y.shape)
    m = metric_report(y, const)
    print(f"CONST       MAE={m['macro_mae_original_scale']:.6f} "
          f"CCC={np.mean([m[d]['ccc'] for d in DIMENSIONS]):.5f}")


if __name__ == '__main__':
    main()
