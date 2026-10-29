#!/usr/bin/env python3
"""One-shot TEST evaluation for EmoBank VAD models.

Reads test_gold.parquet exactly once per model, with the hyper-parameters that
were fixed on DEV. For retrieval-calibrated models the same LA-PLC/VRRC
calibration is applied at inference (bank = labeled TRAIN only).

Usage:
  python eval_test.py --ckpt ... --budget 500 --seed 0 --mode raw|calib \
      [--knn-k 20 --knn-alpha 0.4 --scale 1.2]
"""
import argparse
import json
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


@torch.inference_mode()
def encode(encoder, tokenizer, texts, device, max_length=128, bs=64):
    out = []
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
    ap.add_argument('--budget', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--mode', choices=('raw', 'calib'), default='raw')
    ap.add_argument('--knn-k', type=int, default=20)
    ap.add_argument('--knn-alpha', type=float, default=0.4)
    ap.add_argument('--scale', type=float, default=1.2)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    set_seed(a.seed)

    test = read_gold(a.data_root / 'test_gold.parquet')
    lb = read_gold(a.data_root / f'lb{a.budget}_s{a.seed}' / 'train_lb.parquet', expected_n=a.budget)
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(device)
    ckpt = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    state = {k: v for k, v in ckpt['model'].items() if not k.startswith('arc.')}
    model.load_state_dict(state, strict=True)
    model.eval()

    dl = DataLoader(GoldData(test), batch_size=32, shuffle=False,
                    collate_fn=TextCollator(tok, 128))
    ys, ps, ids = [], [], []
    with torch.inference_mode():
        for t, y, bid in dl:
            t = {k: v.to(device) for k, v in t.items()}
            ps.append(model(**t).cpu().numpy())
            ys.append(y.numpy())
            ids.extend(bid)
    y = np.concatenate(ys)
    p = np.concatenate(ps)

    if a.mode == 'calib':
        lf = encode(model.encoder, tok, lb['text'].tolist(), device)
        tf = encode(model.encoder, tok, test['text'].tolist(), device)
        ly = torch.tensor(lb.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float32))
        sim = tf @ lf.T
        topv, topi = sim.topk(a.knn_k, dim=1)
        w = torch.softmax(topv / 0.1, dim=1)
        knn = (w.unsqueeze(-1) * ly[topi]).sum(1).numpy()
        conf = topv[:, 0].numpy()
        gate = np.clip((conf - 0.3) / 0.7, 0, 1)
        aa = (a.knn_alpha * gate)[:, None]
        p = (1 - aa) * p + aa * knn
        p = p.mean(0) + (p - p.mean(0)) * a.scale

    m = metric_report(y, p)
    ccc = float(np.mean([m[d]['ccc'] for d in DIMENSIONS]))
    r2 = float(np.mean([m[d]['r2'] for d in DIMENSIONS]))
    result = {'ckpt': str(a.ckpt), 'budget': a.budget, 'seed': a.seed, 'mode': a.mode,
              'macro_mae': m['macro_mae_original_scale'], 'macro_ccc': ccc, 'macro_r2': r2,
              'metrics': m}
    print(json.dumps(result, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
