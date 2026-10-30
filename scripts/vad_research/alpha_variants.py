#!/usr/bin/env python3
"""Alpha variants for JACF: current vs fixed/shift/beta-calibrated.

Chooses on meta 80 only; reports probe 100 once. No training, no DEV/TEST.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS, read_gold  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402
from scripts.vad_research.jacf_modes import evidence  # noqa: E402
from scripts.vad_research.source_attribution import encode_eval, mc_predict, collect  # noqa: E402


def mae(alpha, s, y):
    yt = (1 - alpha) * s['mu_t'] + alpha * s['mu_a']
    return float((yt - y).abs().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--pretrain-path', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--mc-passes', type=int, default=5)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    man = json.loads(a.manifest.read_text())
    sp = man['split']
    lb = read_gold(a.data_root / 'lb500_s0' / 'train_lb.parquet', expected_n=500).set_index('id')
    fit = lb.loc[sp['fit_320_ids']].reset_index()
    meta = lb.loc[sp['meta_80_ids']].reset_index()
    probe = lb.loc[sp['probe_100_ids']].reset_index()
    d = torch.load(a.ckpt, map_location='cpu', weights_only=False, mmap=True)
    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(dev)
    model.load_state_dict(d['model'], strict=True)
    del d
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    f_fit = encode_eval(model.encoder, fit['text'].tolist(), tok, dev)
    f_meta = encode_eval(model.encoder, meta['text'].tolist(), tok, dev)
    f_probe = encode_eval(model.encoder, probe['text'].tolist(), tok, dev)
    y_fit = torch.tensor(fit.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=dev)
    y_meta = torch.tensor(meta.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=dev)
    y_probe = torch.tensor(probe.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=dev)
    mc_meta = mc_predict(model, meta['text'].tolist(), tok, dev, a.mc_passes)
    mc_probe = mc_predict(model, probe['text'].tolist(), tok, dev, a.mc_passes)
    sm = collect(evidence(mc_meta, f_meta, f_fit, y_fit, k=a.k, tau=a.tau), y_meta)
    spb = collect(evidence(mc_probe, f_probe, f_fit, y_fit, k=a.k, tau=a.tau), y_probe)
    sm['y'] = y_meta; spb['y'] = y_probe

    def pick(score_meta, score_probe):
        return {'meta_mae': mae(score_meta, sm, y_meta),
                'probe_mae': mae(score_probe, spb, y_probe)}

    out = {'current': pick(sm['alpha'], spb['alpha']),
           'oracle_astar': pick(sm['a_star'], spb['a_star'])}
    # fixed alpha
    grid = np.arange(0.0, 1.0001, 0.05)
    meta_maes = [mae(torch.full_like(sm['alpha'], float(x)), sm, y_meta) for x in grid]
    best_i = int(np.argmin(meta_maes)); best_a = float(grid[best_i])
    out['fixed_best'] = {'alpha': best_a, **pick(torch.full_like(sm['alpha'], best_a),
                                                 torch.full_like(spb['alpha'], best_a))}
    # shift
    sgrid = np.arange(-0.3, 0.5001, 0.05)
    smaes = [mae((sm['alpha'] + float(x)).clamp(0, 1), sm, y_meta) for x in sgrid]
    si = int(np.argmin(smaes)); best_s = float(sgrid[si])
    out['shift_best'] = {'shift': best_s,
                         **pick((sm['alpha'] + best_s).clamp(0, 1),
                                (spb['alpha'] + best_s).clamp(0, 1))}
    # beta calibration logit(alpha)+b
    def beta(alpha, b):
        p = alpha.clamp(1e-4, 1 - 1e-4)
        return torch.sigmoid(torch.log(p / (1 - p)) + b)
    bgrid = np.arange(-2.0, 2.0001, 0.1)
    bmaes = [mae(beta(sm['alpha'], float(x)), sm, y_meta) for x in bgrid]
    bi = int(np.argmin(bmaes)); best_b = float(bgrid[bi])
    out['beta_best'] = {'b': best_b, **pick(beta(sm['alpha'], best_b), beta(spb['alpha'], best_b))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
