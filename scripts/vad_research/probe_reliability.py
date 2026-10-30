#!/usr/bin/env python3
"""Does JACF reliability r actually track pseudo-label error?

Strict TRAIN-only 320/80/100 probe. No DEV/TEST.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS, read_gold  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402
from scripts.vad_research.jacf_modes import evidence, build_objective  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--pretrain-path', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mc-passes', type=int, default=5)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    man = json.loads(a.manifest.read_text())
    lb = read_gold(a.data_root / 'lb500_s0' / 'train_lb.parquet', expected_n=500).set_index('id')
    meta = lb.loc[man['split']['meta_80_ids']].reset_index()
    probe = lb.loc[man['split']['probe_100_ids']].reset_index()
    d = torch.load(a.ckpt, map_location='cpu', weights_only=False, mmap=True)
    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(dev)
    model.load_state_dict(d['model'], strict=True)
    del d
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)

    @torch.inference_mode()
    def enc(texts):
        model.encoder.eval()
        out = []
        for i in range(0, len(texts), 32):
            b = tok(list(texts[i:i + 32]), padding=True, truncation=True,
                    max_length=128, return_tensors='pt')
            b = {k: v.to(dev) for k, v in b.items()}
            st = model.encoder(**b, return_dict=True).last_hidden_state
            m = b['attention_mask'].unsqueeze(-1).to(dtype=st.dtype)
            out.append(F.normalize((st * m).sum(1) / m.sum(1).clamp_min(1), dim=-1).cpu())
        return torch.cat(out).to(dev)

    @torch.no_grad()
    def mc(texts):
        model.train()
        ps = []
        for _ in range(a.mc_passes):
            chunks = []
            for i in range(0, len(texts), 32):
                b = tok(list(texts[i:i + 32]), padding=True, truncation=True,
                        max_length=128, return_tensors='pt')
                b = {k: v.to(dev) for k, v in b.items()}
                chunks.append(model(**b).cpu())
            ps.append(torch.cat(chunks))
        model.eval()
        return torch.stack(ps).to(dev)

    lf = enc(meta['text'].tolist()); pf = enc(probe['text'].tolist())
    ym = torch.tensor(meta.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=dev)
    yp = torch.tensor(probe.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=dev)
    mcp = mc(probe['text'].tolist())
    ev = evidence(mcp, pf, lf, ym, k=a.k, tau=a.tau)
    target, r = build_objective('jacf', ev)
    err = (target - yp).abs().mean(1)          # [B]
    rmean = r.mean(1)                           # [B]
    rho = float(spearmanr(rmean.cpu().numpy(), err.cpu().numpy()).statistic)
    q = torch.quantile(rmean, torch.tensor([0.25, 0.5, 0.75], device=dev))
    groups = []
    for lo, hi in ((-1e9, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 1e9)):
        sel = (rmean > lo) & (rmean <= hi)
        groups.append({'count': int(sel.sum()), 'mean_error': float(err[sel].mean()) if sel.any() else None,
                       'mean_r': float(rmean[sel].mean()) if sel.any() else None})
    hi_r = rmean >= q[2]
    hi_e = err >= torch.quantile(err, 0.75)
    mu_t, v_t, mu_a, v_a, conf = ev
    alpha = v_t / (v_t + v_a)
    out = {
        'spearman_r_error': rho,
        'quartile_groups': groups,
        'high_r_high_error_fraction': float((hi_r & hi_e).sum() / max(1, hi_e.sum())),
        'overall_error_mean': float(err.mean()),
        'alpha_mean_per_dim': [float(alpha[:, i].mean()) for i in range(3)],
        'alpha_std_per_dim': [float(alpha[:, i].std()) for i in range(3)],
        'r_mean_per_dim': [float(r[:, i].mean()) for i in range(3)],
        'r_std_per_dim': [float(r[:, i].std()) for i in range(3)],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
