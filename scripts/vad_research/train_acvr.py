#!/usr/bin/env python3
"""ACVR-VAD: JACF (joint anchor-consistency fusion) + optional VRRC.

Base framework (RoBERTa encoder + 3-D head + EMA teacher) unchanged; ARC/RDA
removed. The only training-side component is JACF; inference calibration is a
separate switch for A1/A3 comparisons.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    DIMENSIONS, NORM_COLUMNS, GoldData, TextCollator, read_gold, read_unlabeled,
)
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import (  # noqa: E402
    RobertaVADRegressor, load_checked_splits, set_seed, sha256_file, git_commit,
)
from scripts.vad_research.jacf import jacf_fuse  # noqa: E402
from scripts.vad_research.jacf_modes import evidence, build_objective  # noqa: E402

DATA_ROOT = REPO.parent / 'data' / 'emobank_processed'
MODEL_DIR = REPO.parent / 'pretrained' / 'roberta-base'


class UnlabeledTexts(Dataset):
    def __init__(self, texts, ids):
        self.texts, self.ids = texts, ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.ids[i]


def forever(loader):
    while True:
        for batch in loader:
            yield batch


@torch.inference_mode()
def encode(encoder, texts, tokenizer, device, max_length=128, bs=64):
    encoder.eval()
    out = []
    for i in range(0, len(texts), bs):
        tok = tokenizer(list(texts[i:i + bs]), padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt')
        tok = {k: v.to(device) for k, v in tok.items()}
        st = encoder(**tok, return_dict=True).last_hidden_state
        m = tok['attention_mask'].unsqueeze(-1).to(dtype=st.dtype)
        out.append(F.normalize((st * m).sum(1) / m.sum(1).clamp_min(1), dim=-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def ema_step(teacher, student, alpha):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.lerp_(s.detach(), 1.0 - alpha)
    for t, s in zip(teacher.buffers(), student.buffers()):
        if t.is_floating_point():
            t.lerp_(s.detach(), 1.0 - alpha)
        else:
            t.copy_(s)


@torch.no_grad()
def mc_predict(model, tokens, passes):
    was = model.training
    model.train()
    preds = []
    for _ in range(passes):
        preds.append(model(**tokens).detach())
    model.train(was)
    return torch.stack(preds)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    ys, ps, ids = [], [], []
    for tok, y, bid in loader:
        tok = {k: v.to(device) for k, v in tok.items()}
        ps.append(model(**tok).cpu().numpy())
        ys.append(y.numpy())
        ids.extend(bid)
    return np.concatenate(ys), np.concatenate(ps), ids


def knn_calibrate(pred, query_feat, l_feat, l_y, k, temp, alpha, c0):
    sim = query_feat @ l_feat.T
    kk = min(k, l_feat.shape[0])
    topv, topi = sim.topk(kk, dim=1)
    w = torch.softmax(topv / temp, dim=1)
    knn = (w.unsqueeze(-1) * l_y[topi]).sum(1)
    conf = topv[:, 0].clamp(0.0, 1.0)
    gate = torch.clamp((conf - c0) / max(1e-6, 1.0 - c0), 0.0, 1.0)
    a = (alpha * gate).unsqueeze(-1)
    return (1.0 - a) * pred + a * knn


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budget', type=int, choices=(250, 500, 1000), default=500)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--data-root', type=Path, default=DATA_ROOT)
    p.add_argument('--pretrain-path', type=Path, default=MODEL_DIR)
    p.add_argument('--p1-dir', type=Path, default=None)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--steps', type=int, default=160)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--ulb-batch-size', type=int, default=64)
    p.add_argument('--eval-batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--clip-grad-norm', type=float, default=1.0)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--ema-decay', type=float, default=0.99)
    p.add_argument('--lambda-u', type=float, default=1.0)
    p.add_argument('--mode', choices=('multi', 'anchor', 'seq', 'jacf', 'no_rel'), default='jacf')
    p.add_argument('--seq-alpha', type=float, default=0.4)
    p.add_argument('--gcf-tau', type=float, default=0.05)
    p.add_argument('--mc-passes', type=int, default=5)
    p.add_argument('--jacf-k', type=int, default=20)
    p.add_argument('--jacf-tau', type=float, default=0.1)
    p.add_argument('--jacf-eps-t', type=float, default=1e-4)
    p.add_argument('--jacf-eps-a', type=float, default=1e-4)
    p.add_argument('--jacf-v0', type=float, default=1e-3)
    p.add_argument('--eval-knn-alpha', type=float, default=0.0)
    p.add_argument('--eval-knn-k', type=int, default=20)
    p.add_argument('--eval-scale', type=float, default=1.0)
    p.add_argument('--mask-prob', type=float, default=0.15)
    p.add_argument('--save-ckpt', action='store_true', default=True)
    p.add_argument('--no-save-ckpt', dest='save_ckpt', action='store_false')
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    a = p.parse_args()
    if a.p1_dir is None:
        a.p1_dir = REPO / 'saved_models' / 'vad_research' / f'p1_roberta_lb{a.budget}_s{a.seed}'
    if a.out is None:
        a.out = REPO / 'saved_models' / 'vad_research' / f'acvr_lb{a.budget}_s{a.seed}'
    return a


def mask_tokens(tokens, tokenizer, prob):
    ids = tokens['input_ids'].clone()
    keep = torch.ones_like(ids, dtype=torch.bool)
    for sid in set(tokenizer.all_special_ids):
        keep &= ids != sid
    m = keep & (torch.rand(ids.shape, device=ids.device) < prob)
    ids[m] = tokenizer.mask_token_id
    out = dict(tokens)
    out['input_ids'] = ids
    return out


def main():
    a = cli()
    set_seed(a.seed)
    lb, dev, ulb, paths = load_checked_splits(a)
    ckpt_path = a.p1_dir / 'model_best.pth'
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    device = torch.device('cpu' if a.device == 'cpu' or not torch.cuda.is_available() else 'cuda')
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path.resolve()), local_files_only=True, use_fast=True)

    model = RobertaVADRegressor(str(a.pretrain_path.resolve()), dropout=a.dropout, offline=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=True)
    del ckpt
    model.to(device)
    teacher = copy.deepcopy(model).to(device)
    teacher.requires_grad_(False)
    teacher.eval()
    optim = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    lb_dl = DataLoader(GoldData(lb), batch_size=a.batch_size, shuffle=True,
                       collate_fn=TextCollator(tok, 128),
                       generator=torch.Generator().manual_seed(17_000 + a.seed))
    dev_dl = DataLoader(GoldData(dev), batch_size=a.eval_batch_size, shuffle=False,
                        collate_fn=TextCollator(tok, 128))

    def ulb_collate(batch):
        texts, ids = zip(*batch)
        return tok(list(texts), padding=True, truncation=True, max_length=128,
                   return_tensors='pt'), list(ids)

    ulb_dl = DataLoader(UnlabeledTexts(ulb['text'].tolist(), ulb['id'].tolist()),
                        batch_size=a.ulb_batch_size, shuffle=True, collate_fn=ulb_collate,
                        generator=torch.Generator().manual_seed(19_000 + a.seed))

    l_feat = encode(model.encoder, lb['text'].tolist(), tok, device).to(device)
    l_y = torch.tensor(lb.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float32), device=device)
    u_feat = encode(model.encoder, ulb['text'].tolist(), tok, device).to(device)
    id2row = {k: i for i, k in enumerate(ulb['id'].tolist())}
    dev_feat = None
    if a.eval_knn_alpha > 0:
        dev_feat = encode(model.encoder, dev['text'].tolist(), tok, device).to(device)

    out.mkdir(parents=True, exist_ok=True)
    conf = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    conf.update({'git_commit': git_commit(), 'device': str(device),
                 'p1_sha256': sha256_file(ckpt_path)})
    (out / 'run_config.json').write_text(json.dumps(conf, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    lb_iter, ulb_iter = forever(lb_dl), forever(ulb_dl)
    history = []
    best, best_step = float('inf'), -1
    for step in range(0, a.steps + 1):
        if step == 0 or step % a.eval_every == 0 or step == a.steps:
            y, p_raw, ids = predict(model, dev_dl, device)
            m_raw = metric_report(y, p_raw)
            p = p_raw
            if dev_feat is not None:
                pt = torch.tensor(p, device=device)
                pt = knn_calibrate(pt, dev_feat, l_feat, l_y, a.eval_knn_k,
                                   a.jacf_tau, a.eval_knn_alpha, 0.3)
                p = pt.cpu().numpy()
            if a.eval_scale != 1.0:
                p = p.mean(0) + (p - p.mean(0)) * a.eval_scale
            m = metric_report(y, p)
            mae = m['macro_mae_original_scale']
            raw_mae = m_raw['macro_mae_original_scale']
            ccc = float(np.mean([m[d]['ccc'] for d in DIMENSIONS]))
            r2 = float(np.mean([m[d]['r2'] for d in DIMENSIONS]))
            raw_ccc = float(np.mean([m_raw[d]['ccc'] for d in DIMENSIONS]))
            raw_r2 = float(np.mean([m_raw[d]['r2'] for d in DIMENSIONS]))
            history.append({'step': step, 'macro_mae': mae, 'macro_ccc': ccc, 'macro_r2': r2,
                            'raw_mae': raw_mae, 'raw_ccc': raw_ccc, 'raw_r2': raw_r2})
            print(f'step={step:03d} dev_MAE={mae:.6f} raw_MAE={raw_mae:.6f} '
                  f'CCC={ccc:.5f} raw_CCC={raw_ccc:.5f}', flush=True)
            if mae < best - 1e-8:
                best, best_step = mae, step
                if a.save_ckpt:
                    torch.save({'model': model.state_dict(), 'step': step,
                                'dev_metrics': m, 'raw_metrics': m_raw},
                               out / 'model_best.pth')
                (out / 'dev_metrics_best.json').write_text(
                    json.dumps({'step': step, 'metrics': m, 'raw_metrics': m_raw},
                               indent=2) + '\n', encoding='utf-8')
        if step == a.steps:
            break
        model.train()
        lb_tok, lb_y, _ = next(lb_iter)
        lb_tok = {k: v.to(device) for k, v in lb_tok.items()}
        lb_y = lb_y.to(device)
        ulb_tok_w, ulb_ids = next(ulb_iter)
        ulb_tok_w = {k: v.to(device) for k, v in ulb_tok_w.items()}
        ulb_tok_s = mask_tokens(ulb_tok_w, tok, a.mask_prob)

        optim.zero_grad(set_to_none=True)
        pred_lb = model(**lb_tok)
        loss = F.l1_loss(pred_lb, lb_y)

        mc = mc_predict(teacher, ulb_tok_w, a.mc_passes)
        rows = torch.tensor([id2row[i] for i in ulb_ids], device=device)
        ev = evidence(mc, u_feat[rows], l_feat, l_y,
                      k=a.jacf_k, tau=a.jacf_tau,
                      eps_t=a.jacf_eps_t, eps_a=a.jacf_eps_a)
        target, rel = build_objective(a.mode, ev, v0=a.jacf_v0,
                                      alpha=a.seq_alpha, c0=0.3, gcf_tau=a.gcf_tau)
        pred_s = model(**ulb_tok_s)
        err = (pred_s - target).abs()
        unsup = (rel * err).sum() / rel.sum().clamp_min(1e-6)
        total = loss + a.lambda_u * unsup
        if not torch.isfinite(total):
            raise RuntimeError('Nonfinite loss')
        total.backward()
        if a.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad_norm)
        optim.step()
        ema_step(teacher, model, a.ema_decay)

    pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
    print(f'BEST best_step={best_step} best_dev_MAE={best:.6f}')
    print('OUTPUT:', out)


if __name__ == '__main__':
    main()
