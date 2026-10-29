#!/usr/bin/env python3
"""Module-swap playground for EmoBank VAD semi-supervised regression.

Built on the frozen P1 checkpoint + P2 protocol (160 continuation steps).
Every run starts from the same P1 weights and sees the same labeled stream.

Swappable modules (all off by default => exact P2 behaviour):
  --sup-loss l1|l1_ccc        labelled loss (CCC term from dimensional-emotion literature)
  --ulb-loss l1|l1_ccc        pseudo-label loss
  --rank-weight W             RankUp/RCUS-style pairwise ranking consistency on unlabeled
  --uncertainty hetero        HPL-style heteroscedastic head (log-variance)
  --ulb-weight none|disagree|agree|unc
  --filter none|topq          keep top-q pseudo labels by reliability weight

Only TRAIN (lb/ulb) and DEV are read; TEST is never opened.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    DIMENSIONS, NORM_COLUMNS, GoldData, TextCollator, UnlabeledData, read_gold, read_unlabeled,
)
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import (  # noqa: E402
    RobertaVADRegressor, load_checked_splits, set_seed, sha256_file, git_commit,
)

DATA_ROOT = REPO.parent / 'data' / 'emobank_processed'
MODEL_DIR = REPO.parent / 'pretrained' / 'roberta-base'


# ----------------------------------------------------------------------------- losses
def ccc_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - mean per-dimension CCC. Both tensors are [B, 3] on the normalized scale."""
    mp, mt = pred.mean(0), target.mean(0)
    vp = pred.var(0, unbiased=False)
    vt = target.var(0, unbiased=False)
    cov = ((pred - mp) * (target - mt)).mean(0)
    ccc = 2.0 * cov / (vp + vt + (mp - mt) ** 2 + eps)
    return 1.0 - ccc.mean()


def pairwise_rank_loss(pred: torch.Tensor, pseudo: torch.Tensor,
                       margin: float = 0.05) -> torch.Tensor:
    """Pseudo-order consistency: student should reproduce the teacher's pairwise order."""
    b = pred.shape[0]
    idx = torch.triu_indices(b, b, offset=1)
    total = pred.new_zeros(())
    n = 0
    for d in range(pred.shape[1]):
        pi = pred[:, d]
        ti = pseudo[:, d]
        diff = ti[idx[0]] - ti[idx[1]]
        keep = diff.abs() > margin
        if keep.sum() == 0:
            continue
        sign = torch.sign(diff[keep])
        pair = (pi[idx[0]] - pi[idx[1]])[keep]
        total = total + F.relu(margin - sign * pair).mean()
        n += 1
    return total / max(n, 1)


class HeteroVAD(nn.Module):
    """RobertaVADRegressor + a log-variance head (HPL-style heteroscedastic regression)."""

    def __init__(self, base: RobertaVADRegressor):
        super().__init__()
        self.encoder = base.encoder
        self.dropout = base.dropout
        self.head = base.head
        hidden = base.encoder.config.hidden_size
        self.var_head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.var_head.weight)
        nn.init.constant_(self.var_head.bias, -1.0)

    def _pooled(self, inputs):
        states = self.encoder(**inputs, return_dict=True).last_hidden_state
        m = inputs['attention_mask'].unsqueeze(-1).to(dtype=states.dtype)
        return (states * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)

    def forward(self, **inputs):
        h = self.dropout(self._pooled(inputs))
        return self.head(h), self.var_head(h)


class UnlabeledCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts, ids = zip(*batch)
        return self.tokenizer(list(texts), padding=True, truncation=True,
                              max_length=self.max_length, return_tensors='pt'), list(ids)


def mean_forward(model, tok):
    out = model(**tok)
    return out[0] if isinstance(out, tuple) else out


def forever(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def ema_step(teacher, student, alpha):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.lerp_(s.detach(), 1.0 - alpha)
    for t, s in zip(teacher.buffers(), student.buffers()):
        if t.is_floating_point():
            t.lerp_(s.detach(), 1.0 - alpha)
        else:
            t.copy_(s)


@torch.inference_mode()
def predict_mean(model, loader, device):
    model.eval()
    gold, preds, ids = [], [], []
    for tok, y, batch_ids in loader:
        tok = {k: v.to(device) for k, v in tok.items()}
        p = mean_forward(model, tok)
        gold.append(y.numpy())
        preds.append(p.detach().cpu().numpy())
        ids.extend(batch_ids)
    return np.concatenate(gold), np.concatenate(preds), ids


@torch.inference_mode()
def encode_texts(encoder, texts, tokenizer, device, max_length, batch_size=64):
    """Frozen mean-pooled features used for label-anchored pseudo-label calibration."""
    feats = []
    for i in range(0, len(texts), batch_size):
        tok = tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt')
        tok = {k: v.to(device) for k, v in tok.items()}
        states = encoder(**tok, return_dict=True).last_hidden_state
        m = tok['attention_mask'].unsqueeze(-1).to(dtype=states.dtype)
        pooled = (states * m).sum(1) / m.sum(1).clamp_min(1)
        feats.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(feats)


def knn_calibrate(pseudo, u_idx, u_feat, l_feat, l_y, k, temp, alpha, conf_floor):
    """Blend teacher pseudo-labels with similarity-weighted labelled neighbours.

    alpha is scaled by the nearest-neighbour similarity so that text with no
    close labelled neighbour keeps the teacher label.
    """
    sim = u_feat[u_idx] @ l_feat.T                       # [B, N_lb]
    topv, topi = sim.topk(k, dim=1)
    w = torch.softmax(topv / temp, dim=1)
    knn_y = (w.unsqueeze(-1) * l_y[topi]).sum(1)         # [B, 3]
    conf = topv[:, 0].clamp(0.0, 1.0)                    # cosine similarity of NN
    gate = torch.clamp((conf - conf_floor) / max(1e-6, 1.0 - conf_floor), 0.0, 1.0)
    a = (alpha * gate).unsqueeze(-1)
    return (1.0 - a) * pseudo + a * knn_y, float(gate.mean().item())


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('control', 'mean_teacher'), default='mean_teacher')
    p.add_argument('--sup-loss', choices=('l1', 'l1_ccc'), default='l1')
    p.add_argument('--ulb-loss', choices=('l1', 'l1_ccc'), default='l1')
    p.add_argument('--ccc-weight', type=float, default=0.3)
    p.add_argument('--rank-weight', type=float, default=0.0)
    p.add_argument('--rank-margin', type=float, default=0.05)
    p.add_argument('--uncertainty', choices=('none', 'hetero'), default='none')
    p.add_argument('--ulb-weight', choices=('none', 'disagree', 'agree', 'unc'), default='none')
    p.add_argument('--weight-tau', type=float, default=0.05)
    p.add_argument('--filter', choices=('none', 'topq'), default='none')
    p.add_argument('--filter-q', type=float, default=0.5)
    p.add_argument('--knn-calib', type=float, default=0.0, help='blend weight for label-anchored kNN calibration')
    p.add_argument('--knn-k', type=int, default=10)
    p.add_argument('--knn-temp', type=float, default=0.1)
    p.add_argument('--knn-conf-floor', type=float, default=0.3)
    p.add_argument('--da', action='store_true', help='distribution alignment of pseudo-labels')
    p.add_argument('--da-alpha', type=float, default=1.0)
    p.add_argument('--eval-knn-alpha', type=float, default=0.0,
                   help='optional inference-time retrieval calibration weight')
    p.add_argument('--eval-knn-k', type=int, default=20)
    p.add_argument('--eval-scale', type=float, default=1.0,
                   help='variance rescaling of calibrated predictions (CCC protection)')
    p.add_argument('--diag-teacher', action='store_true')
    p.add_argument('--budget', type=int, choices=(250, 500, 1000), default=500)
    p.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--data-root', type=Path, default=DATA_ROOT)
    p.add_argument('--pretrain-path', type=Path, default=MODEL_DIR)
    p.add_argument('--p1-dir', type=Path, default=None)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--ulb-batch-size', type=int, default=64)
    p.add_argument('--eval-batch-size', type=int, default=32)
    p.add_argument('--steps', type=int, default=160)
    p.add_argument('--eval-every', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--clip-grad-norm', type=float, default=1.0)
    p.add_argument('--lambda-u', type=float, default=0.2)
    p.add_argument('--ema-decay', type=float, default=0.99)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if a.p1_dir is None:
        a.p1_dir = REPO / 'saved_models' / 'vad_research' / f'p1_roberta_lb{a.budget}_s{a.seed}'
    if a.out is None:
        a.out = REPO / 'saved_models' / 'vad_research' / f'swap_{a.mode}_lb{a.budget}_s{a.seed}'
    if not 0.0 <= a.filter_q <= 1.0:
        p.error('--filter-q must be in [0,1]')
    if a.ulb_weight == 'disagree' and a.mode != 'mean_teacher':
        p.error('--ulb-weight disagree needs --mode mean_teacher')
    return a


def main():
    a = cli()
    set_seed(a.seed)
    lb, dev, ulb, paths = load_checked_splits(a)
    if len(lb) != a.budget or len(ulb) != 8062 - a.budget or len(dev) != 1000:
        raise ValueError('Unexpected split counts')
    ckpt_path = a.p1_dir / 'model_best.pth'
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    p1 = json.loads((a.p1_dir / 'run_config.json').read_text(encoding='utf-8'))
    for k in ('budget', 'seed', 'max_length'):
        if int(p1[k]) != int(getattr(a, k)):
            raise ValueError(f'P1 config mismatch for {k}')
    if abs(float(p1.get('dropout', 0.1)) - a.dropout) > 1e-12:
        raise ValueError('dropout mismatch')
    hashes = {k: sha256_file(v) for k, v in paths.items()}
    if 'split_sha256' in p1 and hashes != p1['split_sha256']:
        raise ValueError('data hashes differ from frozen P1')
    print(f'mode={a.mode} sup={a.sup_loss} ulb={a.ulb_loss} rank={a.rank_weight} '
          f'unc={a.uncertainty} w={a.ulb_weight} filter={a.filter}')
    if a.check_only:
        print('PASS: P1/splits verified, nothing written')
        return

    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Nonempty output: {out}')
    device = torch.device('cpu' if a.device == 'cpu' or not torch.cuda.is_available() else 'cuda')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    tokenizer = AutoTokenizer.from_pretrained(str(a.pretrain_path.resolve()), local_files_only=True, use_fast=True)
    lb_dl = DataLoader(GoldData(lb), batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                       collate_fn=TextCollator(tokenizer, a.max_length),
                       generator=torch.Generator().manual_seed(17_000 + a.seed))
    dev_dl = DataLoader(GoldData(dev), batch_size=a.eval_batch_size, shuffle=False, num_workers=a.num_workers,
                        collate_fn=TextCollator(tokenizer, a.max_length))
    ulb_dl = None
    if a.mode == 'mean_teacher':
        ulb_dl = DataLoader(UnlabeledData(ulb), batch_size=a.ulb_batch_size, shuffle=True,
                            num_workers=a.num_workers, collate_fn=UnlabeledCollator(tokenizer, a.max_length),
                            generator=torch.Generator().manual_seed(19_000 + a.seed))

    base = RobertaVADRegressor(str(a.pretrain_path.resolve()), dropout=a.dropout, offline=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    base.load_state_dict(ckpt['model'], strict=True)
    del ckpt
    if a.uncertainty == 'hetero':
        student = HeteroVAD(base).to(device)
    else:
        student = base.to(device)
    teacher = None
    if a.mode == 'mean_teacher':
        teacher = copy.deepcopy(student).to(device)
        teacher.requires_grad_(False)
        teacher.eval()

    knn_bank = None
    l_feat = l_y = None
    if a.knn_calib > 0 or a.eval_knn_alpha > 0:
        l_feat = encode_texts(student.encoder, lb['text'].tolist(), tokenizer, device, a.max_length).to(device)
        l_y = torch.tensor(lb.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float32), device=device)
        if a.knn_calib > 0:
            u_feat = encode_texts(student.encoder, ulb['text'].tolist(), tokenizer, device, a.max_length).to(device)
            id2row = {k: i for i, k in enumerate(ulb['id'].tolist())}
            knn_bank = (u_feat, id2row)
        print(f'knn bank: labeled={l_feat.shape[0]} dim={l_feat.shape[1]}')
    dev_feat = None
    if a.eval_knn_alpha > 0:
        dev_feat = encode_texts(student.encoder, dev['text'].tolist(), tokenizer, device, a.max_length).to(device)
    lb_y_all = torch.tensor(lb.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float32), device=device)

    optim = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    out.mkdir(parents=True, exist_ok=True)
    conf = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    conf.update({'git_commit': git_commit(), 'device': str(device),
                 'p1_checkpoint_sha256': sha256_file(ckpt_path), 'split_sha256': hashes,
                 'training_design': 'P1 init + 160 continuation steps; modules swapped'})
    (out / 'run_config.json').write_text(json.dumps(conf, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    labeled_iter = forever(lb_dl)
    unlabeled_iter = forever(ulb_dl) if ulb_dl is not None else None
    visited_ulb = set()
    best, best_step = float('inf'), -1
    history = []
    sum_sup = sum_unsup = 0.0
    sum_w = 0.0
    n_w = 0
    sum_gate = 0.0
    n_gate = 0
    filt_frac = 0.0
    n_filt = 0

    def evaluate_and_save(step):
        nonlocal best, best_step
        y, p, ids = predict_mean(student, dev_dl, device)
        if dev_feat is not None:
            pt = torch.tensor(p, device=device)
            idx = torch.arange(pt.shape[0], device=device)
            pt, _ = knn_calibrate(pt, idx, dev_feat, l_feat, l_y,
                                  a.eval_knn_k, a.knn_temp, a.eval_knn_alpha, a.knn_conf_floor)
            p = pt.cpu().numpy()
        if a.eval_scale != 1.0:
            p = p.mean(0) + (p - p.mean(0)) * a.eval_scale
        m = metric_report(y, p)
        mae = m['macro_mae_original_scale']
        ccc = float(np.mean([m[d]['ccc'] for d in DIMENSIONS]))
        r2 = float(np.mean([m[d]['r2'] for d in DIMENSIONS]))
        stats = {}
        for idx, d in enumerate(DIMENSIONS):
            stats[d] = {'gold_mean': float(y[:, idx].mean() * 4 + 1),
                        'pred_mean': float(p[:, idx].mean() * 4 + 1),
                        'gold_std': float(y[:, idx].std() * 4),
                        'pred_std': float(p[:, idx].std() * 4)}
        t_mae = None
        if a.diag_teacher and teacher is not None:
            yt, pt, _ = predict_mean(teacher, dev_dl, device)
            t_mae = metric_report(yt, pt)['macro_mae_original_scale']
        row = {'step': step, 'macro_mae': mae, 'macro_ccc': ccc, 'macro_r2': r2,
               'sup_l1_norm_avg': sum_sup / max(1, step),
               'unsup_l1_norm_avg': sum_unsup / max(1, step),
               'unique_ulb_used': len(visited_ulb),
               'mean_pseudo_weight': sum_w / max(1, n_w),
               'knn_gate': sum_gate / max(1, n_gate),
               'filtered_frac': filt_frac / max(1, n_filt)}
        if t_mae is not None:
            row['teacher_dev_mae'] = t_mae
        for d in DIMENSIONS:
            row[f'{d}_mae'] = m[d]['mae']
            row[f'{d}_ccc'] = m[d]['ccc']
            row[f'{d}_r2'] = m[d]['r2']
            row[f'{d}_pred_std_ratio'] = stats[d]['pred_std'] / max(stats[d]['gold_std'], 1e-12)
        history.append(row)
        print(f'step={step:03d} dev_MAE={mae:.6f} CCC={ccc:.5f} R2={r2:.4f} '
              f'w={row["mean_pseudo_weight"]:.3f} filt={row["filtered_frac"]:.2f} '
              + ' '.join(f'{d}={m[d]["mae"]:.4f}' for d in DIMENSIONS), flush=True)
        if mae < best - 1e-8:
            best, best_step = mae, step
            torch.save({'model': student.state_dict(), 'step': step, 'dev_metrics': m, 'run_config': conf},
                       out / 'model_best.pth')
            (out / 'dev_metrics_best.json').write_text(
                json.dumps({'step': step, 'metrics': m, 'stats': stats}, indent=2) + '\n', encoding='utf-8')
            frame = pd.DataFrame({'id': ids})
            for idx, d in enumerate(DIMENSIONS):
                frame[f'gold_{d}'] = y[:, idx] * 4 + 1
                frame[f'pred_{d}'] = p[:, idx] * 4 + 1
            frame.to_csv(out / 'dev_predictions_best.csv', index=False)

    evaluate_and_save(0)
    for step in range(1, a.steps + 1):
        student.train()
        lb_tok, lb_y, _ = next(labeled_iter)
        lb_tok = {k: v.to(device) for k, v in lb_tok.items()}
        lb_y = lb_y.to(device)
        optim.zero_grad(set_to_none=True)
        out_s = student(**lb_tok)
        pred = out_s[0] if isinstance(out_s, tuple) else out_s
        sup_loss = F.l1_loss(pred, lb_y)
        if a.sup_loss == 'l1_ccc':
            sup_loss = sup_loss + a.ccc_weight * ccc_loss(pred, lb_y)
        total = sup_loss
        unsup_value = 0.0
        if a.mode == 'mean_teacher':
            ulb_tok, ulb_ids = next(unlabeled_iter)
            visited_ulb.update(ulb_ids)
            ulb_tok = {k: v.to(device) for k, v in ulb_tok.items()}
            teacher.eval()
            used_ids = list(ulb_ids)
            with torch.no_grad():
                pseudo = mean_forward(teacher, ulb_tok).detach()
            if knn_bank is not None:
                u_feat, id2row = knn_bank
                u_idx = torch.tensor([id2row[i] for i in used_ids], device=device)
                pseudo, gate_mean = knn_calibrate(pseudo, u_idx, u_feat, l_feat, l_y,
                                                  a.knn_k, a.knn_temp, a.knn_calib, a.knn_conf_floor)
                sum_gate += gate_mean
                n_gate += 1
            if a.da:
                mu_p, sd_p = pseudo.mean(0), pseudo.std(0).clamp_min(1e-6)
                aligned = (pseudo - mu_p) / sd_p * lb_y_all.std(0) + lb_y_all.mean(0)
                pseudo = a.da_alpha * aligned + (1.0 - a.da_alpha) * pseudo
            out_u = student(**ulb_tok)
            pred_u = out_u[0] if isinstance(out_u, tuple) else out_u
            logvar = out_u[1] if isinstance(out_u, tuple) else None

            weight = None
            if a.ulb_weight == 'disagree':
                teacher.train()
                with torch.no_grad():
                    p1 = mean_forward(teacher, ulb_tok)
                    p2 = mean_forward(teacher, ulb_tok)
                teacher.eval()
                dis = (p1 - p2).abs().mean(1)
                weight = torch.exp(-dis / a.weight_tau)
            elif a.ulb_weight == 'agree':
                with torch.no_grad():
                    disagree = (pred_u.detach() - pseudo).abs().mean(1)
                weight = torch.exp(-disagree / a.weight_tau)

            if a.ulb_weight == 'unc' and logvar is not None:
                # Heteroscedastic Laplace: per-dimension scale learned from data.
                s = logvar.clamp(-6.0, 4.0)
                err = (pred_u - pseudo).abs()
                unsup_loss = (torch.exp(-s) * err + s).mean()
                w_mean = float(torch.exp(-s.detach()).mean().item())
            else:
                per = (pred_u - pseudo).abs().mean(1)  # sample-level L1
                if weight is not None:
                    if a.filter == 'topq':
                        k = max(1, int(round(a.filter_q * per.numel())))
                        thresh = torch.topk(weight, k).values.min()
                        weight = weight * (weight >= thresh)
                    wl = (weight * per).sum() / weight.sum().clamp_min(1e-8)
                else:
                    if a.filter == 'topq':
                        k = max(1, int(round(a.filter_q * per.numel())))
                        keep = torch.zeros_like(per, dtype=torch.bool)
                        keep[torch.topk(per, k, largest=False).indices] = True
                        wl = per[keep].mean()
                    else:
                        wl = per.mean()
                    weight = torch.ones_like(per)
                unsup_loss = wl + (a.ccc_weight * ccc_loss(pred_u, pseudo) if a.ulb_loss == 'l1_ccc' else 0.0)
                w_mean = float(weight.mean().item())

            sum_w += w_mean
            n_w += 1
            if a.filter == 'topq':
                n_filt += 1
                filt_frac += 1.0 - w_mean

            total = total + a.lambda_u * unsup_loss
            unsup_value = float(unsup_loss.detach().item())
            if a.rank_weight > 0:
                total = total + a.rank_weight * pairwise_rank_loss(pred_u, pseudo, a.rank_margin)
        if not torch.isfinite(total):
            raise RuntimeError('Nonfinite loss')
        total.backward()
        if a.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), a.clip_grad_norm)
        optim.step()
        if teacher is not None:
            ema_step(teacher, student, a.ema_decay)
        sum_sup += float(sup_loss.detach().item())
        sum_unsup += unsup_value
        if step % a.eval_every == 0 or step == a.steps:
            evaluate_and_save(step)

    pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
    print(f'BEST mode={a.mode} best_step={best_step} best_dev_MAE={best:.6f}')
    print('OUTPUT:', out)


if __name__ == '__main__':
    main()
