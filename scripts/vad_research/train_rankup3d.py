#!/usr/bin/env python3
"""RankUp-3D: faithful 3-dimensional adaptation of RankUp + RDA (NeurIPS 2024).

Base framework : RankUp_Net (per-sample 2-class ARC, pairwise ranking CE) + RDA
                 (regression distribution alignment), from
                 https://github.com/pm25/semi-supervised-regression
Adaptation     : one ARC per V/A/D dimension, RDA per dimension, text weak/strong
                 views via token masking (EmoBank has no back-translation views).
Swap-ins       : --filter gcf  -> GaussianMatch-style multi-view Gaussian filter
                 --reg-target rpl -> label-anchored retrieval calibration (LA-PLC)
                 --eval-knn-alpha/--eval-scale -> VRRC inference calibration

Only TRAIN (lb/ulb) and DEV are read; TEST is never opened here.
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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    DIMENSIONS, NORM_COLUMNS, GoldData, TextCollator, read_gold, read_unlabeled,
)
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import (  # noqa: E402
    load_checked_splits, set_seed, sha256_file, git_commit,
)

DATA_ROOT = REPO.parent / 'data' / 'emobank_processed'
MODEL_DIR = REPO.parent / 'pretrained' / 'roberta-base'


# --------------------------------------------------------------------------- model
class RankUpVAD(nn.Module):
    """RoBERTa encoder + 3-D regression head + per-dimension 2-class ARC head."""

    def __init__(self, model_name, dropout=0.1, offline=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=offline)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.arc = nn.Linear(hidden, 6)  # 3 dims x 2 classes

    def pooled(self, **inputs):
        states = self.encoder(**inputs, return_dict=True).last_hidden_state
        m = inputs['attention_mask'].unsqueeze(-1).to(dtype=states.dtype)
        return (states * m).sum(1) / m.sum(1).clamp_min(1)

    def forward(self, **inputs):
        h = self.dropout(self.pooled(**inputs))
        return self.head(h), self.arc(h).view(-1, 3, 2)


# --------------------------------------------------------------------------- helpers
def pair_logits(arc, d):
    """arc: [B,3,2] -> [B,B,2] with element [i,j] = a_i - a_j for dimension d."""
    a = arc[:, d, :]
    return a.unsqueeze(0) - a.unsqueeze(1)


def soft_ce(logits, target):
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1)


def label_pair_targets(y):
    """[B] -> [B,B,2] soft one-hot of (y_i > y_j)."""
    t = (1.0 + torch.sign(y.unsqueeze(0) - y.unsqueeze(1))) / 2.0
    return torch.stack([t, 1.0 - t], dim=-1)


def mask_tokens(tokens, tokenizer, prob):
    ids = tokens['input_ids'].clone()
    special = set(tokenizer.all_special_ids)
    keep = torch.ones_like(ids, dtype=torch.bool)
    for sid in special:
        keep &= ids != sid
    rand = torch.rand(ids.shape, device=ids.device)
    m = keep & (rand < prob)
    ids[m] = tokenizer.mask_token_id
    out = dict(tokens)
    out['input_ids'] = ids
    return out


def rda_align(pseudo_raw, sorted_lb):
    """Per-dimension quantile alignment to the labeled marginal (RankUp RDA hook)."""
    n = pseudo_raw.shape[0]
    cdf_p = np.linspace(0, 1, n)
    out = torch.empty_like(pseudo_raw)
    for d in range(3):
        cdf_t = np.linspace(0, 1, sorted_lb.shape[0])
        vals = np.interp(cdf_p, cdf_t, sorted_lb[:, d].numpy())
        order = torch.argsort(pseudo_raw[:, d])
        out[order, d] = torch.tensor(vals, dtype=pseudo_raw.dtype, device=pseudo_raw.device)
    return out


def knn_calibrate(pseudo, u_idx, u_feat, l_feat, l_y, k, temp, alpha, conf_floor):
    sim = u_feat[u_idx] @ l_feat.T
    topv, topi = sim.topk(k, dim=1)
    w = torch.softmax(topv / temp, dim=1)
    knn_y = (w.unsqueeze(-1) * l_y[topi]).sum(1)
    conf = topv[:, 0].clamp(0.0, 1.0)
    gate = torch.clamp((conf - conf_floor) / max(1e-6, 1.0 - conf_floor), 0.0, 1.0)
    a = (alpha * gate).unsqueeze(-1)
    return (1.0 - a) * pseudo + a * knn_y


@torch.inference_mode()
def encode_texts(encoder, texts, tokenizer, device, max_length=128, bs=64):
    feats = []
    for i in range(0, len(texts), bs):
        tok = tokenizer(list(texts[i:i + bs]), padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt')
        tok = {k: v.to(device) for k, v in tok.items()}
        st = encoder(**tok, return_dict=True).last_hidden_state
        m = tok['attention_mask'].unsqueeze(-1).to(dtype=st.dtype)
        feats.append(F.normalize((st * m).sum(1) / m.sum(1).clamp_min(1), dim=-1).cpu())
    return torch.cat(feats)


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
    ys, ps, ids = [], [], []
    for tok, y, bid in loader:
        tok = {k: v.to(device) for k, v in tok.items()}
        pred, _ = model(**tok)
        ys.append(y.numpy())
        ps.append(pred.cpu().numpy())
        ids.extend(bid)
    return np.concatenate(ys), np.concatenate(ps), ids


class UnlabeledTexts(Dataset):
    def __init__(self, texts, ids):
        self.texts, self.ids = texts, ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.ids[i]


def ulb_collate(tokenizer, max_length):
    def fn(batch):
        texts, ids = zip(*batch)
        tok = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt')
        return tok, list(ids)
    return fn


def forever(loader):
    while True:
        for batch in loader:
            yield batch


# --------------------------------------------------------------------------- cli
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
    p.add_argument('--mask-prob', type=float, default=0.15)
    # RankUp / RDA
    p.add_argument('--arc', action='store_true', default=True)
    p.add_argument('--no-arc', dest='arc', action='store_false')
    p.add_argument('--arc-loss-ratio', type=float, default=1.0)
    p.add_argument('--arc-ulb-ratio', type=float, default=1.0)
    p.add_argument('--p-cutoff', type=float, default=0.95)
    p.add_argument('--reg-target', choices=('rda', 'raw', 'rpl'), default='rda')
    p.add_argument('--rda-every', type=int, default=32)
    p.add_argument('--unsup-warmup', type=float, default=0.4)
    p.add_argument('--ulb-loss-ratio', type=float, default=1.0)
    # GaussianMatch-style filter
    p.add_argument('--filter', choices=('arc', 'gcf', 'agds', 'none'), default='arc')
    p.add_argument('--gcf-views', type=int, default=3)
    p.add_argument('--gcf-tau', type=float, default=0.05)
    p.add_argument('--agds-views', type=int, default=3)
    p.add_argument('--agds-lambda0', type=float, default=3.0)
    p.add_argument('--agds-lambda1', type=float, default=1.0)
    p.add_argument('--agds-warm', type=float, default=0.5)
    p.add_argument('--agds-ema', type=float, default=0.9)
    p.add_argument('--agds-soft', action='store_true', default=True)
    p.add_argument('--agds-hard', dest='agds_soft', action='store_false')
    # retrieval calibration (LA-PLC / VRRC)
    p.add_argument('--knn-calib', type=float, default=0.0)
    p.add_argument('--knn-k', type=int, default=20)
    p.add_argument('--knn-temp', type=float, default=0.1)
    p.add_argument('--knn-conf-floor', type=float, default=0.3)
    p.add_argument('--eval-knn-alpha', type=float, default=0.0)
    p.add_argument('--eval-knn-k', type=int, default=20)
    p.add_argument('--eval-scale', type=float, default=1.0)
    p.add_argument('--diag-teacher', action='store_true')
    p.add_argument('--save-ckpt', action='store_true', default=True)
    p.add_argument('--no-save-ckpt', dest='save_ckpt', action='store_false')
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    a = p.parse_args()
    if a.p1_dir is None:
        a.p1_dir = REPO / 'saved_models' / 'vad_research' / f'p1_roberta_lb{a.budget}_s{a.seed}'
    if a.out is None:
        a.out = REPO / 'saved_models' / 'vad_research' / f'rankup3d_lb{a.budget}_s{a.seed}'
    return a


# --------------------------------------------------------------------------- train
def main():
    a = cli()
    set_seed(a.seed)
    lb, dev, ulb, paths = load_checked_splits(a)
    ckpt_path = a.p1_dir / 'model_best.pth'
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    p1 = json.loads((a.p1_dir / 'run_config.json').read_text(encoding='utf-8'))
    if int(p1['budget']) != a.budget or int(p1['seed']) != a.seed:
        raise ValueError('P1 budget/seed mismatch')
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Nonempty output: {out}')
    device = torch.device('cpu' if a.device == 'cpu' or not torch.cuda.is_available() else 'cuda')
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path.resolve()), local_files_only=True, use_fast=True)

    model = RankUpVAD(str(a.pretrain_path.resolve()), dropout=a.dropout, offline=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if unexpected or any(not k.startswith('arc.') for k in missing):
        raise RuntimeError(f'Unexpected P1 keys: missing={missing} unexpected={unexpected}')
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
    ulb_ds = UnlabeledTexts(ulb['text'].tolist(), ulb['id'].tolist())
    ulb_dl = DataLoader(ulb_ds, batch_size=a.ulb_batch_size, shuffle=True,
                        collate_fn=ulb_collate(tok, 128),
                        generator=torch.Generator().manual_seed(19_000 + a.seed))

    # ---- retrieval bank (LA-PLC / VRRC)
    l_feat = l_y = u_feat = dev_feat = None
    id2row = {k: i for i, k in enumerate(ulb['id'].tolist())}
    if a.knn_calib > 0 or a.eval_knn_alpha > 0 or a.reg_target == 'rpl':
        l_feat = encode_texts(model.encoder, lb['text'].tolist(), tok, device).to(device)
        l_y = torch.tensor(lb.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float32), device=device)
        if a.reg_target == 'rpl' or a.knn_calib > 0:
            u_feat = encode_texts(model.encoder, ulb['text'].tolist(), tok, device).to(device)
        if a.eval_knn_alpha > 0:
            dev_feat = encode_texts(model.encoder, dev['text'].tolist(), tok, device).to(device)

    # ---- RDA initial table from the frozen teacher
    pseudo_raw = pseudo_refine = None
    sorted_lb = None
    if a.reg_target == 'rda':
        with torch.inference_mode():
            chunks = []
            for bt, _ in ulb_dl:
                bt = {k: v.to(device) for k, v in bt.items()}
                pred, _ = teacher(**bt)
                chunks.append(pred.cpu())
        pseudo_raw = torch.cat(chunks)
        sorted_lb = torch.sort(torch.tensor(
            lb.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float32)))[0]
        pseudo_refine = rda_align(pseudo_raw, sorted_lb)

    out.mkdir(parents=True, exist_ok=True)
    conf = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    conf.update({'git_commit': git_commit(), 'device': str(device),
                 'p1_sha256': sha256_file(ckpt_path),
                 'design': 'RankUp-3D ARC + per-dim RDA; optional GCF / LA-PLC / VRRC'})
    (out / 'run_config.json').write_text(json.dumps(conf, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    history = []
    best, best_step = float('inf'), -1
    scale_ema = None
    lb_iter = forever(lb_dl)
    ulb_iter = forever(ulb_dl)

    def evaluate_and_save(step):
        nonlocal best, best_step
        y, p, ids = predict_mean(model, dev_dl, device)
        if dev_feat is not None:
            pt = torch.tensor(p, device=device)
            idx = torch.arange(pt.shape[0], device=device)
            pt = knn_calibrate(pt, idx, dev_feat, l_feat, l_y,
                               a.eval_knn_k, a.knn_temp, a.eval_knn_alpha, a.knn_conf_floor)
            p = pt.cpu().numpy()
        if a.eval_scale != 1.0:
            p = p.mean(0) + (p - p.mean(0)) * a.eval_scale
        m = metric_report(y, p)
        mae = m['macro_mae_original_scale']
        ccc = float(np.mean([m[d]['ccc'] for d in DIMENSIONS]))
        r2 = float(np.mean([m[d]['r2'] for d in DIMENSIONS]))
        row = {'step': step, 'macro_mae': mae, 'macro_ccc': ccc, 'macro_r2': r2}
        for d in DIMENSIONS:
            row[f'{d}_mae'] = m[d]['mae']
            row[f'{d}_ccc'] = m[d]['ccc']
        history.append(row)
        print(f'step={step:03d} dev_MAE={mae:.6f} CCC={ccc:.5f} R2={r2:.4f} '
              + ' '.join(f'{d}={m[d]["mae"]:.4f}' for d in DIMENSIONS), flush=True)
        if mae < best - 1e-8:
            best, best_step = mae, step
            if a.save_ckpt:
                torch.save({'model': model.state_dict(), 'step': step, 'dev_metrics': m},
                           out / 'model_best.pth')
            (out / 'dev_metrics_best.json').write_text(
                json.dumps({'step': step, 'metrics': m}, indent=2) + '\n', encoding='utf-8')

    evaluate_and_save(0)
    for step in range(1, a.steps + 1):
        model.train()
        lb_tok, lb_y, _ = next(lb_iter)
        lb_tok = {k: v.to(device) for k, v in lb_tok.items()}
        lb_y = lb_y.to(device)
        ulb_tok_w, ulb_ids = next(ulb_iter)
        ulb_tok_w = {k: v.to(device) for k, v in ulb_tok_w.items()}
        ulb_tok_s = mask_tokens(ulb_tok_w, tok, a.mask_prob)

        optim.zero_grad(set_to_none=True)
        reg_lb, arc_lb = model(**lb_tok)
        loss = F.l1_loss(reg_lb, lb_y)
        if a.arc:
            arc_sup = 0.0
            for d in range(3):
                pl = pair_logits(arc_lb, d)
                tgt = label_pair_targets(lb_y[:, d])
                arc_sup = arc_sup + soft_ce(pl, tgt).mean()
            arc_sup = arc_sup / 3.0
        else:
            arc_sup = torch.zeros((), device=device)

        with torch.no_grad():
            reg_t_w, arc_t_w = teacher(**ulb_tok_w)
        reg_s, arc_s = model(**ulb_tok_s)

        rows = torch.tensor([id2row[i] for i in ulb_ids], device=device)

        # ---- regression target (RDA / LA-PLC / raw teacher)
        if a.reg_target == 'rda':
            src = reg_t_w
            if a.knn_calib > 0:
                src = knn_calibrate(reg_t_w.detach(), rows, u_feat, l_feat, l_y,
                                    a.knn_k, a.knn_temp, a.knn_calib, a.knn_conf_floor)
            pseudo_raw[rows.cpu()] = src.detach().cpu()
            if step % a.rda_every == 0:
                pseudo_refine = rda_align(pseudo_raw, sorted_lb)
            target = pseudo_refine[rows.cpu()].to(device)
        elif a.reg_target == 'rpl':
            target = knn_calibrate(reg_t_w.detach(), rows, u_feat, l_feat, l_y,
                                   a.knn_k, a.knn_temp, a.knn_calib, a.knn_conf_floor)
        else:
            target = reg_t_w.detach()

        warm = min(step / max(1.0, a.unsup_warmup * a.steps), 1.0)

        # GaussianMatch-style multi-view consistency filter
        if a.filter == 'gcf':
            views = []
            for _ in range(a.gcf_views):
                rv, _ = model(**ulb_tok_w)
                views.append(rv)
            views = torch.stack(views)
            reg_weak = views.mean(0)
            sigma = views.std(0, unbiased=False)
            weight = torch.exp(-sigma / a.gcf_tau)
        elif a.filter == 'agds':
            views = []
            for _ in range(a.agds_views):
                rv, _ = model(**mask_tokens(ulb_tok_w, tok, a.mask_prob))
                views.append(rv)
            views = torch.stack(views)                 # [K,B,3]
            reg_weak = views.mean(0)
            sigma = views.std(0, unbiased=False)       # [B,3]
            cur = sigma.mean(0).detach()
            if scale_ema is None:
                scale_ema = cur
            else:
                scale_ema = a.agds_ema * scale_ema + (1.0 - a.agds_ema) * cur
            lam = a.agds_lambda0 + (a.agds_lambda1 - a.agds_lambda0) * min(
                step / max(1.0, a.agds_warm * a.steps), 1.0)
            ratio = sigma / (lam * scale_ema.clamp_min(1e-6))
            weight = torch.exp(-0.5 * ratio ** 2) if a.agds_soft else (ratio <= 1.0).float()
        else:
            reg_weak, _ = model(**ulb_tok_w)
            weight = None

        unsup = ((reg_weak - target).abs() * (weight if weight is not None else 1.0)).mean()
        arc_unsup = torch.zeros((), device=device)
        if a.arc:
            for d in range(3):
                tp = pair_logits(arc_t_w, d)
                prob = torch.softmax(tp, dim=-1)
                conf, pseudo = prob.max(-1)
                mask = conf > a.p_cutoff
                if mask.sum() == 0:
                    continue
                sp = pair_logits(arc_s, d)
                onehot = F.one_hot(pseudo, 2).to(sp.dtype)
                ce = -(onehot * F.log_softmax(sp, -1)).sum(-1)
                arc_unsup = arc_unsup + (ce * mask).sum() / mask.sum().clamp_min(1.0)
            arc_unsup = arc_unsup / 3.0
            arc_loss = arc_sup + a.arc_ulb_ratio * arc_unsup
        else:
            arc_loss = torch.zeros((), device=device)

        total = loss + warm * a.ulb_loss_ratio * unsup + a.arc_loss_ratio * arc_loss
        if not torch.isfinite(total):
            raise RuntimeError('Nonfinite loss')
        total.backward()
        if a.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad_norm)
        optim.step()
        ema_step(teacher, model, a.ema_decay)

        if step % a.eval_every == 0 or step == a.steps:
            evaluate_and_save(step)

    pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
    print(f'BEST best_step={best_step} best_dev_MAE={best:.6f}')
    print('OUTPUT:', out)


if __name__ == '__main__':
    main()
