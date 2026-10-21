#!/usr/bin/env python3
"""P2 controlled supervised-continuation versus Mean-Teacher experiments.

Use after P1 RoBERTa-VAD best checkpoints are frozen. Runs only EmoBank
train_lb, train_ulb (text only) and dev_gold. NEVER opens test_gold.

Both arms load the same P1 checkpoint, use the same labeled minibatch order,
optimizer, LR and number of supervised steps. The teacher arm additionally
trains via EMA pseudo-targets on unlabeled text with dropout-only augmentation.
No OOF, confidence filtering, label correction, or ranking is used here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    DIMENSIONS, GoldData, TextCollator, UnlabeledData, read_gold, read_unlabeled,
)
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import (  # noqa: E402
    RobertaVADRegressor, load_checked_splits, predict, set_seed, sha256_file, git_commit,
)

DATA_ROOT = REPO.parent / 'data' / 'emobank_processed'
MODEL_DIR = REPO.parent / 'pretrained' / 'roberta-base'


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('control', 'mean_teacher'), required=True)
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
        a.out = REPO / 'saved_models' / 'vad_research' / f'p2_{a.mode}_lb{a.budget}_s{a.seed}'
    for k in ('max_length', 'batch_size', 'ulb_batch_size', 'eval_batch_size', 'steps', 'eval_every'):
        if getattr(a,k) <= 0:
            p.error(f'--{k.replace("_", "-")} must be positive')
    if a.lr <= 0 or a.weight_decay < 0 or a.clip_grad_norm < 0 or a.lambda_u < 0:
        p.error('Invalid optimizer or loss settings')
    if not 0.0 < a.ema_decay < 1.0:
        p.error('--ema-decay must be between 0 and 1')
    return a


class UnlabeledCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts, ids = zip(*batch)
        return self.tokenizer(list(texts), padding=True, truncation=True,
                              max_length=self.max_length, return_tensors='pt'), list(ids)


def forever(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def ema_step(teacher, student, alpha):
    """EMA parameter update; copy non-floating buffers exactly."""
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.lerp_(s.detach(), 1.0 - alpha)
    for t, s in zip(teacher.buffers(), student.buffers()):
        if t.is_floating_point():
            t.lerp_(s.detach(), 1.0 - alpha)
        else:
            t.copy_(s)


def dev_evaluate(student, dev_dl, device):
    y, pred, ids = predict(student, dev_dl, device)
    m = metric_report(y, pred)
    stats = {}
    for idx, d in enumerate(DIMENSIONS):
        stats[d] = {
            'gold_mean': float(y[:, idx].mean() * 4 + 1),
            'pred_mean': float(pred[:, idx].mean() * 4 + 1),
            'gold_std': float(y[:, idx].std() * 4),
            'pred_std': float(pred[:, idx].std() * 4),
        }
    return m, stats, ids, y, pred


def main():
    a = cli()
    set_seed(a.seed)
    lb, dev, ulb, paths = load_checked_splits(a)
    if len(lb) != a.budget or len(ulb) != 8062-a.budget or len(dev) != 1000:
        raise ValueError('Unexpected split counts')
    ckpt_path = a.p1_dir / 'model_best.pth'
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'P1 checkpoint not found: {ckpt_path}')
    if not a.pretrain_path.is_dir():
        raise FileNotFoundError(f'Pretrained RoBERTa missing: {a.pretrain_path}')
    p1_config_path = a.p1_dir / 'run_config.json'
    p1 = json.loads(p1_config_path.read_text(encoding='utf-8'))
    for k in ('budget', 'seed', 'max_length'):
        if int(p1[k]) != int(getattr(a,k)):
            raise ValueError(f'P1 checkpoint config mismatch for {k}: {p1[k]} != {getattr(a,k)}')
    if abs(float(p1.get('dropout',0.1))-a.dropout)>1e-12:
        raise ValueError('P1 dropout and P2 dropout mismatch')
    if Path(p1['model']).resolve() != a.pretrain_path.resolve():
        raise ValueError(f'P1 and P2 use different pretrained model paths: {p1["model"]} vs {a.pretrain_path}')
    hashes = {k: sha256_file(v) for k,v in paths.items()}
    if 'split_sha256' in p1 and hashes != p1['split_sha256']:
        raise ValueError('P2 data hashes differ from frozen P1!')
    print(f'mode={a.mode} seed={a.seed} labeled={len(lb)} unlabeled={len(ulb)} dev={len(dev)}')
    print('P1:', ckpt_path, 'sha256:', sha256_file(ckpt_path))
    print('split_sha256:', json.dumps(hashes,sort_keys=True))
    if a.check_only:
        print('PASS: P1 config, checkpoint path, and training/dev splits verified. NO TEST loaded or files written.')
        return

    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {out}')
    if a.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    device = torch.device('cpu' if a.device=='cpu' or not torch.cuda.is_available() else 'cuda')
    model_name = str(a.pretrain_path.resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=True)
    lb_dl = DataLoader(GoldData(lb), batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                       collate_fn=TextCollator(tokenizer,a.max_length),
                       generator=torch.Generator().manual_seed(17_000+a.seed))
    dev_dl = DataLoader(GoldData(dev), batch_size=a.eval_batch_size, shuffle=False, num_workers=a.num_workers,
                        collate_fn=TextCollator(tokenizer,a.max_length))
    ulb_dl = None
    if a.mode=='mean_teacher':
        ulb_dl = DataLoader(UnlabeledData(ulb), batch_size=a.ulb_batch_size, shuffle=True,
                           num_workers=a.num_workers, collate_fn=UnlabeledCollator(tokenizer,a.max_length),
                           generator=torch.Generator().manual_seed(19_000+a.seed))

    student = RobertaVADRegressor(model_name, dropout=a.dropout, offline=True).to(device)
    # The checkpoint was generated locally in P1, hence trusted. Never load a random external .pth file.
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    student.load_state_dict(ckpt['model'], strict=True)
    del ckpt
    teacher = None
    if a.mode=='mean_teacher':
        teacher = copy.deepcopy(student).to(device)
        teacher.requires_grad_(False)
        teacher.eval()
    optim = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    out.mkdir(parents=True,exist_ok=True)
    conf = {k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    p1_best_info = json.loads((a.p1_dir / 'dev_metrics_best.json').read_text(encoding='utf-8'))
    conf.update({'git_commit':git_commit(),'device':str(device),'p1_checkpoint_sha256':sha256_file(ckpt_path),
                 'split_sha256':hashes,'p1_selected_epoch':p1_best_info.get('epoch'),
                 'training_design':'same initialized P1 student and labeled steps; MT adds EMA unlabeled L1 loss',
                 'weak_strong':'same original text; teacher eval, student train dropout only',
                 'model_selection':'lowest dev macro MAE, including step 0'})
    (out/'run_config.json').write_text(json.dumps(conf,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    labeled_iter=forever(lb_dl)
    unlabeled_iter=forever(ulb_dl) if ulb_dl is not None else None
    visited_ulb=set()
    best=float('inf')
    best_step=-1
    history=[]
    sum_sup=sum_unsup=0.0

    def evaluate_and_save(step):
        nonlocal best,best_step
        m,stats,ids,y,p=dev_evaluate(student,dev_dl,device)
        mae=m['macro_mae_original_scale']
        ccc=float(np.mean([m[d]['ccc'] for d in DIMENSIONS]))
        row={'step':step,'macro_mae':mae,'macro_ccc':ccc,
             'sup_l1_norm_avg':sum_sup/max(1,step),
             'unsup_l1_norm_avg':sum_unsup/max(1,step),
             'unique_ulb_used':len(visited_ulb)}
        for d in DIMENSIONS:
            row[f'{d}_mae']=m[d]['mae'];row[f'{d}_ccc']=m[d]['ccc']
            row[f'{d}_pred_std_ratio']=stats[d]['pred_std']/max(stats[d]['gold_std'],1e-12)
        history.append(row)
        print(f'step={step:03d} dev_MAE={mae:.6f} dev_CCC={ccc:.5f} '
              f'sup_L1={row["sup_l1_norm_avg"]:.5f} unsup_L1={row["unsup_l1_norm_avg"]:.5f} '
              f'unique_ulb={len(visited_ulb)} '
              + ' '.join(f'{d}_CCC={m[d]["ccc"]:.3f}' for d in DIMENSIONS), flush=True)
        if mae < best-1e-8:
            best=mae; best_step=step
            torch.save({'model':student.state_dict(),'step':step,'dev_metrics':m,'run_config':conf},out/'model_best.pth')
            (out/'dev_metrics_best.json').write_text(
                json.dumps({'step':step,'metrics':m,'stats':stats},indent=2)+'\n',encoding='utf-8')
            pred_frame=pd.DataFrame({'id':ids})
            for idx,d in enumerate(DIMENSIONS):
                pred_frame[f'gold_{d}']=y[:,idx]*4+1
                pred_frame[f'pred_{d}']=p[:,idx]*4+1
            pred_frame.to_csv(out/'dev_predictions_best.csv',index=False)

    evaluate_and_save(0)
    for step in range(1,a.steps+1):
        student.train()
        lb_tok,lb_y,_=next(labeled_iter)
        lb_tok={k:v.to(device) for k,v in lb_tok.items()}
        lb_y=lb_y.to(device)
        optim.zero_grad(set_to_none=True)
        pred=student(**lb_tok)
        if pred.shape!=lb_y.shape:
            raise RuntimeError('Labeled output has unexpected shape')
        sup_loss=F.l1_loss(pred,lb_y)
        total=sup_loss
        unsup_value=0.0
        if a.mode=='mean_teacher':
            ulb_tok,ulb_ids=next(unlabeled_iter)
            visited_ulb.update(ulb_ids)
            ulb_tok={k:v.to(device) for k,v in ulb_tok.items()}
            teacher.eval()
            with torch.no_grad():
                pseudo=teacher(**ulb_tok).detach()
            pred_ulb=student(**ulb_tok)
            unsup_loss=F.l1_loss(pred_ulb,pseudo)
            total=total+a.lambda_u*unsup_loss
            unsup_value=float(unsup_loss.detach().item())
        if not torch.isfinite(total):
            raise RuntimeError('Nonfinite loss')
        total.backward()
        if a.clip_grad_norm>0:
            torch.nn.utils.clip_grad_norm_(student.parameters(),a.clip_grad_norm)
        optim.step()
        if teacher is not None:
            ema_step(teacher,student,a.ema_decay)
        sum_sup+=float(sup_loss.detach().item())
        sum_unsup+=unsup_value
        if step%a.eval_every==0 or step==a.steps:
            evaluate_and_save(step)
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    print(f'BEST mode={a.mode} seed={a.seed} best_step={best_step} '
          f'best_dev_MAE={best:.6f} unique_unlabeled_used={len(visited_ulb)}',flush=True)
    print('NO TEST file was loaded. P1 checkpoint was not modified.')
    print('OUTPUT:',out)

if __name__=='__main__':
    main()
