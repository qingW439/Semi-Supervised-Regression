#!/usr/bin/env python3
"""Hetero-VAD online EMA Teacher mechanism ablation (not exact NeurIPS 2025 reproduction).

Runs against qingW439/Semi-Supervised-Regression, branch emobank-vad.
Modes:
    control : labeled L1 only
    h0      : labeled L1 + ordinary continuous pseudo-label MSE
    h1      : labeled L1 + uncertainty-weighted pseudo-label regression,
              with one-step differentiable HEAD-ONLY bi-level meta update.

Safety: only TRAIN labeled/unlabeled files and DEV gold are ever opened; no TEST.
No pseudo-label calibration fitted on DEV; no leakage of unlabeled target columns.

Key adaptation restrictions:
- An EMA Teacher follows the Student after each optimizer update, rather than remaining frozen.
- EMA is our controlled adaptation, not the exact same-model online Teacher used in the paper.
- "Strong view" is training-time dropout on the SAME text, not token deletion.
- Three-dimensional (V,A,D) uncertainty, not scalar.
- The meta outer objective has a small L2 penalty on log variances for
  stability, rather than claiming exact equivalence to the paper's code.
- Same total gradient update budget; labeled L1 by default to match the E1 checkpoint.
- Optional supervised MSE for auditing the original pilot; not the primary run.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (
    GoldData, TextCollator, UnlabeledData, read_gold, read_unlabeled,
)
from semilearn.nets.bert.bert_vad import BertVADRegressor
from scripts.emobank.train_emobank_supervised import metric_report, set_seed

DIMS = ("V", "A", "D")
DEFAULT_DATA = REPO.parent / "data" / "emobank_processed"
DEFAULT_MODEL = "google/bert_uncased_L-4_H-512_A-8"

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


class UncertaintyLearner(nn.Module):
    """Predict three log-variances from [student-teacher difference, student prediction]."""
    def __init__(self, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, width), nn.ReLU(), nn.Linear(width, 3))
        # Start with z=0: all three dimensions initially receive weight 1.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, student_pred: torch.Tensor, teacher_pred: torch.Tensor) -> torch.Tensor:
        x = torch.cat([(student_pred - teacher_pred).detach(), student_pred.detach()], dim=-1)
        return self.net(x).clamp(-2.0, 2.0)


def supervised_criterion(pred, gold, args):
    """Matched labeled loss for control/H0/H1, including meta inner/outer labeled gold."""
    if args.supervised_loss == 'l1':
        return F.l1_loss(pred, gold)
    if args.supervised_loss == 'mse':
        return F.mse_loss(pred, gold)
    raise ValueError(f'Unexpected supervised loss: {args.supervised_loss}')


def params():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['control', 'h0', 'h1'], required=True)
    p.add_argument('--budget', type=int, choices=[250, 500, 1000], default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    p.add_argument('--pretrain-path', default=DEFAULT_MODEL)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--unlabeled-batch-size', type=int, default=16)
    p.add_argument('--eval-batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--steps-per-epoch', type=int, default=64)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--warmup-epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--unc-lr', type=float, default=1e-3)
    p.add_argument('--meta-inner-lr', type=float, default=0.05)
    p.add_argument('--meta-every', type=int, default=1)
    p.add_argument('--meta-reg', type=float, default=1e-3)
    p.add_argument('--unsup-weight', type=float, default=0.2)
    p.add_argument('--ema-decay', type=float, default=0.99,
                   help='EMA coefficient for Teacher updates after each Student optimizer step')
    p.add_argument('--supervised-loss', choices=['l1','mse'], default='l1')
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--clip-grad-norm', type=float, default=1.0)
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.batch_size < 4 or a.batch_size % 2 or a.unlabeled_batch_size < 1:
        p.error('batch-size must be an even integer >=4; unlabeled-batch-size >=1')
    if any(getattr(a,k) <= 0 for k in ('epochs', 'steps_per_epoch', 'patience', 'meta_every')):
        p.error('epochs, steps-per-epoch, patience, meta-every must be positive')
    if a.warmup_epochs < 0 or a.warmup_epochs >= a.epochs:
        p.error('warmup-epochs must be >=0 and less than epochs')
    if not (0 <= a.ema_decay < 1):
        p.error('ema-decay must be in [0,1)')
    if any(getattr(a,k)<0 for k in ('unsup_weight','meta_reg')):
        p.error('unsup-weight and meta-reg must be nonnegative')
    if a.out is None:
        a.out = REPO / 'saved_models' / 'emobank' / f'hetero_pilot_{a.mode}_lb{a.budget}_s{a.seed}'
    if a.checkpoint is None:
        a.checkpoint = REPO / 'saved_models' / 'emobank' / f'supervised_lb{a.budget}_s{a.seed}' / 'model_best.pth'
    return a


def unlabeled_collator(tokenizer, max_length):
    def collate(rows):
        texts, ids = zip(*rows)
        tokens = tokenizer(list(texts), truncation=True, max_length=max_length,
                           padding=True, return_tensors='pt')
        return tokens, list(ids)
    return collate


def batch_to_device(tokens, device):
    return {k: v.to(device, non_blocking=True) for k, v in tokens.items()}


class CyclicLoader:
    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


def one_step_meta(learner, head, features_lb, y_lb, features_u, pred_u,
                  pseudo, args):
    """Differentiate one HEAD-only update through the uncertainty network.

    First half of labeled batch is inner gold; distinct second half is outer gold.
    Encoder features are detached for meta efficiency; encoder is updated normally
    during the main student step.
    """
    half = y_lb.shape[0] // 2
    z = learner(pred_u.detach(), pseudo.detach())
    w = torch.exp(-z)
    original = dict(head.named_parameters())
    inner_pred = functional_call(head, original, (features_lb[:half].detach(),))
    inner_u = functional_call(head, original, (features_u.detach(),))
    inner_loss = supervised_criterion(inner_pred, y_lb[:half], args) + args.unsup_weight * (
        w * (inner_u - pseudo.detach()).square()
    ).mean()
    grads = torch.autograd.grad(inner_loss, tuple(original.values()), create_graph=True)
    updated = {name: value - args.meta_inner_lr * grad
               for (name, value), grad in zip(original.items(), grads)}
    outer_pred = functional_call(head, updated, (features_lb[half:].detach(),))
    outer = supervised_criterion(outer_pred, y_lb[half:], args) + args.meta_reg * z.square().mean()
    return outer, z, w


def evaluate(model, dev_loader, device):
    model.eval()
    gold, pred, ids = [], [], []
    with torch.inference_mode():
        for tokens, y, batch_ids in dev_loader:
            outputs = model(**batch_to_device(tokens, device))
            gold.append(y.numpy())
            pred.append(outputs.cpu().numpy())
            ids.extend(batch_ids)
    metrics = metric_report(np.concatenate(gold), np.concatenate(pred))
    ccc_vals = [metrics[d]['ccc'] for d in DIMS]
    metrics['macro_ccc'] = float(np.mean([x for x in ccc_vals if x is not None]))
    return metrics, np.concatenate(gold), np.concatenate(pred), ids


def main():
    a = params()
    set_seed(a.seed)
    device = torch.device('cuda' if (a.device == 'auto' and torch.cuda.is_available()) else
                          'cpu' if a.device == 'auto' else a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    if not a.checkpoint.is_file():
        raise FileNotFoundError(f'E1 checkpoint not found: {a.checkpoint}')
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing overwrite of nonempty output directory: {out}')
    lb = read_gold(a.data_root / f'lb{a.budget}_s{a.seed}' / 'train_lb.parquet', expected_n=a.budget)
    ulb = read_unlabeled(a.data_root / f'lb{a.budget}_s{a.seed}' / 'train_ulb.parquet', labeled_ids=lb['id'])
    dev = read_gold(a.data_root / 'dev_gold.parquet', expected_n=1000)
    if len(ulb) != 8062-a.budget or set(dev['id']) & (set(lb['id']) | set(ulb['id'])):
        raise ValueError('Split count or ID overlap validation failed')
    out.mkdir(parents=True, exist_ok=True)
    run_cfg = {k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    run_cfg.update({'adaptation_not_exact_paper_reproduction':True,
                    'teacher_type':'EMA updated after each optimizer step',
                    'teacher_dropout_eval_mode':True,
                    'diagnostic_purpose':'isolate learned unlabeled weights under E1-matched labeled loss',
                    'frozen_e1_teacher':False,
                    'augmentation':'student train-mode dropout; unchanged text',
                    'evaluation':'DEV only; TEST never opened'})
    (out/'run_config.json').write_text(json.dumps(run_cfg, ensure_ascii=False,indent=2)+'\n')
    tok = AutoTokenizer.from_pretrained(a.pretrain_path, use_fast=True, local_files_only=a.offline)
    gold_loader = DataLoader(GoldData(lb), batch_size=a.batch_size, shuffle=True,
                             drop_last=True, collate_fn=TextCollator(tok, a.max_length),
                             generator=torch.Generator().manual_seed(a.seed+101))
    ulb_loader = DataLoader(UnlabeledData(ulb), batch_size=a.unlabeled_batch_size,
                            shuffle=True, drop_last=True,
                            collate_fn=unlabeled_collator(tok,a.max_length),
                            generator=torch.Generator().manual_seed(a.seed+202))
    dev_loader = DataLoader(GoldData(dev), batch_size=a.eval_batch_size, shuffle=False,
                            collate_fn=TextCollator(tok, a.max_length))
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint,dict) or 'model' not in checkpoint:
        raise ValueError('Expected E1 checkpoint with a model state dict')
    student = BertVADRegressor(a.pretrain_path, dropout=a.dropout, offline=a.offline).to(device)
    student.load_state_dict(checkpoint['model'], strict=True)
    teacher = None
    if a.mode != 'control':
        # Deep-copy after strict checkpoint loading: no extra random initialization.
        teacher = copy.deepcopy(student).to(device)
        teacher.eval()
        for par in teacher.parameters():
            par.requires_grad_(False)
    # Restore RNG after initializing only-H1 parameters to keep the student
    # dropout stream matched with H0 as closely as practical.
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    learner = UncertaintyLearner().to(device) if a.mode == 'h1' else None
    torch.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)
    if teacher is not None:
        # Keep stateful Teacher entirely frozen for autograd. Its parameters will
        # nevertheless be updated explicitly by exponential moving average.
        teacher.eval()

    opt_student = torch.optim.AdamW(student.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    opt_unc = torch.optim.Adam(learner.parameters(),lr=a.unc_lr) if learner else None
    label_iter = CyclicLoader(gold_loader)
    ulb_iter = CyclicLoader(ulb_loader)
    best_mae, stale, hist, total_steps, total_ulb, meta_steps = math.inf, 0, [], 0, 0, 0
    initial_metrics, initial_y, initial_p, initial_ids = evaluate(student, dev_loader, device)
    # Allow the best model to remain the E1 initialization if all fine-tuning degrades it.
    best_mae = initial_metrics['macro_mae_original_scale']
    best_epoch = 0
    torch.save({'model':student.state_dict(),'epoch':0,'dev_metrics':initial_metrics,
                'run_config':run_cfg},out/'model_best.pth')
    (out/'dev_metrics_best.json').write_text(
        json.dumps({'epoch':0,**initial_metrics},indent=2)+'\n')
    initial_df=pd.DataFrame({'id':initial_ids})
    for i,d in enumerate(DIMS):
        initial_df[f'gold_{d}']=initial_y[:,i]*4+1
        initial_df[f'pred_{d}']=initial_p[:,i]*4+1
    initial_df.to_csv(out/'dev_predictions_best.csv',index=False)
    if learner is not None:
        torch.save({'model':learner.state_dict(),'epoch':0},out/'uncertainty_best.pth')
    print(f'CONFIG: mode={a.mode}, device={device}, gold={len(lb)}, ulb={len(ulb)}, '
          f'steps/epoch={a.steps_per_epoch}, epochs={a.epochs}, '
          f'ema_teacher={teacher is not None}, ema_decay={a.ema_decay}, '
          f'labeled_loss={a.supervised_loss}',flush=True)
    print(f'INITIAL: MAE={initial_metrics["macro_mae_original_scale"]:.5f} '
          f'CCC={initial_metrics["macro_ccc"]:.5f}',flush=True)
    seen_unlabeled_ids=set()
    for epoch in range(1,a.epochs+1):
        student.train()
        total_supervised = total_unsup = total_meta = total_logvar = total_weight = total_grad = 0.0
        meta_count = 0
        active_batches = 0
        weight_dim_sum = torch.zeros(3)
        logvar_dim_sum = torch.zeros(3)
        for step in range(a.steps_per_epoch):
            tokens_l, y_l, _ = label_iter.next()
            tokens_l=batch_to_device(tokens_l,device)
            y_l=y_l.to(device)
            z_l=student.encode(**tokens_l)
            p_l=student.regress(z_l)
            sup=supervised_criterion(p_l,y_l,a)
            active=a.mode!='control' and epoch>a.warmup_epochs and a.unsup_weight>0
            if active:
                tokens_u,batch_ulb_ids=ulb_iter.next()
                seen_unlabeled_ids.update(batch_ulb_ids)
                tokens_u=batch_to_device(tokens_u,device)
                with torch.no_grad():
                    pseudo=teacher(**tokens_u).detach()
                z_u=student.encode(**tokens_u)
                p_u=student.regress(z_u)
                if a.mode=='h1' and total_steps%a.meta_every==0:
                    opt_unc.zero_grad(set_to_none=True)
                    outer,s,w=one_step_meta(learner,student.head,z_l,y_l,z_u,p_u,pseudo,a)
                    outer.backward()
                    grad_norm=float(torch.nn.utils.clip_grad_norm_(learner.parameters(),a.clip_grad_norm))
                    if not math.isfinite(grad_norm):
                        raise RuntimeError('Nonfinite meta gradient norm')
                    opt_unc.step()
                    total_meta+=float(outer.detach())
                    total_grad+=grad_norm
                    meta_count+=1
                    meta_steps+=1
                if a.mode=='h1':
                    with torch.no_grad():
                        s=learner(p_u.detach(),pseudo.detach())
                        w=torch.exp(-s)
                    # logvar penalty is constant w.r.t student because s is detached.
                    unsup=(w*(p_u-pseudo).square()+s).mean()
                    total_logvar+=float(s.mean().detach())
                    total_weight+=float(w.mean().detach())
                    weight_dim_sum += w.detach().mean(0).cpu()
                    logvar_dim_sum += s.detach().mean(0).cpu()
                else:
                    unsup=F.mse_loss(p_u,pseudo)
                    total_weight+=1.0
                    weight_dim_sum += torch.ones(3)
                active_batches += 1
                total_unsup+=float(unsup.detach())
                total_ulb+=len(pseudo)
            else:
                unsup=torch.zeros((),device=device)
            loss=sup+a.unsup_weight*unsup if active else sup
            if not torch.isfinite(loss).all():
                raise RuntimeError(f'Nonfinite loss at epoch={epoch} step={step}')
            opt_student.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(),a.clip_grad_norm)
            opt_student.step()
            if teacher is not None:
                # Online Teacher updated only AFTER the Student's real step,
                # never through the meta unroll or an unlabeled target gradient.
                with torch.no_grad():
                    for teacher_p, student_p in zip(teacher.parameters(),student.parameters()):
                        teacher_p.mul_(a.ema_decay).add_(student_p,alpha=1-a.ema_decay)
                    for teacher_b, student_b in zip(teacher.buffers(),student.buffers()):
                        if torch.is_floating_point(teacher_b):
                            teacher_b.mul_(a.ema_decay).add_(student_b,alpha=1-a.ema_decay)
                        else:
                            teacher_b.copy_(student_b)
            total_supervised+=float(sup.detach())
            total_steps+=1
        report,y,p,ids=evaluate(student,dev_loader,device)
        mae=report['macro_mae_original_scale']
        head_distance = 0.0
        if teacher is not None:
            with torch.no_grad():
                head_distance = float(torch.sqrt(sum(
                    (x-y).square().sum() for x,y in
                    zip(student.head.parameters(),teacher.head.parameters())
                )).item())
        row={'epoch':epoch,'total_steps':total_steps,'macro_mae':mae,
             'student_teacher_head_distance':head_distance,
             'macro_ccc':report['macro_ccc'],'sup_loss':total_supervised/a.steps_per_epoch,
             'unsup_loss':total_unsup/a.steps_per_epoch,
             'meta_loss':total_meta/max(1,meta_count),
             'meta_grad_norm':total_grad/max(1,meta_count),
             'meta_steps':meta_count,'mean_logvar':total_logvar/max(1,a.steps_per_epoch),
             'mean_weight':total_weight/max(1,active_batches),
             'active_unlabeled_batches':active_batches,
             'unique_unlabeled_seen':len(seen_unlabeled_ids),
             **{f'{d}_weight':float(weight_dim_sum[k]/max(1,active_batches)) for k,d in enumerate(DIMS)},
             **{f'{d}_logvar':float(logvar_dim_sum[k]/max(1,active_batches)) for k,d in enumerate(DIMS)},
             **{f'{d}_mae':report[d]['mae'] for d in DIMS},
             **{f'{d}_ccc':report[d]['ccc'] for d in DIMS}}
        hist.append(row)
        print(f'EPOCH {epoch:02d}: DEV MAE={mae:.5f} CCC={report["macro_ccc"]:.5f} '
              f'sup={row["sup_loss"]:.5f} unsup={row["unsup_loss"]:.5f} '
              f'meta_grad={row["meta_grad_norm"]:.8f} '
              f'weight={row["mean_weight"]:.4f} '
          f'head_dist={row["student_teacher_head_distance"]:.5f}',flush=True)
        if mae<best_mae-1e-4:
            best_mae,stale=mae,0
            best_epoch=epoch
            torch.save({'model':student.state_dict(),'epoch':epoch,'dev_metrics':report,
                        'run_config':run_cfg},out/'model_best.pth')
            (out/'dev_metrics_best.json').write_text(json.dumps({'epoch':epoch,**report},indent=2)+'\n')
            df=pd.DataFrame({'id':ids})
            for i,d in enumerate(DIMS):
                df[f'gold_{d}']=y[:,i]*4+1
                df[f'pred_{d}']=p[:,i]*4+1
            df.to_csv(out/'dev_predictions_best.csv',index=False)
            if learner is not None:
                torch.save({'model':learner.state_dict(),'epoch':epoch},out/'uncertainty_best.pth')
        else:
            stale+=1
        pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False)
        if stale>=a.patience:
            print(f'EARLY STOP: epoch={epoch}, patience={a.patience}',flush=True)
            break
    summary={'mode':a.mode,'best_dev_mae':best_mae,'best_epoch':best_epoch,
             'initial_dev_mae':initial_metrics['macro_mae_original_scale'],
             'total_steps':total_steps,
             'unlabeled_items_processed':total_ulb,'unlabeled_pool_size':len(ulb),
             'unique_unlabeled_seen':len(seen_unlabeled_ids),
             'meta_steps':meta_steps,'ema_decay':a.ema_decay,
             'best_checkpoint':str(out/'model_best.pth'),
             'test_accessed':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'FINISHED mode={a.mode}; {json.dumps(summary)}',flush=True)


if __name__=='__main__':
    main()
