#!/usr/bin/env python3
"""M1: HPL with optional GGR VLR and TRAIN-only VAD utility control.

The hpl_original path preserves the M0 A1 loss and update order. No TEST path
is loaded. Every run writes a new directory and retains the P1 initialization.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import DIMENSIONS, GoldData, TextCollator, UnlabeledData
from scripts.vad_research.train_roberta_vad_p2 import UnlabeledCollator, dev_evaluate, forever
from scripts.vad_research.train_roberta_vad_supervised import (
    RobertaVADRegressor, git_commit, load_checked_splits, set_seed, sha256_file,
)
from scripts.vad_research.m0.hpl_core import (
    VADUncertainty, bilevel_head_step, weighted_pseudo_mse,
)
from scripts.vad_research.m1.dimensional_utility import (
    disjoint_meta_batch, per_dimension_unlabeled, virtual_utility_matrix,
)
from scripts.vad_research.m1.ggr_vlr import merge_head_gradients
from scripts.vad_research.m1.utility_controller import UtilityController

MODES = ('hpl_original', 'hpl_ggr', 'hpl_diagonal', 'hpl_crossdim', 'hpl_crossdim_ggr')


def gradient_statistics(first, second):
    """Finite head gradient norms and cosine; missing gradients count as zero."""
    if len(first) != len(second):
        raise ValueError('Gradient sequences must have matching lengths')
    first_sq = sum(float(g.detach().square().sum()) for g in first if g is not None)
    second_sq = sum(float(g.detach().square().sum()) for g in second if g is not None)
    dot = sum(float((a.detach()*b.detach()).sum()) for a, b in zip(first, second)
              if a is not None and b is not None)
    if not all(math.isfinite(v) for v in (first_sq, second_sq, dot)):
        raise FloatingPointError('Nonfinite gradient diagnostic')
    first_norm, second_norm = math.sqrt(first_sq), math.sqrt(second_sq)
    cosine = dot/(first_norm*second_norm) if first_norm and second_norm else 0.0
    return first_norm, second_norm, max(-1.0, min(1.0, cosine))


def sample_statistics(values):
    """Population statistics for a finite per-sample diagnostic vector."""
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError('Expected nonempty finite 1D values')
    values = values.detach()
    p10, p50, p90 = torch.quantile(values, values.new_tensor((0.1, 0.5, 0.9)))
    return {'mean': float(values.mean()), 'std': float(values.std(unbiased=False)),
            'p10': float(p10), 'p50': float(p50), 'p90': float(p90)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--budget', type=int, choices=(500,), default=500)
    parser.add_argument('--seed', type=int, choices=(0,), default=0)
    parser.add_argument('--data-root', type=Path, default=REPO.parent/'data'/'emobank_processed')
    parser.add_argument('--pretrain-path', type=Path, default=REPO.parent/'pretrained'/'roberta-base')
    parser.add_argument('--p1-dir', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--eval-every', type=int, default=32)
    parser.add_argument('--meta-every', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--ulb-batch-size', type=int, default=64)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--unc-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--lambda-u', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--clip-grad-norm', type=float, default=1.0)
    args = parser.parse_args()
    if args.p1_dir is None:
        args.p1_dir = REPO/'saved_models'/'vad_research'/f'p1_roberta_lb{args.budget}_s{args.seed}'
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        args.out = REPO/'saved_models'/'vad_research'/'m1'/f'{args.mode}_lb{args.budget}_s{args.seed}_{stamp}'
    for name in ('steps', 'eval_every', 'meta_every', 'batch_size', 'ulb_batch_size',
                 'eval_batch_size', 'max_length', 'lr', 'unc_lr'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if min(args.weight_decay, args.lambda_u, args.lambda2, args.clip_grad_norm) < 0:
        parser.error('loss weights, decay and clipping must be nonnegative')
    return args


def pooled_features(model, tokens):
    states = model.encoder(**tokens, return_dict=True).last_hidden_state
    mask = tokens['attention_mask'].unsqueeze(-1).to(dtype=states.dtype)
    pooled = (states*mask).sum(1)/mask.sum(1).clamp_min(1)
    return model.dropout(pooled)


def move(tokens, device):
    return {key: value.to(device) for key, value in tokens.items()}


def validated_inputs(args):
    lb, dev, ulb, paths = load_checked_splits(args)
    if (len(lb), len(ulb), len(dev)) != (500, 7562, 1000):
        raise ValueError('M1 requires the 500/7562/1000 EmoBank protocol')
    hashes = {key: sha256_file(value) for key, value in paths.items()}
    config_path = args.p1_dir/'run_config.json'
    checkpoint = args.p1_dir/'model_best.pth'
    if not config_path.is_file() or not checkpoint.is_file() or not args.pretrain_path.is_dir():
        raise FileNotFoundError('P1 config/checkpoint or pretrained RoBERTa missing')
    p1 = json.loads(config_path.read_text(encoding='utf-8'))
    if any(int(p1[key]) != int(getattr(args, key)) for key in ('budget', 'seed', 'max_length')):
        raise ValueError('P1 configuration mismatch')
    if abs(float(p1.get('dropout', 0.1))-args.dropout)>1e-12:
        raise ValueError('P1 dropout mismatch')
    if Path(p1['model']).resolve()!=args.pretrain_path.resolve():
        raise ValueError('P1 pretrained model mismatch')
    if p1.get('split_sha256')!=hashes:
        raise ValueError('P1 data hashes differ')
    return lb, dev, ulb, hashes, checkpoint


def main():
    args = parse_args()
    set_seed(args.seed)
    lb, dev, ulb, hashes, checkpoint = validated_inputs(args)
    print(json.dumps({'mode': args.mode, 'seed': args.seed, 'labeled': len(lb),
                      'unlabeled': len(ulb), 'dev': len(dev), 'split_sha256': hashes,
                      'p1_checkpoint_sha256': sha256_file(checkpoint)}, sort_keys=True))
    if args.check_only:
        print('PASS: P1 and train/dev inputs verified; no model, TEST or output writes')
        return
    if args.out.exists():
        raise FileExistsError(f'refusing to overwrite output: {args.out}')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.pretrain_path), local_files_only=True, use_fast=True)
    labeled = DataLoader(GoldData(lb), batch_size=args.batch_size, shuffle=True,
                         collate_fn=TextCollator(tokenizer, args.max_length),
                         generator=torch.Generator().manual_seed(17_000+args.seed))
    meta = DataLoader(GoldData(lb), batch_size=args.batch_size, shuffle=True,
                      collate_fn=TextCollator(tokenizer, args.max_length),
                      generator=torch.Generator().manual_seed(23_000+args.seed))
    development = DataLoader(GoldData(dev), batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=TextCollator(tokenizer, args.max_length))
    unlabeled = DataLoader(UnlabeledData(ulb), batch_size=args.ulb_batch_size, shuffle=True,
                           collate_fn=UnlabeledCollator(tokenizer, args.max_length),
                           generator=torch.Generator().manual_seed(19_000+args.seed))
    model = RobertaVADRegressor(str(args.pretrain_path), dropout=args.dropout, offline=True).to(device)
    trusted = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(trusted['model'], strict=True)
    del trusted
    uncertainty = VADUncertainty().to(device)
    # higher==0.2.1 supports differentiable Adam, but not AdamW.
    # A0 and A1 share both optimizers; this differs from the historical P2 runs.
    encoder_opt = torch.optim.Adam(model.encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    head_opt = torch.optim.Adam(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    unc_opt = torch.optim.Adam(uncertainty.parameters(), lr=args.unc_lr, weight_decay=1e-5) if uncertainty else None
    labeled_iter, meta_iter = forever(labeled), forever(meta)
    unlabeled_iter = forever(unlabeled)
    utility_enabled = args.mode in ('hpl_diagonal', 'hpl_crossdim', 'hpl_crossdim_ggr')
    ggr_enabled = args.mode in ('hpl_ggr', 'hpl_crossdim_ggr')
    controller = UtilityController('diagonal' if args.mode == 'hpl_diagonal' else 'crossdim') if utility_enabled else None
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = {key:(str(value) if isinstance(value, Path) else value) for key,value in vars(args).items()}
    config.update({'git_commit': git_commit(), 'p1_checkpoint_sha256': sha256_file(checkpoint),
                   'split_sha256': hashes, 'supervised_loss': 'normalized MSE',
                   'weak_strong': 'same text; eval versus train dropout',
                   'selection': 'minimum Dev Macro-MAE including step 0',
                   'hpl_reference': 'https://github.com/sxq11/Heteroscedastic-Pseudo-Labels/blob/main/sts/main_ours.py',
                   'hpl_reference_blob_sha': '978f26892608e27f62451634dbc0501660c005c6',
                   'ggr_reference': 'https://github.com/JiaheChen2002/GGR/blob/main/semilearn/algorithms/utils/ggr.py',
                   'ggr_reference_blob_sha': '1db09f7e76649816526e246d82b54c5f6ef811c8',
                   'ggr_mode': 'vlr', 'ggr_scope': 'head', 'ggr_surgery_alpha': 1.0,
                   'virtual_update': 'stateless head SGD directional probe at head lr',
                   'utility_momentum': controller.momentum if controller else None})
    (out/'run_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False)+'\n')
    history = []
    diagnostics = []
    utility_rows = []
    best = float('inf')
    used_ids = set()
    sum_sup = sum_unsup = sum_meta = 0.0
    sum_unsup_head_grad = 0.0
    effective_steps = 0
    extra_compute_seconds = 0.0
    sum_logvar = {dim:0.0 for dim in DIMENSIONS}
    sum_weight = {dim:0.0 for dim in DIMENSIONS}
    sum_abs_residual = {dim:0.0 for dim in DIMENSIONS}

    def evaluate(step):
        nonlocal best
        metrics, stats, ids, gold, pred = dev_evaluate(model, development, device)
        mae = metrics['macro_mae_original_scale']
        row = {'step': step, 'macro_mae': mae, 'supervised_mse_avg': sum_sup/max(step,1),
               'unlabeled_weighted_mse_avg': sum_unsup/max(step,1),
               'meta_loss_avg': sum_meta/max(step,1), 'unique_unlabeled': len(used_ids),
               'unsup_head_grad_norm_avg': sum_unsup_head_grad/max(step,1),
               'effective_unlabeled_update_rate': effective_steps/max(step,1),
               'extra_compute_seconds': extra_compute_seconds}
        for dim in DIMENSIONS:
            for metric in ('mae', 'rmse', 'pearson', 'ccc'):
                row[f'{dim}_{metric}'] = metrics[dim][metric]
            row[f'{dim}_pred_std'] = stats[dim]['pred_std']
            row[f'{dim}_pseudo_logvar_avg'] = sum_logvar[dim]/max(step,1)
            row[f'{dim}_pseudo_weight_avg'] = sum_weight[dim]/max(step,1)
            row[f'{dim}_pseudo_abs_residual_avg'] = sum_abs_residual[dim]/max(step,1)
        history.append(row)
        pd.DataFrame(history).to_csv(out/'training_history.csv', index=False)
        if diagnostics:
            pd.DataFrame(diagnostics).to_csv(out/'training_diagnostics.csv', index=False)
        pd.DataFrame(utility_rows, columns=['step', 'meta_ids', 'supervised_ids',
                    *[f'M_{d}_{k}' for d in DIMENSIONS for k in DIMENSIONS],
                    *[f'weight_{d}' for d in DIMENSIONS], 'compute_seconds']).to_csv(
                        out/'utility_matrix_history.csv', index=False)
        print(f'{args.mode} step={step} Dev Macro-MAE={mae:.6f}', flush=True)
        if mae < best-1e-8:
            best = mae
            torch.save({'model': model.state_dict(),
                        'uncertainty': uncertainty.state_dict() if uncertainty else None,
                        'utility_controller': controller.state_dict() if controller else None,
                        'step': step, 'dev_metrics': metrics, 'run_config': config}, out/'model_best.pth')
            (out/'dev_metrics_best.json').write_text(json.dumps({'step': step, 'metrics': metrics,
                                                                  'stats': stats}, indent=2)+'\n')
            frame = pd.DataFrame({'id': ids})
            for index, dim in enumerate(DIMENSIONS):
                frame[f'gold_{dim}'] = gold[:,index]*4+1
                frame[f'pred_{dim}'] = pred[:,index]*4+1
            frame.to_csv(out/'dev_predictions_best.csv', index=False)

    evaluate(0)
    for step in range(1, args.steps+1):
        model.train()
        tokens, targets, supervised_ids = next(labeled_iter)
        tokens, targets = move(tokens, device), targets.to(device)
        encoder_opt.zero_grad(set_to_none=True)
        head_opt.zero_grad(set_to_none=True)
        torch.manual_seed(args.seed*100000+step)
        supervised = F.mse_loss(model(**tokens), targets)
        total = supervised
        unsup_value = meta_value = unsup_head_grad = 0.0
        diagnostic = {'step': step, 'supervised_mse': float(supervised.detach())}
        head_parameters = tuple(model.head.parameters())
        projected_head_grads = None
        ulb_tokens, ulb_ids = next(unlabeled_iter)
        used_ids.update(ulb_ids)
        ulb_tokens = move(ulb_tokens, device)
        devices = [torch.cuda.current_device()] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.seed*100000+50000+step)
            if step%args.meta_every==0:
                meta_tokens, meta_targets, _ = next(meta_iter)
                meta_tokens, meta_targets = move(meta_tokens, device), meta_targets.to(device)
                with torch.no_grad():
                    model.train()
                    lb_features = pooled_features(model, tokens)
                    meta_features = pooled_features(model, meta_tokens)
                    model.eval()
                    weak_features = pooled_features(model, ulb_tokens)
                    model.train()
                    strong_features = pooled_features(model, ulb_tokens)
                info = bilevel_head_step(model.head, head_opt, uncertainty, unc_opt,
                                         lb_features, targets, weak_features, strong_features,
                                         meta_features, meta_targets,
                                         lambda_u=args.lambda_u, lambda2=args.lambda2)
                meta_value = info['meta_loss']
                for key in ('meta_loss', 'meta_mse', 'meta_logvar_mean', 'meta_regularization_term',
                            'meta_mse_uncertainty_grad_norm',
                            'meta_regularization_uncertainty_grad_norm'):
                    if key in info:
                        diagnostic[key] = info[key]
                encoder_opt.zero_grad(set_to_none=True)
                head_opt.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad():
                weak = model(**ulb_tokens).detach()
            model.train()
            strong = model(**ulb_tokens)
            logvar = uncertainty(strong-weak, strong)
            dimension_losses = per_dimension_unlabeled(strong, weak, logvar.detach())
            if utility_enabled:
                utility_start = time.perf_counter()
                if step % args.meta_every == 0:
                    utility_tokens, utility_targets, utility_ids = disjoint_meta_batch(
                            meta_iter, supervised_ids, desired_size=args.batch_size)
                    utility_tokens = move(utility_tokens, device)
                    utility_targets = utility_targets.to(device)
                    with torch.no_grad():
                        utility_features = pooled_features(model, utility_tokens)
                    matrix = virtual_utility_matrix(model.head, dimension_losses,
                                                    utility_features, utility_targets,
                                                    head_opt.param_groups[0]['lr'])
                    weights = controller.update(matrix).to(device=device, dtype=strong.dtype)
                    utility_row = {'step': step, 'meta_ids': '|'.join(map(str, utility_ids)),
                                   'supervised_ids': '|'.join(map(str, supervised_ids))}
                    utility_row.update({f'M_{d}_{k}': float(matrix[i, j])
                                        for i, d in enumerate(DIMENSIONS)
                                        for j, k in enumerate(DIMENSIONS)})
                    utility_row.update({f'weight_{d}': float(weights[i])
                                        for i, d in enumerate(DIMENSIONS)})
                    utility_row['compute_seconds'] = time.perf_counter() - utility_start
                    utility_rows.append(utility_row)
                else:
                    weights = controller.weights().to(device=device, dtype=strong.dtype)
                unlabeled_loss = (dimension_losses * weights).mean()
                extra_compute_seconds += time.perf_counter() - utility_start
            else:
                weights = torch.ones(3, device=device, dtype=strong.dtype)
                unlabeled_loss = weighted_pseudo_mse(strong, weak, logvar.detach())
            total = total+args.lambda_u*unlabeled_loss
            unsup_value = float(unlabeled_loss.detach())
            supervised_grads = torch.autograd.grad(supervised, head_parameters,
                                                   retain_graph=True, allow_unused=True)
            unsup_grads = torch.autograd.grad(args.lambda_u*unlabeled_loss,
                                              head_parameters,
                                              retain_graph=True, allow_unused=True)
            sup_head_grad, unsup_head_grad, grad_cosine = gradient_statistics(
                supervised_grads, unsup_grads)
            if ggr_enabled:
                ggr_start = time.perf_counter()
                projected_head_grads, ggr_stats = merge_head_gradients(
                    head_parameters, supervised_grads, unsup_grads)
                diagnostic.update(ggr_stats)
                extra_compute_seconds += time.perf_counter() - ggr_start
            if unsup_head_grad > 0 and bool((weights > 0).any()):
                effective_steps += 1
            diagnostic.update({'unlabeled_weighted_mse': unsup_value,
                               'supervised_head_grad_norm': sup_head_grad,
                               'unsup_head_grad_norm': unsup_head_grad,
                               'head_grad_cosine': grad_cosine})
            for index, dim in enumerate(DIMENSIONS):
                diagnostic[f'{dim}_utility_weight'] = float(weights[index])
                diagnostic[f'{dim}_unlabeled_loss'] = float(dimension_losses[index].detach())
                diagnostic[f'{dim}_weighted_contribution'] = float(
                    (args.lambda_u * weights[index] * dimension_losses[index] / 3).detach())
            detached_logvar = logvar.detach()
            detached_weight = 0.5*torch.exp(-detached_logvar)
            detached_diff = (strong-weak).detach()
            detached_residual = detached_diff.abs()
            for index,dim in enumerate(DIMENSIONS):
                sum_logvar[dim] += float(detached_logvar[:,index].mean())
                sum_weight[dim] += float(detached_weight[:,index].mean())
                sum_abs_residual[dim] += float(detached_residual[:,index].mean())
                for name,value in sample_statistics(detached_weight[:,index]).items():
                    diagnostic[f'{dim}_weight_{name}'] = value
                for name,value in sample_statistics(detached_diff[:,index]).items():
                    diagnostic[f'{dim}_weak_strong_diff_{name}'] = value
                diagnostic[f'{dim}_weak_strong_abs_mean'] = float(detached_residual[:,index].mean())
        if not torch.isfinite(total):
            raise FloatingPointError('nonfinite training loss')
        total.backward()
        if projected_head_grads is not None:
            # Replace, never add: total.backward already contains supervised MSE.
            for parameter, gradient in zip(head_parameters, projected_head_grads):
                parameter.grad = gradient.detach().clone()
        if args.clip_grad_norm>0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        encoder_opt.step()
        head_opt.step()
        sum_sup += float(supervised.detach())
        sum_unsup += unsup_value
        sum_meta += meta_value
        sum_unsup_head_grad += unsup_head_grad
        if diagnostic is not None:
            diagnostics.append(diagnostic)
        if step%args.eval_every==0 or step==args.steps:
            evaluate(step)
    torch.save({'model': model.state_dict(), 'uncertainty': uncertainty.state_dict(),
                'utility_controller': controller.state_dict() if controller else None,
                'step': args.steps, 'run_config': config}, out/'model_last.pth')


if __name__=='__main__':
    main()
