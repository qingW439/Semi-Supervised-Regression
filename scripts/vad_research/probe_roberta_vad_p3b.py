#!/usr/bin/env python3
"""P3-B: leakage-safe, training-only 400/100 EmoBank VAD teacher probe.

Run from repository root after copying to scripts/vad_research/probe_roberta_vad_p3b.py.
- Reads ONLY the 500 labeled TRAIN parquet and its committed ID list.
- Splits 400 teacher-training / 100 unseen probe rows reproducibly.
- Initializes RoBERTa from the original pretrained weights, NEVER a P1/P2
  checkpoint trained on all 500 rows.
- Trains a FIXED 320 updates, no probing/early stopping during training.
- Evaluates probe only ONCE after training; does not read DEV/TEST.
- Diagnostic-only; probe gold must not be used to tune M1 or select models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    DIMENSIONS, NORM_COLUMNS, GoldData, TextCollator, read_gold,
)
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import (  # noqa: E402
    RobertaVADRegressor, predict, set_seed, sha256_file, git_commit,
)

DEFAULT_DATA = REPO.parent / 'data' / 'emobank_processed'
DEFAULT_MODEL = REPO.parent / 'pretrained' / 'roberta-base'


def get_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--budget', type=int, choices=[500], default=500)
    ap.add_argument('--seed', type=int, choices=[0, 1, 2], default=0)
    ap.add_argument('--split-seed', type=int, default=20261019)
    ap.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--steps', type=int, default=320)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--eval-batch-size', type=int, default=32)
    ap.add_argument('--max-length', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--clip-grad-norm', type=float, default=1.0)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    if any(getattr(a, k) <= 0 for k in ('steps', 'batch_size', 'eval_batch_size', 'max_length')):
        ap.error('steps, batch sizes and max length must be positive')
    if a.lr <= 0 or a.dropout < 0 or a.weight_decay < 0 or a.clip_grad_norm < 0:
        ap.error('Invalid optimizer or dropout parameters')
    if a.out is None:
        a.out = REPO / 'saved_models' / 'vad_research' / f'p3b_probe400_100_lb500_s{a.seed}'
    return a


def hash_ids(ids):
    payload = ('\n'.join(sorted(str(x) for x in ids)) + '\n').encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def load_probe_split(a):
    path = a.data_root / f'lb{a.budget}_s{a.seed}' / 'train_lb.parquet'
    full = read_gold(path, expected_n=500)
    committed_path = REPO / 'repro_records' / 'emobank' / f'labeled_ids_lb500_s{a.seed}.csv'
    if not committed_path.is_file():
        raise FileNotFoundError(committed_path)
    with committed_path.open(newline='', encoding='utf-8') as f:
        committed = {r['id'] for r in csv.DictReader(f)}
    if committed != set(full['id']):
        raise ValueError('Mismatch between train_lb and committed labeled IDs')
    # Independent of full 500-row order: sort by ID before permuting.
    full = full.sort_values('id').reset_index(drop=True)
    idx = np.random.default_rng(a.split_seed + a.seed).permutation(len(full))
    train = full.iloc[idx[:400]].sort_values('id').reset_index(drop=True)
    probe = full.iloc[idx[400:]].sort_values('id').reset_index(drop=True)
    if len(train) != 400 or len(probe) != 100:
        raise AssertionError('Unexpected split counts')
    if set(train['id']) & set(probe['id']):
        raise AssertionError('Train/probe ID leakage')
    if set(train['id']) | set(probe['id']) != set(full['id']):
        raise AssertionError('Missing label IDs in internal split')
    return path, train, probe


def constants(y_train_norm, y_probe_norm):
    ref = {}
    for name, value in (
        ('train_mean_constant', y_train_norm.mean(axis=0)),
        ('train_median_constant', np.median(y_train_norm, axis=0)),
    ):
        pred = np.broadcast_to(value, y_probe_norm.shape).copy()
        ref[name] = metric_report(y_probe_norm, pred)
    return ref


def as_float(x):
    return None if not np.isfinite(x) else float(x)


def diagnostic_report(y_norm, p_norm, train_norm, ids):
    base = metric_report(y_norm, p_norm)
    y = y_norm.astype(np.float64) * 4.0 + 1.0
    p = p_norm.astype(np.float64) * 4.0 + 1.0
    train_orig = train_norm.astype(np.float64) * 4.0 + 1.0
    diagnostics = {}
    for k, d in enumerate(DIMENSIONS):
        g, v = y[:, k], p[:, k]
        e = v - g
        q20, q80 = np.quantile(train_orig[:, k], [0.2, 0.8])
        cuts = {'low': g < q20, 'middle': (g >= q20) & (g <= q80), 'high': g > q80}
        groups = {}
        for name, mask in cuts.items():
            if not mask.any():
                groups[name] = {'count': 0, 'mae': None, 'signed_bias': None}
            else:
                groups[name] = {'count': int(mask.sum()), 'mae': float(np.abs(e[mask]).mean()),
                                'signed_bias': float(e[mask].mean())}
        diagnostics[d] = {
            'gold_mean': float(g.mean()), 'pred_mean': float(v.mean()),
            'signed_bias': float(e.mean()), 'gold_std': float(g.std()),
            'pred_std': float(v.std()),
            'pred_to_gold_std_ratio': float(v.std()/max(g.std(), 1e-12)),
            'low_high_cutoffs_from_train': [float(q20), float(q80)],
            'true_value_groups': groups,
            'predicted_below_1': int((v < 1.0).sum()),
            'predicted_above_5': int((v > 5.0).sum()),
        }
    return {'n_probe': len(ids), 'metrics': base, 'diagnostics': diagnostics,
            'comparators': constants(train_norm, y_norm)}


def main():
    a = get_args()
    set_seed(a.seed)
    full_path, train, probe = load_probe_split(a)
    info = {
        'source': str(full_path.resolve()), 'source_sha256': sha256_file(full_path),
        'source_git_commit': git_commit(), 'budget': a.budget,
        'seed': a.seed, 'split_seed': a.split_seed,
        'train_count': len(train), 'probe_count': len(probe),
        'train_id_sha256': hash_ids(train['id']),
        'probe_id_sha256': hash_ids(probe['id']),
        'fixed_steps': a.steps,
        'protocol': ('Teacher trained from original RoBERTa weights on 400 rows only; '
                     '100 probe labels withheld until single post-training evaluation; '
                     'neither DEV nor TEST read; no checkpoint selection by probe.'),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    if a.check_only:
        print('PASS: 400/100 IDs verified. NO pretrained weights loaded, NO outputs written, NO DEV/TEST accessed.', flush=True)
        return

    model_dir = a.model.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f'Pretrained RoBERTa directory missing: {model_dir}')
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty directory: {out}')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device('cpu' if a.device == 'cpu' or not torch.cuda.is_available() else 'cuda')

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, local_files_only=True)
    collator = TextCollator(tokenizer, a.max_length)
    train_loader = DataLoader(
        GoldData(train), batch_size=a.batch_size, shuffle=True, collate_fn=collator,
        generator=torch.Generator().manual_seed(a.seed))
    probe_loader = DataLoader(GoldData(probe), batch_size=a.eval_batch_size,
                              shuffle=False, collate_fn=collator)
    model = RobertaVADRegressor(str(model_dir), dropout=a.dropout, offline=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'split_manifest.json').write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    pd.DataFrame({'id': train['id']}).to_csv(out/'teacher_train_ids.csv', index=False)
    pd.DataFrame({'id': probe['id']}).to_csv(out/'probe_ids.csv', index=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    (out / 'run_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

    step = 0
    epochs_completed = 0
    iterator = iter(train_loader)
    loss_sum = 0.0
    while step < a.steps:
        try:
            toks, y, _ = next(iterator)
        except StopIteration:
            epochs_completed += 1
            iterator = iter(train_loader)
            toks, y, _ = next(iterator)
        model.train()
        toks = {k: v.to(device) for k, v in toks.items()}
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(**toks)
        if pred.shape != y.shape or not torch.isfinite(pred).all():
            raise RuntimeError('Invalid model outputs')
        loss = nn.functional.l1_loss(pred, y)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        if a.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad_norm)
        optimizer.step()
        step += 1
        loss_sum += float(loss.detach().item())
        if step % 40 == 0 or step == a.steps:
            print(f'train step {step}/{a.steps} mean_norm_L1={loss_sum/step:.6f}', flush=True)

    # Entire 100-probe gold is touched ONLY after the last optimization step.
    g, p, ids = predict(model, probe_loader, device)
    if len(ids) != 100 or set(ids) != set(probe['id']):
        raise RuntimeError('Probe predictions do not align with 100 reserved IDs')
    train_norm = train.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float32)
    report = diagnostic_report(g, p, train_norm, ids)
    report['protocol'] = info
    report['train_steps'] = step
    report['completed_full_data_passes'] = epochs_completed
    (out / 'probe_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    frame = pd.DataFrame({'id': ids})
    for k, d in enumerate(DIMENSIONS):
        frame[f'gold_{d}'] = g[:, k]*4+1
        frame[f'pred_{d}'] = p[:, k]*4+1
        frame[f'abs_error_{d}'] = np.abs(frame[f'pred_{d}']-frame[f'gold_{d}'])
    frame.to_csv(out/'probe_predictions.csv', index=False)
    # Checkpoint is diagnostic-only, and was NOT initialized from a P1 model that saw all 500 labels.
    torch.save({'model': model.state_dict(), 'train_steps': step,
                'source': '400-label diagnostic teacher, never trained on probe rows',
                'split_manifest': info, 'config': config}, out / 'teacher_probe_only.pth')
    print('\n=== P3B RESULTS / 100 unseen TRAIN probe rows ===', flush=True)
    print('Probe Macro-MAE:', f'{report["metrics"]["macro_mae_original_scale"]:.6f}', flush=True)
    print('Constant mean Macro-MAE:', f'{report["comparators"]["train_mean_constant"]["macro_mae_original_scale"]:.6f}', flush=True)
    print('Constant median Macro-MAE:', f'{report["comparators"]["train_median_constant"]["macro_mae_original_scale"]:.6f}', flush=True)
    for d in DIMENSIONS:
        m=report['metrics'][d]; q=report['diagnostics'][d]
        print(f'{d}: MAE={m["mae"]:.5f} CCC={m["ccc"]:.4f} '
              f'Pearson={m["pearson"]:.4f} R2={m["r2"]:.4f} '
              f'bias={q["signed_bias"]:+.5f} std_ratio={q["pred_to_gold_std_ratio"]:.3f}', flush=True)
        for group, item in q['true_value_groups'].items():
            print(f'  {group}: n={item["count"]} MAE={item["mae"]} bias={item["signed_bias"]}', flush=True)
    print('RESULT_DIR=', out, flush=True)
    print('No DEV or TEST read. The 100 probe labels were never used for optimization.', flush=True)


if __name__ == '__main__':
    main()
