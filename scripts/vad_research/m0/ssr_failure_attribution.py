#!/usr/bin/env python3
"""One fixed TRAIN-only resampled-covariate control for the failed spectral gate."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.emobank.train_emobank_supervised import metric_report
from scripts.vad_research.m0.spectral_core import fit_nystrom_spectral
from semilearn.datasets.nlp_datasets.emobank import DIMENSIONS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-dir', type=Path, required=True)
    p.add_argument('--spectral-params', type=Path, required=True)
    p.add_argument('--spectral-oof', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out-json', type=Path, required=True)
    p.add_argument('--out-csv', type=Path, required=True)
    a = p.parse_args()
    for path in (a.out_json, a.out_csv):
        if path.exists():
            raise FileExistsError(path)
    protocol_bytes = a.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    assert protocol['version'] == 'single-fixed-negative-control-v1'
    assert protocol['landmarks'] == 256 and protocol['rank'] == 64 and protocol['ridge_alpha'] == 100
    params = json.loads(a.spectral_params.read_text())
    assert params['selected_shared_rank'] == 64 and params['selected_alpha'] == {'b1': 100.0, 'b2': 100.0}
    with np.load(a.feature_dir / 'pretrained_features.npz', allow_pickle=False) as z:
        x = normalize(z['lb_x'].astype(np.float64))
        u = normalize(z['ulb_x'].astype(np.float64))
        y = z['lb_y'].astype(np.float64)
        ids = z['lb_ids'].astype(str)
    assert x.shape == (500, 768) and u.shape == (7562, 768) and y.shape == (500, 3)
    previous = pd.read_csv(a.spectral_oof)
    assert np.array_equal(previous.id.astype(str).to_numpy(), ids)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=0).split(x))
    oof = {mode: np.full_like(y, np.nan) for mode in ('b1', 'b2', 'resampled_control')}
    fold_of = np.full(500, -1, dtype=int)
    fold_diag = []
    for fold, (fit_idx, val_idx) in enumerate(folds):
        fit_x, val_x = x[fit_idx], x[val_idx]
        gamma = params['fold_diagnostics'][fold]['gamma']
        anchors = fit_x[np.random.default_rng(fold).choice(400, size=256, replace=False)]
        control_u = fit_x[np.random.default_rng(314159 + fold).choice(400, size=7562, replace=True)]
        covariates = {'b1': fit_x, 'b2': np.concatenate((fit_x, u)),
                      'resampled_control': np.concatenate((fit_x, control_u))}
        bases = {}
        one = {'fold': fold, 'gamma': float(gamma)}
        for mode, basis_x in covariates.items():
            basis = fit_nystrom_spectral(basis_x, gamma=gamma, rank=64,
                                         landmarks=256, seed=fold, external_landmarks=anchors)
            bases[mode] = basis
            model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
            model.fit(basis.transform(fit_x), y[fit_idx])
            oof[mode][val_idx] = model.predict(basis.transform(val_x))
        ref = bases['b1'].directions
        for mode in ('b2', 'resampled_control'):
            one['b1_' + mode + '_rank64_projector_overlap'] = float(np.linalg.norm(ref.T @ bases[mode].directions, 'fro') ** 2 / 64)
        fold_of[val_idx] = fold
        fold_diag.append(one)
        print('attribution fold', fold, one, flush=True)
    assert all(np.isfinite(v).all() for v in oof.values())
    assert np.all(fold_of >= 0)
    max_abs_reproduce = {}
    for mode in ('b1', 'b2'):
        saved = np.column_stack([previous[mode + '_pred_' + dim].to_numpy() for dim in DIMENSIONS])
        max_abs_reproduce[mode] = float(np.max(np.abs(4 * oof[mode] + 1 - saved)))
        if max_abs_reproduce[mode] > 1e-7:
            raise RuntimeError(f'{mode} reproduction failed: {max_abs_reproduce[mode]}')
    metrics = {mode: metric_report(y, pred) for mode, pred in oof.items()}
    std_ratio = {mode: {dim: float(np.std(pred[:, d]) / np.std(y[:, d]))
                        for d, dim in enumerate(DIMENSIONS)} for mode, pred in oof.items()}
    fold_mae = []
    for fold, (_, val_idx) in enumerate(folds):
        row = {'fold': fold}
        for mode, pred in oof.items():
            row[mode + '_macro_mae'] = float(np.abs(4 * (pred[val_idx] - y[val_idx])).mean())
        fold_mae.append(row)
    table = pd.DataFrame({'id': ids, 'fold': fold_of})
    for d, dim in enumerate(DIMENSIONS):
        table['gold_' + dim] = 4 * y[:, d] + 1
        for mode, pred in oof.items():
            table[mode + '_pred_' + dim] = 4 * pred[:, d] + 1
    result = {'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
              'max_abs_reproduced_prediction_error_original_scale': max_abs_reproduce,
              'fold_diagnostics': fold_diag, 'fold_macro_mae': fold_mae,
              'oof_metrics': metrics, 'pred_to_gold_std_ratio': std_ratio,
              'b1_b2_projector_overlap_mean': float(np.mean([v['b1_b2_rank64_projector_overlap'] for v in fold_diag])),
              'b1_control_projector_overlap_mean': float(np.mean([v['b1_resampled_control_rank64_projector_overlap'] for v in fold_diag])),
              'dev_test_access': False}
    a.out_json.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    table.to_csv(a.out_csv, index=False)
    print('macro', {mode: metrics[mode]['macro_mae_original_scale'] for mode in metrics}, flush=True)


if __name__ == '__main__':
    main()
