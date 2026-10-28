#!/usr/bin/env python3
"""Locked TRAIN-only B1/B2 spectral gate. Never loads DEV or TEST arrays."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.emobank.train_emobank_supervised import metric_report
from scripts.vad_research.m0.run_spectral_ridge import ALPHAS
from scripts.vad_research.m0.spectral_core import fit_nystrom_spectral, rbf_kernel
from semilearn.datasets.nlp_datasets.emobank import DIMENSIONS, NORM_COLUMNS, read_gold, read_unlabeled


def distance_gamma(x, seed, count):
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, len(x), size=(count, 2))
    distances = []
    for block in np.array_split(pairs, math.ceil(count / 2048)):
        block = block[block[:, 0] != block[:, 1]]
        diff = x[block[:, 0]] - x[block[:, 1]]
        d2 = np.einsum('ij,ij->i', diff, diff)
        distances.extend(d2[d2 > 0].tolist())
    median = float(np.median(distances))
    gamma = float(np.log(2.0) / median)
    if not (len(distances) > count * .95 and np.isfinite(median) and
            0 < median <= 4.000001 and np.isfinite(gamma) and gamma > 0):
        raise RuntimeError('invalid squared-distance median or gamma')
    return {'median_nonzero_d2': median, 'gamma': gamma, 'nonzero_pairs': len(distances)}


def score(y, pred):
    return float(4 * mean_absolute_error(y, pred))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-dir', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--out-json', type=Path, required=True)
    p.add_argument('--out-csv', type=Path, required=True)
    a = p.parse_args()
    for path in (a.out_json, a.out_csv):
        if path.exists():
            raise FileExistsError(path)
    protocol_bytes = a.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    assert protocol['protocol_version'] == 'train-only-v1-locked-before-computation'
    assert protocol['ridge_alphas'] == list(ALPHAS)
    assert protocol['rank_candidates'] == [8, 16, 32, 64]
    assert protocol['nystrom']['landmarks'] == 256
    assert protocol['outer_cv'] == {'type': 'KFold', 'n_splits': 5, 'shuffle': True, 'random_state': 0}
    with np.load(a.feature_dir / 'pretrained_features.npz', allow_pickle=False) as z:
        x = normalize(z['lb_x'].astype(np.float64))
        u = normalize(z['ulb_x'].astype(np.float64))
        y = z['lb_y'].astype(np.float64)
        ids = z['lb_ids'].astype(str)
        ulb_ids = z['ulb_ids'].astype(str)
    assert x.shape == (500, 768) and u.shape == (7562, 768) and y.shape == (500, 3)
    assert len(set(ids)) == 500 and len(set(ulb_ids)) == 7562 and not set(ids).intersection(ulb_ids)
    assert all(np.isfinite(z).all() for z in (x, u, y))
    assert np.allclose(np.linalg.norm(x, axis=1), 1., atol=1e-12)
    assert np.allclose(np.linalg.norm(u, axis=1), 1., atol=1e-12)
    data_root = REPO.parent / 'data' / 'emobank_processed' / 'lb500_s0'
    lb = read_gold(data_root / 'train_lb.parquet', expected_n=500)
    ulb = read_unlabeled(data_root / 'train_ulb.parquet', labeled_ids=lb['id'])
    assert np.array_equal(ids, lb['id'].astype(str).to_numpy())
    assert np.array_equal(ulb_ids, ulb['id'].astype(str).to_numpy())
    assert np.array_equal(y.astype(np.float32), lb.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float32))
    n_pairs = protocol['gamma']['pair_sample_count']
    gamma_base = protocol['gamma']['pair_rng_seed_base']
    final_gamma = distance_gamma(x, gamma_base, n_pairs)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=0).split(x))
    features = {'b1': [], 'b2': []}
    diagnostics = []
    for fold, (fit_idx, val_idx) in enumerate(folds):
        fit_x, val_x = x[fit_idx], x[val_idx]
        g = distance_gamma(fit_x, gamma_base + fold, n_pairs)
        anchor_idx = np.random.default_rng(fold).choice(len(fit_x), size=256, replace=False)
        anchors = fit_x[anchor_idx].copy()
        eig = np.linalg.eigvalsh(rbf_kernel(anchors, anchors, g['gamma']))
        diag = {'fold': fold, **g, 'fit_count': len(fit_idx), 'valid_count': len(val_idx),
                'min_landmark_eigenvalue': float(eig.min()), 'clipped_eigenvalues': int(np.sum(eig < 1e-8)),
                'anchor_id_sha256': hashlib.sha256('\n'.join(ids[fit_idx][anchor_idx]).encode()).hexdigest()}
        for mode, basis_x in (('b1', fit_x), ('b2', np.concatenate((fit_x, u)))):
            basis = fit_nystrom_spectral(basis_x, gamma=g['gamma'], rank=64,
                                         landmarks=256, seed=fold, external_landmarks=anchors)
            assert np.array_equal(basis.landmarks, anchors)
            fit_phi, val_phi = basis.transform(fit_x), basis.transform(val_x)
            assert fit_phi.shape == (400, 64) and val_phi.shape == (100, 64)
            assert np.isfinite(fit_phi).all() and np.isfinite(val_phi).all()
            probe = fit_x[:128]
            exact = rbf_kernel(probe, probe, g['gamma'])
            projected = basis.transform(probe)
            approx = projected @ projected.T
            diag[mode + '_rank64_relative_kernel_error'] = float(np.linalg.norm(exact - approx) / np.linalg.norm(exact))
            features[mode].append((fit_idx, val_idx, fit_phi, val_phi))
        diagnostics.append(diag)
        print('basis fold', fold, 'gamma', g['gamma'], 'errors', diag['b1_rank64_relative_kernel_error'], diag['b2_rank64_relative_kernel_error'], flush=True)
    ranks, alphas = protocol['rank_candidates'], protocol['ridge_alphas']
    cv_scores = {'b1': {}, 'b2': {}}
    cv_predictions = {'b1': {}, 'b2': {}}
    for mode in ('b1', 'b2'):
        for rank in ranks:
            cv_scores[mode][str(rank)] = {}
            cv_predictions[mode][rank] = {}
            for alpha in alphas:
                oof = np.full_like(y, np.nan)
                fold_scores = []
                for fit_idx, val_idx, fit_phi, val_phi in features[mode]:
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                    model.fit(fit_phi[:, :rank], y[fit_idx])
                    pred = model.predict(val_phi[:, :rank])
                    oof[val_idx] = pred
                    fold_scores.append(score(y[val_idx], pred))
                assert np.isfinite(oof).all()
                cv_scores[mode][str(rank)][str(alpha)] = float(np.mean(fold_scores))
                cv_predictions[mode][rank][alpha] = oof
    b1_best = {rank: min(alphas, key=lambda alpha: (cv_scores['b1'][str(rank)][str(alpha)], alpha)) for rank in ranks}
    rank = min(ranks, key=lambda r: (cv_scores['b1'][str(r)][str(b1_best[r])], r))
    chosen_alpha = {mode: min(alphas, key=lambda alpha: (cv_scores[mode][str(rank)][str(alpha)], alpha)) for mode in ('b1', 'b2')}
    pred = {mode: cv_predictions[mode][rank][chosen_alpha[mode]] for mode in ('b1', 'b2')}
    reports = {mode: metric_report(y, pred[mode]) for mode in ('b1', 'b2')}
    fold_of = np.full(500, -1, dtype=int)
    paired = []
    for fold, (_, val_idx) in enumerate(folds):
        fold_of[val_idx] = fold
        m1, m2 = (metric_report(y[val_idx], pred[mode][val_idx]) for mode in ('b1', 'b2'))
        paired.append({'fold': fold, 'macro_mae_b2_minus_b1': float(m2['macro_mae_original_scale'] - m1['macro_mae_original_scale']),
                       'per_dimension': {dim: {'mae_b2_minus_b1': float(m2[dim]['mae'] - m1[dim]['mae']),
                                              'ccc_b2_minus_b1': float(m2[dim]['ccc'] - m1[dim]['ccc']),
                                              'pearson_b2_minus_b1': float(m2[dim]['pearson'] - m1[dim]['pearson'])} for dim in DIMENSIONS}})
    assert np.all(fold_of >= 0)
    delta = float(reports['b2']['macro_mae_original_scale'] - reports['b1']['macro_mae_original_scale'])
    shrink = {dim: float(np.std(pred['b2'][:, i]) / np.std(pred['b1'][:, i])) for i, dim in enumerate(DIMENSIONS)}
    improved_folds = int(sum(d['macro_mae_b2_minus_b1'] < 0 for d in paired))
    finite = all(np.isfinite(pred[mode]).all() for mode in pred) and all(np.isfinite(list(shrink.values())))
    if delta >= 0:
        gate = 'NO-GO'
    elif (-delta >= .002 and improved_folds >= 4 and
          all(reports['b2'][dim]['mae'] - reports['b1'][dim]['mae'] <= .005 for dim in DIMENSIONS) and
          all(reports['b2'][dim]['ccc'] - reports['b1'][dim]['ccc'] >= -.02 for dim in DIMENSIONS) and
          all(v >= .5 for v in shrink.values()) and finite):
        gate = 'GO-FORMAL-VALIDATION'
    else:
        gate = 'INCONCLUSIVE'
    table = pd.DataFrame({'id': ids, 'fold': fold_of})
    for i, dim in enumerate(DIMENSIONS):
        table['gold_' + dim] = 4 * y[:, i] + 1
        for mode in ('b1', 'b2'):
            table[mode + '_pred_' + dim] = 4 * pred[mode][:, i] + 1
    result = {'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
              'feature_dir': str(a.feature_dir.resolve()), 'counts': {'labeled': 500, 'unlabeled': 7562},
              'final_labeled_train_gamma': final_gamma, 'fold_diagnostics': diagnostics,
              'selected_shared_rank': rank, 'selected_alpha': chosen_alpha,
              'b1_rank_selection_alpha': {str(k): v for k, v in b1_best.items()},
              'cv_scores': cv_scores, 'oof_metrics': reports, 'paired_folds': paired,
              'oof_macro_mae_b2_minus_b1': delta, 'b2_over_b1_prediction_std': shrink,
              'improved_fold_count': improved_folds, 'gate': gate,
              'selection_conditioned_oof': True, 'dev_test_access': False}
    a.out_json.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    table.to_csv(a.out_csv, index=False)
    print('SELECTED rank', rank, 'alpha', chosen_alpha, 'delta', delta, 'gate', gate, flush=True)


if __name__ == '__main__':
    main()
