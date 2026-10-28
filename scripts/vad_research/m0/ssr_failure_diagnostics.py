#!/usr/bin/env python3
"""Fixed TRAIN-only representation/target diagnostics; no model training."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.emobank.train_emobank_supervised import metric_report
from semilearn.datasets.nlp_datasets.emobank import DIMENSIONS, NORM_COLUMNS, read_gold


def numbers(values):
    return [float(x) for x in values]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-dir', type=Path, required=True)
    p.add_argument('--probe-dir', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--spectral-params', type=Path, required=True)
    p.add_argument('--out-csv', type=Path, required=True)
    p.add_argument('--out-json', type=Path, required=True)
    a = p.parse_args()
    for path in (a.out_csv, a.out_json):
        if path.exists():
            raise FileExistsError(path)
    protocol_bytes = a.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    assert protocol['version'] == 'train-only-fixed-v1' and protocol['k'] == 8
    assert protocol['label_permutations'] == 1000
    assert protocol['query_bootstrap_replicates'] == 10000
    meta = json.loads((a.feature_dir / 'feature_metadata.json').read_text())
    assert meta['budget'] == 500 and meta['seed'] == 0
    assert meta['model'].startswith('frozen pretrained RoBERTa-base')
    with np.load(a.feature_dir / 'pretrained_features.npz', allow_pickle=False) as z:
        x = normalize(z['lb_x'].astype(np.float64))
        y = z['lb_y'].astype(np.float64) * 4 + 1
        ids = z['lb_ids'].astype(str)
    assert x.shape == (500, 768) and y.shape == (500, 3)
    assert len(set(ids)) == 500 and np.isfinite(x).all() and np.isfinite(y).all()
    train_source = REPO.parent / 'data' / 'emobank_processed' / 'lb500_s0' / 'train_lb.parquet'
    source = read_gold(train_source, expected_n=500)
    assert np.array_equal(ids, source.id.astype(str).to_numpy())
    assert np.allclose(y, source.loc[:, list(DIMENSIONS)].to_numpy(), atol=2e-7)
    probe = pd.read_csv(a.probe_dir / 'probe_predictions.csv')
    assert len(probe) == 100 and probe.id.is_unique
    probe_ids = probe.id.astype(str).to_numpy()
    assert set(probe_ids).issubset(ids)
    id_to_index = {value: i for i, value in enumerate(ids)}
    qi = np.array([id_to_index[value] for value in probe_ids])
    ri = np.array([i for i, value in enumerate(ids) if value not in set(probe_ids)])
    assert len(ri) == 400 and len(qi) == 100 and not set(ri).intersection(qi)
    qx, rx, qy, ry = x[qi], x[ri], y[qi], y[ri]
    for d, dim in enumerate(DIMENSIONS):
        assert np.allclose(probe['gold_' + dim].to_numpy(), qy[:, d], atol=2e-5)
    cosine = np.clip(1 - qx @ rx.T, 0, 2)
    knn = np.argsort(cosine, axis=1, kind='stable')[:, :8]
    target_abs = np.abs(qy[:, None, :] - ry[None, :, :])
    nearest_abs = np.take_along_axis(target_abs, knn[:, :, None], axis=1).mean(axis=1)
    random_expect = target_abs.mean(axis=1)
    rng = np.random.default_rng(protocol['random_seed'])
    random_idx = np.vstack([rng.choice(400, size=8, replace=False) for _ in range(100)])
    random_draw_abs = np.take_along_axis(target_abs, random_idx[:, :, None], axis=1).mean(axis=1)
    gain = random_expect - nearest_abs
    observed = nearest_abs.mean(axis=0)
    permutation = np.empty((1000, 3))
    for b in range(1000):
        shuffled = ry[rng.permutation(400)]
        permutation[b] = np.abs(qy[:, None, :] - shuffled[knn]).mean(axis=(0, 1))
    p_lower = (1 + np.sum(permutation <= observed, axis=0)) / 1001
    bootstrap_idx = rng.integers(0, 100, size=(10000, 100))
    bootstrap = gain[bootstrap_idx].mean(axis=1)
    ci = np.quantile(bootstrap, [.025, .975], axis=0)
    macro_gain = gain.mean(axis=1)
    macro_ci = np.quantile(macro_gain[bootstrap_idx].mean(axis=1), [.025, .975])
    macro_perm = permutation.mean(axis=1)
    macro_p = float((1 + np.sum(macro_perm <= observed.mean())) / 1001)
    target_macro = target_abs.mean(axis=2)
    correlation = {}
    for d, dim in enumerate((*DIMENSIONS, 'macro')):
        target = target_abs[:, :, d] if d < 3 else target_macro
        within = [spearmanr(cosine[i], target[i]).statistic for i in range(100)]
        correlation[dim] = {'pooled_pair_spearman_descriptive': float(spearmanr(cosine.ravel(), target.ravel()).statistic),
                            'mean_within_probe_spearman': float(np.mean(within)),
                            'median_within_probe_spearman': float(np.median(within))}
    # Density threshold comes entirely from the 400 reference labels.
    ref_norm, query_norm = (ry - 1) / 4, (qy - 1) / 4
    ref_dist = cdist(ref_norm, ref_norm)
    np.fill_diagonal(ref_dist, np.inf)
    ref_radius = np.partition(ref_dist, 7, axis=1)[:, 7]
    threshold = float(np.quantile(ref_radius, .75))
    query_radius = np.partition(cdist(query_norm, ref_norm), 7, axis=1)[:, 7]
    sparse = query_radius > threshold
    pred = np.column_stack([probe['pred_' + dim].to_numpy(dtype=float) for dim in DIMENSIONS])
    absolute_error = np.abs(pred - qy)
    p3b_metrics = metric_report((qy - 1) / 4, (pred - 1) / 4)
    p3b_std_ratio = {dim: float(np.std(pred[:, d]) / np.std(qy[:, d])) for d, dim in enumerate(DIMENSIONS)}
    region = {}
    for name, mask in (('dense', ~sparse), ('sparse', sparse)):
        region[name] = {'count': int(mask.sum()),
                        'mae': {dim: float(absolute_error[mask, d].mean()) for d, dim in enumerate(DIMENSIONS)},
                        'macro_mae': float(absolute_error[mask].mean()),
                        'mean_label_radius': float(query_radius[mask].mean())}
    spectral = json.loads(a.spectral_params.read_text())
    spectral_context = {mode: {'macro_mae': spectral['oof_metrics'][mode]['macro_mae_original_scale'],
                                'per_dimension': {dim: {key: spectral['oof_metrics'][mode][dim][key] for key in ('mae', 'ccc', 'pearson')}
                                                  for dim in DIMENSIONS}}
                        for mode in ('b1', 'b2')}
    frame = pd.DataFrame({'id': probe_ids, 'cosine_nn_mean': np.take_along_axis(cosine, knn, axis=1).mean(axis=1),
                          'cosine_random_mean': cosine.mean(axis=1),
                          'label_radius_8th': query_radius,
                          'coverage': np.where(sparse, 'sparse', 'dense')})
    for d, dim in enumerate(DIMENSIONS):
        frame['gold_' + dim] = qy[:, d]
        frame['p3b_pred_' + dim] = pred[:, d]
        frame['p3b_abs_error_' + dim] = absolute_error[:, d]
        frame['knn_abs_target_diff_' + dim] = nearest_abs[:, d]
        frame['random_expected_abs_target_diff_' + dim] = random_expect[:, d]
        frame['random_draw_abs_target_diff_' + dim] = random_draw_abs[:, d]
        frame['target_alignment_gain_' + dim] = gain[:, d]
    frame['target_alignment_gain_macro'] = macro_gain
    result = {'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
              'feature_pretrain_sha256': meta['pretrain_sha256'],
              'train_source': str(train_source),
              'counts': {'reference': 400, 'probe': 100}, 'k': 8,
              'representation_distance': {'mean_nearest_cosine': float(np.take_along_axis(cosine, knn, axis=1).mean()),
                                          'mean_all_reference_cosine': float(cosine.mean())},
              'target_alignment': {dim: {'nearest_mean_abs_diff': float(observed[d]),
                                         'random_expected_mean_abs_diff': float(random_expect[:, d].mean()),
                                         'random_draw_mean_abs_diff': float(random_draw_abs[:, d].mean()),
                                         'gain_random_expected_minus_nearest': float(gain[:, d].mean()),
                                         'bootstrap_95_ci': numbers(ci[:, d]),
                                         'permuted_reference_label_p_lower': float(p_lower[d])}
                                   for d, dim in enumerate(DIMENSIONS)},
              'macro_alignment': {'gain': float(macro_gain.mean()), 'bootstrap_95_ci': numbers(macro_ci),
                                  'permuted_reference_label_p_lower': macro_p},
              'distance_target_correlation': correlation,
              'coverage': {'reference_75pct_radius_threshold': threshold, 'dense': region['dense'], 'sparse': region['sparse']},
              'p3b_probe_metrics': p3b_metrics, 'p3b_pred_to_gold_std_ratio': p3b_std_ratio,
              'spectral_selection_conditioned_oof_context': spectral_context,
              'dev_test_access': False}
    a.out_json.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    frame.to_csv(a.out_csv, index=False)
    print('macro alignment gain', result['macro_alignment'], flush=True)
    print('coverage', result['coverage'], flush=True)


if __name__ == '__main__':
    main()
