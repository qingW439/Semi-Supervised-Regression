#!/usr/bin/env python3
"""Seed0 GO/NO-GO gate for the LC-RankUp-VAD replacement.

Compares C0 (no calibration), C1 (RDA), M1 (local calibration) on DEV metrics
and applies the pre-registered thresholds from the research plan.
"""
import argparse
import json
from pathlib import Path


def load(d):
    p = Path(d) / 'dev_metrics_best.json'
    return json.loads(p.read_text())


def macro(m, key):
    vals = [m['metrics'][d][key] for d in ('V', 'A', 'D')]
    return sum(vals) / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--mae-min-improve', type=float, default=0.002)
    ap.add_argument('--ccc-max-drop', type=float, default=0.005)
    ap.add_argument('--r2-max-drop', type=float, default=0.01)
    ap.add_argument('--max-dim-mae-drop', type=float, default=0.003)
    ap.add_argument('--require-improved-dims', type=int, default=2)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    c0, c1, m1 = load(a.control), load(a.baseline), load(a.candidate)
    mae0, mae1, maem = (x['metrics']['macro_mae_original_scale'] for x in (c0, c1, m1))
    ccc0, ccc1, cccm = (macro(x, 'ccc') for x in (c0, c1, m1))
    r20, r21, r2m = (macro(x, 'r2') for x in (c0, c1, m1))
    dims = ('V', 'A', 'D')
    dim_delta = {d: c0['metrics'][d]['mae'] - m1['metrics'][d]['mae'] for d in dims}
    improved = sum(1 for d in dims if dim_delta[d] > 0)
    worst = min(dim_delta.values())

    checks = {
        'mae_improve_ge_threshold': (mae0 - maem) >= a.mae_min_improve,
        'ccc_drop_within': (cccm - ccc0) >= -a.ccc_max_drop,
        'r2_drop_within': (r2m - r20) >= -a.r2_max_drop,
        'improved_dims_ge': improved >= a.require_improved_dims,
        'worst_dim_drop_within': worst >= -a.max_dim_mae_drop,
        'best_step_not_zero': int(m1.get('step', 0)) != 0,
        'beats_rda': maem < mae1,
    }
    decision = 'GO' if all(checks.values()) else 'NO-GO'
    report = {
        'decision': decision,
        'control_mae': mae0, 'rda_mae': mae1, 'local_mae': maem,
        'local_minus_control': maem - mae0,
        'control_ccc': ccc0, 'local_ccc': cccm, 'control_r2': r20, 'local_r2': r2m,
        'dim_mae_delta': dim_delta,
        'improved_dims': improved,
        'checks': checks,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
