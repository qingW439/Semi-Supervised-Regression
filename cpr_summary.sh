#!/bin/bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-cpr
/root/autodl-tmp/rankup_project/envs/rankup/bin/python - <<'PY'
import json, statistics
BASE='/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research'
def load(p):
    d=json.load(open(p)); m=d['metrics']
    return (d['step'], m['macro_mae_original_scale'],
            statistics.mean(m[k]['ccc'] for k in ('V','A','D')),
            statistics.mean(m[k]['r2'] for k in ('V','A','D')))
def table(name, paths):
    rows=[load(p) for p in paths]
    print(f'== {name} ==')
    for i,r in enumerate(rows):
        print(f'  s{i}: MAE={r[1]:.6f} CCC={r[2]:.4f} R2={r[3]:.4f} step={r[0]}')
    print(f'  MEAN MAE={statistics.mean(r[1] for r in rows):.6f} '
          f'CCC={statistics.mean(r[2] for r in rows):.4f} '
          f'R2={statistics.mean(r[3] for r in rows):.4f}')
    return rows
c0=table('C0 P2', [f'{BASE}/p2_control_lb500_s{s}/dev_metrics_best.json' for s in range(3)])
c1=table('C1 original RankUp-3D', [f'saved_models/vad_research/c1_orig_s{s}/dev_metrics_best.json' for s in range(3)])
m1=table('M1 CPR', [f'saved_models/vad_research/m1_cpr_s{s}/dev_metrics_best.json' for s in range(3)])
a1=table('A1 CPR - unlabeled ARC', [f'saved_models/vad_research/a1_noarc_s{s}/dev_metrics_best.json' for s in range(3)])
print('== paired deltas (C0 - M1) ==')
for s in range(3):
    print(f'  s{s}: {c0[s][1]-m1[s][1]:+.6f}')
print('== paired deltas (C1 - M1) ==')
for s in range(3):
    print(f'  s{s}: {c1[s][1]-m1[s][1]:+.6f}')
print('== paired deltas (A1 - M1) ==')
for s in range(3):
    print(f'  s{s}: {a1[s][1]-m1[s][1]:+.6f}')
PY
