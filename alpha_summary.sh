#!/bin/bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-source
/root/autodl-tmp/rankup_project/envs/rankup/bin/python - <<'PY'
import json, statistics
BASE='/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-jacfabl/saved_models/vad_research'
def load(p):
    d=json.load(open(p)); m=d['metrics']; r=d.get('raw_metrics',m)
    def mac(x,k): return statistics.mean(x[dim][k] for dim in ('V','A','D'))
    return {'raw':(r['macro_mae_original_scale'],mac(r,'ccc'),mac(r,'r2')),
            'v':(m['macro_mae_original_scale'],mac(m,'ccc'),mac(m,'r2'))}
rows={'fixed':[load(f'saved_models/vad_research/alpha_fixed075_s{s}/dev_metrics_best.json') for s in range(3)],
      'anchor':[load(f'{BASE}/abl_anchor_s{s}/dev_metrics_best.json') for s in range(3)],
      'jacf':[load(f'{BASE}/abl_jacf_s{s}/dev_metrics_best.json') for s in range(3)]}
for name,rs in rows.items():
    print(name, 'raw', tuple(round(statistics.mean(r['raw'][i] for r in rs),6) for i in range(3)),
          'vrrc', tuple(round(statistics.mean(r['v'][i] for r in rs),6) for i in range(3)))
print('fixed per-seed vrrc mae', [round(r['v'][0],6) for r in rows['fixed']])
print('anchor per-seed vrrc mae', [round(r['v'][0],6) for r in rows['anchor']])
print('jacf per-seed vrrc mae', [round(r['v'][0],6) for r in rows['jacf']])
PY
