#!/bin/bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-jacfabl
/root/autodl-tmp/rankup_project/envs/rankup/bin/python - <<'PY'
import json, statistics
MODES=['multi','anchor','seq','jacf','no_rel']
def load(mode,s):
    d=json.load(open(f'saved_models/vad_research/abl_{mode}_s{s}/dev_metrics_best.json'))
    def mac(x,k): return statistics.mean(x[dim][k] for dim in ('V','A','D'))
    r=d['raw_metrics']; v=d['metrics']
    return {'step':d['step'],
            'raw_mae':r['macro_mae_original_scale'],'raw_ccc':mac(r,'ccc'),'raw_r2':mac(r,'r2'),
            'v_mae':v['macro_mae_original_scale'],'v_ccc':mac(v,'ccc'),'v_r2':mac(v,'r2')}
res={m:[load(m,s) for s in range(3)] for m in MODES}
print(f"{'mode':8s} {'raw_MAE':>9s} {'raw_CCC':>8s} {'raw_R2':>8s} {'VRRC_MAE':>9s} {'VRRC_CCC':>8s} {'VRRC_R2':>8s}")
for m in MODES:
    rows=res[m]
    print(f"{m:8s} {statistics.mean(r['raw_mae'] for r in rows):9.6f} "
          f"{statistics.mean(r['raw_ccc'] for r in rows):8.4f} "
          f"{statistics.mean(r['raw_r2'] for r in rows):8.4f} "
          f"{statistics.mean(r['v_mae'] for r in rows):9.6f} "
          f"{statistics.mean(r['v_ccc'] for r in rows):8.4f} "
          f"{statistics.mean(r['v_r2'] for r in rows):8.4f}")
print('\n== per-seed VRRC MAE ==')
for m in MODES:
    print(m, [round(r['v_mae'],6) for r in res[m]])
print('\n== paired deltas (baseline - jacf), positive = jacf better ==')
for base in ['multi','anchor','seq','no_rel']:
    vr=[res[base][s]['v_mae']-res['jacf'][s]['v_mae'] for s in range(3)]
    raw=[res[base][s]['raw_mae']-res['jacf'][s]['raw_mae'] for s in range(3)]
    print(f'{base:7s} VRRC {[round(x,6) for x in vr]} mean={statistics.mean(vr):+.6f} | '
          f'raw {[round(x,6) for x in raw]} mean={statistics.mean(raw):+.6f}')
PY
