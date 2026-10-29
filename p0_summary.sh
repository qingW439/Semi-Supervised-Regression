#!/bin/bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-lcrankup
/root/autodl-tmp/rankup_project/envs/rankup/bin/python - <<'PY'
import glob, json, os, statistics
def load(p):
    d=json.load(open(p)); m=d['metrics']
    return (d['step'], m['macro_mae_original_scale'],
            statistics.mean(m[k]['ccc'] for k in ('V','A','D')),
            statistics.mean(m[k]['r2'] for k in ('V','A','D')))
print('== ablation (lb500 s0) ==')
for p in sorted(glob.glob('saved_models/vad_research/abl_*/dev_metrics_best.json')):
    n=os.path.basename(os.path.dirname(p)); s,mae,ccc,r2=load(p)
    print(f'{n:14s} MAE={mae:.6f} CCC={ccc:.4f} R2={r2:.4f} step={s}')
print('== budget main ==')
for budget,k in ((250,10),(500,20),(1000,20)):
    rows=[]
    for s in (0,1,2):
        p=f'saved_models/vad_research/rk_lb{budget}_s{s}/dev_metrics_best.json'
        if not os.path.exists(p): continue
        st,mae,ccc,r2=load(p); rows.append((mae,ccc,r2))
        print(f'  lb{budget} s{s}: MAE={mae:.6f} CCC={ccc:.4f} R2={r2:.4f} step={st}')
    if rows:
        print(f'  lb{budget} MEAN: MAE={statistics.mean(r[0] for r in rows):.6f} '
              f'CCC={statistics.mean(r[1] for r in rows):.4f} R2={statistics.mean(r[2] for r in rows):.4f}')
PY
