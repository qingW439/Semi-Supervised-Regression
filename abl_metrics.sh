#!/bin/bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-lcrankup
/root/autodl-tmp/rankup_project/envs/rankup/bin/python - <<'PY'
import glob, json, os, statistics
rows=[]
for p in glob.glob('saved_models/vad_research/abl_*/dev_metrics_best.json'):
    d=json.load(open(p)); m=d['metrics']
    ccc=statistics.mean(m[k]['ccc'] for k in ('V','A','D'))
    r2=statistics.mean(m[k]['r2'] for k in ('V','A','D'))
    rows.append((os.path.basename(os.path.dirname(p)), d['step'], m['macro_mae_original_scale'], ccc, r2))
rows.sort(key=lambda r: r[2])
for n,s,mae,ccc,r2 in rows:
    print(f"{mae:.6f} CCC={ccc:.4f} R2={r2:.4f} step={s:3d} {n}")
PY
