#!/bin/bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-vadswap
/root/autodl-tmp/rankup_project/envs/rankup/bin/python - <<'PY'
import json, statistics
rows=[]
for s in (0,1,2):
    d=json.load(open(f'saved_models/vad_research/test_eval/rkfinal_s{s}.json'))
    m=d['metrics']
    ccc=statistics.mean(m[k]['ccc'] for k in ('V','A','D'))
    r2=statistics.mean(m[k]['r2'] for k in ('V','A','D'))
    rows.append((s,m['macro_mae_original_scale'],ccc,r2))
    print(f"s{s}: MAE={m['macro_mae_original_scale']:.6f} CCC={ccc:.4f} R2={r2:.4f}")
print('mean MAE', statistics.mean(r[1] for r in rows))
print('mean CCC', statistics.mean(r[2] for r in rows))
print('mean R2 ', statistics.mean(r[3] for r in rows))
PY
