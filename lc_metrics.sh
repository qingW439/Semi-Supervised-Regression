#!/bin/bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-lcrankup
/root/autodl-tmp/rankup_project/envs/rankup/bin/python - <<'PY'
import json
for name in ('lcrankup_c0_s0','lcrankup_c1_rda_s0','lcrankup_m1_s0'):
    d=json.load(open(f'saved_models/vad_research/{name}/dev_metrics_best.json'))
    m=d['metrics']
    ccc=sum(m[k]['ccc'] for k in ('V','A','D'))/3
    r2=sum(m[k]['r2'] for k in ('V','A','D'))/3
    print(f"{name}: step={d['step']} MAE={m['macro_mae_original_scale']:.6f} CCC={ccc:.4f} R2={r2:.4f}")
PY
