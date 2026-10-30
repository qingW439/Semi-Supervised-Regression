#!/bin/bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-source
/root/autodl-tmp/rankup_project/envs/rankup/bin/python - <<'PY'
import json, statistics
def load(p):
    d=json.load(open(p)); m=d['metrics']; r=d.get('raw_metrics',m)
    def mac(x,k): return statistics.mean(x[dim][k] for dim in ('V','A','D'))
    return d['step'], r['macro_mae_original_scale'], mac(r,'ccc'), mac(r,'r2'), m['macro_mae_original_scale'], mac(m,'ccc'), mac(m,'r2')
for name in ('alpha_fixed075_s0','alpha_shift05_s0'):
    s,rm,rc,rr,vm,vc,vr=load(f'saved_models/vad_research/{name}/dev_metrics_best.json')
    print(f"{name}: step={s} raw=({rm:.6f},{rc:.4f},{rr:.4f}) vrrc=({vm:.6f},{vc:.4f},{vr:.4f})")
PY
