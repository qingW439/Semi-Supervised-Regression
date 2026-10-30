#!/bin/bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-jacfabl
/root/autodl-tmp/rankup_project/envs/rankup/bin/python - <<'PY'
import glob, json, os, statistics
print(f"{'run':20s} {'step':>4s} {'raw_MAE':>9s} {'raw_CCC':>8s} {'vrrc_MAE':>9s} {'vrrc_CCC':>8s}")
for p in sorted(glob.glob('saved_models/vad_research/abl_*_s0/dev_metrics_best.json')):
    d=json.load(open(p))
    m=d['metrics']; r=d.get('raw_metrics', m)
    def mac(x,k): return statistics.mean(x[dim][k] for dim in ('V','A','D'))
    name=os.path.basename(os.path.dirname(p))
    print(f"{name:20s} {d['step']:4d} {r['macro_mae_original_scale']:9.6f} {mac(r,'ccc'):8.4f} "
          f"{m['macro_mae_original_scale']:9.6f} {mac(m,'ccc'):8.4f}")
PY
