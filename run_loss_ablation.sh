#!/bin/bash
# Phase-1 loss-function diagnostic: does the L1-only objective handicap the base recipe?
R=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-comp2
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
DATA=/root/autodl-tmp/rankup_project/data/emobank_processed
PRE=/root/autodl-tmp/rankup_project/pretrained/roberta-base
P1=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0
cd $R || exit 1
mkdir -p outputs/auto_search/logs saved_models/vad_research
for L in l1 mse huber; do
  for CFG in sup jacf; do
    if [ "$CFG" = "sup" ]; then LU=0.0; else LU=1.0; fi
    OUT=saved_models/vad_research/abl_${L}_${CFG}_s0
    if [ -d "$OUT" ]; then echo "SKIP $OUT"; continue; fi
    echo "=== RUN $L $CFG $(date +%H:%M:%S) ==="
    $PY scripts/vad_research/train_acvr.py --budget 500 --seed 0 --steps 160 \
      --mode fixed --alpha-fixed 0.75 --lambda-u $LU \
      --sup-loss $L --unsup-loss l1 --huber-delta 0.1 \
      --p1-dir $P1 --data-root $DATA --pretrain-path $PRE \
      --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda --no-save-ckpt \
      --out $OUT >> outputs/auto_search/logs/loss_ablation.log 2>&1
    grep -E '^BEST' outputs/auto_search/logs/loss_ablation.log | tail -1
  done
done
echo "DONE $(date +%H:%M:%S)"

# summarise raw metrics of every abl_* dir
$PY - <<'PYEOF'
import glob, json, os
import numpy as np
rows = []
for d in sorted(glob.glob('saved_models/vad_research/abl_*_s0')):
    f = os.path.join(d, 'dev_metrics_best.json')
    if not os.path.exists(f):
        continue
    j = json.load(open(f)); raw = j['raw_metrics']
    dims = ('V', 'A', 'D')
    m = {k: float(np.mean([raw[x][k] for x in dims])) for k in ('mae', 'pearson', 'ccc', 'r2')}
    rows.append((os.path.basename(d), j['step'], m))
print(f"{'run':26s} {'step':>4s} {'MAE':>8s} {'r':>7s} {'CCC':>7s} {'R2':>7s}")
for n, s, m in rows:
    print(f"{n:26s} {s:4d} {m['mae']:8.5f} {m['pearson']:7.4f} {m['ccc']:7.4f} {m['r2']:7.4f}")
PYEOF
