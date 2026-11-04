#!/bin/bash
# E0* candidate 2: the supervised MLP trained longer (same lr/bs as P1, 25 epochs instead of 10).
# Strict OOF: fixed step count, no dev, no unlabeled data.
R=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-comp2
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
DATA=/root/autodl-tmp/rankup_project/data/emobank_processed
PRE=/root/autodl-tmp/rankup_project/pretrained/roberta-base
MAIN=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
STEPS=${STEPS:-640}
TAG=${TAG:-s640}
cd $R || exit 1
mkdir -p outputs/oof_sup_${TAG} outputs/auto_search/logs
for S in 0 1 2; do
  if [ "$S" = "0" ]; then MAN=$MAIN/outputs/p0/fold_manifest.json
  else MAN=$MAIN/outputs/p0_s${S}/fold_manifest.json; fi
  for F in 0 1 2 3 4; do
    OUT=outputs/oof_sup_${TAG}/seed${S}_fold${F}
    if [ -f "$OUT/sup_predictions.parquet" ]; then echo "SKIP $OUT"; continue; fi
    echo "=== sup-$TAG seed$S fold$F $(date +%H:%M:%S) ==="
    $PY scripts/vad_research/audit_sup_recipe_fold.py --fold $F --manifest $MAN --data-seed $S \
      --data-root $DATA --pretrain-path $PRE --out-dir $OUT --steps $STEPS --device cuda \
      >> outputs/auto_search/logs/sup_${TAG}_oof.log 2>&1
    tail -1 outputs/auto_search/logs/sup_${TAG}_oof.log
  done
done
echo "DONE $(date +%H:%M:%S)"
