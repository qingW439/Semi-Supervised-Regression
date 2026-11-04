#!/bin/bash
# Phase-1b: can the neural readout (same architecture as ridge: frozen RoBERTa + linear
# head) match the closed-form ridge solution once it is trained properly?
# Pure supervised (lambda_u=0) so the comparison isolates the training recipe.
R=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-comp2
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
DATA=/root/autodl-tmp/rankup_project/data/emobank_processed
PRE=/root/autodl-tmp/rankup_project/pretrained/roberta-base
P1=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0
cd $R || exit 1
mkdir -p outputs/auto_search/logs saved_models/vad_research
run () {  # name steps lr batch
  local NAME=$1 STEPS=$2 LR=$3 BS=$4
  local OUT=saved_models/vad_research/$NAME
  if [ -d "$OUT" ]; then echo "SKIP $NAME"; return; fi
  echo "=== $NAME steps=$STEPS lr=$LR bs=$BS $(date +%H:%M:%S) ==="
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed 0 --steps $STEPS \
    --lr $LR --batch-size $BS --eval-every $((STEPS/10)) \
    --mode fixed --alpha-fixed 0.75 --lambda-u 0.0 --sup-loss l1 \
    --p1-dir $P1 --data-root $DATA --pretrain-path $PRE \
    --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda --no-save-ckpt \
    --out $OUT >> outputs/auto_search/logs/recipe_test.log 2>&1
  grep -E '^BEST' outputs/auto_search/logs/recipe_test.log | tail -1
}
run recipe_l1_s640_lr5e6_bs16  640 5e-6 16
run recipe_l1_s640_lr1e4_bs16  640 1e-4 16
run recipe_l1_s640_lr1e4_bs64  640 1e-4 64
run recipe_l1_s1600_lr1e4_bs64 1600 1e-4 64
echo "DONE $(date +%H:%M:%S)"
