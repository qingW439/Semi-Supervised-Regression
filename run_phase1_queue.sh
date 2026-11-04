#!/bin/bash
# Phase-1 queue: wait for the ACLDP OOF matrix to finish, then run the E0* candidates.
# Each candidate: 3 seeds x 5 folds, strict OOF, no checkpoints written.
R=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-comp2
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
DATA=/root/autodl-tmp/rankup_project/data/emobank_processed
PRE=/root/autodl-tmp/rankup_project/pretrained/roberta-base
MAIN=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
cd $R || exit 1
mkdir -p outputs/auto_search/logs

run_recipe () {   # tag steps lr bs
  local TAG=$1 STEPS=$2 LR=$3 BS=$4
  echo "##### recipe $TAG steps=$STEPS lr=$LR bs=$BS  $(date +%H:%M:%S)" \
    >> outputs/auto_search/logs/phase1_queue.log
  for S in 0 1 2; do
    if [ "$S" = "0" ]; then MAN=$MAIN/outputs/p0/fold_manifest.json
    else MAN=$MAIN/outputs/p0_s${S}/fold_manifest.json; fi
    for F in 0 1 2 3 4; do
      OUT=outputs/oof_sup_${TAG}/seed${S}_fold${F}
      if [ -f "$OUT/sup_predictions.parquet" ]; then echo "SKIP $OUT" \
        >> outputs/auto_search/logs/phase1_queue.log; continue; fi
      $PY scripts/vad_research/audit_sup_recipe_fold.py --fold $F --manifest $MAN \
        --data-seed $S --data-root $DATA --pretrain-path $PRE --out-dir $OUT \
        --steps $STEPS --lr $LR --batch-size $BS --device cuda \
        >> outputs/auto_search/logs/sup_${TAG}.log 2>&1
      echo "  $TAG seed$S fold$F $(tail -1 outputs/auto_search/logs/sup_${TAG}.log)" \
        >> outputs/auto_search/logs/phase1_queue.log
    done
  done
}

while pgrep -f run_acldp_oof.sh > /dev/null; do sleep 20; done
echo "ACLDP matrix finished $(date +%H:%M:%S), free: $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}')" \
  >> outputs/auto_search/logs/phase1_queue.log

run_recipe 640_lr1e4_bs64  640  1e-4 64
run_recipe 1600_lr1e4_bs64 1600 1e-4 64
run_recipe 640_lr2e5_bs16  640  2e-5 16
echo "PHASE1 QUEUE DONE $(date +%H:%M:%S)" >> outputs/auto_search/logs/phase1_queue.log
