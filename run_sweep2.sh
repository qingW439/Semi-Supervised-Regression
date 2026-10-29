#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
COPY=$ROOT/Semi-Supervised-Regression-vadswap
ORIG=$ROOT/Semi-Supervised-Regression
ENV=$ROOT/envs/rankup/bin/python
P1=$ORIG/saved_models/vad_research/p1_roberta_lb500_s0
cd "$COPY"
run() {
  name=$1; shift
  echo "===== START $name $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_vad_modules.py \
    --mode mean_teacher --budget 500 --seed 0 --p1-dir "$P1" \
    --out "saved_models/vad_research/$name" --device cuda "$@" \
    > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  grep -E "BEST|Error|Traceback" "logs/$name.log" | tail -3
}
run knn_a02 --knn-calib 0.2 --knn-k 20 --diag-teacher
run knn_a04 --knn-calib 0.4 --knn-k 20 --diag-teacher
run knn_a06 --knn-calib 0.6 --knn-k 20 --diag-teacher
run mt_ema999 --ema-decay 0.999 --diag-teacher
echo "ALL DONE $(date +%H:%M:%S)"
