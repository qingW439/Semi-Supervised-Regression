#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
COPY=$ROOT/Semi-Supervised-Regression-vadswap
ORIG=$ROOT/Semi-Supervised-Regression
ENV=$ROOT/envs/rankup/bin/python
cd "$COPY"
run() {
  name=$1; seed=$2; shift 2
  P1=$ORIG/saved_models/vad_research/p1_roberta_lb500_s$seed
  echo "===== START $name seed=$seed $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_vad_modules.py \
    --budget 500 --seed "$seed" --p1-dir "$P1" \
    --out "saved_models/vad_research/$name" --device cuda "$@" \
    > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  grep -E "BEST|Traceback" "logs/$name.log" | tail -2
}
run combo_a04_ema999 0 --mode mean_teacher --knn-calib 0.4 --knn-k 20 --ema-decay 0.999 --diag-teacher
run combo_a06_ema999 0 --mode mean_teacher --knn-calib 0.6 --knn-k 20 --ema-decay 0.999 --diag-teacher
run knn_a04_s1 1 --mode mean_teacher --knn-calib 0.4 --knn-k 20 --diag-teacher
run knn_a04_s2 2 --mode mean_teacher --knn-calib 0.4 --knn-k 20 --diag-teacher
run mt_ema999_s1 1 --mode mean_teacher --ema-decay 0.999 --diag-teacher
run mt_ema999_s2 2 --mode mean_teacher --ema-decay 0.999 --diag-teacher
echo "ALL DONE $(date +%H:%M:%S)"
