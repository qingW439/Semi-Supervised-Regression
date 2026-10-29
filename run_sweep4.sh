#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
COPY=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-vadswap
ORIG=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression
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
for s in 0 1 2; do
  run final_full_s$s $s --mode mean_teacher --knn-calib 0.4 --knn-k 20 \
      --eval-knn-alpha 0.4 --eval-knn-k 20 --eval-scale 1.2 --diag-teacher
done
for s in 0 1 2; do
  run infer_only_s$s $s --mode control \
      --eval-knn-alpha 0.4 --eval-knn-k 20 --eval-scale 1.2 --diag-teacher
done
echo "ALL DONE $(date +%H:%M:%S)"
