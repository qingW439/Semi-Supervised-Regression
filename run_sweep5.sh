#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
COPY=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-vadswap
ORIG=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression
ENV=$ROOT/envs/rankup/bin/python
cd "$COPY"
run() {
  name=$1; budget=$2; seed=$3; shift 3
  P1=$COPY/saved_models/vad_research/p1_roberta_lb${budget}_s${seed}
  [ -f "$P1/model_best.pth" ] || P1=$ORIG/saved_models/vad_research/p1_roberta_lb${budget}_s${seed}
  echo "===== START $name $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_vad_modules.py \
    --budget "$budget" --seed "$seed" --p1-dir "$P1" \
    --out "saved_models/vad_research/$name" --device cuda "$@" \
    > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  grep -E "BEST|Traceback" "logs/$name.log" | tail -2
}
# 250: k=10 (smaller labeled bank)
for s in 0 1 2; do
  run ctrl_lb250_s$s 250 $s --mode control
done
for s in 0 1 2; do
  run rcmt_lb250_s$s 250 $s --mode mean_teacher --knn-calib 0.4 --knn-k 10 \
      --eval-knn-alpha 0.4 --eval-knn-k 10 --eval-scale 1.2 --diag-teacher
done
# 1000: k=20
for s in 0 1 2; do
  run ctrl_lb1000_s$s 1000 $s --mode control
done
for s in 0 1 2; do
  run rcmt_lb1000_s$s 1000 $s --mode mean_teacher --knn-calib 0.4 --knn-k 20 \
      --eval-knn-alpha 0.4 --eval-knn-k 20 --eval-scale 1.2 --diag-teacher
done
echo "ALL DONE $(date +%H:%M:%S)"
