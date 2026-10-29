#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
COPY=$ROOT/Semi-Supervised-Regression-lcrankup
ORIG=$ROOT/Semi-Supervised-Regression
ENV=$ROOT/envs/rankup/bin/python
P1=$ORIG/saved_models/vad_research/p1_roberta_lb500_s0
cd "$COPY"
mkdir -p saved_models/vad_research logs
run() {
  name=$1; steps=$2; shift 2
  echo "===== START $name steps=$steps $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_rankup3d.py --budget 500 --seed 0 --steps "$steps" \
    --p1-dir "$P1" --out "saved_models/vad_research/$name" --device cuda "$@" \
    > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  tail -3 "logs/$name.log"
}
run lcrankup_smoke5_s0  5  --pl-calib local --arc-loss-ratio 0.05 --knn-k 20 --knn-alpha 0.4 --knn-tau 0.1 --knn-c0 0.3 --no-save-ckpt
run lcrankup_smoke20_s0 20 --pl-calib local --arc-loss-ratio 0.05 --knn-k 20 --knn-alpha 0.4 --knn-tau 0.1 --knn-c0 0.3 --no-save-ckpt
echo "SMOKE DONE $(date +%H:%M:%S)"
