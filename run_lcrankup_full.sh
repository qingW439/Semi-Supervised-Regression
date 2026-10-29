#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
COPY=$ROOT/Semi-Supervised-Regression-lcrankup
ORIG=$ROOT/Semi-Supervised-Regression
ENV=$ROOT/envs/rankup/bin/python
P1=$ORIG/saved_models/vad_research/p1_roberta_lb500_s0
cd "$COPY"
run() {
  name=$1; shift
  echo "===== START $name $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_rankup3d.py --budget 500 --seed 0 --steps 160 \
    --p1-dir "$P1" --out "saved_models/vad_research/$name" --device cuda \
    --arc-loss-ratio 0.05 "$@" > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  grep -E "BEST|Traceback" "logs/$name.log" | tail -2
}
run lcrankup_c0_s0    --pl-calib none
run lcrankup_c1_rda_s0 --pl-calib rda
run lcrankup_m1_s0    --pl-calib local --knn-k 20 --knn-alpha 0.4 --knn-tau 0.1 --knn-c0 0.3
echo "FULL DONE $(date +%H:%M:%S)"
