#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
WT=$ROOT/Semi-Supervised-Regression-lcrankup
DATA_P1=$ROOT/Semi-Supervised-Regression-vadswap
ENV=$ROOT/envs/rankup/bin/python
cd "$WT"
run_b() {
  name=$1; budget=$2; seed=$3; k=$4
  P1=$DATA_P1/saved_models/vad_research/p1_roberta_lb${budget}_s${seed}
  echo "===== START $name $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_rankup3d.py --budget "$budget" --seed "$seed" \
    --p1-dir "$P1" --out "saved_models/vad_research/$name" --steps 160 --device cuda \
    --arc-loss-ratio 0.05 --pl-calib local --knn-k "$k" --knn-alpha 0.4 --knn-tau 0.1 --knn-c0 0.3 \
    --filter gcf --eval-knn-alpha 0.4 --eval-knn-k "$k" --eval-scale 1.2 \
    > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  grep -E "BEST|Traceback" "logs/$name.log" | tail -2
}
for s in 0 1 2; do run_b rk_lb250_s$s 250 $s 10; done
for s in 0 1 2; do run_b rk_lb1000_s$s 1000 $s 20; done
echo "BUDGETS DONE $(date +%H:%M:%S)"
