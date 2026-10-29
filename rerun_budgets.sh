#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
WT=$ROOT/Semi-Supervised-Regression-lcrankup
P1ROOT=$ROOT/Semi-Supervised-Regression-vadswap
ENV=$ROOT/envs/rankup/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WT"
run_b() {
  name=$1; budget=$2; seed=$3; k=$4
  P1=$P1ROOT/saved_models/vad_research/p1_roberta_lb${budget}_s${seed}
  rm -rf "saved_models/vad_research/$name"
  echo "===== START $name $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_rankup3d.py --budget "$budget" --seed "$seed" \
    --p1-dir "$P1" --out "saved_models/vad_research/$name" --steps 160 --device cuda \
    --arc-loss-ratio 0.05 --pl-calib local --knn-k "$k" --knn-alpha 0.4 --knn-tau 0.1 --knn-c0 0.3 \
    --filter gcf --eval-knn-alpha 0.4 --eval-knn-k "$k" --eval-scale 1.2 \
    > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  grep -E "BEST|Traceback" "logs/$name.log" | tail -2
  sleep 5
}
run_b rk_lb250_s2 250 2 10
run_b rk_lb1000_s0 1000 0 20
run_b rk_lb1000_s1 1000 1 20
echo "RERUN DONE $(date +%H:%M:%S)"
