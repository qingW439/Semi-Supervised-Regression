#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
WT=$ROOT/Semi-Supervised-Regression-lcrankup
P1ROOT=$ROOT/Semi-Supervised-Regression
ENV=$ROOT/envs/rankup/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WT"
run_b() {
  name=$1; seed=$2
  P1=$P1ROOT/saved_models/vad_research/p1_roberta_lb500_s${seed}
  rm -rf "saved_models/vad_research/$name"
  echo "===== START $name $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_rankup3d.py --budget 500 --seed "$seed" \
    --p1-dir "$P1" --out "saved_models/vad_research/$name" --steps 160 --device cuda \
    --arc-loss-ratio 0.05 --pl-calib local --knn-k 20 --knn-alpha 0.4 --knn-tau 0.1 --knn-c0 0.3 \
    --filter gcf --eval-knn-alpha 0.4 --eval-knn-k 20 --eval-scale 1.2 \
    > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  grep -E "BEST|Traceback" "logs/$name.log" | tail -2
  sleep 5
}
run_b rk_lb500_s1 1
run_b rk_lb500_s2 2
echo "LB500 DONE $(date +%H:%M:%S)"
