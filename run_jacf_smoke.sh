#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
WT=$ROOT/Semi-Supervised-Regression-acvr
P1=$ROOT/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0
ENV=$ROOT/envs/rankup/bin/python
cd "$WT"
run() {
  name=$1; steps=$2
  echo "===== $name steps=$steps"
  "$ENV" scripts/vad_research/train_acvr.py --budget 500 --seed 0 --steps "$steps" \
    --p1-dir "$P1" --out "saved_models/vad_research/$name" --device cuda \
    --eval-knn-alpha 0.4 --eval-scale 1.2 --no-save-ckpt > "logs/$name.log" 2>&1
  echo "rc=$?"
  grep -E "step=|BEST|Traceback" "logs/$name.log" | tail -6
}
run jacf_smoke5_s0 5
run jacf_smoke20_s0 20
echo DONE
