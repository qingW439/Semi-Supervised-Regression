#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
WT=$ROOT/Semi-Supervised-Regression-cpr
ORIG=$ROOT/Semi-Supervised-Regression
ENV=$ROOT/envs/rankup/bin/python
P1=$ORIG/saved_models/vad_research/p1_roberta_lb500_s0
cd "$WT"
run() {
  name=$1; steps=$2
  echo "===== $name steps=$steps"
  "$ENV" scripts/vad_research/train_rankup3d.py --budget 500 --seed 0 --steps "$steps" \
    --p1-dir "$P1" --out "saved_models/vad_research/$name" --device cuda \
    --arc-loss-ratio 0.05 --pl-calib rda --arc-pseudo-source simreg_calibrated \
    --mc-passes 5 --diag-arc --no-save-ckpt > "logs/$name.log" 2>&1
  echo "rc=$?"
  grep -E "step=|CPR|Error|Traceback" "logs/$name.log" | tail -12
}
run cpr_smoke5_s0 5
run cpr_smoke20_s0 20
echo DONE
