#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
WT=$ROOT/Semi-Supervised-Regression-cpr
ORIG=$ROOT/Semi-Supervised-Regression
ENV=$ROOT/envs/rankup/bin/python
cd "$WT"
run() {
  name=$1; seed=$2; shift 2
  P1=$ORIG/saved_models/vad_research/p1_roberta_lb500_s$seed
  echo "===== START $name $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_rankup3d.py --budget 500 --seed "$seed" --steps 160 \
    --p1-dir "$P1" --out "saved_models/vad_research/$name" --device cuda \
    --arc-loss-ratio 0.05 --pl-calib rda "$@" > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  grep -E "BEST|Traceback" "logs/$name.log" | tail -2
  sleep 3
}
for s in 0 1 2; do
  run c1_orig_s$s $s --arc-pseudo-source original --no-save-ckpt
done
for s in 0 1 2; do
  run m1_cpr_s$s $s --arc-pseudo-source simreg_calibrated --mc-passes 5 --diag-arc
done
for s in 0 1 2; do
  run a1_noarc_s$s $s --arc-pseudo-source simreg_calibrated --mc-passes 5 --arc-ulb-ratio 0 --no-save-ckpt
done
echo "CPR 3SEEDS DONE $(date +%H:%M:%S)"
