#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
COPY=$ROOT/Semi-Supervised-Regression-vadswap
ORIG=$ROOT/Semi-Supervised-Regression
ENV=$ROOT/envs/rankup/bin/python
P1=$ORIG/saved_models/vad_research/p1_roberta_lb500_s0
cd "$COPY"
run() {
  name=$1; shift
  echo "===== START $name $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_rankup3d.py --budget 500 --seed 0 --p1-dir "$P1" \
    --out "saved_models/vad_research/$name" --steps 160 --no-save-ckpt --device cuda "$@" \
    > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  grep -E "BEST|Traceback|Error" "logs/$name.log" | tail -3
}
run rk_noarc_raw   --no-arc --reg-target raw
run rk_noarc_rda   --no-arc --reg-target rda
run rk_arc01_raw   --arc-loss-ratio 0.1 --reg-target raw
run rk_arc01_rda   --arc-loss-ratio 0.1 --reg-target rda
run rk_arc01_rpl   --arc-loss-ratio 0.1 --reg-target rpl --knn-calib 0.4
run rk_arc005_rda  --arc-loss-ratio 0.05 --reg-target rda --eval-knn-alpha 0.4 --eval-scale 1.2
echo "ALL DONE $(date +%H:%M:%S)"
