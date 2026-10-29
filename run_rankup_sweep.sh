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
    --out "saved_models/vad_research/$name" --steps 160 --device cuda "$@" \
    > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  grep -E "BEST|Traceback|Error" "logs/$name.log" | tail -3
}
run rk_rda        --reg-target rda
run rk_rda_rpl    --reg-target rda --knn-calib 0.4
run rk_rda_eval   --reg-target rda --eval-knn-alpha 0.4 --eval-scale 1.2
run rk_gcf_raw    --filter gcf --reg-target raw
run rk_gcf_rpl    --filter gcf --reg-target rpl --knn-calib 0.4
run rk_gcf_rda    --filter gcf --reg-target rda --knn-calib 0.4
echo "ALL DONE $(date +%H:%M:%S)"
