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
  grep -E "BEST|Traceback" "logs/$name.log" | tail -3
}
run rk0_gcf_rpl_ev   --no-arc --filter gcf --reg-target rpl --knn-calib 0.4 --eval-knn-alpha 0.4 --eval-scale 1.2
run rk001_gcf_rpl_ev --arc-loss-ratio 0.01 --filter gcf --reg-target rpl --knn-calib 0.4 --eval-knn-alpha 0.4 --eval-scale 1.2
run rk5_gcf_tau02    --arc-loss-ratio 0.05 --filter gcf --gcf-tau 0.02 --reg-target rpl --knn-calib 0.4 --eval-knn-alpha 0.4 --eval-scale 1.2
echo "ALL DONE $(date +%H:%M:%S)"
