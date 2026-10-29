#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
COPY=$ROOT/Semi-Supervised-Regression-lcrankup
ORIG=$ROOT/Semi-Supervised-Regression-vadswap
ENV=$ROOT/envs/rankup/bin/python
P1_500=$ROOT/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0
cd "$COPY"
mkdir -p saved_models/vad_research logs
run() {
  name=$1; shift
  echo "===== START $name $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_rankup3d.py --budget 500 --seed 0 --steps 160 \
    --p1-dir "$P1_500" --out "saved_models/vad_research/$name" --device cuda \
    --arc-loss-ratio 0.05 --no-save-ckpt "$@" > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  grep -E "BEST|Traceback" "logs/$name.log" | tail -2
}
run abl_base      --pl-calib none
run abl_gcf       --pl-calib none --filter gcf
run abl_lpl       --pl-calib local --knn-k 20 --knn-alpha 0.4 --knn-tau 0.1 --knn-c0 0.3
run abl_vrrc      --pl-calib none --eval-knn-alpha 0.4 --eval-scale 1.2
run abl_gcf_lpl   --pl-calib local --knn-k 20 --knn-alpha 0.4 --knn-tau 0.1 --knn-c0 0.3 --filter gcf
run abl_gcf_vrrc  --pl-calib none --filter gcf --eval-knn-alpha 0.4 --eval-scale 1.2
run abl_lpl_vrrc  --pl-calib local --knn-k 20 --knn-alpha 0.4 --knn-tau 0.1 --knn-c0 0.3 --eval-knn-alpha 0.4 --eval-scale 1.2
run abl_full      --pl-calib local --knn-k 20 --knn-alpha 0.4 --knn-tau 0.1 --knn-c0 0.3 --filter gcf --eval-knn-alpha 0.4 --eval-scale 1.2
echo "ABLATION DONE $(date +%H:%M:%S)"
