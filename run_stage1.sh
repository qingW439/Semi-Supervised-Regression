#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
WT=$ROOT/Semi-Supervised-Regression-lcrankup
P1=$ROOT/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0
ENV=$ROOT/envs/rankup/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WT"
run() {
  name=$1; shift
  echo "===== START $name $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_rankup3d.py --budget 500 --seed 0 --steps 160 \
    --p1-dir "$P1" --out "saved_models/vad_research/$name" --device cuda \
    --arc-loss-ratio 0.05 --filter gcf --eval-knn-alpha 0.4 --eval-scale 1.2 \
    --no-save-ckpt "$@" > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  grep -E "BEST|Traceback" "logs/$name.log" | tail -2
  sleep 4
}
# B0 = abl_full (exists): ARC + local + GCF + VRRC
run b1_noarc_local --no-arc --pl-calib local --knn-k 20 --knn-alpha 0.4 --knn-tau 0.1 --knn-c0 0.3
run b2_arc_rda     --arc --pl-calib rda
run b3_noarc_rda   --no-arc --pl-calib rda
echo "STAGE1 DONE $(date +%H:%M:%S)"
