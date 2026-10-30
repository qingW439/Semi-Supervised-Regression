#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
WT=$ROOT/Semi-Supervised-Regression-source
P1=$ROOT/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0
ENV=$ROOT/envs/rankup/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WT"
run() {
  name=$1; shift
  rm -rf "saved_models/vad_research/$name"
  echo "===== START $name $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_acvr.py --budget 500 --seed 0 --steps 160 \
    --p1-dir "$P1" --out "saved_models/vad_research/$name" --device cuda \
    --eval-knn-alpha 0.4 --eval-scale 1.2 --save-ckpt "$@" > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  grep -E "BEST|Traceback" "logs/$name.log" | tail -2
  sleep 4
}
run alpha_fixed075_s0 --mode fixed --alpha-fixed 0.75
run alpha_shift05_s0 --mode shift --alpha-shift 0.5
echo DONE
