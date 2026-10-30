#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
WT=$ROOT/Semi-Supervised-Regression-source
P1ROOT=$ROOT/Semi-Supervised-Regression/saved_models/vad_research
ENV=$ROOT/envs/rankup/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$WT"
for s in 1 2; do
  name="alpha_fixed075_s$s"
  rm -rf "saved_models/vad_research/$name"
  echo "===== START $name $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_acvr.py --budget 500 --seed "$s" --steps 160 \
    --mode fixed --alpha-fixed 0.75 --p1-dir "$P1ROOT/p1_roberta_lb500_s$s" \
    --out "saved_models/vad_research/$name" --device cuda \
    --eval-knn-alpha 0.4 --eval-scale 1.2 --no-save-ckpt > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  grep -E "BEST|Traceback" "logs/$name.log" | tail -2
  sleep 3
done
echo DONE
