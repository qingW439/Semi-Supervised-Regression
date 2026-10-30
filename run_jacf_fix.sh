#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
AC=$ROOT/Semi-Supervised-Regression-acvr
P1ROOT=$ROOT/Semi-Supervised-Regression/saved_models/vad_research
ENV=$ROOT/envs/rankup/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$AC"
for s in 0 1 2; do
  rm -rf "saved_models/vad_research/jacf_fix_s$s"
  echo "===== JACF fixed s$s"
  "$ENV" scripts/vad_research/train_acvr.py --budget 500 --seed "$s" --steps 160 \
    --p1-dir "$P1ROOT/p1_roberta_lb500_s$s" --out "saved_models/vad_research/jacf_fix_s$s" \
    --device cuda --eval-knn-alpha 0.4 --eval-scale 1.2 --no-save-ckpt \
    > "logs/jacf_fix_s$s.log" 2>&1
  echo "rc=$?"; grep BEST "logs/jacf_fix_s$s.log" | tail -1
  sleep 4
done
echo DONE
