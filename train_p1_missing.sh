#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
COPY=$ROOT/Semi-Supervised-Regression-vadswap
ENV=$ROOT/envs/rankup/bin/python
MODEL=$ROOT/pretrained/roberta-base
cd "$COPY"
mkdir -p logs
for b in 250 1000; do
  for s in 0 1 2; do
    out="saved_models/vad_research/p1_roberta_lb${b}_s${s}"
    if [ -f "$out/model_best.pth" ]; then
      echo "SKIP $out (exists)"
      continue
    fi
    echo "===== P1 budget=$b seed=$s start $(date +%H:%M:%S)"
    "$ENV" scripts/vad_research/train_roberta_vad_supervised.py \
      --budget "$b" --seed "$s" --device cuda --offline \
      --model "$MODEL" --out "$out" > "logs/p1_lb${b}_s${s}.log" 2>&1
    echo "===== P1 budget=$b seed=$s rc=$?"
    grep -E "BEST|Traceback" "logs/p1_lb${b}_s${s}.log" | tail -2
  done
done
echo "P1 ALL DONE $(date +%H:%M:%S)"
