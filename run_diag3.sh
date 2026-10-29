#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
COPY=$ROOT/Semi-Supervised-Regression-vadswap
ENV=$ROOT/envs/rankup/bin/python
cd "$COPY"
run_one() {
  seed=$1; ckpt=$2; out=$3
  "$ENV" diag_knn.py --ckpt "$ckpt" \
    --data-root "$ROOT/data/emobank_processed" \
    --pretrain-path "$ROOT/pretrained/roberta-base" \
    --budget 500 --seed "$seed" --device cuda > "$out" 2>/dev/null
  echo "done $out"
}
run_one 0 saved_models/vad_research/knn_a04/model_best.pth diag_knn_a04_s0.txt
run_one 1 saved_models/vad_research/knn_a04_s1/model_best.pth diag_knn_a04_s1.txt
run_one 2 saved_models/vad_research/knn_a04_s2/model_best.pth diag_knn_a04_s2.txt
echo ALLDONE
