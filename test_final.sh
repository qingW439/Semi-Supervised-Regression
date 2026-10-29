#!/bin/bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-vadswap
ENV=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
for s in 0 1 2; do
  echo "==== rkfinal_s$s"
  "$ENV" eval_test.py --ckpt "saved_models/vad_research/rkfinal_s$s/model_best.pth" \
    --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
    --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
    --budget 500 --seed "$s" --mode calib --knn-k 20 --knn-alpha 0.4 --scale 1.2 \
    --device cuda --out "saved_models/vad_research/test_eval/rkfinal_s$s.json" 2>/dev/null | tail -6
done
