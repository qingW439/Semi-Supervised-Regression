#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
COPY=$ROOT/Semi-Supervised-Regression-vadswap
ORIG=$ROOT/Semi-Supervised-Regression
ENV=$ROOT/envs/rankup/bin/python
DATA=$ROOT/data/emobank_processed
MODEL=$ROOT/pretrained/roberta-base
cd "$COPY"
mkdir -p saved_models/vad_research/test_eval

ev() {
  name=$1; ckpt=$2; budget=$3; seed=$4; mode=$5; k=$6; alpha=$7; scale=$8
  echo "===== $name"
  "$ENV" eval_test.py --ckpt "$ckpt" --data-root "$DATA" --pretrain-path "$MODEL" \
    --budget "$budget" --seed "$seed" --mode "$mode" \
    --knn-k "$k" --knn-alpha "$alpha" --scale "$scale" --device cuda \
    --out "saved_models/vad_research/test_eval/$name.json" 2>/dev/null | \
    "$ENV" -c "import json,sys; d=json.load(sys.stdin); print(f\"TEST {d['budget']} s{d['seed']} {d['mode']}: MAE={d['macro_mae']:.6f} CCC={d['macro_ccc']:.4f} R2={d['macro_r2']:.4f}\")"
}

for s in 0 1 2; do
  ev ctrl_lb250_s$s saved_models/vad_research/ctrl_lb250_s$s/model_best.pth 250 $s raw 0 0 1.0
done
for s in 0 1 2; do
  ev rcmt_lb250_s$s saved_models/vad_research/rcmt_lb250_s$s/model_best.pth 250 $s calib 10 0.4 1.2
done
for s in 0 1 2; do
  ev ctrl_lb500_s$s $ORIG/saved_models/vad_research/p2_control_lb500_s$s/model_best.pth 500 $s raw 0 0 1.0
done
for s in 0 1 2; do
  ev rcmt_lb500_s$s saved_models/vad_research/final_full_s$s/model_best.pth 500 $s calib 20 0.4 1.2
done
for s in 0 1 2; do
  ev ctrl_lb1000_s$s saved_models/vad_research/ctrl_lb1000_s$s/model_best.pth 1000 $s raw 0 0 1.0
done
for s in 0 1 2; do
  ev rcmt_lb1000_s$s saved_models/vad_research/rcmt_lb1000_s$s/model_best.pth 1000 $s calib 20 0.4 1.2
done
echo "TEST EVAL DONE"
