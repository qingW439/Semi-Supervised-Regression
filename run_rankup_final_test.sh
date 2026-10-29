#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
COPY=$ROOT/Semi-Supervised-Regression-vadswap
ORIG=$ROOT/Semi-Supervised-Regression
ENV=$ROOT/envs/rankup/bin/python
DATA=$ROOT/data/emobank_processed
MODEL=$ROOT/pretrained/roberta-base
cd "$COPY"
for s in 0 1 2; do
  P1=$ORIG/saved_models/vad_research/p1_roberta_lb500_s$s
  echo "===== TRAIN rkfinal_s$s $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_rankup3d.py --budget 500 --seed "$s" --p1-dir "$P1" \
    --out "saved_models/vad_research/rkfinal_s$s" --steps 160 --device cuda \
    --arc-loss-ratio 0.05 --filter gcf --reg-target rpl --knn-calib 0.4 \
    --eval-knn-alpha 0.4 --eval-scale 1.2 > "logs/rkfinal_s$s.log" 2>&1
  echo "rc=$?"; grep BEST "logs/rkfinal_s$s.log" | tail -1
done
mkdir -p saved_models/vad_research/test_eval
for s in 0 1 2; do
  echo "===== TEST rkfinal_s$s"
  "$ENV" eval_test.py --ckpt "saved_models/vad_research/rkfinal_s$s/model_best.pth" \
    --data-root "$DATA" --pretrain-path "$MODEL" --budget 500 --seed "$s" \
    --mode calib --knn-k 20 --knn-alpha 0.4 --scale 1.2 --device cuda \
    --out "saved_models/vad_research/test_eval/rkfinal_s$s.json" 2>/dev/null | \
    "$ENV" -c "import json,sys; d=json.load(sys.stdin); print(f\"TEST {d['budget']} s{d['seed']}: MAE={d['macro_mae']:.6f} CCC={d['macro_ccc']:.4f} R2={d['macro_r2']:.4f}\")"
done
echo "ALL DONE $(date +%H:%M:%S)"
