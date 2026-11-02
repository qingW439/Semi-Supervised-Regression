set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
for w in 1 10 50; do
  out=saved_models/vad_research/rda_${w}_s0
  if [ -f "$out/model_step96.pth" ]; then echo "SKIP $w"; continue; fi
  echo "=== RDA weight $w $(date) ==="
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed 0 --steps 160 \
    --mode fixed --rel-mode default --alpha-fixed 0.75 --lambda-u 1.0 \
    --rda-weight "$w" --save-at-step 96 --no-save-ckpt \
    --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0 \
    --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda --out "$out"
done > outputs/alpha_sweep/rda_train.log 2>&1
echo ALL_DONE
