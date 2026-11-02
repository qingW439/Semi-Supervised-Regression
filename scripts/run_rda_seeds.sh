set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
for s in 1 2; do
  out=saved_models/vad_research/rda_0.5_s${s}
  if [ -f "$out/model_step96.pth" ]; then echo "SKIP $s"; continue; fi
  echo "=== RDA 0.5 seed $s $(date) ==="
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed "$s" --steps 160 \
    --mode fixed --rel-mode default --alpha-fixed 0.75 --lambda-u 1.0 \
    --rda-weight 0.5 --save-at-step 96 --no-save-ckpt \
    --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s${s} \
    --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda --out "$out"
done > outputs/alpha_sweep/rda_seeds.log 2>&1
echo ALL_DONE
