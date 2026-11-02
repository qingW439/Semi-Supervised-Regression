set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
M0=saved_models/vad_research/alpha_sweep/sup_s0/model_step96.pth
for w in 0.02 0.05; do
  tag=$(echo "$w" | tr -d '.')
  out=saved_models/vad_research/diff_s${tag}_s0
  if [ -f "$out/model_step96.pth" ]; then echo "SKIP $w"; continue; fi
  echo "=== DIFF small $w $(date) ==="
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed 0 --steps 160 \
    --mode fixed --rel-mode default --alpha-fixed 0.75 --lambda-u 1.0 \
    --m0-ckpt "$M0" --diff-weight "$w" \
    --save-at-step 96 --no-save-ckpt \
    --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0 \
    --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda --out "$out"
done > outputs/alpha_sweep/diff_small.log 2>&1
echo ALL_DONE
