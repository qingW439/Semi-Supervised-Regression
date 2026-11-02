set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
for spec in "0.9,0.62,0.53:g2" "0.8,0.55,0.45:g3"; do
  g=${spec%%:*}; tag=${spec##*:}
  out=saved_models/vad_research/lvdp_${tag}_s0
  if [ -f "$out/model_step96.pth" ]; then echo "SKIP $tag"; continue; fi
  echo "=== LVDP $tag gamma=$g ==="
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed 0 --steps 160 \
    --mode fixed --gamma-dims "$g" --lambda-u 1.0 \
    --save-at-step 96 --no-save-ckpt \
    --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0 \
    --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda --out "$out"
done > outputs/alpha_sweep/lvdp_grid.log 2>&1
echo ALL_DONE
