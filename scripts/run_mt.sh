set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
for s in 0 1 2; do
  P1=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s${s}
  out=saved_models/vad_research/alpha_sweep/mt_s${s}
  echo "=== MT seed $s $(date) ==="
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed "$s" --steps 160 \
    --mode teacher --rel-mode default --lambda-u 1.0 \
    --save-at-step 96 --no-save-ckpt \
    --p1-dir "$P1" --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda \
    --out "$out"
done > outputs/alpha_sweep/mt_train.log 2>&1
echo ALL_DONE
