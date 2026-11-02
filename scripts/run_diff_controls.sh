set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
for s in 0 1 2; do
  out=saved_models/vad_research/diff_shuf05_s${s}
  if [ ! -f "$out/model_step96.pth" ]; then
    echo "=== SHUF 0.05 seed $s $(date) ==="
    $PY scripts/vad_research/train_acvr.py --budget 500 --seed "$s" --steps 160 \
      --mode fixed --rel-mode default --alpha-fixed 0.75 --lambda-u 1.0 \
      --m0-ckpt saved_models/vad_research/alpha_sweep/sup_s${s}/model_step96.pth \
      --diff-weight 0.05 --diff-shuffle --save-at-step 96 --no-save-ckpt \
      --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s${s} \
      --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda --out "$out"
  fi
  out=saved_models/vad_research/diff_center05_s${s}
  if [ ! -f "$out/model_step96.pth" ]; then
    echo "=== CENTER 0.05 seed $s $(date) ==="
    $PY scripts/vad_research/train_acvr.py --budget 500 --seed "$s" --steps 160 \
      --mode fixed --rel-mode default --alpha-fixed 0.75 --lambda-u 1.0 \
      --m0-ckpt saved_models/vad_research/alpha_sweep/sup_s${s}/model_step96.pth \
      --diff-weight 0.05 --diff-mode center --save-at-step 96 --no-save-ckpt \
      --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s${s} \
      --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda --out "$out"
  fi
done > outputs/alpha_sweep/diff_controls.log 2>&1
echo ALL_DONE
