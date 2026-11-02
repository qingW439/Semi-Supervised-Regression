set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
M0=saved_models/vad_research/alpha_sweep/sup_s0/model_step96.pth
for spec in "0.1::d1" "0.5::d5" "1.0::d10" "0.5:shuffle:d5s"; do
  w=${spec%%:*}; rest=${spec#*:}; sh=${rest%%:*}; tag=${rest##*:}
  out=saved_models/vad_research/diff_${tag}_s0
  if [ -f "$out/model_step96.pth" ]; then echo "SKIP $tag"; continue; fi
  extra=""
  if [ "$sh" = "shuffle" ]; then extra="--diff-shuffle"; fi
  echo "=== DIFF $tag w=$w $sh $(date) ==="
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed 0 --steps 160 \
    --mode fixed --rel-mode default --alpha-fixed 0.75 --lambda-u 1.0 \
    --m0-ckpt "$M0" --diff-weight "$w" $extra \
    --save-at-step 96 --no-save-ckpt \
    --p1-dir /root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0 \
    --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda --out "$out"
done > outputs/alpha_sweep/diff_train.log 2>&1
echo ALL_DONE
