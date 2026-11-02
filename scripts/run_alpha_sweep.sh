set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
mkdir -p saved_models/vad_research/alpha_sweep outputs/alpha_sweep
for a in 0.55 0.60 0.65 0.70 0.75; do
  for s in 0 1 2; do
    out=saved_models/vad_research/alpha_sweep/a${a}_s${s}
    if [ -f "$out/model_step96.pth" ]; then echo "SKIP a$a s$s"; continue; fi
    P1=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s${s}
    echo "=== alpha $a seed $s $(date) ==="
    $PY scripts/vad_research/train_acvr.py --budget 500 --seed "$s" --steps 160 \
      --mode fixed --rel-mode default --alpha-fixed "$a" --lambda-u 1.0 \
      --save-at-step 96 --no-save-ckpt \
      --p1-dir "$P1" --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda \
      --out "$out"
  done
done > outputs/alpha_sweep/train.log 2>&1
for s in 0 1 2; do
  out=saved_models/vad_research/alpha_sweep/sup_s${s}
  if [ -f "$out/model_step96.pth" ]; then echo "SKIP sup s$s"; continue; fi
  P1=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s${s}
  echo "=== sup seed $s $(date) ==="
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed "$s" --steps 160 \
    --mode fixed --rel-mode default --alpha-fixed 0.75 --lambda-u 0.0 \
    --save-at-step 96 --no-save-ckpt \
    --p1-dir "$P1" --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda \
    --out "$out"
done >> outputs/alpha_sweep/train.log 2>&1
echo ALL_DONE
