set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
mkdir -p outputs/oof
for f in 1 2 3 4; do
  out=outputs/oof/fold${f}
  if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then echo "SKIP fold $f"; continue; fi
  echo "=== FOLD $f $(date) ==="
  $PY scripts/vad_research/train_oof_fold.py --fold "$f" \
    --manifest outputs/p0/fold_manifest.json \
    --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
    --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
    --out-dir "$out" --p1-epochs 10 --jacf-steps 96 --device cuda
  echo "=== DONE fold $f $(date) ==="
done > outputs/oof/all_folds.log 2>&1
echo ALL_DONE
