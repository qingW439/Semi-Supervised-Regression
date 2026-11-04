#!/bin/bash
# Build per-fold supervised (P1) checkpoints for seeds 1 and 2 so that E0* can be audited
# under the SAME protocol as seed 0: the encoder must be re-trained on each fold's 400
# labels (using a globally-trained encoder would leak those very labels into the OOF).
R=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-comp2
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
DATA=/root/autodl-tmp/rankup_project/data/emobank_processed
PRE=/root/autodl-tmp/rankup_project/pretrained/roberta-base
cd $R || exit 1
for S in 1 2; do
  MAN=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc/outputs/p0_s${S}/fold_manifest.json
  for F in 0 1 2 3 4; do
    OUT=outputs/oof_seed${S}/fold${F}
    if [ -f "$OUT/jacf_final.pth" ]; then echo "SKIP seed$S fold$F"; continue; fi
    echo "=== seed$S fold$F $(date +%H:%M:%S) ==="
    $PY scripts/vad_research/train_oof_fold.py --fold $F --manifest $MAN \
      --data-root $DATA --pretrain-path $PRE --out-dir $OUT --device cuda \
      --data-seed $S --p1-epochs 10 --jacf-steps 96 --alpha-fixed 0.75 --lambda-u 1.0 \
      >> outputs/auto_search/logs/oof_seed${S}.log 2>&1
    tail -2 outputs/auto_search/logs/oof_seed${S}.log
  done
done
echo "DONE $(date +%H:%M:%S)"
