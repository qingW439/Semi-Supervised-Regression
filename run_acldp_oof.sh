#!/bin/bash
# ACLDP re-verification under the strict OOF protocol.
#
# The difference reference must be trained on the SAME 400 labels as the student, so it is
# built inside the fold trainer from that fold's own supervised model:
#   m0    : the fold's supervised P1 network (predicts the unlabelled pool)
#   ridge : closed-form ridge fitted on the fold's labelled features (alpha by inner CV)
# plus a negative control that shuffles the ridge reference within each batch.
#
# Arms: 3 seeds x 5 folds x {m0, ridge, ridge_shuffle}
R=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-comp2
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
DATA=/root/autodl-tmp/rankup_project/data/emobank_processed
PRE=/root/autodl-tmp/rankup_project/pretrained/roberta-base
MAIN=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
LAMBDA=0.05
cd $R || exit 1
mkdir -p outputs/oof_acldp outputs/auto_search/logs
for S in 0 1 2; do
  if [ "$S" = "0" ]; then MAN=$MAIN/outputs/p0/fold_manifest.json
  else MAN=$MAIN/outputs/p0_s${S}/fold_manifest.json; fi
  for F in 0 1 2 3 4; do
    for REF in m0 ridge ridge_shuffle; do
      base=$REF
      extra=""
      if [ "$REF" = "ridge_shuffle" ]; then base=ridge; extra="--diff-shuffle"; fi
      OUT=outputs/oof_acldp/seed${S}_${REF}_fold${F}
      if [ -f "$OUT/oof_predictions.parquet" ]; then echo "SKIP $OUT"; continue; fi
      echo "=== seed$S fold$F ref=$REF $(date +%H:%M:%S) ==="
      $PY scripts/vad_research/train_oof_fold.py --fold $F --manifest $MAN --data-seed $S \
        --data-root $DATA --pretrain-path $PRE --out-dir $OUT --device cuda \
        --p1-epochs 10 --jacf-steps 96 --alpha-fixed 0.75 --lambda-u 1.0 \
        --diff-weight $LAMBDA --diff-ref $base --diff-mode pairs $extra \
        >> outputs/auto_search/logs/acldp_oof.log 2>&1
      grep -E '\[diff-ref' outputs/auto_search/logs/acldp_oof.log | tail -1
    done
  done
done
echo "DONE $(date +%H:%M:%S)"
