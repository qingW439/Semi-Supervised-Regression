#!/bin/bash
# Strong-base M1 grid (v2) -- run only when the GPU is actually present.
#
# For each seed x fold, from ONE supervised base with the frozen E0* recipe:
#   base    : E0* itself                        (supervised, 1600 steps)
#   sup96   : base + 96 further SUPERVISED steps (lambda_u = 0)   <- matched control (E5)
#   jacf    : base + 96 JACF steps              (component 1)
#   m1      : base + 96 JACF + ACLDP(ridge ref) (full M1)
#   m1_shuf : same as m1 but the ridge reference is shuffled within each batch (negative control)
#
# All five arms share the identical base weights, so every comparison is exactly paired and
# the unlabeled-data contribution is isolated by `m1 - sup96`.
set -euo pipefail
R=${R:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
DATA=/root/autodl-tmp/rankup_project/data/emobank_processed
PRE=/root/autodl-tmp/rankup_project/pretrained/roberta-base
MAIN=${MAIN:-$R}
BASE_STEPS=${BASE_STEPS:-1600}
BASE_LR=${BASE_LR:-1e-4}
BASE_BS=${BASE_BS:-64}
SUP_STEPS=${SUP_STEPS:-96}
TAG=${TAG:-strong_b${BASE_STEPS}}
cd $R || exit 1
mkdir -p outputs/auto_search/logs outputs/$TAG

$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "ABORT: CUDA not available"; exit 1; }

echo "##### $TAG start $(date +%H:%M:%S)" >> outputs/auto_search/logs/strong_base.log
for S in 0 1 2; do
  if [ "$S" = "0" ]; then MAN=$MAIN/outputs/p0/fold_manifest.json
  else MAN=$MAIN/outputs/p0_s${S}/fold_manifest.json; fi
  for F in 0 1 2 3 4; do
    ROOT=outputs/$TAG/seed${S}_fold${F}
    mkdir -p $ROOT
    # 1) supervised base (also the E0* arm)
    if [ ! -s "$ROOT/sup_predictions.parquet" ]; then
      $PY scripts/vad_research/audit_sup_recipe_fold.py --fold $F --manifest $MAN --data-seed $S \
        --data-root $DATA --pretrain-path $PRE --out-dir $ROOT --steps $BASE_STEPS --lr $BASE_LR \
        --batch-size $BASE_BS --device cuda --save-ckpt >> outputs/auto_search/logs/strong_base.log 2>&1
    fi
    [ -f "$ROOT/base.pth" ] || { echo "  MISSING BASE $ROOT" >> outputs/auto_search/logs/strong_base.log; continue; }
    run_branch () {  # name extra-args...
      local NAME=$1; shift
      if [ -s "$ROOT/$NAME/oof_predictions.parquet" ]; then return; fi
      $PY scripts/vad_research/train_oof_fold.py --fold $F --manifest $MAN --data-seed $S \
        --data-root $DATA --pretrain-path $PRE --out-dir $ROOT/$NAME --device cuda \
        --init-ckpt $ROOT/base.pth --jacf-steps $SUP_STEPS --alpha-fixed 0.75 "$@" \
        >> outputs/auto_search/logs/strong_base.log 2>&1
    }
    run_branch sup96  --lambda-u 0.0 --diff-weight 0.0
    run_branch jacf   --lambda-u 1.0 --diff-weight 0.0
    run_branch m1     --lambda-u 1.0 --diff-weight 0.05 --diff-ref ridge --diff-mode pairs
    run_branch m1_shuf --lambda-u 1.0 --diff-weight 0.05 --diff-ref ridge --diff-mode pairs --diff-shuffle
    echo "  done $ROOT $(date +%H:%M:%S)" >> outputs/auto_search/logs/strong_base.log
    # Retain base.pth for exact paired reruns and checkpoint provenance.
  done
done
echo "STRONG BASE GRID DONE $(date +%H:%M:%S)" >> outputs/auto_search/logs/strong_base.log
