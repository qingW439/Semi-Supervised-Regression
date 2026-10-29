#!/bin/bash
ROOT=/root/autodl-tmp/rankup_project
COPY=$ROOT/Semi-Supervised-Regression-vadswap
ORIG=$ROOT/Semi-Supervised-Regression
ENV=$ROOT/envs/rankup/bin/python
P1=$ORIG/saved_models/vad_research/p1_roberta_lb500_s0
cd "$COPY"
mkdir -p logs
run() {
  name=$1; shift
  echo "===== START $name $(date +%H:%M:%S)"
  "$ENV" scripts/vad_research/train_vad_modules.py \
    --mode mean_teacher --budget 500 --seed 0 --p1-dir "$P1" \
    --out "saved_models/vad_research/$name" --device cuda "$@" \
    > "logs/$name.log" 2>&1
  echo "===== END $name rc=$?"
  tail -2 "logs/$name.log"
}
run sweep_mt_ccc_sup   --sup-loss l1_ccc --ccc-weight 0.3
run sweep_mt_ccc_both  --sup-loss l1_ccc --ulb-loss l1_ccc --ccc-weight 0.3
run sweep_mt_rank01    --rank-weight 0.1
run sweep_mt_rank03    --rank-weight 0.3
run sweep_mt_disagree  --ulb-weight disagree --weight-tau 0.05
run sweep_mt_disq_filt --ulb-weight disagree --filter topq --filter-q 0.5
run sweep_mt_unc       --uncertainty hetero --ulb-weight unc
run sweep_mt_agree     --ulb-weight agree --weight-tau 0.05
echo "ALL DONE $(date +%H:%M:%S)"
