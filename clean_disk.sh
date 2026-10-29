#!/bin/bash
set -e
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-vadswap/saved_models/vad_research
echo "== before =="
df -h /root/autodl-tmp | tail -1
du -sh .
# remove checkpoints of failed / superseded screening runs, keep metrics JSON/CSV
for d in smoke_mt smoke_rankup3d repro_p2_control_lb500_s0 \
         knn_a02 knn_a04 knn_a04_s1 knn_a04_s2 knn_a06 \
         mt_ema999 mt_ema999_s1 mt_ema999_s2 \
         combo_a04_ema999 combo_a06_ema999 \
         infer_only_s0 infer_only_s1 infer_only_s2 \
         rk_rda rk_rda_rpl; do
  [ -d "$d" ] && rm -f "$d/model_best.pth" "$d/model_last.pth" && echo "cleaned $d"
done
for d in sweep_*; do
  [ -d "$d" ] && rm -f "$d/model_best.pth" "$d/model_last.pth" && echo "cleaned $d"
done
echo "== after =="
du -sh .
df -h /root/autodl-tmp | tail -1
