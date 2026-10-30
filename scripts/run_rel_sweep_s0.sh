set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-relstudy
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
P1=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0
mkdir -p outputs saved_models/vad_research
run() {
  tag=$1; mode=$2; extra=$3
  out=saved_models/vad_research/rel_${tag}_s0
  if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then echo "SKIP $tag"; return; fi
  echo "=== RUN $tag $(date) ==="
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed 0 --steps 160 \
    --mode jacf --rel-mode "$mode" --alpha-fixed 0.75 $extra \
    --p1-dir "$P1" --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda \
    --out "$out"
  echo "=== DONE $tag $(date) ==="
}
{
  run none none ""
  run anchor_var anchor_var ""
  run prop_var prop_var ""
  run anchor_dimnorm anchor_var "--rel-dimnorm"
  run prop_dimnorm prop_var "--rel-dimnorm"
} > outputs/rel_sweep_s0.log 2>&1
echo ALL_DONE
