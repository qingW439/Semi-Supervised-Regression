set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-relstudy
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
P1=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0
run() {
  tag=$1; rel=$2; extra=$3
  out=saved_models/vad_research/steps320_${tag}_s0
  if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then echo "SKIP $tag"; return; fi
  echo "=== RUN $tag $(date) ==="
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed 0 --steps 320 \
    --mode jacf --rel-mode "$rel" --alpha-fixed 0.75 $extra \
    --oracle-labels outputs/oracle_labels_lb500_s0.npz --oracle-eps 0.005 \
    --p1-dir "$P1" --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda \
    --out "$out"
  echo "=== DONE $tag $(date) ==="
}
{
  run none none ""
  run oracle oracle ""
} > outputs/steps320_s0.log 2>&1
echo ALL_DONE
