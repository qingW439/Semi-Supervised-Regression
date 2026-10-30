set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-relstudy
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
P1=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0
run() {
  tag=$1; extra=$2
  out=saved_models/vad_research/oracle_${tag}_s0
  if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then echo "SKIP $tag"; return; fi
  echo "=== RUN $tag $(date) ==="
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed 0 --steps 160 \
    --mode jacf --rel-mode oracle --alpha-fixed 0.75 \
    --oracle-labels outputs/oracle_labels_lb500_s0.npz --oracle-eps 0.005 $extra \
    --p1-dir "$P1" --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda \
    --out "$out"
  echo "=== DONE $tag $(date) ==="
}
{
  run inv ""
  run inv_dimnorm "--rel-dimnorm"
} > outputs/oracle_sweep_s0.log 2>&1
echo ALL_DONE
