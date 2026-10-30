set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-relstudy
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
run() {
  tag=$1; rel=$2; extra=$3; s=$4
  P1=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s${s}
  out=saved_models/vad_research/rel_${tag}_s${s}
  if [ "$tag" = "oracle_inv_dimnorm" ]; then out=saved_models/vad_research/oracle_inv_dimnorm_s${s}; fi
  if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then echo "SKIP $tag s$s"; return; fi
  echo "=== RUN $tag s$s $(date) ==="
  $PY scripts/vad_research/train_acvr.py --budget 500 --seed "$s" --steps 160 \
    --mode jacf --rel-mode "$rel" --alpha-fixed 0.75 $extra \
    --oracle-labels outputs/oracle_labels_all_train.npz --oracle-eps 0.005 \
    --p1-dir "$P1" --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda \
    --out "$out"
  echo "=== DONE $tag s$s $(date) ==="
}
{
  for s in 1 2; do
    run none none "" "$s"
    run prop_dimnorm prop_var "--rel-dimnorm" "$s"
    run oracle_inv_dimnorm oracle "--rel-dimnorm" "$s"
  done
} > outputs/seed_confirm.log 2>&1
echo ALL_DONE
