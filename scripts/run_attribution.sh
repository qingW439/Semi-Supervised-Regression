set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
mkdir -p saved_models/vad_research outputs/attribution
for s in 0 1 2; do
  P1=/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s${s}
  for arm in sup jacf; do
    out=saved_models/vad_research/attr_${arm}_s${s}
    if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then echo "SKIP $arm s$s"; continue; fi
    lam=0.0; [ "$arm" = "jacf" ] && lam=1.0
    echo "=== $arm s$s $(date) ==="
    $PY scripts/vad_research/train_acvr.py --budget 500 --seed "$s" --steps 160 \
      --mode fixed --rel-mode default --alpha-fixed 0.75 --lambda-u "$lam" \
      --p1-dir "$P1" --eval-knn-alpha 0.4 --eval-scale 1.2 --device cuda \
      --out "$out"
  done
done > outputs/attribution/train.log 2>&1
echo ALL_DONE
