set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
for s in 1 2; do
  $PY scripts/make_oof_folds.py --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
    --out-dir outputs/p0_s${s} --data-seed ${s}
done
