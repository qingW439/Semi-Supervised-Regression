set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
for s in 1 2; do
  echo "=== audit seed $s $(date) ==="
  $PY scripts/vad_research/m0_oof_audit.py \
    --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
    --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
    --manifest outputs/p0_s${s}/fold_manifest.json \
    --data-seed ${s} \
    --out outputs/m0_audit/audit_seed${s}.json --device cuda
done > outputs/m0_audit/run_seeds12.log 2>&1
echo ALL_DONE
