set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
for s in 0 1 2; do
  echo "SEED $s"
  $PY scripts/vad_research/eval_single.py \
    --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
    --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
    --ckpt saved_models/vad_research/rda_0.5_s${s}/model_step96.pth \
    --out outputs/alpha_sweep/rda_0.5_s${s}.json --data-seed ${s} --device cuda
done
