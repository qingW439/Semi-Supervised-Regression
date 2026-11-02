set -eu
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
$PY scripts/vad_research/eval_single.py \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --ckpt saved_models/vad_research/diff_s005_s0/model_step96.pth \
  --out outputs/alpha_sweep/diff_s5_s0.json --data-seed 0 --device cuda
for s in 1 2; do
  $PY scripts/vad_research/eval_single.py \
    --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
    --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
    --ckpt saved_models/vad_research/diff_s5_s${s}/model_step96.pth \
    --out outputs/alpha_sweep/diff_s5_s${s}.json --data-seed ${s} --device cuda
done
