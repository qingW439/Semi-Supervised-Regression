#!/bin/bash
cd /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-acvr
mkdir -p outputs
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
echo "== make model-only =="
$PY -u make_model_only.py
echo "== probe =="
$PY -u scripts/vad_research/probe_anchor_relative.py \
  --manifest /root/autodl-tmp/rankup_project/Semi-Supervised-Regression-m1-validity-v2/docs/m1/M1_TARGET_UTILITY_VALIDITY_V2_RUN_MANIFEST.json \
  --ckpt outputs/reference_320_model_only.pth \
  --data-root /root/autodl-tmp/rankup_project/data/emobank_processed \
  --pretrain-path /root/autodl-tmp/rankup_project/pretrained/roberta-base \
  --out outputs/arcl_probe.json \
  --k 20 --tau 0.1 --mc-passes 5 --bootstrap 1000
echo "rc=$?"
