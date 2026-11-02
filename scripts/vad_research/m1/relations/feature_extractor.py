"""Frozen-feature extraction and fold-model prediction helpers for M1-R0."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402


@torch.inference_mode()
def extract_frozen(texts, pretrain_path, device, bs=64, max_length=128):
    tok = AutoTokenizer.from_pretrained(str(pretrain_path), local_files_only=True, use_fast=True)
    enc = AutoModel.from_pretrained(str(pretrain_path), local_files_only=True).to(device).eval()
    feats = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True,
                max_length=max_length, return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        st = enc(**b, return_dict=True).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(dtype=st.dtype)
        pooled = (st * m).sum(1) / m.sum(1).clamp_min(1)
        feats.append(F.normalize(pooled.float(), dim=-1).cpu())
    del enc
    torch.cuda.empty_cache()
    return torch.cat(feats).numpy()


@torch.inference_mode()
def predict_fold_model(ckpt, texts, pretrain_path, device, bs=64, max_length=128):
    tok = AutoTokenizer.from_pretrained(str(pretrain_path), local_files_only=True, use_fast=True)
    model = RobertaVADRegressor(str(pretrain_path), dropout=0.1, offline=True).to(device)
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(d["model"], strict=True)
    del d
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True,
                max_length=max_length, return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        out.append(model(**b).float().cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(out).numpy()
@torch.inference_mode()
def extract_fold_features(ckpt, texts, pretrain_path, device, bs=64, max_length=128):
    from scripts.vad_research.train_acvr import encode
    tok = AutoTokenizer.from_pretrained(str(pretrain_path), local_files_only=True, use_fast=True)
    model = RobertaVADRegressor(str(pretrain_path), dropout=0.1, offline=True).to(device)
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(d["model"], strict=True)
    del d
    feat = encode(model.encoder, texts, tok, device).numpy()
    del model
    torch.cuda.empty_cache()
    return feat
