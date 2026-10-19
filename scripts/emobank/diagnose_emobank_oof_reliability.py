#!/usr/bin/env python3
"""M1-P0 pilot: train-only OOF VAD error-risk model; DEV diagnosis, no TEST.

Usage:
  python scripts/emobank/diagnose_emobank_oof_reliability.py --offline \
      --budget 500 --seed 0 --epochs 14 \
      --out saved_models/emobank/m1_oof_diag_lb500_s0

Partial smoke (does NOT fit the risk model):
  python scripts/emobank/diagnose_emobank_oof_reliability.py --offline \
      --budget 500 --seed 0 --epochs 1 --max-folds 1 \
      --out saved_models/emobank/m1_oof_smoke_lb500_s0

This tests a *candidate* reliability estimator, not a published method reproduction.
The OOF teachers train on 4/5 of the labeled training set, and full E1 teacher
trains on all labels; this distribution difference limits portability.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from semilearn.nets.bert.bert_vad import BertVADRegressor
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, read_gold
from scripts.emobank.train_emobank_supervised import set_seed

DIMS = ("V", "A", "D")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget", type=int, choices=(250, 500, 1000), default=500)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-folds", type=int, default=5,
                   help="Number of OOF folds to run; less than --folds runs smoke only")
    p.add_argument("--epochs", type=int, default=14,
                   help="Fixed fold training epochs. No OOF fold gold used for model selection")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--mc-passes", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--error-limit", type=float, default=0.25,
                   help="Original 1-5 scale; error > limit is a large error")
    p.add_argument("--data-root", type=Path, default=ROOT.parent / "data" / "emobank_processed")
    p.add_argument("--pretrain-path", default="google/bert_uncased_L-4_H-512_A-8")
    p.add_argument("--full-checkpoint", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    a = p.parse_args()
    if not (2 <= a.folds <= a.budget and 1 <= a.max_folds <= a.folds):
        p.error("Require 2 <= folds <= budget and 1 <= max-folds <= folds")
    if min(a.epochs, a.batch_size, a.eval_batch_size, a.mc_passes, a.max_length) <= 0:
        p.error("Epochs, batch sizes, mc-passes and max-length must be positive")
    if a.error_limit <= 0:
        p.error("--error-limit must be positive")
    if a.full_checkpoint is None:
        a.full_checkpoint = ROOT / "saved_models" / "emobank" / f"supervised_lb{a.budget}_s{a.seed}" / "model_best.pth"
    if not a.full_checkpoint.is_file() and a.max_folds == a.folds:
        p.error(f"Full E1 checkpoint not found: {a.full_checkpoint}")
    if a.out.exists() and any(a.out.iterdir()):
        p.error(f"Refusing to overwrite nonempty output directory: {a.out}")
    return a


def as_loader(frame, tokenizer, a, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(GoldData(frame), batch_size=a.batch_size if shuffle else a.eval_batch_size,
                      shuffle=shuffle, num_workers=0,
                      collate_fn=TextCollator(tokenizer, a.max_length),
                      generator=generator)


def predict_mc(model, frame, tokenizer, a, device, reference_mean):
    """Get deterministic and K dropout draws, all returned on original [1,5] scale."""
    ids, golds, means, stds, dets = [], [], [], [], []
    loader = as_loader(frame, tokenizer, a)
    for toks, y, batch_ids in loader:
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.inference_mode():
            model.eval()
            det = model(**toks).detach().cpu().numpy()
            model.train()  # activate stochastic dropout, but NEVER update weights
            draws = torch.stack([model(**toks) for _ in range(a.mc_passes)]).detach().cpu().numpy()
            model.eval()
        ids.extend(str(v) for v in batch_ids)
        golds.append(y.numpy() * 4 + 1)
        means.append(draws.mean(axis=0) * 4 + 1)
        stds.append(draws.std(axis=0) * 4)
        dets.append(det * 4 + 1)
    golds, means, stds, dets = map(np.concatenate, (golds, means, stds, dets))
    ret = pd.DataFrame({"id": ids})
    texts = frame["text"].tolist()
    ret["text_chars"] = [len(t) for t in texts]
    for k, d in enumerate(DIMS):
        ret[f"gold_{d}"] = golds[:, k]
        ret[f"pred_{d}"] = means[:, k]
        ret[f"pred_det_{d}"] = dets[:, k]
        ret[f"mc_std_{d}"] = stds[:, k]
        ret[f"mean_dist_{d}"] = np.abs(means[:, k] - reference_mean[k])
    return ret


def feature_matrix(df, dim):
    # Features are all available WITHOUT a real label for the evaluated text.
    return df[[f"mc_std_{dim}", f"pred_{dim}", f"mean_dist_{dim}", "text_chars"]].to_numpy(dtype=np.float64)


def safe_auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def selected_report(df, dim, score, fraction, full_train_mean):
    n = max(1, int(len(df) * fraction))
    sel = np.argsort(score, kind="stable")[:n]  # smaller predicted risk = safer
    gold = df[f"gold_{dim}"].to_numpy()[sel]
    prediction = df[f"pred_{dim}"].to_numpy()[sel]
    return {"count":int(n),"pseudo_mae":float(np.abs(prediction-gold).mean()),
            "constant_train_mean_mae_on_same_subset":float(np.abs(full_train_mean-gold).mean())}


def run(a):
    set_seed(a.seed)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if (a.device == "cuda" or
                         (a.device == "auto" and torch.cuda.is_available())) else "cpu")
    a.out.mkdir(parents=True, exist_ok=True)
    labeled = read_gold(a.data_root / f"lb{a.budget}_s{a.seed}" / "train_lb.parquet", expected_n=a.budget)
    # Read DEV ONLY for final risk-model assessment; NOT in fitting the OOF model.
    tokenizer = AutoTokenizer.from_pretrained(a.pretrain_path, use_fast=True, local_files_only=a.offline)
    split = KFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    all_oof = []
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}
    (a.out / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"OOF: budget={a.budget}, folds={a.folds}, max_folds={a.max_folds}, fixed_epochs={a.epochs}")
    for fold, (train_idx, held_idx) in enumerate(split.split(labeled)):
        if fold >= a.max_folds:
            break
        set_seed(a.seed + 1000 + fold)
        train = labeled.iloc[train_idx].reset_index(drop=True)
        held = labeled.iloc[held_idx].reset_index(drop=True)
        ref = train[list(DIMS)].mean().to_numpy(dtype=np.float64)
        model = BertVADRegressor(a.pretrain_path, dropout=a.dropout, offline=a.offline).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        train_loader = as_loader(train, tokenizer, a, shuffle=True, seed=a.seed+2000+fold)
        for epoch in range(1, a.epochs+1):
            model.train()
            sums, examples = 0., 0
            for toks, y, _ in train_loader:
                toks = {k:v.to(device) for k,v in toks.items()}
                y = y.to(device)
                optim.zero_grad(set_to_none=True)
                loss = torch.nn.functional.l1_loss(model(**toks),y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),a.clip_grad_norm)
                optim.step()
                sums += float(loss.detach()) * len(y)
                examples += len(y)
            if epoch in (1,a.epochs):
                print(f"fold={fold} epoch={epoch} train_norm_L1={sums/examples:.5f}",flush=True)
        oo = predict_mc(model,held,tokenizer,a,device,ref)
        oo["fold"] = fold
        oo.to_csv(a.out/f"oof_fold{fold}.csv",index=False)
        all_oof.append(oo)
        print(f"fold={fold} heldout={len(held)} (no heldout gold used for fitting/early stopping)", flush=True)
        del model,optim
        if device.type == "cuda": torch.cuda.empty_cache()
    oof = pd.concat(all_oof,ignore_index=True)
    oof.to_csv(a.out / "oof_predictions.csv",index=False)
    if a.max_folds < a.folds:
        print(f"SMOKE ONLY: {len(oof)} OOF rows; skipping calibrator/DEV; no TEST access.")
        return
    if len(oof) != len(labeled) or oof["id"].nunique() != len(labeled):
        raise RuntimeError("OOF incomplete or contains duplicate IDs")

    # Load FULL teacher for DEV; never train on DEV.
    dev = read_gold(a.data_root / "dev_gold.parquet", expected_n=1000)
    if set(map(str,dev["id"])) & set(map(str,labeled["id"])):
        raise RuntimeError("DEV IDs intersect labeled train")
    model = BertVADRegressor(a.pretrain_path, dropout=a.dropout, offline=a.offline).to(device)
    checkpoint = torch.load(a.full_checkpoint,map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint["model"],strict=True)
    train_mean = labeled[list(DIMS)].mean().to_numpy(dtype=np.float64)
    set_seed(a.seed)  # independent MC draws from fold-training RNG state
    dev_pred = predict_mc(model, dev, tokenizer, a, device, train_mean)
    dev_pred.to_csv(a.out/"dev_diagnostic_predictions.csv",index=False)
    del model
    print("Training-only reliability fitting; DEV used solely for evaluation.",flush=True)
    report = {"purpose":"diagnostic: train-only OOF risk estimator evaluated on DEV",
              "full_teacher_epoch":checkpoint.get("epoch"),
              "oof_train_size_per_fold":len(labeled)-int(np.ceil(len(labeled)/a.folds)),
              "full_teacher_labeled_size":len(labeled),
              "calibration_target":f"abs(MC mean prediction - gold) > {a.error_limit} on original [1,5]",
              "test_accessed":False,"caveat":"OOF teachers are trained on fewer examples than the full teacher; risk predictions may be shifted."}
    for k,dim in enumerate(DIMS):
        oof_error = np.abs(oof[f"pred_{dim}"].to_numpy()-oof[f"gold_{dim}"].to_numpy())
        dv_error = np.abs(dev_pred[f"pred_{dim}"].to_numpy()-dev_pred[f"gold_{dim}"].to_numpy())
        y_train = (oof_error > a.error_limit).astype(int)
        y_dev = (dv_error > a.error_limit).astype(int)
        if len(np.unique(y_train)) != 2:
            raise ValueError(f"OOF error classes degenerate for {dim}; cannot fit logistic risk model")
        risk = make_pipeline(StandardScaler(), LogisticRegression(C=0.3,max_iter=1000,random_state=a.seed))
        risk.fit(feature_matrix(oof,dim),y_train)
        score = risk.predict_proba(feature_matrix(dev_pred,dim))[:,1]
        mcstd = dev_pred[f"mc_std_{dim}"].to_numpy()
        mean_dist = dev_pred[f"mean_dist_{dim}"].to_numpy()
        report[dim] = {
            "oof_large_error_rate":float(y_train.mean()),"dev_large_error_rate":float(y_dev.mean()),
            "full_teacher_dev_mae_mc":float(dv_error.mean()),
            "auroc_mc_std":safe_auc(y_dev,mcstd),
            "auroc_train_mean_distance":safe_auc(y_dev,mean_dist),
            "auroc_oof_risk_estimator":safe_auc(y_dev,score),
            "auprc_oof_risk_estimator":float(average_precision_score(y_dev,score)),
            "brier_oof_risk_estimator":float(brier_score_loss(y_dev,score)),
            "risk_selection_at_50pct":selected_report(dev_pred,dim,score,0.5,train_mean[k]),
            "mcstd_selection_at_50pct":selected_report(dev_pred,dim,mcstd,0.5,train_mean[k]),
        }
        print(f"{dim}: AUROC MCstd={report[dim]['auroc_mc_std']:.4f} "
              f"MeanDist={report[dim]['auroc_train_mean_distance']:.4f} "
              f"OOF_risk={report[dim]['auroc_oof_risk_estimator']:.4f} "
              f"Selected50MAE={report[dim]['risk_selection_at_50pct']['pseudo_mae']:.4f} ",flush=True)
    (a.out/"reliability_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(f"REPORT={a.out/'reliability_report.json'} | TEST never opened")


if __name__=="__main__":
    run(arguments())
