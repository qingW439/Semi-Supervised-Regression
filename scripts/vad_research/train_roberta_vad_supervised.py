#!/usr/bin/env python3
"""P1: RoBERTa-base supervised three-dimensional EmoBank VAD baseline.

Run this script from the repository root, after copying it to
scripts/vad_research/train_roberta_vad_supervised.py.

Reads only labeled TRAIN and DEV files. NEVER reads TEST.
Does not alter any archived E1/E2 scripts or checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import transformers

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    DIMENSIONS, NORM_COLUMNS, GoldData, TextCollator, read_gold,
)
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402

DATA_ROOT = REPO.parent / "data" / "emobank_processed"
MODEL_NAME = "FacebookAI/roberta-base"


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--budget", type=int, choices=(250, 500, 1000), default=500)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--check-only", action="store_true", help="Validate split and labels without loading pretrained weights or training")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    for name in ("max_length", "batch_size", "eval_batch_size", "epochs", "patience"):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if a.lr <= 0 or a.weight_decay < 0 or a.min_delta < 0:
        p.error("Invalid optimizer/stopping arguments")
    if a.out is None:
        a.out = REPO / "saved_models" / "vad_research" / f"p1_roberta_lb{a.budget}_s{a.seed}"
    return a


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            digest.update(part)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_commit():
    p = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def load_checked_splits(a):
    paths = {
        "lb": a.data_root / f"lb{a.budget}_s{a.seed}" / "train_lb.parquet",
        "ulb": a.data_root / f"lb{a.budget}_s{a.seed}" / "train_ulb.parquet",
        "dev": a.data_root / "dev_gold.parquet",
    }
    lb = read_gold(paths["lb"], expected_n=a.budget)
    dev = read_gold(paths["dev"], expected_n=1000)
    from semilearn.datasets.nlp_datasets.emobank import read_unlabeled
    ulb = read_unlabeled(paths["ulb"], labeled_ids=lb["id"])
    if len(ulb) != 8062 - a.budget:
        raise ValueError(f"Unexpected unlabeled count: {len(ulb)}")
    if set(dev["id"]) & (set(lb["id"]) | set(ulb["id"])):
        raise ValueError("Dev data overlaps train data")
    record = REPO / "repro_records" / "emobank" / f"labeled_ids_lb{a.budget}_s{a.seed}.csv"
    if not record.is_file():
        raise FileNotFoundError(f"Missing committed labeled-ID record: {record}")
    with record.open(newline="", encoding="utf-8") as f:
        observed = {row["id"] for row in csv.DictReader(f)}
    if observed != set(lb["id"]):
        raise ValueError("Actual labeled IDs differ from committed labeled-ID list")
    return lb, dev, ulb, paths


def report_constants(lb, dev):
    gold = dev.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float32)
    x = lb.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float32)
    out = {}
    for method, vec in (("mean", np.mean(x, axis=0)), ("median", np.median(x, axis=0))):
        out[method] = metric_report(gold, np.broadcast_to(vec, gold.shape).copy())
    return out


class RobertaVADRegressor(nn.Module):
    """AutoModel backbone, attention-mask-aware mean pooling, 3-value regression head."""
    def __init__(self, model_name, dropout=0.1, offline=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=offline)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, **inputs):
        states = self.encoder(**inputs, return_dict=True).last_hidden_state
        m = inputs["attention_mask"].unsqueeze(-1).to(dtype=states.dtype)
        pooled = (states * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
        return self.head(self.dropout(pooled))


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    gold, preds, ids = [], [], []
    for tok, y, batch_ids in loader:
        tok = {k: v.to(device) for k, v in tok.items()}
        p = model(**tok)
        if p.shape != y.shape or not torch.isfinite(p).all():
            raise RuntimeError("Bad prediction shape or nonfinite values")
        gold.append(y.numpy())
        preds.append(p.detach().cpu().numpy())
        ids.extend(batch_ids)
    return np.concatenate(gold), np.concatenate(preds), ids


def main():
    a = args_parser()
    set_seed(a.seed)
    lb, dev, ulb, paths = load_checked_splits(a)
    hashes = {k: sha256_file(v) for k, v in paths.items()}
    const = report_constants(lb, dev)
    print(f"git={git_commit()} labeled={len(lb)} unlabeled={len(ulb)} dev={len(dev)}")
    print("split sha256:", json.dumps(hashes, sort_keys=True))
    print("constant Dev MAE [1,5]:", {k: round(v["macro_mae_original_scale"], 6) for k,v in const.items()})
    if a.check_only:
        print("PASS: splits and gold labels checked; NO model loaded, NO Test accessed, NO files written")
        return

    out = a.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {out}. Refusing to overwrite")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device("cuda" if torch.cuda.is_available() and a.device != "cpu" else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True, local_files_only=a.offline)
    collator = TextCollator(tokenizer, a.max_length)
    generator = torch.Generator().manual_seed(a.seed)
    train_dl = DataLoader(GoldData(lb), batch_size=a.batch_size, shuffle=True,
                          num_workers=a.num_workers, collate_fn=collator, generator=generator)
    dev_dl = DataLoader(GoldData(dev), batch_size=a.eval_batch_size, shuffle=False,
                        num_workers=a.num_workers, collate_fn=collator)
    model = RobertaVADRegressor(a.model, dropout=a.dropout, offline=a.offline).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    config = {**vars(a), "out":str(out), "data_root":str(a.data_root.resolve()),
              "git_commit":git_commit(),"torch":torch.__version__,"transformers":transformers.__version__,
              "split_sha256":hashes,"device":str(device)}
    config = {k:(str(v) if isinstance(v,Path) else v) for k,v in config.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out/"run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    (out/"constant_baselines_dev.json").write_text(json.dumps(const, indent=2)+"\n", encoding="utf-8")
    best, best_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, a.epochs+1):
        model.train()
        total, count = 0.0, 0
        for tokens, target, _ in train_dl:
            tokens = {k:v.to(device) for k,v in tokens.items()}
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**tokens)
            if output.shape != target.shape or not torch.isfinite(output).all():
                raise RuntimeError("Nonfinite or mismatched training output")
            loss = nn.functional.l1_loss(output, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            if a.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad_norm)
            optimizer.step()
            total += loss.item()*len(target)
            count += len(target)
        yt, yp, ids = predict(model, dev_dl, device)
        metrics = metric_report(yt,yp)
        value = metrics["macro_mae_original_scale"]
        per_dim = {d:metrics[d] for d in DIMENSIONS}
        stats = {d:{"gold_mean":float(np.mean(yt[:,k])*4+1),
                    "pred_mean":float(np.mean(yp[:,k])*4+1),
                    "gold_std":float(np.std(yt[:,k])*4),
                    "pred_std":float(np.std(yp[:,k])*4)} for k,d in enumerate(DIMENSIONS)}
        history.append({"epoch":epoch,"train_loss_norm":total/count,"dev_mae":value,
                        "dev_macro_ccc":float(np.mean([per_dim[d]["ccc"] for d in DIMENSIONS])),
                        **{f"{d}_mae":per_dim[d]["mae"] for d in DIMENSIONS}})
        print(f"epoch {epoch:02d} trainL1={total/count:.5f} Dev MAE={value:.5f} " +
              " ".join(f"{d}_ccc={per_dim[d]['ccc']:.3f}" for d in DIMENSIONS), flush=True)
        if value < best - a.min_delta:
            best, best_epoch, stale = value, epoch, 0
            torch.save({"model":model.state_dict(),"epoch":epoch,
                        "dev_metrics":metrics,"run_config":config},out/"model_best.pth")
            (out/"dev_metrics_best.json").write_text(json.dumps({"epoch":epoch,"metrics":metrics,"stats":stats},indent=2)+"\n",encoding="utf-8")
            preds = pd.DataFrame({"id":ids})
            for k,d in enumerate(DIMENSIONS):
                preds[f"gold_{d}"] = yt[:,k]*4+1
                preds[f"pred_{d}"] = yp[:,k]*4+1
            preds.to_csv(out/"dev_predictions_best.csv",index=False)
        else:
            stale += 1
            if stale>=a.patience:
                print(f"early stopping at epoch={epoch} best_epoch={best_epoch}")
                break
    pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
    print(f"BEST epoch={best_epoch} Dev Macro-MAE={best:.6f}")
    print(f"RESULT_DIR={out}")
    print("No TEST file loaded")


if __name__ == "__main__":
    main()
