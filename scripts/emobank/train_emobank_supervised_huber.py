#!/usr/bin/env python3
"""E1: three-dimensional supervised EmoBank VAD baseline (NO TEST evaluation).

Designed for qingW439/Semi-Supervised-Regression, emobank-vad branch.
Run under the project's RankUp venv from the repository root.

This deliberately uses a standalone training loop so the scalar-only
AlgorithmBase/scaler/ARC are not inadvertently reused for vector labels.
The same BertVADRegressor should be shared by subsequent RankUp-VAD experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from transformers import AutoTokenizer

DIMENSIONS = ("V", "A", "D")
NORM_COLUMNS = tuple(f"{d}_norm" for d in DIMENSIONS)
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.nets.bert.bert_vad import BertVADRegressor
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, read_gold
DEFAULT_DATA_ROOT = REPO.parent / "data" / "emobank_processed"
DEFAULT_MODEL = "google/bert_uncased_L-4_H-512_A-8"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--budget", type=int, choices=(250, 500, 1000), default=250)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--pretrain-path", default=DEFAULT_MODEL)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--offline", action="store_true", help="Only use pretrained weights already cached locally")
    args = p.parse_args()
    if args.max_length <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0 or args.epochs <= 0:
        p.error("batch sizes, max length and epochs must be positive")
    if args.patience < 1:
        p.error("patience must be >= 1")
    if args.out is None:
        args.out = REPO / "saved_models" / "emobank" / f"supervised_lb{args.budget}_s{args.seed}"
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def metric_report(y_norm: np.ndarray, p_norm: np.ndarray) -> dict:
    """Report all metrics on ORIGINAL [1,5] scale; do not clamp model predictions."""
    yt = np.asarray(y_norm, dtype=np.float64) * 4.0 + 1.0
    yp = np.asarray(p_norm, dtype=np.float64) * 4.0 + 1.0
    if yt.shape != yp.shape or yt.ndim != 2 or yt.shape[1] != 3:
        raise ValueError(f"Invalid [N,3] labels: true={yt.shape}, pred={yp.shape}")
    results = {}
    for index, dim in enumerate(DIMENSIONS):
        y, p = yt[:, index], yp[:, index]
        err = y - p
        ymean, pmean = np.mean(y), np.mean(p)
        yvar, pvar = np.mean((y-ymean)**2), np.mean((p-pmean)**2)
        cov = np.mean((y-ymean) * (p-pmean))
        denom = yvar + pvar + (ymean-pmean)**2
        corr = cov / np.sqrt(yvar*pvar) if yvar > 1e-12 and pvar > 1e-12 else None
        srcc = spearmanr(y, p).statistic if yvar > 1e-12 and pvar > 1e-12 else None
        results[dim] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "pearson": float(corr) if corr is not None else None,
            "srcc": float(srcc) if srcc is not None and np.isfinite(srcc) else None,
            "ccc": float(2.0*cov/denom) if denom > 1e-12 else None,
            "r2": float(1.0-np.sum(err**2)/np.sum((y-ymean)**2)) if yvar > 1e-12 else None,
        }
    results["macro_mae_original_scale"] = float(np.mean([results[d]["mae"] for d in DIMENSIONS]))
    results["out_of_range_count"] = int(np.sum((yp < 1) | (yp > 5)))
    return results


def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    gold, preds, ids = [], [], []
    with torch.inference_mode():
        for tokens, targets, batch_ids in loader:
            tokens = {k:v.to(device) for k,v in tokens.items()}
            predictions = model(**tokens).cpu().numpy()
            gold.append(targets.numpy())
            preds.append(predictions)
            ids.extend(batch_ids)
    return np.concatenate(gold), np.concatenate(preds), ids


def current_git_commit():
    result = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    out = args.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory already contains files: {out}. Select a new --out; refusing to overwrite.")

    lb_path = args.data_root / f"lb{args.budget}_s{args.seed}" / "train_lb.parquet"
    ulb_path = args.data_root / f"lb{args.budget}_s{args.seed}" / "train_ulb.parquet"
    dev_path = args.data_root / "dev_gold.parquet"
    lb = read_gold(lb_path, expected_n=args.budget)
    dev = read_gold(dev_path, expected_n=1000)
    if not ulb_path.is_file():
        raise FileNotFoundError(ulb_path)
    ulb_ids = pd.read_parquet(ulb_path, columns=["id"])["id"].tolist()
    if len(ulb_ids) != 8062-args.budget or len(set(ulb_ids)) != len(ulb_ids):
        raise ValueError("Unlabeled pool count/ID uniqueness mismatch")
    if set(lb["id"]) & set(ulb_ids) or set(lb["id"]) & set(dev["id"]):
        raise ValueError("Label leakage detected")

    # Keep a definitive record of the gold-label budget in case data was regenerated.
    ids_file = REPO / "repro_records" / "emobank" / f"labeled_ids_lb{args.budget}_s{args.seed}.csv"
    if ids_file.is_file():
        with ids_file.open(encoding="utf-8", newline="") as f:
            recorded_ids = {r["id"] for r in csv.DictReader(f)}
        if recorded_ids != set(lb["id"]):
            raise ValueError("Gold labeled IDs differ from the committed experiment split")

    print(f"device={device}, seed={args.seed}, gold={len(lb)}, unlabeled={len(ulb_ids)}, dev={len(dev)}")
    print("Train means [1,5] scale:", lb.loc[:, DIMENSIONS].mean().round(3).to_dict())
    print("Train std [1,5] scale:", lb.loc[:, DIMENSIONS].std().round(3).to_dict())

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path, use_fast=True, local_files_only=args.offline)
    collator = TextCollator(tokenizer, args.max_length)
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(GoldData(lb), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collator, generator=train_generator)
    dev_loader = DataLoader(GoldData(dev), batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collator)

    model = BertVADRegressor(args.pretrain_path, dropout=args.dropout, offline=args.offline).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_mae, best_epoch, stale = float("inf"), 0, 0
    history = []
    out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "data_root":str(args.data_root.resolve()),"out":str(out),
              "repository_commit":current_git_commit(),"torch_version":torch.__version__}
    config = {k:(str(v) if isinstance(v,Path) else v) for k,v in config.items()}
    config["training_loss"] = "smooth_l1_beta0.05_normalized"
    (out/"run_config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

    baseline_pred = np.broadcast_to(lb[list(NORM_COLUMNS)].mean().to_numpy(dtype=np.float32),
                                    (len(dev),3)).copy()
    baseline_metrics = metric_report(dev[list(NORM_COLUMNS)].to_numpy(dtype=np.float32),baseline_pred)
    (out/"constant_baseline_dev.json").write_text(json.dumps(baseline_metrics,indent=2)+"\n",encoding="utf-8")
    print(f"constant-mean dev macro MAE [1,5] = {baseline_metrics['macro_mae_original_scale']:.5f}")

    for epoch in range(1,args.epochs+1):
        model.train()
        total_loss, total_num = 0.0, 0
        for tokens, targets, _ in train_loader:
            tokens = {k:v.to(device) for k,v in tokens.items()}
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**tokens)
            if output.shape != targets.shape or not torch.isfinite(output).all():
                raise RuntimeError(f"Nonfinite or mismatched output {output.shape} vs {targets.shape}")
            loss = torch.nn.functional.smooth_l1_loss(output, targets, beta=0.05)
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad_norm)
            optimizer.step()
            total_loss += loss.item() * len(targets)
            total_num += len(targets)

        y,p,ids = predict(model, dev_loader, device)
        report = metric_report(y,p)
        mae = report["macro_mae_original_scale"]
        history.append({"epoch":epoch,"train_huber_norm":total_loss/total_num,
                        "dev_macro_mae_original":mae,
                        **{f"dev_{d}_mae":report[d]["mae"] for d in DIMENSIONS}})
        print(f"epoch {epoch:03d} train_huber_norm={total_loss/total_num:.5f} "
              f"dev_macro_MAE_1to5={mae:.5f} "
              + " ".join(f"{d}={report[d]['mae']:.4f}" for d in DIMENSIONS),flush=True)

        if mae < best_mae - args.min_delta:
            best_mae,best_epoch,stale = mae,epoch,0
            torch.save({"model":model.state_dict(),"epoch":epoch,"dev_metrics":report,"run_config":config},
                       out/"model_best.pth")
            (out/"dev_metrics_best.json").write_text(json.dumps({"epoch":epoch,**report},indent=2)+"\n",encoding="utf-8")
            pred_frame = pd.DataFrame({"id":ids})
            for index,d in enumerate(DIMENSIONS):
                pred_frame[f"gold_{d}"] = y[:,index]*4+1
                pred_frame[f"pred_{d}"] = p[:,index]*4+1
            pred_frame.to_csv(out/"dev_predictions_best.csv",index=False)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop epoch={epoch}, best_epoch={best_epoch}")
                break

    pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
    print(f"BEST epoch={best_epoch}, dev_macro_MAE_1to5={best_mae:.5f}")
    print(f"outputs: {out}")
    print("The independent TEST split was never loaded or evaluated.")


if __name__=="__main__":
    main()
