#!/usr/bin/env python3
"""Read-only E1 gradient comparison on labeled EmoBank train batches.

Never creates an optimizer or reads Dev/Test. The rank head is newly initialized
under the fixed seed; the BERT and regression head strictly load from E1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, read_gold
from semilearn.nets.bert.bert_vad_rank import supervised_rank_loss
from scripts.emobank.train_emobank_rank3d import (
    _move, _sha256, load_model, parse_args as parse_train_args,
)
from scripts.emobank.train_emobank_supervised import current_git_commit, set_seed


def parameter_digest(model):
    digest = hashlib.sha256()
    for tensor in model.parameters():
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def grad_statistics(reg_grad, rank_grad):
    reg_sq = rank_sq = dot = 0.0
    for g, h in zip(reg_grad, rank_grad):
        if g is not None:
            reg_sq += float(g.detach().double().square().sum())
        if h is not None:
            rank_sq += float(h.detach().double().square().sum())
        if g is not None and h is not None:
            dot += float((g.detach().double() * h.detach().double()).sum())
    reg_norm, rank_norm = math.sqrt(reg_sq), math.sqrt(rank_sq)
    return reg_norm, rank_norm, dot / (reg_norm * rank_norm) if reg_norm and rank_norm else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    cli = parser.parse_args()
    if cli.batches < 1 or cli.out.exists():
        parser.error("batches must be positive and out must not already exist")
    a = parse_train_args(["--config", str(cli.config), "--device", "cuda"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(a.seed)
    device = torch.device("cuda")
    labeled_path = a.data_root / f"lb{a.budget}_s{a.seed}" / "train_lb.parquet"
    labeled = read_gold(labeled_path, expected_n=a.budget)
    record = Path(__file__).resolve().parents[2] / "repro_records" / "emobank" / f"labeled_ids_lb{a.budget}_s{a.seed}.csv"
    with record.open(encoding="utf-8", newline="") as stream:
        recorded = {row["id"] for row in csv.DictReader(stream)}
    if recorded != set(labeled["id"]):
        raise ValueError("Labeled IDs differ from committed split record")
    tokenizer = AutoTokenizer.from_pretrained(a.pretrain_path, use_fast=True,
                                               local_files_only=a.offline)
    loader = DataLoader(GoldData(labeled), batch_size=a.batch_size,
                        shuffle=True, num_workers=0,
                        collate_fn=TextCollator(tokenizer, a.max_length),
                        generator=torch.Generator().manual_seed(a.seed))
    model, source_epoch, _ = load_model(a, device)
    before = parameter_digest(model)
    params = tuple(p for p in model.backbone.bert.parameters() if p.requires_grad)
    rows = []
    for index, (tokens, targets, ids) in enumerate(loader, start=1):
        model.train()
        outputs = model(**_move(tokens, device))
        targets = targets.to(device)
        regression = F.l1_loss(outputs["regression"], targets)
        gold_rank, ties = supervised_rank_loss(outputs["rank_logits"], targets)
        reg_grad = torch.autograd.grad(regression, params, retain_graph=True,
                                       allow_unused=True)
        rank_grad = torch.autograd.grad(gold_rank, params, allow_unused=True)
        reg_norm, rank_norm, cosine = grad_statistics(reg_grad, rank_grad)
        if reg_norm == 0 or rank_norm == 0 or not all(
                math.isfinite(v) for v in (reg_norm, rank_norm, cosine)):
            raise RuntimeError("Nonfinite gradient statistic")
        row = {"batch": index, "ids": ids, "regression_loss": float(regression.detach()),
               "supervised_rank_loss": float(gold_rank.detach()),
               "regression_grad_norm": reg_norm, "rank_grad_norm": rank_norm,
               "gradient_cosine": cosine, "gold_ties_VAD": ties.cpu().tolist(),
               "weighted_rank_to_reg_0p2": 0.2 * rank_norm / reg_norm if reg_norm else None,
               "weighted_rank_to_reg_0p02": 0.02 * rank_norm / reg_norm if reg_norm else None}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if index >= cli.batches:
            break
    if parameter_digest(model) != before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Diagnostic changed model parameters or populated .grad")
    keys = ("regression_loss", "supervised_rank_loss", "regression_grad_norm",
            "rank_grad_norm", "gradient_cosine", "weighted_rank_to_reg_0p2",
            "weighted_rank_to_reg_0p02")
    summary = {key: {"mean": float(np.mean([r[key] for r in rows])),
                     "std": float(np.std([r[key] for r in rows])),
                     "min": float(np.min([r[key] for r in rows])),
                     "max": float(np.max([r[key] for r in rows]))} for key in keys}
    result = {"repository_commit": current_git_commit(), "source_checkpoint": str(a.checkpoint),
              "checkpoint_sha256": _sha256(a.checkpoint), "source_epoch": source_epoch,
              "labeled_split_sha256": _sha256(labeled_path), "seed": a.seed,
              "batch_size": a.batch_size, "batches": len(rows), "dropout_mode": "train",
              "encoder": "shared BERT parameters only", "optimizer_steps": 0,
              "test_access": "none", "parameters_unchanged": True,
              "summary": summary, "per_batch": rows}
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("NO_OPTIMIZER_UPDATE=PASS", "OUTPUT", cli.out, flush=True)


if __name__ == "__main__":
    main()
