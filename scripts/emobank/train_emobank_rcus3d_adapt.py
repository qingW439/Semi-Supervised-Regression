#!/usr/bin/env python3
"""Warmup-initialized A0/A1/A2 Close Mixup continuation; never opens Test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    DIMENSIONS, GoldData, TextCollator, UnlabeledData,
)
from semilearn.nets.bert.bert_vad import BertVADRegressor  # noqa: E402
from semilearn.nets.bert.bert_vad_rcus_adapt import (  # noqa: E402
    BertVADRCUS3D, supervised_close_mixup_loss, unsupervised_close_mixup_loss,
)
from scripts.emobank.rcus3d_adapt_common import E1, load_splits_no_test  # noqa: E402
from scripts.emobank.train_emobank_supervised import (  # noqa: E402
    current_git_commit, metric_report, predict, set_seed,
)


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO / path).resolve()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="JSON defaults; CLI flags override them")
    p.add_argument("--data-root", default="../data/emobank_processed")
    p.add_argument("--checkpoint", required=False, help="New head-only warmup checkpoint")
    p.add_argument("--gradient-record", required=False)
    p.add_argument("--calibration", required=False)
    p.add_argument("--pretrain-path", default="google/bert_uncased_L-4_H-512_A-8")
    p.add_argument("--mode", choices=("a0", "a1", "a2"), default="a1")
    p.add_argument("--budget", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--unlabeled-ratio", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--supervised-mixup-weight", type=float, default=None)
    p.add_argument("--unsupervised-mixup-weight", type=float, default=1.0)
    p.add_argument("--mixup-alpha", type=float, default=0.5)
    p.add_argument("--confidence-cutoff", type=float, default=0.95)
    p.add_argument("--diagnostic-epochs", nargs="*", type=int, default=[],
                   help="Preset epochs for extra rank-quality checkpoints")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--out", help="New empty output directory")
    p.add_argument("--smoke-steps", type=int, default=0,
                   help="CPU-friendly short run; no checkpoint or metrics files saved")
    initial, _ = p.parse_known_args(argv)
    if initial.config:
        config = json.loads(initial.config.read_text(encoding="utf-8"))
        unknown = set(config) - {action.dest for action in p._actions}
        if unknown:
            p.error(f"Unknown config keys: {sorted(unknown)}")
        p.set_defaults(**config)
    a = p.parse_args(argv)
    if not a.checkpoint or not a.gradient_record:
        p.error("Warmup checkpoint and gradient record are required")
    if a.budget not in (250, 500, 1000) or a.seed not in (0, 1, 2):
        p.error("Unsupported budget or seed")
    if min(a.batch_size, a.eval_batch_size, a.unlabeled_ratio, a.max_length,
           a.epochs) < 1 or a.smoke_steps < 0:
        p.error("Batch/length/epoch values must be positive")
    if not (0.5 <= a.confidence_cutoff <= 1.0):
        p.error("--confidence-cutoff must be in [0.5,1]")
    if a.unsupervised_mixup_weight < 0 or a.lr <= 0 or a.weight_decay < 0 or a.mixup_alpha <= 0:
        p.error("Invalid loss or optimizer weights")
    if a.mode == "a0" and a.unsupervised_mixup_weight != 0:
        p.error("A0 requires unsupervised_mixup_weight=0")
    if a.mode != "a0" and a.unsupervised_mixup_weight != 1.0:
        p.error("A1/A2 use the same fixed unsupervised weight 1.0")
    if a.mode == "a2" and not a.calibration:
        p.error("A2 requires training-only OOF calibration")
    if len(set(a.diagnostic_epochs)) != len(a.diagnostic_epochs) or any(
            epoch < 1 or epoch > a.epochs for epoch in a.diagnostic_epochs):
        p.error("diagnostic_epochs must be unique and within [1, epochs]")
    a.data_root = _repo_path(a.data_root)
    a.checkpoint = _repo_path(a.checkpoint)
    a.gradient_record = _repo_path(a.gradient_record)
    weight_record = json.loads(a.gradient_record.read_text())
    a.supervised_mixup_weight = float(weight_record["supervised_mixup_weight"])
    if not a.supervised_mixup_weight > 0:
        p.error("Gradient-derived supervised weight must be positive")
    a.calibration = _repo_path(a.calibration) if a.calibration else None
    if a.mode == "a2":
        calibration = json.loads(a.calibration.read_text())["dimensions"]
        a.temperatures = [float(calibration[d]["temperature"]) for d in DIMENSIONS]
        a.cutoffs = [float(calibration[d]["threshold"]) for d in DIMENSIONS]
        a.active = [bool(calibration[d]["enabled"]) for d in DIMENSIONS]
    else:
        a.temperatures = [1., 1., 1.]
        a.cutoffs = [a.confidence_cutoff] * 3
        a.active = [True] * 3
    a.out = _repo_path(a.out or f"saved_models/emobank/{a.mode}_lb{a.budget}_s{a.seed}")
    return a


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class UnlabeledCollator:
    """Identical tokenization to E1; no text edits or token masking."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        texts, ids = zip(*rows)
        tokens = self.tokenizer(list(texts), truncation=True,
                                max_length=self.max_length, padding=True,
                                return_tensors="pt")
        return tokens, list(ids)


def load_model(a, device):
    backbone = BertVADRegressor(a.pretrain_path, dropout=a.dropout, offline=a.offline)
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = BertVADRCUS3D(backbone)
    model.load_state_dict(checkpoint["model"], strict=True)
    epoch0_metrics = torch.load(E1, map_location="cpu", weights_only=False)["dev_metrics"]
    return model.to(device), checkpoint.get("epochs"), epoch0_metrics


def _move(tokens, device):
    return {key: value.to(device) for key, value in tokens.items()}


def train_step(model, optimizer, labeled_batch, unlabeled_batch, a, device,
               supervised_rng, unsupervised_rng):
    labeled_tokens, targets, _ = labeled_batch
    labeled_tokens, targets = _move(labeled_tokens, device), targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    labeled = model(**labeled_tokens)
    regression = torch.nn.functional.l1_loss(labeled["regression"], targets)
    lam_l = float(supervised_rng.beta(a.mixup_alpha, a.mixup_alpha))
    lam_l = max(lam_l, 1.0 - lam_l)
    supervised, ties = supervised_close_mixup_loss(
        model, labeled["features"], targets, lam_l)
    unsupervised = regression.new_zeros(())
    accepted = torch.zeros(3, dtype=torch.long, device=device)
    possible = torch.zeros(3, dtype=torch.long, device=device)
    if a.mode in ("a1", "a2"):
        if unlabeled_batch is None:
            raise ValueError("A1/A2 need unlabeled data")
        unlabeled_tokens = _move(unlabeled_batch[0], device)
        # Text/tokens are identical in both views. Eval teacher is deterministic,
        # student differs only by dropout. Fork keeps labeled RNG matched to R0.
        cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            model.eval()
            with torch.no_grad():
                teacher = model(**unlabeled_tokens)["rank_logits"]
            model.train()
            student = model(**unlabeled_tokens)["features"]
            lam_u = float(unsupervised_rng.beta(a.mixup_alpha, a.mixup_alpha))
            lam_u = max(lam_u, 1.0 - lam_u)
            unsupervised, accepted, possible = unsupervised_close_mixup_loss(
                model, teacher, student, lam_u, a.cutoffs,
                temperature=a.temperatures, active=a.active)
    loss = (regression + a.supervised_mixup_weight * supervised
            + a.unsupervised_mixup_weight * unsupervised)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite training loss")
    loss.backward()
    if a.clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad_norm)
    optimizer.step()
    return {"loss": float(loss.detach()), "regression": float(regression.detach()),
            "supervised_mixup": float(supervised.detach()),
            "unsupervised_mixup": float(unsupervised.detach()),
            "weighted_supervised_mixup": float((a.supervised_mixup_weight * supervised).detach()),
            "weighted_unsupervised_mixup": float((a.unsupervised_mixup_weight * unsupervised).detach()),
            "accepted": accepted.cpu().tolist(), "possible": possible.cpu().tolist(),
            "ties": ties.cpu().tolist(),
            "regression_shape": list(labeled["regression"].shape),
            "rank_shape": list(labeled["rank_logits"].shape)}


def main(argv=None):
    a = parse_args(argv)
    set_seed(a.seed)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if a.device == "cuda" or
                          (a.device == "auto" and torch.cuda.is_available()) else "cpu")
    if not a.smoke_steps and a.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {a.out}")
    labeled, unlabeled, dev, split_hashes = load_splits_no_test()
    print(f"SPLITS labeled={len(labeled)} unlabeled={len(unlabeled)} dev={len(dev)} "
          "train/dev IDs disjoint; Test unopened", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(a.pretrain_path, use_fast=True,
                                               local_files_only=a.offline)
    gold_collate = TextCollator(tokenizer, a.max_length)
    labeled_loader = DataLoader(GoldData(labeled), batch_size=a.batch_size,
                                shuffle=True, num_workers=0, collate_fn=gold_collate,
                                generator=torch.Generator().manual_seed(a.seed))
    unlabeled_loader = None
    if a.mode in ("a1", "a2"):
        unlabeled_loader = DataLoader(UnlabeledData(unlabeled),
                                      batch_size=a.batch_size * a.unlabeled_ratio,
                                      shuffle=True, num_workers=0,
                                      collate_fn=UnlabeledCollator(tokenizer, a.max_length),
                                      generator=torch.Generator().manual_seed(a.seed + 10000))
    dev_loader = DataLoader(GoldData(dev), batch_size=a.eval_batch_size,
                            shuffle=False, num_workers=0, collate_fn=gold_collate)
    model, source_epoch, epoch0_metrics = load_model(a, device)
    print(f"STRICT_CHECKPOINT_LOAD=PASS source_epoch={source_epoch} "
          f"optimizer_steps_per_epoch={len(labeled_loader)} mode={a.mode}", flush=True)
    print(f"EPOCH_0_DEV_MACRO_MAE_1TO5={epoch0_metrics['macro_mae_original_scale']:.6f}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    config = {**vars(a), "data_root": str(a.data_root), "checkpoint": str(a.checkpoint),
              "gradient_record": str(a.gradient_record),
              "calibration": str(a.calibration) if a.calibration else None,
              "out": str(a.out), "config": str(a.config) if a.config else None,
              "repository_commit": current_git_commit(), "checkpoint_sha256": _sha256(a.checkpoint),
              "gradient_record_sha256": _sha256(a.gradient_record),
              "calibration_sha256": _sha256(a.calibration) if a.calibration else None,
              "initial_rank_head_sha256": hashlib.sha256(b"".join(
                  tensor.detach().cpu().numpy().tobytes()
                  for tensor in model.rank_head.parameters())).hexdigest(),
              "split_sha256": split_hashes,
              "source_epoch": source_epoch, "steps_per_epoch": len(labeled_loader),
              "rng_control": "independent labeled/unlabeled sampler generators; pseudo-view dropout in fork_rng",
              "pseudo_loss_normalization": "masked two-local-rank sum / all unique candidate pairs across V,A,D",
              "mixup_rng_control": "separate labeled and unlabeled NumPy generators",
              "test_access": "none"}
    supervised_rng = np.random.default_rng(a.seed + 21001)
    unsupervised_rng = np.random.default_rng(a.seed + 31001)
    if a.smoke_steps:
        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None
        for step, batch in enumerate(labeled_loader, start=1):
            ub = next(unlabeled_iter) if a.mode != "a0" else None
            report = train_step(model, optimizer, batch, ub, a, device,
                                supervised_rng, unsupervised_rng)
            print(f"SMOKE step={step} {report}", flush=True)
            if step >= a.smoke_steps:
                break
        model.eval()
        with torch.inference_mode():
            tokens, targets, _ = next(iter(dev_loader))
            output = model(**_move(tokens, device))["regression"].cpu().numpy()
        metrics = metric_report(targets.numpy(), output)
        print(f"SMOKE_DEV_BATCH_MAE_1TO5={metrics['macro_mae_original_scale']:.6f}", flush=True)
        print("SMOKE_ONLY no output files; full Dev and Test not evaluated", flush=True)
        return

    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (a.out / "epoch0_reference.json").write_text(
        json.dumps({"source_checkpoint": str(a.checkpoint), "source_epoch": source_epoch,
                    "checkpoint_sha256": config["checkpoint_sha256"],
                    "dev_metrics": epoch0_metrics}, indent=2) + "\n", encoding="utf-8")
    best, best_epoch = math.inf, 0
    history = [{"epoch": 0, "steps": 0,
                "dev_macro_mae_1to5": epoch0_metrics["macro_mae_original_scale"],
                "dev_macro_ccc": float(np.mean([epoch0_metrics[d]["ccc"] for d in DIMENSIONS])),
                **{f"dev_{d}_{k}": epoch0_metrics[d][k] for d in DIMENSIONS
                   for k in ("mae", "rmse", "pearson", "ccc")}}]
    pd.DataFrame(history).to_csv(a.out / "training_history.csv", index=False)
    unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None
    for epoch in range(1, a.epochs + 1):
        sums = {key: 0.0 for key in (
            "loss", "regression", "supervised_mixup", "unsupervised_mixup",
            "weighted_supervised_mixup", "weighted_unsupervised_mixup")}
        accepted = np.zeros(3, dtype=np.int64)
        possible = np.zeros(3, dtype=np.int64)
        ties = np.zeros(3, dtype=np.int64)
        steps = 0
        for labeled_batch in labeled_loader:
            ub = None
            if a.mode != "a0":
                try:
                    ub = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    ub = next(unlabeled_iter)
            result = train_step(model, optimizer, labeled_batch, ub, a, device,
                                supervised_rng, unsupervised_rng)
            for key in sums:
                sums[key] += result[key]
            accepted += result["accepted"]
            possible += result["possible"]
            ties += result["ties"]
            steps += 1
        gold, predictions, ids = predict(model.backbone, dev_loader, device)
        metrics = metric_report(gold, predictions)
        mae = metrics["macro_mae_original_scale"]
        row = {"epoch": epoch, "steps": steps, "dev_macro_mae_1to5": mae,
               "dev_macro_ccc": float(np.mean([metrics[d]["ccc"] for d in DIMENSIONS])),
               **{f"dev_{d}_{k}": metrics[d][k] for d in DIMENSIONS
                  for k in ("mae", "rmse", "pearson", "ccc")},
               **{f"train_{key}": val / steps for key, val in sums.items()}}
        for index, dim in enumerate(DIMENSIONS):
            row[f"pseudo_accepted_{dim}"] = int(accepted[index])
            row[f"pseudo_possible_{dim}"] = int(possible[index])
            row[f"pseudo_keep_{dim}"] = float(accepted[index] / possible[index]) if possible[index] else 0.0
            row[f"gold_tie_pairs_{dim}"] = int(ties[index])
        history.append(row)
        print(f"epoch={epoch} steps={steps} DevMAE={mae:.6f} "
              + " ".join(f"{d}:MAE={metrics[d]['mae']:.4f},CCC={metrics[d]['ccc']:.3f},keep={row[f'pseudo_keep_{d}']:.3f}"
                         for d in DIMENSIONS), flush=True)
        if mae < best - a.min_delta:
            best, best_epoch = mae, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "dev_metrics": metrics, "run_config": config}, a.out / "model_best.pth")
            (a.out / "dev_metrics_best.json").write_text(
                json.dumps({"epoch": epoch, **metrics}, indent=2) + "\n", encoding="utf-8")
            frame = pd.DataFrame({"id": ids})
            for index, dim in enumerate(DIMENSIONS):
                frame[f"gold_{dim}"] = gold[:, index] * 4 + 1
                frame[f"pred_{dim}"] = predictions[:, index] * 4 + 1
            frame.to_csv(a.out / "dev_predictions_best.csv", index=False)
        if epoch in a.diagnostic_epochs:
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "dev_metrics": metrics, "run_config": config},
                       a.out / f"model_epoch_{epoch:02d}.pth")
        pd.DataFrame(history).to_csv(a.out / "training_history.csv", index=False)
    print(f"BEST epoch={best_epoch} DevMacroMAE_1to5={best:.6f} outputs={a.out}", flush=True)
    print("TEST path never opened", flush=True)


if __name__ == "__main__":
    main()
