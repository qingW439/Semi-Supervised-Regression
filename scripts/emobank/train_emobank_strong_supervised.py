#!/usr/bin/env python3
"""Frozen BERT + Ridge S0 and controlled S1/S2 supervised fine-tuning.

Only lb500/s0 training labels and Dev labels are read. Historical Ridge inputs
are read-only, and all run directories must be absent before a full run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    DIMENSIONS, NORM_COLUMNS, GoldData, TextCollator, read_gold,
)
from semilearn.nets.bert.bert_vad import BertVADRegressor  # noqa: E402
from scripts.emobank.train_emobank_supervised import (  # noqa: E402
    current_git_commit, metric_report, set_seed,
)

MODEL = "google/bert_uncased_L-4_H-512_A-8"
DATA = REPO.parent / "data" / "emobank_processed"
DEFAULT_RIDGE = REPO.parent / "strong_supervised_inputs" / "ridge_base"
FALLBACK_RIDGE = REPO.parent / "Semi-Supervised-Regression-rcus3d-adapt" / "saved_models/emobank/rcus3d_adapt/ridge_base"
RESULT_ROOT = REPO / "saved_models/emobank/strong_supervised"
EXPECTED_S0_MAE = 0.1819347408215205


class RidgeInitializedVAD(BertVADRegressor):
    """Exactly the E1 token encoder and pooling with a folded linear Ridge head."""

    def __init__(self, model_path=MODEL):
        super().__init__(model_path, dropout=0.1, offline=True)
        if self.num_features != 512:
            raise ValueError(f"Expected BERT-512, got {self.num_features}")
        self.head = nn.Linear(self.num_features, 3)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_fingerprint(model: nn.Module) -> str:
    """Stable hash of every initial tensor, including the frozen encoder."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def choose_ridge_dir(argument: Path | None) -> Path:
    chosen = argument or (DEFAULT_RIDGE if DEFAULT_RIDGE.is_dir() else FALLBACK_RIDGE)
    chosen = chosen.expanduser().resolve()
    required = ("base_pretrained_features.npz", "ridge_base_audit.json", "dev_predictions.csv")
    if not all((chosen / name).is_file() for name in required):
        raise FileNotFoundError(f"Incomplete read-only Ridge package: {chosen}")
    return chosen


def load_protocol_data():
    paths = {"train": DATA / "lb500_s0/train_lb.parquet", "dev": DATA / "dev_gold.parquet"}
    train = read_gold(paths["train"], expected_n=500)
    dev = read_gold(paths["dev"], expected_n=1000)
    record = REPO / "repro_records/emobank/labeled_ids_lb500_s0.csv"
    with record.open(encoding="utf-8", newline="") as stream:
        ids = {row["id"] for row in csv.DictReader(stream)}
    if ids != set(train.id.astype(str)):
        raise ValueError("Training IDs differ from the committed split record")
    assert_disjoint_ids(train.id.astype(str), dev.id.astype(str))
    return train, dev, {name: file_hash(path) for name, path in paths.items()}


def assert_disjoint_ids(train_ids, dev_ids):
    if set(train_ids) & set(dev_ids):
        raise ValueError("Train/Dev ID leakage")


def load_ridge_package(ridge_dir: Path, train: pd.DataFrame, dev: pd.DataFrame):
    audit = json.loads((ridge_dir / "ridge_base_audit.json").read_text(encoding="utf-8"))
    if audit["selected_alpha"] != 1000.0:
        raise ValueError("Historical Ridge alpha is not 1000")
    if abs(audit["dev_metrics"]["macro_mae_original_scale"] - EXPECTED_S0_MAE) > 1e-10:
        raise ValueError("Historical S0 metric differs from the fixed reference")
    with np.load(ridge_dir / "base_pretrained_features.npz", allow_pickle=False) as saved:
        feature = {name: saved[name].copy() for name in (
            "x_train", "y_train", "x_dev", "y_dev", "train_ids", "dev_ids")}
    if feature["x_train"].shape != (500, 512) or feature["x_dev"].shape != (1000, 512):
        raise ValueError("Unexpected BERT feature shape")
    if feature["y_train"].shape != (500, 3) or feature["y_dev"].shape != (1000, 3):
        raise ValueError("Unexpected Ridge label shape")
    for split, frame in (("train", train), ("dev", dev)):
        if list(feature[f"{split}_ids"].astype(str)) != list(frame.id.astype(str)):
            raise ValueError(f"Cached {split} feature IDs or order differ from Parquet")
        if not np.allclose(feature[f"y_{split}"], frame[list(NORM_COLUMNS)].to_numpy(), atol=1e-7, rtol=0):
            raise ValueError(f"Cached {split} labels differ from Parquet")
    historical = pd.read_csv(ridge_dir / "dev_predictions.csv", dtype={"id": str})
    if list(historical.id) != list(feature["dev_ids"].astype(str)):
        raise ValueError("Historical Dev prediction IDs differ from feature cache")
    historical_norm = (historical[[f"pred_{dim}" for dim in DIMENSIONS]].to_numpy(dtype=np.float64) - 1) / 4
    return feature, audit, historical_norm


def fit_fixed_ridge(feature: dict):
    # The historical probe used this exact pipeline; alpha is fixed, never selected here.
    probe = make_pipeline(StandardScaler(), Ridge(alpha=1000.0))
    probe.fit(feature["x_train"], feature["y_train"])
    scaler, ridge = probe.steps[0][1], probe.steps[1][1]
    weight = np.asarray(ridge.coef_, dtype=np.float64) / np.asarray(scaler.scale_, dtype=np.float64)[None, :]
    bias = np.asarray(ridge.intercept_, dtype=np.float64) - weight @ np.asarray(scaler.mean_, dtype=np.float64)
    return probe, weight, bias


def install_linear_head(model: RidgeInitializedVAD, weight: np.ndarray, bias: np.ndarray):
    if weight.shape != (3, 512) or bias.shape != (3,):
        raise ValueError("Ridge coefficient shape mismatch")
    with torch.no_grad():
        model.head.weight.copy_(torch.as_tensor(weight, dtype=model.head.weight.dtype))
        model.head.bias.copy_(torch.as_tensor(bias, dtype=model.head.bias.dtype))


def assert_prediction_equivalence(reference: np.ndarray, candidate: np.ndarray,
                                  *, name: str, atol: float):
    if reference.shape != candidate.shape or not np.isfinite(candidate).all():
        raise ValueError(f"{name}: prediction shape or finiteness mismatch")
    error = float(np.max(np.abs(reference.astype(np.float64) - candidate.astype(np.float64))))
    if error > atol:
        raise ValueError(f"{name}: max normalized prediction difference {error:.9g} > {atol:.9g}")
    return error


def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels, predictions, ids = [], [], []
    with torch.inference_mode():
        for tokens, targets, batch_ids in loader:
            tokens = {name: value.to(device) for name, value in tokens.items()}
            output = model(**tokens)
            if output.shape != targets.shape or not torch.isfinite(output).all().item():
                raise RuntimeError("Invalid Dev prediction shape or nonfinite values")
            predictions.append(output.cpu().numpy())
            labels.append(targets.numpy())
            ids.extend(str(item) for item in batch_ids)
    return np.concatenate(labels), np.concatenate(predictions), ids


def make_report(y: np.ndarray, p: np.ndarray):
    report = metric_report(y, p)
    for index, dim in enumerate(DIMENSIONS):
        report[dim]["pred_std"] = float(np.std(p[:, index].astype(np.float64) * 4 + 1))
        report[dim]["target_std"] = float(np.std(y[:, index].astype(np.float64) * 4 + 1))
    return report


def freeze_for_stage(model: RidgeInitializedVAD, stage: str):
    if stage not in {"S1", "S2"}:
        raise ValueError(stage)
    for param in model.bert.parameters():
        param.requires_grad_(stage == "S2")
    if stage == "S1":
        for param in model.bert.encoder.layer[-1].parameters():
            param.requires_grad_(True)
    for param in model.head.parameters():
        param.requires_grad_(True)
    encoder_params = [p for p in model.bert.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    if not encoder_params or not head_params:
        raise RuntimeError("Missing trainable encoder or head parameters")
    return encoder_params, head_params


def assert_trainable_gradients(model: nn.Module, encoder_params, head_params):
    for label, params in (("encoder", encoder_params), ("head", head_params)):
        grads = [param.grad for param in params]
        if not grads or not any(g is not None and torch.any(g != 0).item() for g in grads):
            raise RuntimeError(f"No nonzero {label} gradient")
        if any(g is not None and not torch.isfinite(g).all().item() for g in grads):
            raise RuntimeError(f"Nonfinite {label} gradient")
    if any(p.grad is not None for p in model.bert.parameters() if not p.requires_grad):
        raise RuntimeError("Frozen BERT parameter received a gradient")


def history_row(epoch: int, train_l1: float | None, report: dict):
    row = {"epoch": epoch, "train_l1_norm": train_l1,
           "dev_macro_mae_original_scale": report["macro_mae_original_scale"]}
    for dim in DIMENSIONS:
        for metric in ("mae", "rmse", "pearson", "ccc", "pred_std", "target_std"):
            row[f"{dim}_{metric}"] = report[dim][metric]
    return row


def improves(candidate: float, best: float) -> bool:
    return bool(np.isfinite(candidate) and candidate < best)


def best_history_epoch(rows: list[dict]) -> int:
    """Keep epoch zero when all later Dev MAEs are no better."""
    return min(rows, key=lambda row: row["dev_macro_mae_original_scale"])["epoch"]


def write_best(out: Path, model: nn.Module, epoch: int, source: str,
               report: dict, y: np.ndarray, pred: np.ndarray, ids: list[str]):
    torch.save({"model": model.state_dict(), "epoch": epoch, "source": source,
                "dev_metrics": report}, out / "model_best.pth")
    (out / "dev_metrics_best.json").write_text(
        json.dumps({"epoch": epoch, "source": source, **report}, indent=2) + "\n", encoding="utf-8")
    prediction_frame(y, pred, ids).to_csv(out / "dev_predictions_best.csv", index=False)


def prediction_frame(y: np.ndarray, pred: np.ndarray, ids: list[str]):
    frame = pd.DataFrame({"id": ids})
    for index, dim in enumerate(DIMENSIONS):
        frame[f"gold_{dim}"] = y[:, index] * 4 + 1
        frame[f"pred_{dim}"] = pred[:, index] * 4 + 1
    return frame


def validate_config(config: dict):
    required = {"stage", "budget", "seed", "model", "max_length", "batch_size",
                "eval_batch_size", "epochs", "encoder_lr", "head_lr", "weight_decay",
                "clip_grad_norm", "dropout", "output_dir"}
    if not required.issubset(config):
        raise ValueError(f"Missing config keys: {required - set(config)}")
    if config["stage"] not in {"S1", "S2"} or config["budget"] != 500 or config["seed"] != 0:
        raise ValueError("This protocol is fixed to S1/S2, lb500, seed0")
    expected_lr = 1e-6 if config["stage"] == "S1" else 3e-7
    fixed = {"model": MODEL, "max_length": 128, "batch_size": 8,
             "eval_batch_size": 16, "weight_decay": 5e-4,
             "epochs": 25, "encoder_lr": expected_lr, "head_lr": 1e-5,
             "clip_grad_norm": 1.0, "dropout": 0.1}
    for name, value in fixed.items():
        if config[name] != value:
            raise ValueError(f"Protocol violation: {name}={config[name]!r}, expected {value!r}")
    out = (REPO / config["output_dir"]).resolve()
    expected_out = RESULT_ROOT / ("s1_lb500_s0" if config["stage"] == "S1" else "s2_lb500_s0")
    if out != expected_out.resolve():
        raise ValueError(f"Output must be the fixed stage path {expected_out}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ridge-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_steps < 0:
        parser.error("--smoke-steps must be nonnegative")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    out = validate_config(config)
    if not args.smoke_steps and out.exists():
        raise FileExistsError(f"New result directory already exists: {out}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    ridge_dir = choose_ridge_dir(args.ridge_dir)
    train, dev, hashes = load_protocol_data()
    feature, audit, historical = load_ridge_package(ridge_dir, train, dev)
    if hashes["train"] != audit["data_sha256"]["labeled"] or hashes["dev"] != audit["data_sha256"]["dev"]:
        raise ValueError("Current Parquet hashes differ from historical Ridge inputs")

    probe, weight, bias = fit_fixed_ridge(feature)
    ridge_pred = probe.predict(feature["x_dev"])
    errors = {"ridge_replay": assert_prediction_equivalence(
        historical, ridge_pred, name="historical Ridge replay", atol=2e-6)}
    set_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    collator = TextCollator(tokenizer, config["max_length"])
    dev_loader = DataLoader(GoldData(dev), batch_size=config["eval_batch_size"],
                            shuffle=False, collate_fn=collator)
    train_loader = DataLoader(GoldData(train), batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collator,
                              generator=torch.Generator().manual_seed(0))
    model = RidgeInitializedVAD(MODEL)
    install_linear_head(model, weight, bias)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        from_features = model.head(torch.as_tensor(feature["x_dev"], device=device)).cpu().numpy()
    errors["folded_head"] = assert_prediction_equivalence(
        ridge_pred, from_features, name="folded PyTorch head", atol=5e-6)
    y0, p0, ids0 = predict(model, dev_loader, device)
    if ids0 != list(feature["dev_ids"].astype(str)):
        raise ValueError("Fresh BERT Dev IDs differ from historical feature order")
    errors["full_model"] = assert_prediction_equivalence(
        ridge_pred, p0, name="full BERT + folded head", atol=5e-5)
    s0 = make_report(y0, p0)
    if abs(s0["macro_mae_original_scale"] - EXPECTED_S0_MAE) > 2e-5:
        raise ValueError("Fresh S0 Macro-MAE differs from historical Ridge reference")
    print("S0_EQUIVALENCE", json.dumps({"max_abs_normalized": errors,
           "dev_macro_mae": s0["macro_mae_original_scale"]}), flush=True)
    initial_state_sha256 = state_fingerprint(model)

    encoder_params, head_params = freeze_for_stage(model, config["stage"])
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": config["encoder_lr"]},
        {"params": head_params, "lr": config["head_lr"]}],
        weight_decay=config["weight_decay"])
    history = [history_row(0, None, s0)]
    best_mae, best_epoch, best_source = s0["macro_mae_original_scale"], 0, "S0_frozen_BERT_Ridge"
    if not args.smoke_steps:
        out.mkdir(parents=True, exist_ok=False)
        metadata = {"config": config, "ridge_dir": str(ridge_dir),
                    "ridge_sha256": {name: file_hash(ridge_dir / name) for name in (
                        "base_pretrained_features.npz", "ridge_base_audit.json", "dev_predictions.csv")},
                    "data_sha256": hashes, "ridge_equivalence_max_abs_norm": errors,
                    "initial_model_state_sha256": initial_state_sha256,
                    "git_commit": current_git_commit(), "test_access": "none"}
        (out / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (out / "s0_reference.json").write_text(json.dumps({"epoch": 0,
            "source": "S0_frozen_BERT_Ridge", **s0}, indent=2) + "\n", encoding="utf-8")
        prediction_frame(y0, p0, ids0).to_csv(out / "s0_dev_predictions.csv", index=False)
        torch.save({"model": model.state_dict(), "epoch": 0,
                    "source": "S0_frozen_BERT_Ridge", "dev_metrics": s0}, out / "model_epoch0.pth")
        write_best(out, model, 0, best_source, s0, y0, p0, ids0)
    steps = 0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss, count = 0.0, 0
        for tokens, target, _ in train_loader:
            tokens = {name: value.to(device) for name, value in tokens.items()}
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(**tokens)
            loss = torch.nn.functional.l1_loss(pred, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            if args.smoke_steps:
                assert_trainable_gradients(model, encoder_params, head_params)
            torch.nn.utils.clip_grad_norm_(encoder_params + head_params, config["clip_grad_norm"])
            optimizer.step()
            total_loss += loss.item() * len(target)
            count += len(target)
            steps += 1
            if args.smoke_steps and steps >= args.smoke_steps:
                print("SMOKE_OK", json.dumps({"stage": config["stage"], "steps": steps,
                      "last_loss_norm": loss.item(), "s0_macro_mae": best_mae}), flush=True)
                return
        y, pred, ids = predict(model, dev_loader, device)
        report = make_report(y, pred)
        history.append(history_row(epoch, total_loss / count, report))
        mae = report["macro_mae_original_scale"]
        if improves(mae, best_mae):
            best_mae, best_epoch, best_source = mae, epoch, config["stage"]
            write_best(out, model, epoch, best_source, report, y, pred, ids)
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
        print(f"{config['stage']} epoch={epoch} train_L1_norm={total_loss/count:.6f} "
              f"dev_macro_MAE={mae:.6f} best_epoch={best_epoch}", flush=True)
    if best_history_epoch(history) != best_epoch:
        raise RuntimeError("Saved best epoch differs from the full Dev history")
    (out / "summary.json").write_text(json.dumps({"stage": config["stage"],
        "s0_macro_mae": s0["macro_mae_original_scale"], "best_epoch": best_epoch,
        "best_source": best_source, "best_macro_mae": best_mae,
        "initial_model_state_sha256": initial_state_sha256,
        "epochs_completed": config["epochs"], "test_access": "none"}, indent=2) + "\n", encoding="utf-8")
    print("FINISHED", config["stage"], "best", best_epoch, best_mae, "output", out, flush=True)


if __name__ == "__main__":
    main()
