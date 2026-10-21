#!/usr/bin/env python3
"""Read-only audit, head-only warmup, gradient diagnosis, strict 5-fold OOF.

All new outputs require absent paths. No Test path is opened in this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    DIMENSIONS, NORM_COLUMNS, GoldData, TextCollator,
)
from semilearn.nets.bert.bert_vad import BertVADRegressor  # noqa: E402
from semilearn.nets.bert.bert_vad_rcus_adapt import (  # noqa: E402
    BertVADRCUS3D, pairwise_rank_loss, supervised_close_mixup_loss, unique_pairs,
)
from scripts.emobank.rcus3d_adapt_common import B0, E1, REPO as COMMON_REPO, load_splits_no_test, sha256  # noqa: E402
from scripts.emobank.train_emobank_supervised import metric_report, set_seed  # noqa: E402

MODEL = "google/bert_uncased_L-4_H-512_A-8"
ROOT = REPO / "saved_models/emobank/rcus3d_adapt"


def absent(path: Path):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.mkdir(parents=True, exist_ok=False)


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)


def e1_model(device):
    backbone = BertVADRegressor(MODEL, dropout=.1, offline=True)
    ckpt = torch.load(E1, map_location="cpu", weights_only=False)
    backbone.load_state_dict(ckpt["model"], strict=True)
    return backbone.to(device)


def features(model, frame, tok, device, batch_size=16):
    loader = DataLoader(GoldData(frame), batch_size=batch_size, shuffle=False,
                        collate_fn=TextCollator(tok, 128))
    model.eval()
    xs, ys, ids = [], [], []
    with torch.inference_mode():
        for tokens, y, batch_ids in loader:
            x = model.encode(**{k: v.to(device) for k, v in tokens.items()})
            xs.append(x.cpu().numpy())
            ys.append(y.numpy())
            ids.extend(batch_ids)
    return np.concatenate(xs), np.concatenate(ys), ids


def compare_metrics(gold, pred):
    m = metric_report(gold, pred)
    for k, d in enumerate(DIMENSIONS):
        m[d]["pred_std"] = float(np.std(pred[:, k] * 4 + 1))
        m[d]["target_std"] = float(np.std(gold[:, k] * 4 + 1))
        m[d]["pred_bias"] = float(np.mean((pred[:, k] - gold[:, k]) * 4))
    return m


def audit(out: Path, device):
    lb, _, dev, hashes = load_splits_no_test()
    model = e1_model(device)
    tok = tokenizer()
    x_train, y_train, train_ids = features(model, lb, tok, device)
    x_dev, y_dev, dev_ids = features(model, dev, tok, device)
    absent(out)
    np.savez_compressed(out / "e1_frozen_features.npz", x_train=x_train,
                        y_train=y_train, x_dev=x_dev, y_dev=y_dev,
                        train_ids=np.asarray(train_ids), dev_ids=np.asarray(dev_ids))
    candidates = [0.01, 0.1, 1., 10., 100., 1000.]
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    cv = []
    for alpha in candidates:
        scores = []
        for tr, val in folds.split(x_train):
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            ridge.fit(x_train[tr], y_train[tr])
            scores.append(float(np.mean(np.abs((ridge.predict(x_train[val]) - y_train[val]) * 4))))
        cv.append({"alpha": alpha, "fold_macro_mae": scores, "mean_macro_mae": float(np.mean(scores))})
    chosen = min(cv, key=lambda row: (row["mean_macro_mae"], row["alpha"]))["alpha"]
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=chosen)).fit(x_train, y_train)
    preds = {"train_mean": np.broadcast_to(y_train.mean(0), y_dev.shape).copy(),
             "train_median": np.broadcast_to(np.median(y_train, axis=0), y_dev.shape).copy(),
             "ridge_frozen_e1": ridge.predict(x_dev)}
    source = {"B0": B0 / "dev_predictions_best.csv",
              "E1": E1.parent / "dev_predictions_best.csv"}
    for name, path in source.items():
        frame = pd.read_csv(path).set_index("id").loc[dev_ids]
        preds[name] = (frame[[f"pred_{d}" for d in DIMENSIONS]].to_numpy() - 1) / 4
    results = {name: compare_metrics(y_dev, value) for name, value in preds.items()}
    prediction_frame = pd.DataFrame({"id": dev_ids})
    for name, value in preds.items():
        for index, dim in enumerate(DIMENSIONS):
            prediction_frame[f"{name}_{dim}"] = value[:, index] * 4 + 1
    prediction_frame.to_csv(out / "supervised_audit_dev_predictions.csv", index=False)
    save_json(out / "supervised_audit.json", {
        "source_e1_sha256": sha256(E1), "source_b0_metrics_sha256": sha256(B0 / "dev_metrics_best.json"),
        "data_sha256": hashes, "ridge_cv": cv, "ridge_selected_alpha": chosen,
        "ridge_cv_training_only": True, "results": results, "test_access": "none"})
    print("AUDIT", {k: round(v["macro_mae_original_scale"], 6) for k, v in results.items()},
          "ridge_alpha", chosen, flush=True)


def ridge_base(out: Path, device):
    """Leakage-free Ridge probe: base pretrained BERT never saw EmoBank labels."""
    lb, _, dev, hashes = load_splits_no_test()
    set_seed(0)
    backbone = BertVADRegressor(MODEL, dropout=.1, offline=True).to(device)
    tok = tokenizer()
    x_train, y_train, train_ids = features(backbone, lb, tok, device)
    x_dev, y_dev, dev_ids = features(backbone, dev, tok, device)
    grid = [0.01, 0.1, 1., 10., 100., 1000.]
    cv = []
    for alpha in grid:
        fold_mae = []
        for tr, held in KFold(n_splits=5, shuffle=True, random_state=0).split(x_train):
            probe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            probe.fit(x_train[tr], y_train[tr])
            fold_mae.append(float(np.mean(np.abs((probe.predict(x_train[held]) - y_train[held]) * 4))))
        cv.append({"alpha": alpha, "fold_macro_mae": fold_mae,
                   "mean_macro_mae": float(np.mean(fold_mae))})
    chosen = min(cv, key=lambda row: (row["mean_macro_mae"], row["alpha"]))["alpha"]
    probe = make_pipeline(StandardScaler(), Ridge(alpha=chosen)).fit(x_train, y_train)
    pred = probe.predict(x_dev)
    absent(out)
    np.savez_compressed(out / "base_pretrained_features.npz", x_train=x_train, y_train=y_train,
                        x_dev=x_dev, y_dev=y_dev, train_ids=np.asarray(train_ids), dev_ids=np.asarray(dev_ids))
    frame = pd.DataFrame({"id": dev_ids})
    for i, dim in enumerate(DIMENSIONS):
        frame[f"pred_{dim}"] = pred[:, i] * 4 + 1
    frame.to_csv(out / "dev_predictions.csv", index=False)
    metrics = compare_metrics(y_dev, pred)
    save_json(out / "ridge_base_audit.json", {"ridge_cv": cv, "selected_alpha": chosen,
              "dev_metrics": metrics, "data_sha256": hashes,
              "feature_source": "base pretrained BERT; no EmoBank label training",
              "test_access": "none"})
    print("RIDGE_BASE", chosen, metrics["macro_mae_original_scale"], flush=True)


def rank_diagnostics(head: BertVADRCUS3D, x: torch.Tensor, y: torch.Tensor):
    head.eval()
    with torch.no_grad():
        rank = head.rank(x)
        binary = rank[..., 0] - rank[..., 1]
        i, j = unique_pairs(len(y), y.device)
        logit = binary[i] - binary[j]
        delta = y[i] - y[j]
        target = (delta > 0).to(logit.dtype) + .5 * (delta == 0).to(logit.dtype)
        probs = torch.sigmoid(logit)
        conf = torch.maximum(probs, 1 - probs)
        loss, ties = pairwise_rank_loss(rank, y)
        result = {"overall_rank_bce": float(loss)}
        for k, dim in enumerate(DIMENSIONS):
            non_tie = delta[:, k] != 0
            q = conf[:, k].cpu().numpy()
            result[dim] = {"accuracy_excluding_ties": float(((probs[:, k] > .5) == (delta[:, k] > 0))[non_tie].float().mean()),
                           "tie_pairs": int(ties[k]),
                           "confidence_quantiles": np.quantile(q, [0, .25, .5, .75, .9, .95, .99, 1]).tolist(),
                           "keep_at_095": float((conf[:, k] >= .95).float().mean())}
        return result


def fit_head(head: BertVADRCUS3D, x: torch.Tensor, y: torch.Tensor,
             epochs: int, seed: int, lr: float):
    for p in head.backbone.parameters():
        p.requires_grad_(False)
    head.backbone.eval()
    optimizer = torch.optim.AdamW(head.rank_head.parameters(), lr=lr, weight_decay=0)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        head.rank_head.train()
        total = 0.
        steps = 0
        for idx in torch.randperm(len(x), generator=generator).split(8):
            if len(idx) < 2:
                continue
            logits = head.rank(x[idx.to(x.device)])
            loss, _ = pairwise_rank_loss(logits, y[idx.to(y.device)])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        history.append({"epoch": epoch, "train_rank_bce_mean": total / steps, "steps": steps})
    return history


def warmup(out: Path, audit_dir: Path, device):
    _, _, _, hashes = load_splits_no_test()
    cache = np.load(audit_dir / "e1_frozen_features.npz")
    set_seed(0)
    backbone = e1_model(device)
    model = BertVADRCUS3D(backbone).to(device)
    x = torch.from_numpy(cache["x_train"]).to(device)
    y = torch.from_numpy(cache["y_train"]).to(device)
    initial_backbone = {k: v.detach().clone() for k, v in backbone.state_dict().items()}
    before = rank_diagnostics(model, x, y)
    history = fit_head(model, x, y, 3, 3700, 1e-3)
    after = rank_diagnostics(model, x, y)
    if not all(torch.equal(v, backbone.state_dict()[k]) for k, v in initial_backbone.items()):
        raise RuntimeError("Frozen E1 backbone changed during head-only warmup")
    absent(out)
    ckpt = out / "warmup_model.pth"
    torch.save({"model": model.state_dict(), "source_e1_sha256": sha256(E1),
                "epochs": 3, "data_sha256": hashes}, ckpt)
    save_json(out / "warmup_record.json", {
        "before": before, "after": after, "history": history, "data_sha256": hashes,
        "e1_sha256": sha256(E1), "warmup_checkpoint_sha256": sha256(ckpt),
        "head_lr": 1e-3, "batch_size": 8, "epochs": 3, "backbone_and_regression_frozen": True,
        "test_access": "none"})
    print("WARMUP", {d: (before[d]["accuracy_excluding_ties"], after[d]["accuracy_excluding_ties"])
                     for d in DIMENSIONS}, flush=True)


def warm_model(path: Path, device):
    backbone = BertVADRegressor(MODEL, dropout=.1, offline=True)
    model = BertVADRCUS3D(backbone)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    for p in model.parameters():
        p.requires_grad_(True)
    return model.to(device)


def gradient(out: Path, warmup_path: Path, device):
    lb, _, _, hashes = load_splits_no_test()
    set_seed(0)
    model = warm_model(warmup_path, device)
    loader = DataLoader(GoldData(lb), batch_size=8, shuffle=True,
                        collate_fn=TextCollator(tokenizer(), 128),
                        generator=torch.Generator().manual_seed(8311))
    rng = np.random.default_rng(4922)
    params = tuple(model.backbone.bert.parameters())
    rows = []
    for batch_index, (tokens, labels, _) in enumerate(loader):
        if batch_index >= 8:
            break
        model.train()
        tokens = {k: v.to(device) for k, v in tokens.items()}
        labels = labels.to(device)
        output = model(**tokens)
        l1 = torch.nn.functional.l1_loss(output["regression"], labels)
        lam = float(rng.beta(.5, .5)); lam = max(lam, 1 - lam)
        mix, _ = supervised_close_mixup_loss(model, output["features"], labels, lam)
        g1 = torch.autograd.grad(l1, params, retain_graph=True, allow_unused=True)
        g2 = torch.autograd.grad(mix, params, allow_unused=True)
        n1 = torch.sqrt(sum((g.detach() ** 2).sum() for g in g1 if g is not None)).item()
        n2 = torch.sqrt(sum((g.detach() ** 2).sum() for g in g2 if g is not None)).item()
        dot = sum((a.detach() * b.detach()).sum() for a, b in zip(g1, g2)
                  if a is not None and b is not None).item()
        rows.append({"batch": batch_index, "l1": float(l1), "supervised_mixup": float(mix),
                     "norm_l1": n1, "norm_mixup": n2, "cosine": dot / (n1 * n2) if n1*n2 else None,
                     "lambda": lam})
    if len(rows) != 8 or min(row["norm_mixup"] for row in rows) <= 0:
        raise RuntimeError("Gradient diagnosis incomplete or zero mixup gradient")
    weight = .1 * np.median([r["norm_l1"] for r in rows]) / np.median([r["norm_mixup"] for r in rows])
    absent(out)
    save_json(out / "gradient_weight.json", {
        "target_encoder_gradient_ratio": .1, "supervised_mixup_weight": float(weight),
        "aggregation": "0.1 * median(L1 encoder norm) / median(Close Mixup encoder norm)",
        "rows": rows, "optimizer_step_count": 0, "data_sha256": hashes,
        "warmup_sha256": sha256(warmup_path), "test_access": "none"})
    print("GRADIENT_WEIGHT", weight, "median_cosine", np.median([r["cosine"] for r in rows]), flush=True)


def reliability(conf, correct, bins=10):
    edges = np.linspace(.5, 1., bins + 1)
    rows = []
    ece = 0.
    for idx in range(bins):
        mask = (conf >= edges[idx]) & ((conf < edges[idx+1]) if idx < bins-1 else (conf <= edges[idx+1]))
        count = int(mask.sum())
        acc = float(correct[mask].mean()) if count else None
        avg = float(conf[mask].mean()) if count else None
        if count:
            ece += count / len(conf) * abs(acc - avg)
        rows.append({"lower": float(edges[idx]), "upper": float(edges[idx+1]),
                     "count": count, "accuracy": acc, "mean_confidence": avg})
    return rows, float(ece)


def oof(out: Path, device):
    lb, _, _, hashes = load_splits_no_test()
    tok = tokenizer()
    dataset = GoldData(lb)
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    absent(out)
    all_logits, all_targets, all_folds = [], [], []
    fold_records = []
    for fold, (train_idx, held_idx) in enumerate(kfold.split(lb)):
        set_seed(9100 + fold)
        fold_dir = out / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=False)
        backbone = BertVADRegressor(MODEL, dropout=.1, offline=True).to(device)
        optimizer = torch.optim.AdamW(backbone.parameters(), lr=1e-5, weight_decay=5e-4)
        loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=8, shuffle=True,
                            collate_fn=TextCollator(tok, 128),
                            generator=torch.Generator().manual_seed(10100 + fold))
        for epoch in range(14):
            backbone.train()
            for tokens, labels, _ in loader:
                tokens = {k: v.to(device) for k, v in tokens.items()}
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.l1_loss(backbone(**tokens), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.)
                optimizer.step()
        x_tr, y_tr, ids_tr = features(backbone, lb.iloc[train_idx].reset_index(drop=True), tok, device)
        x_held, y_held, ids_held = features(backbone, lb.iloc[held_idx].reset_index(drop=True), tok, device)
        assert not set(ids_tr) & set(ids_held)
        model = BertVADRCUS3D(backbone).to(device)
        fit_head(model, torch.from_numpy(x_tr).to(device), torch.from_numpy(y_tr).to(device),
                 3, 11100 + fold, 1e-3)
        model.eval()
        with torch.inference_mode():
            logits = model.rank(torch.from_numpy(x_held).to(device))
            binary = logits[..., 0] - logits[..., 1]
            i, j = unique_pairs(len(held_idx), device)
            pair_logits = (binary[i] - binary[j]).cpu().numpy()
            pair_targets = ((torch.from_numpy(y_held).to(device)[i] - torch.from_numpy(y_held).to(device)[j]).cpu().numpy())
        all_logits.append(pair_logits)
        all_targets.append(pair_targets)
        all_folds.append(np.full(len(i), fold, dtype=np.int16))
        checkpoint = fold_dir / "model.pth"
        torch.save({"model": model.state_dict(), "train_ids": ids_tr, "heldout_ids": ids_held,
                    "fold": fold, "encoder_epochs": 14, "head_epochs": 3}, checkpoint)
        fold_records.append({"fold": fold, "train_size": len(train_idx), "heldout_size": len(held_idx),
                             "heldout_pairs": len(i), "checkpoint_sha256": sha256(checkpoint),
                             "train_id_hash": __import__("hashlib").sha256("\n".join(ids_tr).encode()).hexdigest(),
                             "heldout_id_hash": __import__("hashlib").sha256("\n".join(ids_held).encode()).hexdigest()})
        print("OOF_FOLD", fold, "pairs", len(i), flush=True)
        del model, backbone, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    fold_ids = np.concatenate(all_folds)
    np.savez_compressed(out / "oof_pairs.npz", logits=logits, target_deltas=targets, fold=fold_ids)
    dimensions = {}
    grid = [.5, .6, .7, .8, .85, .9, .925, .95, .975, .99]
    for idx, dim in enumerate(DIMENSIONS):
        valid = targets[:, idx] != 0
        scores = logits[valid, idx]
        gold = (targets[valid, idx] > 0).astype(np.float64)
        groups = fold_ids[valid]
        objective = lambda t: float(np.mean(np.logaddexp(0, scores / t) - gold * scores / t))
        opt = minimize_scalar(objective, bounds=(.05, 20.), method="bounded")
        temp = float(opt.x)
        prob = 1 / (1 + np.exp(-np.clip(scores / temp, -100, 100)))
        conf = np.maximum(prob, 1-prob)
        correct = (prob >= .5) == gold
        bins, ece = reliability(conf, correct)
        candidates = []
        for threshold in grid:
            mask = conf >= threshold
            counts = [int(((groups == f) & mask).sum()) for f in range(5)]
            fold_accuracy = [float(correct[(groups == f) & mask].mean()) if count else None
                             for f, count in enumerate(counts)]
            n = int(mask.sum())
            accuracy = float(correct[mask].mean()) if n else None
            qualifies = (n >= 200 and all(c >= 20 for c in counts)
                         and accuracy >= .8 and all(v >= .7 for v in fold_accuracy))
            candidates.append({"threshold": threshold, "kept": n,
                               "coverage": n / len(conf), "accuracy": accuracy,
                               "fold_counts": counts, "fold_accuracy": fold_accuracy,
                               "passes_quality_gate": qualifies})
        selected = next((r for r in candidates if r["passes_quality_gate"]), None)
        dimensions[dim] = {"temperature": temp, "enabled": selected is not None,
                           "threshold": selected["threshold"] if selected else 1.0,
                           "oof_accuracy": float(correct.mean()), "oof_ece": ece,
                           "non_tie_pairs": len(conf), "tie_pairs": int((~valid).sum()),
                           "reliability_bins": bins, "threshold_candidates": candidates}
    save_json(out / "calibration.json", {
        "dimensions": dimensions, "folds": fold_records, "data_sha256": hashes,
        "encoder_training": "5 independent BERT encoders; each starts from base pretrained weights; 14 fixed supervised epochs on fold train only",
        "rank_training": "3 head-only epochs on fold train only; holdout pairs scored after training",
        "threshold_gate": "lowest predeclared grid threshold with pooled accuracy>=0.8, each-fold accuracy>=0.7, >=200 pooled pairs and >=20 each-fold pairs",
        "threshold_grid": grid, "test_access": "none", "dev_label_access": "none"})
    print("OOF_CALIBRATION", {d: {k: dimensions[d][k] for k in ("temperature", "enabled", "threshold", "oof_accuracy", "oof_ece")}
                               for d in DIMENSIONS}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("audit", "ridge_base", "warmup", "gradient", "oof"))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = p.parse_args()
    device = torch.device(a.device)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if a.stage == "audit":
        audit(ROOT / "audit", device)
    elif a.stage == "ridge_base":
        ridge_base(ROOT / "ridge_base", device)
    elif a.stage == "warmup":
        warmup(ROOT / "warmup", ROOT / "audit", device)
    elif a.stage == "gradient":
        gradient(ROOT / "gradient", ROOT / "warmup/warmup_model.pth", device)
    else:
        oof(ROOT / "oof", device)


if __name__ == "__main__":
    main()
