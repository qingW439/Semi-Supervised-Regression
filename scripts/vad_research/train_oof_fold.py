#!/usr/bin/env python3
"""P1-A: train one strict OOF fold (400 labeled + 7562 unlabeled) and predict
the held-out 100 labeled TRAIN items.

No DEV/TEST access.  No hidden unlabeled labels.  P1 is trained for a fixed
number of epochs and JACF for a fixed number of steps; no early stopping and no
DEV-based checkpoint selection.

Outputs (per fold):
  p1_final.pth, jacf_final.pth
  oof_predictions.parquet   id, fold, y_*, raw_*, vrr_*
  features_holdout.npy, features_labeled.npy
  centering_stats.json
  label_isolation.json
  run_config.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    DIMENSIONS, NORM_COLUMNS, GoldData, TextCollator, read_gold,
)
from scripts.vad_research.train_roberta_vad_supervised import (  # noqa: E402
    RobertaVADRegressor, git_commit, set_seed, sha256_file,
)
from scripts.vad_research.train_acvr import (  # noqa: E402
    UnlabeledTexts, ema_step, encode, knn_calibrate, mask_tokens, mc_predict, predict,
)
from scripts.vad_research.jacf_modes import evidence, fixed_alpha_objective  # noqa: E402

DIMS = ("V", "A", "D")


class Texts(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


def collate_texts(tokenizer, max_length=128):
    def f(batch):
        return tokenizer(list(batch), padding=True, truncation=True,
                         max_length=max_length, return_tensors="pt")
    return f


@torch.no_grad()
def predict_texts(model, texts, tok, device, bs=64):
    model.eval()
    out = []
    dl = DataLoader(Texts(texts), batch_size=bs, shuffle=False,
                    collate_fn=collate_texts(tok))
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        out.append(model(**b).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--p1-epochs", type=int, default=10)
    ap.add_argument("--jacf-steps", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ulb-batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr-p1", type=float, default=2e-5)
    ap.add_argument("--lr-jacf", type=float, default=5e-6)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip-grad-norm", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--lambda-u", type=float, default=1.0)
    ap.add_argument("--mc-passes", type=int, default=5)
    ap.add_argument("--jacf-k", type=int, default=20)
    ap.add_argument("--jacf-tau", type=float, default=0.1)
    ap.add_argument("--alpha-fixed", type=float, default=0.75)
    ap.add_argument("--eval-knn-alpha", type=float, default=0.4)
    ap.add_argument("--eval-knn-k", type=int, default=20)
    ap.add_argument("--eval-scale", type=float, default=1.2)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    set_seed(a.fold)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    out = a.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    lb = read_gold(a.data_root / "lb500_s0" / "train_lb.parquet", expected_n=500)
    ulb = pd.read_parquet(a.data_root / "lb500_s0" / "train_ulb.parquet")
    if set(ulb.columns) != {"id", "text"}:
        raise ValueError("train_ulb must be id/text only")
    man = json.loads(a.manifest.read_text())
    hold_ids = man["fold_ids"][str(a.fold)]
    train_ids = [i for i in lb["id"] if i not in set(hold_ids)]
    if len(hold_ids) != 100 or len(train_ids) != 400:
        raise ValueError("bad fold sizes")
    if set(train_ids) & set(hold_ids):
        raise ValueError("held-out ids leaked into training ids")
    lb_train = lb[lb["id"].isin(train_ids)].reset_index(drop=True)
    lb_hold = lb[lb["id"].isin(hold_ids)].reset_index(drop=True)
    if len(lb_train) != 400 or len(lb_hold) != 100:
        raise ValueError("bad labeled split")
    # labels of the held-out 100 are loaded for output/evaluation only; they are
    # never passed to the model, optimizer, anchor bank, or calibrator fit.
    y_hold = lb_hold.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32)

    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(device)

    # ---------------- P1: supervised, fixed epochs, no DEV ----------------
    p1_dl = DataLoader(GoldData(lb_train), batch_size=a.batch_size, shuffle=True,
                       collate_fn=TextCollator(tok, 128),
                       generator=torch.Generator().manual_seed(17_000 + a.fold))
    opt1 = torch.optim.AdamW(model.parameters(), lr=a.lr_p1, weight_decay=a.weight_decay)
    p1_losses = []
    for epoch in range(a.p1_epochs):
        model.train()
        total, n = 0.0, 0
        for t, y, _ in p1_dl:
            t = {k: v.to(device) for k, v in t.items()}
            y = y.to(device)
            pred = model(**t)
            loss = F.l1_loss(pred, y)
            opt1.zero_grad(set_to_none=True)
            loss.backward()
            if a.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad_norm)
            opt1.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        p1_losses.append(total / max(1, n))
        print(f"P1 epoch {epoch + 1}/{a.p1_epochs} L1={p1_losses[-1]:.6f}", flush=True)
        if a.dry_run and epoch + 1 >= 1:
            break
    torch.save({"model": model.state_dict(), "fold": a.fold, "p1_epochs": a.p1_epochs,
                "p1_losses": p1_losses}, out / "p1_final.pth")

    # ---------------- JACF: fixed alpha, fixed steps, no DEV ----------------
    teacher = copy.deepcopy(model).to(device)
    teacher.requires_grad_(False)
    teacher.eval()
    opt2 = torch.optim.AdamW(model.parameters(), lr=a.lr_jacf, weight_decay=a.weight_decay)
    l_feat = encode(model.encoder, lb_train["text"].tolist(), tok, device).to(device)
    u_feat = encode(model.encoder, ulb["text"].tolist(), tok, device).to(device)
    l_y = torch.tensor(lb_train.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32), device=device)
    id2row = {k: i for i, k in enumerate(ulb["id"].tolist())}
    lb_iter = iter(DataLoader(GoldData(lb_train), batch_size=a.batch_size, shuffle=True,
                              collate_fn=TextCollator(tok, 128),
                              generator=torch.Generator().manual_seed(17_000 + a.fold)))
    ulb_iter = iter(DataLoader(UnlabeledTexts(ulb["text"].tolist(), ulb["id"].tolist()),
                               batch_size=a.ulb_batch_size, shuffle=True,
                               collate_fn=lambda batch: (
                                   tok([t for t, _ in batch], padding=True, truncation=True,
                                       max_length=128, return_tensors="pt"),
                                   [i for _, i in batch]),
                               generator=torch.Generator().manual_seed(19_000 + a.fold)))
    jacf_losses = []
    for step in range(a.jacf_steps):
        model.train()
        try:
            lb_tok, lb_y, _ = next(lb_iter)
        except StopIteration:
            lb_iter = iter(DataLoader(GoldData(lb_train), batch_size=a.batch_size, shuffle=True,
                                      collate_fn=TextCollator(tok, 128),
                                      generator=torch.Generator().manual_seed(17_000 + a.fold + step + 1)))
            lb_tok, lb_y, _ = next(lb_iter)
        try:
            ulb_tok, ulb_ids = next(ulb_iter)
        except StopIteration:
            ulb_iter = iter(DataLoader(UnlabeledTexts(ulb["text"].tolist(), ulb["id"].tolist()),
                                       batch_size=a.ulb_batch_size, shuffle=True,
                                       collate_fn=lambda batch: (
                                           tok([t for t, _ in batch], padding=True, truncation=True,
                                               max_length=128, return_tensors="pt"),
                                           [i for _, i in batch]),
                                       generator=torch.Generator().manual_seed(19_000 + a.fold + step + 1)))
            ulb_tok, ulb_ids = next(ulb_iter)
        lb_tok = {k: v.to(device) for k, v in lb_tok.items()}
        lb_y = lb_y.to(device)
        ulb_tok = {k: v.to(device) for k, v in ulb_tok.items()}
        ulb_tok_s = mask_tokens(ulb_tok, tok, a.mask_prob)
        mc = mc_predict(teacher, ulb_tok, a.mc_passes)
        rows = torch.tensor([id2row[i] for i in ulb_ids], device=device)
        ev = evidence(mc, u_feat[rows], l_feat, l_y, k=a.jacf_k, tau=a.jacf_tau,
                      eps_t=1e-4, eps_a=1e-4)
        target, rel = fixed_alpha_objective(ev, alpha=a.alpha_fixed, rel_mode="none")
        pred_lb = model(**lb_tok)
        loss_sup = F.l1_loss(pred_lb, lb_y)
        pred_s = model(**ulb_tok_s)
        unsup = ((pred_s - target).abs() * rel).sum() / rel.sum().clamp_min(1e-6)
        loss = loss_sup + a.lambda_u * unsup
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        if a.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad_norm)
        opt2.step()
        ema_step(teacher, model, a.ema_decay)
        jacf_losses.append(float(loss.detach()))
        if (step + 1) % 16 == 0 or step == 0:
            print(f"JACF step {step + 1}/{a.jacf_steps} loss={jacf_losses[-1]:.6f}", flush=True)
        if a.dry_run and step + 1 >= 1:
            break
    torch.save({"model": model.state_dict(), "fold": a.fold, "jacf_steps": a.jacf_steps,
                "jacf_losses": jacf_losses, "alpha_fixed": a.alpha_fixed}, out / "jacf_final.pth")

    # ---------------- held-out prediction + frozen VRRC ----------------
    hold_dl = DataLoader(GoldData(lb_hold), batch_size=a.eval_batch_size, shuffle=False,
                         collate_fn=TextCollator(tok, 128))
    y_h, raw_h, hold_ids_pred = predict(model, hold_dl, device)
    hold_feat = encode(model.encoder, lb_hold["text"].tolist(), tok, device).to(device)
    raw_ulb = predict_texts(model, ulb["text"].tolist(), tok, device, bs=a.ulb_batch_size)
    mu_ulb = raw_ulb.mean(0).to(device)
    sigma_ulb = raw_ulb.std(0, unbiased=False).to(device)
    q = knn_calibrate(torch.tensor(raw_h, device=device), hold_feat, l_feat, l_y,
                      a.eval_knn_k, a.jacf_tau, a.eval_knn_alpha, 0.3)
    q = mu_ulb + (q - mu_ulb) * a.eval_scale
    vrr_h = q.cpu().numpy()

    df = pd.DataFrame({"id": hold_ids_pred, "fold": a.fold})
    for j, d in enumerate(DIMS):
        df[f"y_{d}"] = y_h[:, j]
        df[f"raw_{d}"] = raw_h[:, j]
        df[f"vrr_{d}"] = vrr_h[:, j]
    df.to_parquet(out / "oof_predictions.parquet", index=False)
    np.save(out / "features_holdout.npy", hold_feat.cpu().numpy())
    np.save(out / "features_labeled.npy", l_feat.cpu().numpy())
    (out / "centering_stats.json").write_text(json.dumps({
        "mu_ulb_norm": mu_ulb.cpu().tolist(),
        "sigma_ulb_norm": sigma_ulb.cpu().tolist(),
        "mu_ulb_original": (mu_ulb.cpu() * 4 + 1).tolist(),
        "sigma_ulb_original": (sigma_ulb.cpu() * 4).tolist(),
        "n_ulb": len(ulb),
    }, indent=2) + "\n")
    (out / "label_isolation.json").write_text(json.dumps({
        "fold": a.fold,
        "train_n": len(lb_train), "holdout_n": len(lb_hold),
        "train_holdout_overlap": len(set(train_ids) & set(hold_ids)),
        "anchor_bank_n": len(lb_train),
        "holdout_labels_used_in_training": False,
        "raw_csv_read": False,
        "test_read": False,
        "dev_read": False,
        "p1_epochs": a.p1_epochs, "jacf_steps": a.jacf_steps,
        "note": "held-out labels are used only to write the OOF output row",
    }, indent=2) + "\n")
    (out / "run_config.json").write_text(json.dumps({
        **vars(a), "device": str(device), "git_commit": git_commit(),
        "fold": a.fold, "p1_losses": p1_losses, "jacf_loss_last": jacf_losses[-1],
        "train_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "holdout_ids_sha256": hashlib.sha256("\n".join(hold_ids).encode()).hexdigest(),
    }, indent=2, default=str) + "\n")
    print("DONE", out)


if __name__ == "__main__":
    main()
