#!/usr/bin/env python3
"""Nested OOF comparison of local residual correction mechanisms.

Outer 5-fold: 400 train / 100 eval.
Inner 5-fold: 320 teacher-training labels / 80 calibration anchors.
For each inner fold:
  * train a supervised RoBERTa teacher on 320 labels (fixed 10 epochs);
  * predict the 80 anchors (out-of-sample) and the outer 100 eval items;
  * fit residual / anchor corrections on the 80 anchors;
  * predict the outer 100 eval items.
Average the five inner-model predictions for each outer eval item, then
aggregate the five outer folds into 500 strictly nested OOF predictions.

Methods:
  T0 teacher raw
  T1 JACF (0.25 teacher + 0.75 kNN anchor target)
  T2 BC (teacher + 0.75 * kNN residual)
  T3 KNN-R (teacher + alpha * kNN residual; alpha selected inside the 80
            anchors by 4-fold CV)
  T4 global residual (teacher + mean anchor residual)
  T5 residual permutation (shuffle anchor residuals before correction)

Neighbour graphs always use one frozen pretrained RoBERTa feature space.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    GoldData, NORM_COLUMNS, TextCollator, read_gold,
)
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402
from scripts.vad_research.train_acvr import encode  # noqa: E402

DIMS = ("V", "A", "D")
METHODS = ("T0", "T1", "T2", "T3", "T4", "T5")


@torch.no_grad()
def predict_texts(model, texts, tok, device, bs=64):
    model.eval(); out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=128, return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        out.append(model(**b).float().cpu())
    return torch.cat(out)


def train_teacher(lb_train, tok, pretrain, device, epochs=10, seed=0):
    model = RobertaVADRegressor(str(pretrain), dropout=0.1, offline=True).to(device)
    dl = DataLoader(GoldData(lb_train), batch_size=16, shuffle=True,
                    collate_fn=TextCollator(tok, 128),
                    generator=torch.Generator().manual_seed(17_000 + seed))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    for _ in range(epochs):
        model.train()
        for t, y, _ in dl:
            t = {k: v.to(device) for k, v in t.items()}
            y = y.to(device)
            pred = model(**t)
            loss = F.l1_loss(pred, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    return model


def knn_w(q, bank, k=20, tau=0.1):
    sim = q @ bank.T
    kk = min(k, bank.shape[0])
    topv, topi = sim.topk(kk, dim=1)
    return torch.softmax(topv / tau, dim=1), topi


def knn_residual(q, bank_feat, r_bank, k=20, tau=0.1):
    w, idx = knn_w(q, bank_feat, k, tau)
    return (w.unsqueeze(-1) * r_bank[idx]).sum(1)


def select_alpha(q_anchor, t_anchor, y_anchor, r_bank, k=20, tau=0.1, seed=0):
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    kf = KFold(4, shuffle=True, random_state=seed)
    best, best_a = float("inf"), 0.5
    for a in alphas:
        errs = []
        for tr, va in kf.split(q_anchor):
            pred_r = knn_residual(q_anchor[va], q_anchor[tr], r_bank[tr], k, tau)
            pred = t_anchor[va] + a * pred_r
            errs.append(float(torch.mean(torch.abs(pred - y_anchor[va]))))
        e = float(np.mean(errs))
        if e < best:
            best, best_a = e, a
    return best_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    lb = read_gold(a.data_root / "lb500_s0" / "train_lb.parquet", expected_n=500)
    y = lb.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32)
    man = json.loads(a.manifest.read_text())
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    enc = AutoModel.from_pretrained(str(a.pretrain_path), local_files_only=True).to(device).eval()
    feat = encode(enc, lb["text"].tolist(), tok, device).cpu().numpy()
    del enc; torch.cuda.empty_cache()

    rows = []
    all_alphas = {}
    for f in [int(x) for x in a.folds.split(",")]:
        hold_ids = man["fold_ids"][str(f)]
        train_ids = [i for i in lb["id"] if i not in set(hold_ids)]
        tr_pos = lb.index[lb["id"].isin(train_ids)].to_numpy()
        ev_pos = lb.index[lb["id"].isin(hold_ids)].to_numpy()
        acc = {m: np.zeros((len(ev_pos), 3), np.float32) for m in METHODS}
        acc_fixed = np.zeros((len(ev_pos), 3), np.float32)
        alphas = []
        kf = KFold(a.inner_folds, shuffle=True, random_state=1000 + f)
        for j, (itr, iva) in enumerate(kf.split(tr_pos)):
            pos_tr, pos_anchor = tr_pos[itr], tr_pos[iva]
            lb_tr = lb.iloc[pos_tr].reset_index(drop=True)
            model = train_teacher(lb_tr, tok, a.pretrain_path, device, a.epochs, seed=2000 + 10 * f + j)
            t_anchor = predict_texts(model, lb.iloc[pos_anchor]["text"].tolist(), tok, device).numpy()
            t_eval = predict_texts(model, lb.iloc[ev_pos]["text"].tolist(), tok, device).numpy()
            del model; torch.cuda.empty_cache()
            y_anchor = y[pos_anchor]
            q_anchor = torch.tensor(feat[pos_anchor], device=device)
            q_eval = torch.tensor(feat[ev_pos], device=device)
            r_bank = torch.tensor(y_anchor - t_anchor, device=device)
            y_a = torch.tensor(y_anchor, device=device)
            t_a = torch.tensor(t_anchor, device=device)
            # T0
            acc["T0"] += t_eval
            # T1 JACF anchor target
            w, idx = knn_w(q_eval, q_anchor, a.k, a.tau)
            anchor_target = (w.unsqueeze(-1) * y_a[idx]).sum(1)
            acc["T1"] += ((1 - a.alpha) * torch.tensor(t_eval, device=device) + a.alpha * anchor_target).cpu().numpy()
            # T2 BC fixed alpha
            r_pred = (w.unsqueeze(-1) * r_bank[idx]).sum(1)
            t_eval_t = torch.tensor(t_eval, device=device)
            acc["T2"] += (t_eval_t + a.alpha * r_pred).cpu().numpy()
            acc_fixed += (t_eval_t + a.alpha * r_pred).cpu().numpy()  # equivalence probe
            # T3 tuned alpha
            alpha = select_alpha(q_anchor, t_a, y_a, r_bank, a.k, a.tau, seed=3000 + 10 * f + j)
            alphas.append(alpha)
            acc["T3"] += (t_eval_t + alpha * r_pred).cpu().numpy()
            # T4 global residual
            acc["T4"] += (t_eval_t + r_bank.mean(0)).cpu().numpy()
            # T5 permuted residual
            g = torch.Generator().manual_seed(4000 + 10 * f + j)
            perm = torch.randperm(len(r_bank), generator=g)
            r_perm = r_bank[perm]
            r_pred_perm = (w.unsqueeze(-1) * r_perm[idx]).sum(1)
            acc["T5"] += (t_eval_t + a.alpha * r_pred_perm).cpu().numpy()
        for m in METHODS:
            acc[m] /= a.inner_folds
        acc_fixed /= a.inner_folds
        all_alphas[f] = alphas
        for i, pos in enumerate(ev_pos):
            row = {"id": lb.iloc[pos]["id"], "outer_fold": f,
                   **{f"y_{d}": y[pos, k] for k, d in enumerate(DIMS)}}
            for m in METHODS:
                for k, d in enumerate(DIMS):
                    row[f"{m}_{d}"] = acc[m][i, k]
            for k, d in enumerate(DIMS):
                row[f"T3fix_{d}"] = acc_fixed[i, k]
            rows.append(row)
        print(f"outer fold {f} done; alphas={alphas}", flush=True)

    d = pd.DataFrame(rows)
    d.to_parquet(a.out_dir / "nested_oof_predictions.parquet", index=False)
    # equivalence probe
    eq = float(np.max(np.abs(d[[f"T2_{x}" for x in DIMS]].to_numpy() -
                             d[[f"T3fix_{x}" for x in DIMS]].to_numpy())))
    out = {"config": vars(a), "n": len(d), "equivalence_max_abs_T2_T3fix": eq,
           "alphas_per_outer_fold": all_alphas, "methods": list(METHODS)}
    (a.out_dir / "nested_oof_config.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print("DONE", a.out_dir, "rows", len(d), "equiv", eq)


if __name__ == "__main__":
    main()
