#!/usr/bin/env python3
"""DIAGNOSTIC ONLY (not an M1 model): how does explained variance scale with the
number of labels when the frozen RoBERTa features are read out by a closed-form
ridge?  This separates 'label-limited' from 'representation-limited'.

Protocol notes
- Uses the seed-0 label ordering of the fixed lb500 split; larger budgets are
  supersets, so the curve is nested and no new split is created.
- Ridge alpha is chosen by 5-fold CV *inside* the labelled subset only.
- DEV (1000) is used for reporting; nothing is selected on DEV.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import read_gold, NORM_COLUMNS  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402

DIMS = ("V", "A", "D")
S = 4.0


@torch.inference_mode()
def encode(encoder, texts, tok, device, bs=128):
    encoder.eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=128,
                return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        st = encoder(**b, return_dict=True).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(dtype=st.dtype)
        out.append(F.normalize((st * m).sum(1) / m.sum(1).clamp_min(1), dim=-1).cpu())
    return torch.cat(out).numpy()


def metrics(p, y):
    mp, my = p.mean(), y.mean()
    vp, vy = p.var(), y.var()
    cov = ((p - mp) * (y - my)).mean()
    return dict(mae=float(np.abs(p - y).mean() * S),
                pearson=float(cov / np.sqrt(vp * vy)),
                sd_ratio=float(np.sqrt(vp / vy)),
                ccc=float(2 * cov / (vp + vy + (mp - my) ** 2)),
                r2=float(1 - ((p - y) ** 2).mean() / vy))


def ridge_fit(X, Y, alpha):
    Xm = X.mean(0, keepdims=True)
    Ym = Y.mean(0, keepdims=True)
    Xc = X - Xm
    G = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    W = np.linalg.solve(G, Xc.T @ (Y - Ym))
    return Xm, Ym, W


def cv_alpha(X, Y, alphas, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)
    best, best_err = None, np.inf
    for a in alphas:
        errs = []
        for f in folds:
            tr = np.setdiff1d(idx, f)
            Xm, Ym, W = ridge_fit(X[tr], Y[tr], a)
            pred = (X[f] - Xm) @ W + Ym
            errs.append(np.abs(pred - Y[f]).mean())
        e = float(np.mean(errs))
        if e < best_err:
            best, best_err = a, e
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/rankup_project/data/emobank_processed"))
    ap.add_argument("--pretrain-path", type=Path, default=Path("/root/autodl-tmp/rankup_project/pretrained/roberta-base"))
    ap.add_argument("--p1", type=Path, default=Path("/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0/model_best.pth"))
    ap.add_argument("--out", type=Path, default=Path("outputs/auto_search/diag_label_scaling.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", type=Path, default=Path("outputs/auto_search/features_frozen.npz"))
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    lb = read_gold(a.data_root / "lb500_s0" / "train_lb.parquet", expected_n=500)
    ulb = pd.read_parquet(a.data_root / "lb500_s0" / "train_ulb.parquet")
    dv = read_gold(a.data_root / "dev_gold.parquet", expected_n=1000)

    # labels for the WHOLE train pool (8062) come from emobank.csv; this is a
    # DIAGNOSTIC ceiling probe, no M1 model is trained with more than 500 labels.
    eb = pd.read_csv(a.data_root.parent / "emobank_source" / "EmoBank" / "corpus" / "emobank.csv")
    eb = eb[eb.split == "train"].set_index("id")
    lab_of = {i: np.array([eb.at[i, "V"], eb.at[i, "A"], eb.at[i, "D"]], float) for i in eb.index}

    if a.cache.exists():
        z = np.load(a.cache, allow_pickle=True)
        Ftr, Fdev, order = z["Ftr"], z["Fdev"], [str(x) for x in z["order"]]
        print(f"loaded cached features {Ftr.shape} {Fdev.shape}", flush=True)
    else:
        tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
        model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True)
        ck = torch.load(a.p1, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        del ck
        model.to(dev)
        pool = pd.concat([lb, ulb], ignore_index=True)
        Ftr = encode(model.encoder, pool["text"].tolist(), tok, dev)
        Fdev = encode(model.encoder, dv["text"].tolist(), tok, dev)
        order = pool["id"].tolist()
        a.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.cache, Ftr=Ftr, Fdev=Fdev,
                            order=np.array([str(x) for x in order], dtype=object))
        print(f"encoded {Ftr.shape} {Fdev.shape}", flush=True)

    # reorder so that the first 500 rows are exactly the canonical lb500_s0 labels
    lb_ids = lb["id"].tolist()
    pos = {k: i for i, k in enumerate(order)}
    head = [pos[i] for i in lb_ids]
    tail = [i for i in range(len(order)) if i not in set(head)]
    perm = np.array(head + tail)
    Ftr = Ftr[perm]
    order = [order[i] for i in perm]
    ytr = np.stack([lab_of[i] for i in order])  # original 1..5 scale
    ytr = (ytr - 1.0) / S
    ydev = dv.loc[:, list(NORM_COLUMNS)].to_numpy(np.float64)
    assert ytr.shape == (Ftr.shape[0], 3), ytr.shape
    print(f"label matrix {ytr.shape} min={ytr.min():.3f} max={ytr.max():.3f}", flush=True)

    # lab[0:500] is the canonical labelled set of seed 0; the rest is unlabelled
    alphas = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]
    out = {"note": "diagnostic label-scaling of a closed-form frozen-feature ridge; not an M1 result",
           "n_total_pool": int(Ftr.shape[0]), "curve": []}
    for n in (100, 250, 500, 1000, 2000, 4000, 8062):
        X, Y = Ftr[:n], ytr[:n]
        al = cv_alpha(X, Y, alphas)
        Xm, Ym, W = ridge_fit(X, Y, al)
        pred = (Fdev - Xm) @ W + Ym
        mac = {k: float(np.mean([metrics(pred[:, i], ydev[:, i])[k] for i in range(3)]))
               for k in ("mae", "pearson", "sd_ratio", "ccc", "r2")}
        per = {DIMS[i]: metrics(pred[:, i], ydev[:, i]) for i in range(3)}
        out["curve"].append({"n_labels": n, "alpha": al, "macro": mac, "per_dim": per})
        print(f"n={n:5d} alpha={al:8.2f} MAE={mac['mae']:.5f} r={mac['pearson']:.4f} "
              f"sd={mac['sd_ratio']:.4f} CCC={mac['ccc']:.4f} R2={mac['r2']:.4f}", flush=True)

    # kNN cosine reference (k=20, softmax tau=0.1) at the canonical 500 labels
    def knn_pred(Xl, Yl, Xq, k=20, tau=0.1):
        sim = Xq @ Xl.T
        topv, topi = torch.topk(torch.tensor(sim), min(k, len(Xl)), dim=1)
        w = torch.softmax(topv / tau, dim=1).numpy()
        return (w[:, :, None] * Yl[topi.numpy()]).sum(1)
    for n in (500, 2000, 8062):
        p = knn_pred(Ftr[:n], ytr[:n], Fdev)
        mac = {kk: float(np.mean([metrics(p[:, i], ydev[:, i])[kk] for i in range(3)]))
               for kk in ("mae", "pearson", "sd_ratio", "ccc", "r2")}
        out.setdefault("knn", []).append({"n_labels": n, "macro": mac})
        print(f"kNN n={n:5d} MAE={mac['mae']:.5f} r={mac['pearson']:.4f} sd={mac['sd_ratio']:.4f} "
              f"CCC={mac['ccc']:.4f} R2={mac['r2']:.4f}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
