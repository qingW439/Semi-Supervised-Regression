#!/usr/bin/env python3
"""M1-R0: VAD multi-dimensional relation audit (seed 0 first).

For each outer fold, 40 diagnostic batches of 16 anchors + 32 hidden OOF items +
32 ULB items.  Relations are recovered from a shared frozen-RoBERTa feature
graph; R3 uses per-dimension anchor constraints (GCLSS-style harmonic extension).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))
from feature_extractor import extract_fold_features, extract_frozen, predict_fold_model  # noqa: E402
from graph_ops import cosine_graph, fiedler, gclss, graph_diagnostics, laplacian  # noqa: E402

DIMS = ("V", "A", "D")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--oof-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature-source", choices=("f0", "f1"), default="f0")
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--n-anchors", type=int, default=16)
    ap.add_argument("--knn", type=int, default=0)
    ap.add_argument("--normalized", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ds = a.data_root / f"lb500_s{a.seed}"
    lb = pd.read_parquet(ds / "train_lb.parquet").reset_index(drop=True)
    ulb = pd.read_parquet(ds / "train_ulb.parquet").reset_index(drop=True)
    man = json.loads(a.manifest.read_text())

    # frozen features, cached
    fcache = a.out_dir / f"features_f0_seed{a.seed}.npz"
    if fcache.exists():
        z = np.load(fcache)
        f_lb, f_ulb = z["lb"], z["ulb"]
    else:
        texts = lb["text"].tolist() + ulb["text"].tolist()
        feat = extract_frozen(texts, a.pretrain_path, device)
        f_lb, f_ulb = feat[:len(lb)], feat[len(lb):]
        np.savez(fcache, lb=f_lb, ulb=f_ulb)
    f_lb = f_lb / np.maximum(np.linalg.norm(f_lb, axis=1, keepdims=True), 1e-12)
    f_ulb = f_ulb / np.maximum(np.linalg.norm(f_ulb, axis=1, keepdims=True), 1e-12)
    id2lb = {i: k for k, i in enumerate(lb["id"])}
    id2ulb = {i: k for k, i in enumerate(ulb["id"])}
    y = lb.loc[:, [f"{d}_norm" for d in DIMS]].to_numpy(np.float64)

    pairs = []
    diags = []
    for f in [int(x) for x in a.folds.split(",")]:
        hold = man["fold_ids"][str(f)]
        train = [i for i in lb["id"] if i not in set(hold)]
        tr_pos = np.array([id2lb[i] for i in train])
        ho_pos = np.array([id2lb[i] for i in hold])
        # anchors: per-dim min/max union, fill randomly to 16
        anchor_pos = []
        for di in range(3):
            vals = y[tr_pos, di]
            anchor_pos += [int(tr_pos[np.argmin(vals)]), int(tr_pos[np.argmax(vals)])]
        anchor_pos = list(dict.fromkeys(anchor_pos))
        rng = np.random.default_rng(10_000 + f)
        fill = [int(p) for p in rng.choice(tr_pos, size=len(tr_pos), replace=False) if int(p) not in anchor_pos]
        anchor_pos = (anchor_pos + fill)[:a.n_anchors]
        anchor_pos = np.array(anchor_pos)
        # fold model predictions (student raw proxy for the Teacher ranking)
        pcache = a.out_dir / f"fold{f}_preds_seed{a.seed}.npz"
        ckpt = a.oof_dir / f"fold{f}" / "jacf_final.pth"
        if pcache.exists():
            zp = np.load(pcache); p_lb, p_ulb = zp["lb"], zp["ulb"]
        else:
            p_lb = predict_fold_model(ckpt, lb["text"].tolist(), a.pretrain_path, device)
            p_ulb = predict_fold_model(ckpt, ulb["text"].tolist(), a.pretrain_path, device)
            np.savez(pcache, lb=p_lb, ulb=p_ulb)
        if a.feature_source == "f1":
            f1cache = a.out_dir / f"features_f1_fold{f}_seed{a.seed}.npz"
            if f1cache.exists():
                zf = np.load(f1cache); flb, fulb = zf["lb"], zf["ulb"]
            else:
                ff = extract_fold_features(ckpt, lb["text"].tolist() + ulb["text"].tolist(),
                                           a.pretrain_path, device)
                flb, fulb = ff[:len(lb)], ff[len(lb):]
                np.savez(f1cache, lb=flb, ulb=fulb)
            flb = flb / np.maximum(np.linalg.norm(flb, axis=1, keepdims=True), 1e-12)
            fulb = fulb / np.maximum(np.linalg.norm(fulb, axis=1, keepdims=True), 1e-12)
        else:
            flb, fulb = f_lb, f_ulb
        for b in range(a.batches):
            brng = np.random.default_rng(a.seed * 1_000_000 + f * 10_000 + b)
            hid = brng.choice(ho_pos, size=32, replace=False)
            uid = brng.choice(len(ulb), size=32, replace=False)
            X = np.concatenate([flb[hid], fulb[uid], flb[anchor_pos]], axis=0)
            S = cosine_graph(X, nonnegative=True, knn=a.knn)
            L = laplacian(S, normalized=a.normalized)
            diags.append({"seed": a.seed, "fold": f, "batch": b, **graph_diagnostics(S)})
            # R0 random
            r0 = brng.permutation(64).astype(np.float64)
            # R1 model prediction on 64 unlabeled nodes
            p_u = np.concatenate([p_lb[hid], p_ulb[uid]], axis=0)  # [64,3]
            # R2 CLSS on 64 unlabeled only
            Suu = S[:64, :64]
            r2 = fiedler(Suu, normalized=a.normalized)
            # R3 GCLSS per dim
            r3 = np.zeros((64, 3)); r3lp = np.zeros((64, 3))
            lam = 1e-8
            Luu = L[:64, :64] + lam * np.eye(64)
            Lul = L[:64, 64:]
            for di in range(3):
                yl = y[anchor_pos, di]
                rl = (yl - yl.mean()) / max(yl.std(), 1e-8)
                ru = -np.linalg.pinv(Luu, rcond=1e-10) @ (Lul @ rl)
                r3[:, di] = ru
                r3lp[:, di] = ru
            # eval pairs among unique hidden ids
            uniq = list(dict.fromkeys(hid.tolist()))
            for ii in range(len(uniq)):
                for jj in range(ii + 1, len(uniq)):
                    # positions in the 64-node U block: first occurrence
                    pi = int(np.where(hid == uniq[ii])[0][0]); pj = int(np.where(hid == uniq[jj])[0][0])
                    for di, d in enumerate(DIMS):
                        pairs.append({
                            "seed": a.seed, "fold": f, "batch": b, "dim": d,
                            "i": lb.iloc[uniq[ii]]["id"], "j": lb.iloc[uniq[jj]]["id"],
                            "true_diff": float(y[uniq[ii], di] - y[uniq[jj], di]),
                            "r0": float(r0[pi] - r0[pj]),
                            "r1": float(p_u[pi, di] - p_u[pj, di]),
                            "r2": float(r2[pi] - r2[pj]),
                            "r3": float(r3[pi, di] - r3[pj, di]),
                            "r3lp": float(r3lp[pi, di] - r3lp[pj, di]),
                        })
        print(f"fold {f} done", flush=True)
    pd.DataFrame(pairs).to_parquet(a.out_dir / f"pairs_seed{a.seed}.parquet", index=False)
    (a.out_dir / f"graph_diag_seed{a.seed}.json").write_text(json.dumps(diags, indent=2) + "\n")
    print("DONE", len(pairs), "pair-rows")


if __name__ == "__main__":
    main()
