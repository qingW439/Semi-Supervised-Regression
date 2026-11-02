#!/usr/bin/env python3
"""Stage A: M0 OOF pair-reliability audit (no ACLDP training).

Train five supervised M0 fold models on 400 labels, predict the held-out 100
with MC dropout, then test whether MC difference variance / feature similarity
predict M0 pairwise-difference error.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import GoldData, NORM_COLUMNS, TextCollator, read_gold  # noqa
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa

DIMS = ("V", "A", "D")


@torch.no_grad()
def predict_mc(model, texts, tok, device, passes=10, bs=64):
    model.train()  # keep dropout active
    out = []
    for _ in range(passes):
        ps = []
        for i in range(0, len(texts), bs):
            b = tok(list(texts[i:i + bs]), padding=True, truncation=True,
                    max_length=128, return_tensors="pt")
            b = {k: v.to(device) for k, v in b.items()}
            ps.append(model(**b).float().cpu())
        out.append(torch.cat(ps))
    return torch.stack(out)  # [K,N,3]


def train_teacher(lb, tok, pretrain, device, epochs=10, seed=0):
    model = RobertaVADRegressor(str(pretrain), dropout=0.1, offline=True).to(device)
    dl = DataLoader(GoldData(lb), batch_size=16, shuffle=True,
                    collate_fn=TextCollator(tok, 128),
                    generator=torch.Generator().manual_seed(17_000 + seed))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    for _ in range(epochs):
        model.train()
        for t, y, _ in dl:
            t = {k: v.to(device) for k, v in t.items()}
            y = y.to(device)
            loss = F.l1_loss(model(**t), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--oof", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    random.seed(1000 * a.data_seed)
    np.random.seed(1000 * a.data_seed)
    torch.manual_seed(1000 * a.data_seed)
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    lb = read_gold(a.data_root / f"lb500_s{a.data_seed}" / "train_lb.parquet", expected_n=500)
    y = lb.loc[:, [f"{d}_norm" for d in DIMS]].to_numpy(np.float64)
    if a.features is not None and Path(a.features).is_file():
        feat = np.load(a.features)["lb"]
    else:
        from scripts.vad_research.train_acvr import encode
        enc = AutoModel.from_pretrained(str(a.pretrain_path), local_files_only=True).to(device).eval()
        feat = encode(enc, lb["text"].tolist(), tok, device).numpy()
        del enc; torch.cuda.empty_cache()
    feat = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
    man = json.loads(a.manifest.read_text())
    id2pos = {i: k for k, i in enumerate(lb["id"])}
    oof = pd.read_parquet(a.oof).set_index("id") if a.oof is not None else None
    m0_mc = np.zeros((10, 500, 3), np.float32)
    jacf = np.zeros((500, 3), np.float32)
    for f in range(5):
        hold = man["fold_ids"][str(f)]
        train = lb[~lb["id"].isin(hold)].reset_index(drop=True)
        hold_df = lb[lb["id"].isin(hold)].reset_index(drop=True)
        model = train_teacher(train, tok, a.pretrain_path, device, 10, seed=1000 * a.data_seed + 100 + f)
        mc = predict_mc(model, hold_df["text"].tolist(), tok, device, 10).numpy()
        for k, idx in enumerate(hold_df["id"]):
            pos = id2pos[idx]
            m0_mc[:, pos] = mc[:, k]
            if oof is not None:
                jacf[pos] = oof.loc[idx, ["raw_V", "raw_A", "raw_D"]].to_numpy()
        del model; torch.cuda.empty_cache()
        print("fold", f, "done", flush=True)
    np.savez(a.out.with_suffix(".npz"), m0_mc=m0_mc, m0_mean=m0_mc.mean(0), jacf=jacf, y=y)
    errs = {d: [] for d in DIMS}; unc = {d: [] for d in DIMS}; sims = {d: [] for d in DIMS}
    dirs = {d: [] for d in DIMS}; jacf_dir = {d: [] for d in DIMS}
    for f in range(5):
        hold = man["fold_ids"][str(f)]
        pos = np.array([id2pos[i] for i in hold])
        sub_feat = feat[pos]
        sim = sub_feat @ sub_feat.T
        iu = np.triu_indices(len(pos), 1)
        p0 = m0_mc[:, pos, :].mean(0)
        for di, d in enumerate(DIMS):
            dp0 = p0[iu[0], di] - p0[iu[1], di]
            dy = y[pos][iu[0], di] - y[pos][iu[1], di]
            err = np.abs(dp0 - dy)
            var = m0_mc[:, pos, di][:, iu[0]] - m0_mc[:, pos, di][:, iu[1]]
            u = var.var(0)
            errs[d].append(err); unc[d].append(u); sims[d].append(sim[iu])
            dirs[d].append(np.sign(dp0) == np.sign(dy))
            jd = np.sign(jacf[pos][iu[0], di] - jacf[pos][iu[1], di]) == np.sign(dy)
            jacf_dir[d].append(jd)
    out = {}
    for d in DIMS:
        e = np.concatenate(errs[d]); u = np.concatenate(unc[d]); s = np.concatenate(sims[d])
        dd = np.concatenate(dirs[d]); jd = np.concatenate(jacf_dir[d])
        out[d] = {
            "n_pairs": int(len(e)),
            "mae_diff": float(e.mean()),
            "direction_acc": float(dd.mean()),
            "rho_err_unc": float(spearmanr(e, u).statistic),
            "rho_err_sim": float(spearmanr(e, s).statistic),
            "err_when_jacf_dir_agree": float(e[jd].mean()) if jd.any() else None,
            "err_when_jacf_dir_disagree": float(e[~jd].mean()) if (~jd).any() else None,
            "unc_quintile_err": [float(e[(u >= lo) & (u <= hi)].mean())
                                 for lo, hi in zip(np.quantile(u, [0, .2, .4, .6, .8]),
                                                   np.quantile(u, [.2, .4, .6, .8, 1.0]))],
            "sim_quintile_err": [float(e[(s >= lo) & (s <= hi)].mean())
                                 for lo, hi in zip(np.quantile(s, [0, .2, .4, .6, .8]),
                                                   np.quantile(s, [.2, .4, .6, .8, 1.0]))],
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    for d in DIMS:
        v = out[d]
        print(d, "mae_diff", round(v["mae_diff"], 4), "dir", round(v["direction_acc"], 4),
              "rho_unc", round(v["rho_err_unc"], 3), "rho_sim", round(v["rho_err_sim"], 3))


if __name__ == "__main__":
    main()
