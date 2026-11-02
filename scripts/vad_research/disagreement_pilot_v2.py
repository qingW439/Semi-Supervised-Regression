#!/usr/bin/env python3
"""U0/U1/U2 v2: ridge-anchored disagreement pilot.

U0: ridge mean, constant variance.
U1: ridge mean fixed, learned variance head (Gaussian NLL on individual ratings).
U2: ridge mean + zero-initialised residual mean head + variance head, trained
    with Huber + eta * dispersion MSE.  Trunk is shared, so dispersion gradients
    can influence the residual mean.

The ridge mean is fitted only on the 400 training items of each outer fold; the
inner 80-item split is used for early stopping and eta selection.  Held-out 100
labels are used only for evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS, read_gold  # noqa: E402
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_acvr import encode  # noqa: E402

DIMS = ("V", "A", "D")


class Head(nn.Module):
    def __init__(self, in_dim=64, hidden=32, residual=False, dropout=0.2):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.var = nn.Linear(hidden, 3)
        self.res = nn.Linear(hidden, 3) if residual else None
        if self.res is not None:
            nn.init.zeros_(self.res.weight); nn.init.zeros_(self.res.bias)

    def forward(self, x):
        h = self.trunk(x)
        return self.var(h).clamp(-8, 2), (self.res(h) if self.res is not None else None)


def make_data(emobank_root: Path, lb):
    r = pd.read_csv(emobank_root / "corpus" / "individual_reader_ratings.csv")
    w = pd.read_csv(emobank_root / "corpus" / "individual_writer_ratings.csv")
    r = r[~((r["V"] == 1) & (r["A"] == 1) & (r["D"] == 1))]
    w = w[~((w["V"] == 1) & (w["A"] == 1) & (w["D"] == 1))]
    allr = pd.concat([r, w], ignore_index=True)
    ids = lb["id"].tolist()
    max_n = int(allr.groupby("id").size().max())
    ratings = np.zeros((len(ids), max_n, 3), np.float32)
    mask = np.zeros((len(ids), max_n), np.float32)
    var = np.zeros((len(ids), 3), np.float32)
    for k, i in enumerate(ids):
        arr = allr[allr["id"] == i][list(DIMS)].to_numpy(np.float32)
        arr = (arr - 1) / 4
        ratings[k, :len(arr)] = arr; mask[k, :len(arr)] = 1
        var[k] = arr.var(axis=0, ddof=1) if len(arr) > 1 else 0
    return ratings, mask, var


def ridge_cv(X, y, seed):
    alphas = [1.0, 10.0, 100.0, 1000.0]
    kf = KFold(4, shuffle=True, random_state=seed)
    best, best_a = np.inf, alphas[0]
    for alpha in alphas:
        errs = []
        for tr, va in kf.split(X):
            m = Ridge(alpha=alpha).fit(X[tr], y[tr])
            errs.append(float(np.mean((m.predict(X[va]) - y[va]) ** 2)))
        e = float(np.mean(errs))
        if e < best:
            best, best_a = e, alpha
    return Ridge(alpha=best_a).fit(X, y), best_a


def nll(mu, lv, ratings, mask):
    s2 = torch.exp(lv).clamp(1e-4, 1.0)
    diff = ratings - mu.unsqueeze(1)
    v = 0.5 * lv.unsqueeze(1) + diff.square() / (2 * s2.unsqueeze(1))
    return ((v * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)).mean()


def train_head(Xtr, ytr, rtr, mtr, vtr, mu_tr, mode, eta, seed, max_epochs=400, patience=30):
    g = torch.Generator().manual_seed(seed)
    n = len(Xtr)
    perm = torch.randperm(n, generator=g).to(device)
    val, tr = perm[:80], perm[80:]
    model = Head(residual=(mode == "u2")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_ep, best_state = None, 0, None
    for ep in range(max_epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        lv, res = model(Xtr[tr])
        mu = mu_tr[tr] + (res if res is not None else 0.0)
        if mode == "u1":
            loss = nll(mu, lv, rtr[tr], mtr[tr])
        else:
            loss = F.huber_loss(mu, ytr[tr], delta=0.1) + eta * F.mse_loss(lv, torch.log(vtr[tr] + 1e-3))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % 2 == 0 or ep == max_epochs - 1:
            model.eval()
            with torch.no_grad():
                lv, res = model(Xtr[val])
                mu = mu_tr[val] + (res if res is not None else 0.0)
                if mode == "u1":
                    metric = float(nll(mu, lv, rtr[val], mtr[val]))
                else:
                    metric = float(torch.mean(torch.abs(mu - ytr[val])))
            if best is None or metric < best - 1e-6:
                best, best_ep, best_state = metric, ep, copy.deepcopy(model.state_dict())
            elif ep - best_ep >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_ep, best


def summary(y, p):
    m = metric_report(y, p)
    return {"mae": m["macro_mae_original_scale"],
            "ccc": float(np.mean([m[d]["ccc"] for d in DIMS])),
            "r2": float(np.mean([m[d]["r2"] for d in DIMS])),
            "per_dim_mae": {d: m[d]["mae"] for d in DIMS}}


def dist(mu, lv, ratings, mask):
    s2 = np.exp(np.clip(lv, -8, 2)); sig = np.sqrt(s2)
    diff = ratings - mu[:, None, :]; m = mask[:, :, None]
    nll_v = 0.5 * np.log(2 * np.pi * s2)[:, None, :] + diff ** 2 / (2 * s2[:, None, :])
    z = diff / sig[:, None, :]
    crps = sig[:, None, :] * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    out = {"nll": float((nll_v * m).sum() / m.sum()),
           "crps": float((crps * m).sum() / m.sum())}
    for level in (0.5, 0.8, 0.9):
        q = norm.ppf(0.5 + level / 2)
        cm = np.broadcast_to(m.astype(bool), diff.shape)
        out[f"coverage_{int(level*100)}"] = float(np.mean((np.abs(diff) <= q * sig[:, None, :])[cm]))
        out[f"width_{int(level*100)}"] = float(np.mean(2 * q * sig))
    n = mask.sum(1).clip(min=1); mean_r = (ratings * m).sum(1) / n[:, None]
    var_r = (((ratings - mean_r[:, None, :]) ** 2) * m).sum(1) / np.maximum(n[:, None] - 1, 1)
    obs = np.sqrt(np.maximum(var_r, 0))
    out["rho_sigma_obs"] = {DIMS[d]: float(spearmanr(sig[:, d], obs[:, d]).statistic) for d in range(3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--emobank-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--pretrain-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    global device
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    lb = read_gold(a.data_root / "lb500_s0" / "train_lb.parquet", expected_n=500)
    y = lb.loc[:, list(NORM_COLUMNS)].to_numpy(np.float32)
    ratings, mask, var = make_data(a.emobank_root, lb)
    man = json.loads(a.manifest.read_text())
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    enc = AutoModel.from_pretrained(str(a.pretrain_path), local_files_only=True).to(device).eval()
    feat = encode(enc, lb["text"].tolist(), tok, device).cpu().numpy()
    del enc; torch.cuda.empty_cache()

    configs = ["u0", "u1", "u2", "u2_shuffle"]
    mu_p = {k: np.zeros((500, 3), np.float32) for k in configs}
    lv_p = {k: np.zeros((500, 3), np.float32) for k in configs}
    info = {k: [] for k in configs}
    for f in range(5):
        hold = man["fold_ids"][str(f)]
        train = [i for i in lb["id"] if i not in set(hold)]
        ti = lb.index[lb["id"].isin(train)].to_numpy(); hi = lb.index[lb["id"].isin(hold)].to_numpy()
        sc = StandardScaler().fit(feat[ti]); pca = PCA(64, random_state=0).fit(sc.transform(feat[ti]))
        Xtr = torch.tensor(pca.transform(sc.transform(feat[ti])), dtype=torch.float32, device=device)
        Xho = torch.tensor(pca.transform(sc.transform(feat[hi])), dtype=torch.float32, device=device)
        ytr = torch.tensor(y[ti], device=device); yho = y[hi]
        rtr = torch.tensor(ratings[ti], device=device); mtr = torch.tensor(mask[ti], device=device)
        vtr = torch.tensor(var[ti], device=device)
        # ridge mean
        rm, alpha = ridge_cv(Xtr.cpu().numpy(), ytr.cpu().numpy(), seed=800 + f)
        mu_tr = torch.tensor(rm.predict(Xtr.cpu().numpy()), dtype=torch.float32, device=device)
        mu_ho = rm.predict(Xho.cpu().numpy())
        # U0
        mu_p["u0"][hi] = mu_ho
        lv_p["u0"][hi] = np.log(np.maximum(var[ti].mean(0), 1e-4))[None, :]
        info["u0"].append({"fold": f, "alpha": alpha})
        # U1: fixed ridge mean, learned variance
        m1, ep1, vm1 = train_head(Xtr, ytr, rtr, mtr, vtr, mu_tr, "u1", 0.0, seed=810 + f)
        with torch.no_grad():
            lv, _ = m1(Xho)
        mu_p["u1"][hi] = mu_ho; lv_p["u1"][hi] = lv.cpu().numpy()
        info["u1"].append({"fold": f, "epoch": ep1, "val": vm1})
        # U2 eta grid
        best = None
        for eta in (0.01, 0.03, 0.1, 0.3):
            m, ep, v = train_head(Xtr, ytr, rtr, mtr, vtr, mu_tr, "u2", eta, seed=820 + f)
            if best is None or v < best[2]:
                best = (m, ep, v, eta)
        m2, ep2, vm2, eta2 = best
        with torch.no_grad():
            lv, res = m2(Xho)
        mu_p["u2"][hi] = (torch.tensor(mu_ho, device=device) + res).cpu().numpy()
        lv_p["u2"][hi] = lv.cpu().numpy()
        info["u2"].append({"fold": f, "epoch": ep2, "val": vm2, "eta": eta2, "alpha": alpha})
        # U2-shuffle
        g = torch.Generator().manual_seed(830 + f)
        v_shuf = vtr[torch.randperm(len(ti), generator=g).to(device)]
        ms, eps, vms = train_head(Xtr, ytr, rtr, mtr, v_shuf, mu_tr, "u2", eta2, seed=840 + f)
        with torch.no_grad():
            lv, res = ms(Xho)
        mu_p["u2_shuffle"][hi] = (torch.tensor(mu_ho, device=device) + res).cpu().numpy()
        lv_p["u2_shuffle"][hi] = lv.cpu().numpy()
        info["u2_shuffle"].append({"fold": f, "epoch": eps, "val": vms, "eta": eta2})
        print(f"fold {f} ridge_alpha={alpha} eta2={eta2}", flush=True)

    res = {k: {"mean": summary(y, mu_p[k]), "dist": dist(mu_p[k], lv_p[k], ratings, mask)} for k in configs}
    (a.out_dir / "u_pilot_v2.json").write_text(json.dumps({"results": res, "info": info}, indent=2) + "\n")
    rows = pd.DataFrame({"id": lb["id"]})
    for k in configs:
        for j, d in enumerate(DIMS):
            rows[f"{k}_mu_{d}"] = mu_p[k][:, j]
            rows[f"{k}_sigma_{d}"] = np.exp(0.5 * lv_p[k][:, j])
    rows.to_parquet(a.out_dir / "u_pilot_v2_predictions.parquet", index=False)
    for k in configs:
        m, dd = res[k]["mean"], res[k]["dist"]
        print(f'{k:12s} MAE={m["mae"]:.6f} CCC={m["ccc"]:.4f} NLL={dd["nll"]:.4f} '
              f'CRPS={dd["crps"]:.4f} cov80={dd["coverage_80"]:.3f} '
              f'rhoSD={np.mean(list(dd["rho_sigma_obs"].values())):.3f}')


if __name__ == "__main__":
    main()
