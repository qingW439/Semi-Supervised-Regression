#!/usr/bin/env python3
"""U0/U1/U2 disagreement-aware regression pilot on frozen RoBERTa features.

U0: Huber mean regression, constant variance
U1: Gaussian NLL on individual ratings (mean + log-variance head)
U2: Huber mean + eta * MSE(log-variance, log(sample variance))
U2-shuffle: same as U2 but dispersion targets shuffled within the training fold

Everything is 5-fold OOF on TRAIN-500.  Hyper-parameters (early-stopping epoch,
eta) are selected on an inner 80-item split of each 400-item training fold; the
held-out 100 labels are used only for final evaluation.
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
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS, read_gold  # noqa: E402
from scripts.emobank.train_emobank_supervised import metric_report  # noqa: E402
from scripts.vad_research.train_acvr import encode  # noqa: E402

DIMS = ("V", "A", "D")


class Net(nn.Module):
    def __init__(self, in_dim=256, hidden=32, dropout=0.2):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.mean_linear = nn.Linear(in_dim, 3)
        self.mean_shared = nn.Linear(hidden, 3)
        self.logvar = nn.Linear(hidden, 3)
        nn.init.zeros_(self.mean_shared.weight)
        nn.init.zeros_(self.mean_shared.bias)

    def forward(self, x):
        h = self.shared(x)
        mu = torch.sigmoid(self.mean_linear(x) + self.mean_shared(h))
        return mu, self.logvar(h).clamp(-8.0, 2.0)


def make_data(data_root: Path, emobank_root: Path, lb):
    r = pd.read_csv(emobank_root / "corpus" / "individual_reader_ratings.csv")
    w = pd.read_csv(emobank_root / "corpus" / "individual_writer_ratings.csv")
    r = r[~((r["V"] == 1) & (r["A"] == 1) & (r["D"] == 1))]
    w = w[~((w["V"] == 1) & (w["A"] == 1) & (w["D"] == 1))]
    allr = pd.concat([r, w], ignore_index=True)
    ids = lb["id"].tolist()
    max_n = int(allr.groupby("id").size().max())
    ratings = np.zeros((len(ids), max_n, 3), dtype=np.float32)
    mask = np.zeros((len(ids), max_n), dtype=np.float32)
    var = np.zeros((len(ids), 3), dtype=np.float32)
    n = np.zeros(len(ids), dtype=np.int32)
    for k, i in enumerate(ids):
        g = allr[allr["id"] == i]
        arr = g[list(DIMS)].to_numpy(np.float32)
        arr = (arr - 1.0) / 4.0
        ratings[k, :len(arr)] = arr
        mask[k, :len(arr)] = 1.0
        n[k] = len(arr)
        var[k] = arr.var(axis=0, ddof=1) if len(arr) > 1 else 0.0
    return ratings, mask, var, n


def huber(mu, y):
    return F.huber_loss(mu, y, delta=0.1)


def nll_loss(mu, logvar, ratings, mask):
    sigma2 = torch.exp(logvar).clamp(1e-4, 1.0)
    diff = ratings - mu.unsqueeze(1)
    nll = 0.5 * logvar.unsqueeze(1) + diff.square() / (2.0 * sigma2.unsqueeze(1))
    denom = mask.sum(1).clamp_min(1.0).unsqueeze(-1)
    return ((nll * mask.unsqueeze(-1)).sum(1) / denom).mean()


def dispersion_loss(logvar, var):
    return F.mse_loss(logvar, torch.log(var + 1e-3))


class ModelBundle:
    def __init__(self, seed=0):
        torch.manual_seed(seed)
        self.net = Net().to(device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=5e-4, weight_decay=1e-1)

    def train_step(self, X, y, ratings, mask, var, mode, eta):
        self.net.train()
        self.opt.zero_grad(set_to_none=True)
        mu, logvar = self.net(X)
        if mode == "u0":
            loss = huber(mu, y)
        elif mode == "u1":
            loss = nll_loss(mu, logvar, ratings, mask)
        elif mode in ("u2", "u2_shuffle"):
            loss = huber(mu, y) + eta * dispersion_loss(logvar, var)
        else:
            raise ValueError(mode)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        return float(loss.detach())

    @torch.no_grad()
    def predict(self, X):
        self.net.eval()
        mu, logvar = self.net(X)
        return mu.cpu().numpy(), logvar.cpu().numpy()


def eval_metric(model, X, y, ratings, mask, mode):
    mu, logvar = model.predict(X)
    if mode == "u1":
        # negative log-likelihood of validation individual ratings
        s2 = np.exp(np.clip(logvar, -8, 2))
        diff = ratings.cpu().numpy() - mu[:, None, :]
        nll = 0.5 * np.clip(logvar, -8, 2)[:, None, :] + diff ** 2 / (2 * s2[:, None, :])
        m = mask.cpu().numpy()
        return float(np.mean((nll * m[:, :, None]).sum(1) / np.maximum(m.sum(1), 1)[:, None]))
    return float(np.mean(np.abs(mu - y.cpu().numpy())))


def fit_one(Xtr, ytr, rtr, mtr, vtr, mode, eta, seed, max_epochs=500, patience=50):
    """Train on an inner 320/80 split and return the best-validation state."""
    g = torch.Generator().manual_seed(seed)
    n = len(Xtr)
    perm = torch.randperm(n, generator=g).to(device)
    val = perm[:80]; tr = perm[80:]
    model = ModelBundle(seed)
    best_state, best_epoch, best_metric = None, 0, float("inf")
    for epoch in range(max_epochs):
        model.train_step(Xtr[tr], ytr[tr], rtr[tr], mtr[tr], vtr[tr], mode, eta)
        if epoch % 2 == 0 or epoch == max_epochs - 1:
            metric = eval_metric(model, Xtr[val], ytr[val], rtr[val], mtr[val], mode)
            if metric < best_metric - 1e-6:
                best_metric, best_epoch = metric, epoch
                best_state = copy.deepcopy(model.net.state_dict())
            elif epoch - best_epoch >= patience:
                break
    model.net.load_state_dict(best_state)
    return model, best_epoch, best_metric


def summary(y, p):
    m = metric_report(y, p)
    return {"mae": m["macro_mae_original_scale"],
            "ccc": float(np.mean([m[d]["ccc"] for d in DIMS])),
            "r2": float(np.mean([m[d]["r2"] for d in DIMS])),
            "per_dim_mae": {d: m[d]["mae"] for d in DIMS},
            "per_dim_ccc": {d: m[d]["ccc"] for d in DIMS}}


def dist_metrics(mu, logvar, ratings, mask):
    """NLL / CRPS / coverage / width against individual ratings."""
    s2 = np.exp(np.clip(logvar, -8, 2))
    sigma = np.sqrt(s2)
    diff = ratings - mu[:, None, :]
    m = mask[:, :, None]
    nll = 0.5 * np.log(2 * np.pi * s2)[:, None, :] + diff ** 2 / (2 * s2[:, None, :])
    z = diff / sigma[:, None, :]
    crps = sigma[:, None, :] * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    out = {"nll": float(np.sum(nll * m) / np.sum(m)),
           "crps": float(np.sum(crps * m) / np.sum(m))}
    for level in (0.5, 0.8, 0.9):
        q = norm.ppf(0.5 + level / 2)
        cov_mask = np.broadcast_to(m.astype(bool), diff.shape)
        out[f"coverage_{int(level*100)}"] = float(np.mean((np.abs(diff) <= q * sigma[:, None, :])[cov_mask]))
        out[f"width_{int(level*100)}"] = float(np.mean(2 * q * sigma))
    # Spearman(pred sigma, observed per-item SD), mask-aware
    n = mask.sum(1).clip(min=1)
    mean_r = (ratings * m).sum(1) / n[:, None]
    var_r = (((ratings - mean_r[:, None, :]) ** 2) * m).sum(1) / np.maximum(n[:, None] - 1, 1)
    obs_sd = np.sqrt(np.maximum(var_r, 0))
    out["spearman_sigma_obs_sd"] = {}
    for d in range(3):
        out["spearman_sigma_obs_sd"][DIMS[d]] = float(spearmanr(sigma[:, d], obs_sd[:, d]).statistic)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--emobank-root", type=Path, required=True)
    ap.add_argument("--oof-dir", type=Path, required=True)
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
    ratings, mask, var, n = make_data(a.data_root, a.emobank_root, lb)
    man = json.loads(a.manifest.read_text())

    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    enc = AutoModel.from_pretrained(str(a.pretrain_path), local_files_only=True).to(device).eval()
    feat = encode(enc, lb["text"].tolist(), tok, device).cpu().numpy()
    del enc
    torch.cuda.empty_cache()

    configs = ["u0", "u1", "u2", "u2_shuffle"]
    pred_mu = {k: np.zeros((500, 3), np.float32) for k in configs}
    pred_lv = {k: np.zeros((500, 3), np.float32) for k in configs}
    info = {k: [] for k in configs}
    eta_grid = [0.01, 0.03, 0.1, 0.3]
    for f in range(5):
        hold = man["fold_ids"][str(f)]
        train = [i for i in lb["id"] if i not in set(hold)]
        ti = lb.index[lb["id"].isin(train)].to_numpy()
        hi = lb.index[lb["id"].isin(hold)].to_numpy()
        sc = StandardScaler().fit(feat[ti])
        pca = PCA(n_components=256, random_state=0).fit(sc.transform(feat[ti]))
        Xtr = torch.tensor(pca.transform(sc.transform(feat[ti])), dtype=torch.float32, device=device)
        Xho = torch.tensor(pca.transform(sc.transform(feat[hi])), dtype=torch.float32, device=device)
        ytr = torch.tensor(y[ti], device=device)
        yho = y[hi]
        rtr = torch.tensor(ratings[ti], device=device)
        mtr = torch.tensor(mask[ti], device=device)
        vtr = torch.tensor(var[ti], device=device)
        # U0
        m0, ep0, vm0 = fit_one(Xtr, ytr, rtr, mtr, vtr, "u0", 0.0, seed=700 + f)
        pred_mu["u0"][hi], pred_lv["u0"][hi] = m0.predict(Xho)
        info["u0"].append({"fold": f, "epoch": ep0, "val": vm0})
        # U1
        m1, ep1, vm1 = fit_one(Xtr, ytr, rtr, mtr, vtr, "u1", 0.0, seed=710 + f)
        pred_mu["u1"][hi], pred_lv["u1"][hi] = m1.predict(Xho)
        info["u1"].append({"fold": f, "epoch": ep1, "val": vm1})
        # U2 eta grid
        best_eta, best_ep, best_val, best_state = None, 0, float("inf"), None
        for eta in eta_grid:
            m, ep, v = fit_one(Xtr, ytr, rtr, mtr, vtr, "u2", eta, seed=720 + f)
            if v < best_val:
                best_eta, best_ep, best_val, best_state = eta, ep, v, m
        pred_mu["u2"][hi], pred_lv["u2"][hi] = best_state.predict(Xho)
        info["u2"].append({"fold": f, "epoch": best_ep, "val": best_val, "eta": best_eta})
        # U2-shuffle: same eta, shuffled dispersion target
        g = torch.Generator().manual_seed(730 + f)
        perm = torch.randperm(len(ti), generator=g).to(device)
        v_shuf = vtr[perm]
        m2s, ep2s, vm2s = fit_one(Xtr, ytr, rtr, mtr, v_shuf, "u2_shuffle", best_eta, seed=740 + f)
        pred_mu["u2_shuffle"][hi], pred_lv["u2_shuffle"][hi] = m2s.predict(Xho)
        info["u2_shuffle"].append({"fold": f, "epoch": ep2s, "val": vm2s, "eta": best_eta})
        print(f"fold {f} done", flush=True)

    # U0 constant variance: per-dim variance of individual ratings on all 500 (for metrics only)
    const_lv = np.log(np.maximum(var.mean(0), 1e-4))[None, :].repeat(500, axis=0)
    pred_lv["u0"] = const_lv
    res = {}
    for k in configs:
        res[k] = {"mean": summary(y, pred_mu[k]),
                  "dist": dist_metrics(pred_mu[k], pred_lv[k], ratings, mask)}
    res["u0_constant_logvar"] = const_lv[0].tolist()
    (a.out_dir / "u_pilot.json").write_text(json.dumps({"results": res, "info": info}, indent=2) + "\n")
    rows = pd.DataFrame({"id": lb["id"], "fold": [next(f for f in range(5) if i in set(man["fold_ids"][str(f)])) for i in lb["id"]]})
    for k in configs:
        for j, d in enumerate(DIMS):
            rows[f"{k}_mu_{d}"] = pred_mu[k][:, j]
            rows[f"{k}_sigma_{d}"] = np.exp(0.5 * pred_lv[k][:, j])
    rows.to_parquet(a.out_dir / "u_pilot_predictions.parquet", index=False)
    for k in configs:
        m = res[k]["mean"]; dd = res[k]["dist"]
        print(f'{k:12s} MAE={m["mae"]:.6f} CCC={m["ccc"]:.4f} '
              f'NLL={dd["nll"]:.4f} CRPS={dd["crps"]:.4f} '
              f'cov80={dd["coverage_80"]:.3f} rhoSD={np.mean(list(dd["spearman_sigma_obs_sd"].values())):.3f}')


if __name__ == "__main__":
    main()
