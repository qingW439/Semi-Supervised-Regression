#!/usr/bin/env python3
"""Assemble the E0* audit table from the per-fold OOF artefacts (v2).

Arms, all evaluated on the SAME 100 held-out samples of each outer fold:

  mlp_p1   : that fold's supervised checkpoint `p1_final.pth`, run through its real
             forward pass (un-normalised mean pooling, exactly as trained)
  mlp_jacf : that fold's JACF student (`oof_predictions.parquet`, raw_* columns)
  ridge    : closed-form ridge on `features_labeled.npy` (400) with alpha+prep chosen by
             5-fold CV inside the 400, applied to `features_holdout.npy`

`features_*.npy` are L2-normalised (that is what `encode()` returns and what the kNN
anchor uses), so the ridge arm is a cosine-space linear readout; the MLP arm must be
re-run through the network instead of applying its head to those features.

v2 fix: v1 applied the MLP head to the L2-normalised features and produced MAE ~1.85
(garbage); this version re-runs the fold model's forward pass.
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
from semilearn.datasets.nlp_datasets.emobank import TextCollator, GoldData  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor  # noqa: E402

DIMS = ("V", "A", "D")
SCALE = 4.0
ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
PREPS = ("center", "zscore")


def metrics(p, y):
    mp, my = p.mean(), y.mean()
    vp, vy = p.var(), y.var()
    cov = ((p - mp) * (y - my)).mean()
    return dict(mae=float(np.abs(p - y).mean() * SCALE),
                rmse=float(np.sqrt(((p - y) ** 2).mean()) * SCALE),
                pearson=float(cov / np.sqrt(vp * vy)),
                sd_ratio=float(np.sqrt(vp / vy)),
                ccc=float(2 * cov / (vp + vy + (mp - my) ** 2)),
                r2=float(1 - ((p - y) ** 2).mean() / vy))


def macro(pred, y):
    m = {k: float(np.mean([metrics(pred[:, i], y[:, i])[k] for i in range(3)]))
         for k in ("mae", "rmse", "pearson", "sd_ratio", "ccc", "r2")}
    m["rho"] = m["pearson"] * m["sd_ratio"]
    return m


def fit_ridge(Xtr, Ytr, alpha, prep):
    mu = Xtr.mean(0, keepdims=True)
    Xc = Xtr - mu
    sd = None
    if prep == "zscore":
        sd = Xc.std(0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        Xc = Xc / sd
    ym = Ytr.mean(0, keepdims=True)
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ (Ytr - ym))

    def pred(Xq):
        Z = Xq - mu
        if sd is not None:
            Z = Z / sd
        return Z @ W + ym
    return pred


def inner_cv(X, Y, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(X))
    best, be = (PREPS[0], ALPHAS[0]), np.inf
    for prep in PREPS:
        for al in ALPHAS:
            errs = []
            for f in np.array_split(idx, k):
                tr = np.setdiff1d(idx, f)
                errs.append(np.abs(fit_ridge(X[tr], Y[tr], al, prep)(X[f]) - Y[f]).mean())
            e = float(np.mean(errs))
            if e < be:
                best, be = (prep, al), e
    return best


@torch.inference_mode()
def model_predict(ckpt, pretrain, texts, tok, device):
    model = RobertaVADRegressor(pretrain, dropout=0.1, offline=True)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    out = []
    for i in range(0, len(texts), 64):
        b = tok(list(texts[i:i + 64]), padding=True, truncation=True, max_length=128,
                return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        out.append(model(**b).float().cpu().numpy())
    return np.concatenate(out)


def boot_ci(diff, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(B, n))
    return tuple(float(x) for x in np.percentile(diff[idx].mean(1), [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/rankup_project/data/emobank_processed"))
    ap.add_argument("--pretrain-path", type=Path, default=Path("/root/autodl-tmp/rankup_project/pretrained/roberta-base"))
    ap.add_argument("--mainline", type=Path, default=Path("/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc"))
    ap.add_argument("--out", type=Path, default=Path("outputs/audit_e0/e0_audit.json"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)

    rows, frames = {}, {}
    for s in (0, 1, 2):
        lb = pd.read_parquet(a.data_root / f"lb500_s{s}" / "train_lb.parquet")
        for f in range(5):
            d = (a.mainline / "outputs" / "oof" / f"fold{f}") if s == 0 else \
                (Path(".") / "outputs" / f"oof_seed{s}" / f"fold{f}")
            if not (d / "oof_predictions.parquet").exists():
                print("MISSING", s, f, d)
                continue
            par = pd.read_parquet(d / "oof_predictions.parquet")
            hold_ids = par["id"].tolist()
            lb_hold = lb.set_index("id").loc[hold_ids]
            Yh = lb_hold[["V_norm", "A_norm", "D_norm"]].to_numpy(float)
            assert np.allclose(par[["y_V", "y_A", "y_D"]].to_numpy(float), Yh, atol=1e-6)
            Yt = lb[~lb["id"].isin(set(hold_ids))][["V_norm", "A_norm", "D_norm"]].to_numpy(float)

            Fl, Fh = np.load(d / "features_labeled.npy"), np.load(d / "features_holdout_p1.npy")
            prep, al = inner_cv(Fl, Yt)
            p_ridge = fit_ridge(Fl, Yt, al, prep)(Fh)
            p_p1 = model_predict(d / "p1_final.pth", str(a.pretrain_path),
                                 lb_hold["text"].tolist(), tok, dev)
            p_jacf = par[["raw_V", "raw_A", "raw_D"]].to_numpy(float)
            for arm, pred in (("ridge", p_ridge), ("mlp_p1", p_p1), ("mlp_jacf", p_jacf)):
                rows.setdefault((s, arm), []).append(pd.DataFrame(
                    {"id": hold_ids, "fold": f, **{f"y_{x}": Yh[:, i] for i, x in enumerate(DIMS)},
                     **{f"p_{x}": pred[:, i] for i, x in enumerate(DIMS)},
                     "prep": prep, "alpha": al}))

    out = {"protocol": "per-fold encoder trained from base roberta-base; paired arms on the same 100 holdout samples",
           "per_seed": {}, "pooled": {}}
    for (s, arm), lst in sorted(rows.items()):
        df = pd.concat(lst, ignore_index=True)
        assert len(df) == 500, (s, arm, len(df))
        frames[(s, arm)] = df
        po = df[[f"p_{x}" for x in DIMS]].to_numpy()
        yo = df[[f"y_{x}" for x in DIMS]].to_numpy()
        out["per_seed"].setdefault(f"seed{s}", {})[arm] = dict(
            oof=macro(po, yo), per_dim={DIMS[i]: metrics(po[:, i], yo[:, i]) for i in range(3)},
            prep=str(df["prep"].iloc[0]), alpha=float(df["alpha"].iloc[0]))
        m = out["per_seed"][f"seed{s}"][arm]["oof"]
        print(f"seed{s} {arm:9s} MAE={m['mae']:.5f} r={m['pearson']:.4f} sd={m['sd_ratio']:.4f} "
              f"CCC={m['ccc']:.4f} R2={m['r2']:.4f}", flush=True)

    print("\n=== 3-seed mean (OOF) ===")
    for arm in ("mlp_p1", "mlp_jacf", "ridge"):
        vals = [out["per_seed"][f"seed{s}"][arm]["oof"] for s in (0, 1, 2)]
        mean = {k: float(np.mean([v[k] for v in vals])) for k in ("mae", "pearson", "sd_ratio", "ccc", "r2")}
        sd = {k: float(np.std([v[k] for v in vals])) for k in ("mae", "ccc", "r2")}
        out["pooled"][arm] = dict(mean=mean, sd=sd)
        print(f"{arm:9s} MAE={mean['mae']:.5f}±{sd['mae']:.5f} r={mean['pearson']:.4f} "
              f"sd={mean['sd_ratio']:.4f} CCC={mean['ccc']:.4f}±{sd['ccc']:.4f} "
              f"R2={mean['r2']:.4f}±{sd['r2']:.4f}")

    print("\n=== paired bootstrap: ridge - mlp_p1 / mlp_jacf - mlp_p1 (MAE, 1-5 scale) ===")
    for s in (0, 1, 2):
        base = frames[(s, "mlp_p1")].set_index("id").sort_index()
        yb = base[[f"y_{x}" for x in DIMS]].to_numpy()
        eb = np.abs(base[[f"p_{x}" for x in DIMS]].to_numpy() - yb).mean(1) * SCALE
        for arm in ("ridge", "mlp_jacf"):
            o = frames[(s, arm)].set_index("id").sort_index()
            assert o.index.equals(base.index)
            eo = np.abs(o[[f"p_{x}" for x in DIMS]].to_numpy() - yb).mean(1) * SCALE
            d = eo - eb
            lo, hi = boot_ci(d)
            key = f"{arm}_minus_mlp_p1"
            out["per_seed"][f"seed{s}"][key] = dict(mean_delta_mae=float(d.mean()), ci95=[lo, hi], n=len(d))
            print(f"seed{s} {arm:9s} - mlp_p1 = {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
