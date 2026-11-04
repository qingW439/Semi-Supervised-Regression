#!/usr/bin/env python3
"""Component-level OOF comparison for M1 (JACF vs its control vs ACLDP references).

Arms (all 5-fold OOF, per-fold encoders, identical holdout samples):
  mlp_p1            supervised MLP of that fold (matched control)
  jacf              JACF student of that fold (lambda_u=1)
  acldp_m0          JACF + difference distillation, reference = the fold's supervised MLP
  acldp_ridge       JACF + difference distillation, reference = fold-internal closed-form ridge
  acldp_ridge_shuf  negative control: the ridge reference shuffled within each batch
  ridge             closed-form ridge readout on the fold features (E0* candidate)

Reports per-seed and 3-seed means plus paired bootstrap CIs against JACF and against E0*.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    GoldData, NORM_COLUMNS, TextCollator, read_gold,
)
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
                pearson=float(cov / np.sqrt(vp * vy)),
                sd_ratio=float(np.sqrt(vp / vy)),
                ccc=float(2 * cov / (vp + vy + (mp - my) ** 2)),
                r2=float(1 - ((p - y) ** 2).mean() / vy))


def macro(pred, y):
    m = {k: float(np.mean([metrics(pred[:, i], y[:, i])[k] for i in range(3)]))
         for k in ("mae", "pearson", "sd_ratio", "ccc", "r2")}
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
def mlp_predict(ckpt, pretrain, texts, tok, device):
    model = RobertaVADRegressor(pretrain, dropout=0.1, offline=True)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"],
                          strict=True)
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
    idx = rng.integers(0, len(diff), size=(B, len(diff)))
    return tuple(float(x) for x in np.percentile(diff[idx].mean(1), [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/rankup_project/data/emobank_processed"))
    ap.add_argument("--pretrain-path", type=Path, default=Path("/root/autodl-tmp/rankup_project/pretrained/roberta-base"))
    ap.add_argument("--mainline", type=Path, default=Path("/root/autodl-tmp/rankup_project/Semi-Supervised-Regression-avrrc"))
    ap.add_argument("--acldp-root", type=Path, default=Path("outputs/oof_acldp"))
    ap.add_argument("--out", type=Path, default=Path("outputs/audit_e0/components_oof.json"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)

    arms = ["mlp_p1", "jacf", "jacf_final", "acldp_m0", "acldp_m0_final",
            "acldp_ridge", "acldp_ridge_final", "acldp_ridge_shuffle", "ridge"]
    frames, per_seed = {}, {}
    for s in (0, 1, 2):
        lb = read_gold(a.data_root / f"lb500_s{s}" / "train_lb.parquet", expected_n=500)
        for f in range(5):
            base = (a.mainline / "outputs" / "oof" / f"fold{f}") if s == 0 else \
                (Path(".") / "outputs" / f"oof_seed{s}" / f"fold{f}")
            if not (base / "oof_predictions.parquet").exists():
                print("MISSING base", s, f)
                continue
            par = pd.read_parquet(base / "oof_predictions.parquet")
            hold_ids = par["id"].tolist()
            lbh = lb.set_index("id").loc[hold_ids]
            Yh = lbh[["V_norm", "A_norm", "D_norm"]].to_numpy(float)
            preds = {"jacf": par[["raw_V", "raw_A", "raw_D"]].to_numpy(float),
                     "jacf_final": par[["vrr_V", "vrr_A", "vrr_D"]].to_numpy(float),
                     "mlp_p1": mlp_predict(base / "p1_final.pth", str(a.pretrain_path),
                                           lbh["text"].tolist(), tok, dev)}
            Fl, Fh = np.load(base / "features_labeled.npy"), np.load(base / "features_holdout_p1.npy")
            Yt = lb[~lb["id"].isin(set(hold_ids))][["V_norm", "A_norm", "D_norm"]].to_numpy(float)
            prep, al = inner_cv(Fl, Yt)
            preds["ridge"] = fit_ridge(Fl, Yt, al, prep)(Fh)
            for ref in ("m0", "ridge", "ridge_shuffle"):
                d = a.acldp_root / f"seed{s}_{ref}_fold{f}"
                if (d / "oof_predictions.parquet").exists():
                    p = pd.read_parquet(d / "oof_predictions.parquet")
                    assert p["id"].tolist() == hold_ids, (s, f, ref)
                    preds[f"acldp_{ref}"] = p[["raw_V", "raw_A", "raw_D"]].to_numpy(float)
                    preds[f"acldp_{ref}_final"] = p[["vrr_V", "vrr_A", "vrr_D"]].to_numpy(float)
            for arm, pred in preds.items():
                frames.setdefault((s, arm), []).append(pd.DataFrame(
                    {"id": hold_ids,
                     **{f"y_{x}": Yh[:, i] for i, x in enumerate(DIMS)},
                     **{f"p_{x}": pred[:, i] for i, x in enumerate(DIMS)}}))

    for (s, arm), lst in sorted(frames.items()):
        df = pd.concat(lst, ignore_index=True)
        po = df[[f"p_{x}" for x in DIMS]].to_numpy()
        yo = df[[f"y_{x}" for x in DIMS]].to_numpy()
        per_seed.setdefault(f"seed{s}", {})[arm] = dict(n=len(df), oof=macro(po, yo),
                                                        per_dim={DIMS[i]: metrics(po[:, i], yo[:, i]) for i in range(3)})
        m = per_seed[f"seed{s}"][arm]["oof"]
        print(f"seed{s} {arm:17s} n={len(df):3d} MAE={m['mae']:.5f} r={m['pearson']:.4f} "
              f"sd={m['sd_ratio']:.4f} CCC={m['ccc']:.4f} R2={m['r2']:.4f}", flush=True)

    print("\n=== 3-seed mean ===")
    pooled = {}
    for arm in arms:
        vals = [per_seed[f"seed{s}"][arm]["oof"] for s in (0, 1, 2) if arm in per_seed.get(f"seed{s}", {})]
        if len(vals) != 3:
            print(f"{arm:17s} incomplete ({len(vals)}/3 seeds)")
            continue
        mean = {k: float(np.mean([v[k] for v in vals])) for k in ("mae", "pearson", "sd_ratio", "ccc", "r2")}
        sd = {k: float(np.std([v[k] for v in vals])) for k in ("mae", "ccc", "r2")}
        pooled[arm] = dict(mean=mean, sd=sd)
        print(f"{arm:17s} MAE={mean['mae']:.5f}±{sd['mae']:.5f} r={mean['pearson']:.4f} "
              f"sd={mean['sd_ratio']:.4f} CCC={mean['ccc']:.4f}±{sd['ccc']:.4f} "
              f"R2={mean['r2']:.4f}±{sd['r2']:.4f}")

    print("\n=== paired bootstrap vs jacf (MAE, 1-5 scale) ===")
    deltas = {}
    by_sa = {}
    for (s, arm), lst in frames.items():
        df = pd.concat(lst, ignore_index=True).set_index("id").sort_index()
        by_sa[(s, arm)] = df
    for s in (0, 1, 2):
        key = (s, "jacf")
        if key not in by_sa:
            continue
        base = by_sa[key]
        yb = base[[f"y_{x}" for x in DIMS]].to_numpy()
        eb = np.abs(base[[f"p_{x}" for x in DIMS]].to_numpy() - yb).mean(1) * SCALE
        for arm in arms:
            if arm == "jacf" or (s, arm) not in by_sa:
                continue
            o = by_sa[(s, arm)]
            if not o.index.equals(base.index):
                continue
            eo = np.abs(o[[f"p_{x}" for x in DIMS]].to_numpy() - yb).mean(1) * SCALE
            d = eo - eb
            lo, hi = boot_ci(d)
            deltas.setdefault(arm, {})[f"seed{s}"] = dict(mean=float(d.mean()), ci95=[lo, hi])
            print(f"seed{s} {arm:17s} - jacf = {d.mean():+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]")

    out = {"per_seed": per_seed, "pooled": pooled, "paired_vs_jacf": deltas,
           "note": "all arms share split, per-fold encoder and holdout samples; MAE on 1-5 scale"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
