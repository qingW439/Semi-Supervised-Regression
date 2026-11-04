#!/usr/bin/env python3
"""E0' audit: closed-form ridge under the SAME 5-fold OOF protocol as the neural baselines.

Pre-registered in docs/ADMISSION_V2_PREREGISTERED.md (frozen before this ran):
  * per outer fold, the ridge alpha AND the preprocessing (centre-only vs z-score) are
    chosen by 5-fold CV *inside that fold's 400 training samples* -- never on dev, never
    on the outer holdout;
  * every fold model predicts only its own 100 held-out samples -> 500-sample OOF;
  * dev is predicted for reporting only;
  * both feature variants are reported: L2-normalised mean-pool (the canonical `encode()`)
    and raw mean-pool (what the MLP head actually sees).

Outputs
  outputs/audit_e0/ridge_oof_s{seed}.parquet     id, fold, y_*, pred_*
  outputs/audit_e0/ridge_metrics.json            per-seed and 3-seed summaries
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
                srcc=float(pd.Series(p).corr(pd.Series(y), method="spearman")),
                sd_ratio=float(np.sqrt(vp / vy)),
                ccc=float(2 * cov / (vp + vy + (mp - my) ** 2)),
                r2=float(1 - ((p - y) ** 2).mean() / vy))


def macro(pred, y):
    m = {k: float(np.mean([metrics(pred[:, i], y[:, i])[k] for i in range(3)])) for k in
         ("mae", "rmse", "pearson", "srcc", "sd_ratio", "ccc", "r2")}
    m["rho"] = m["pearson"] * m["sd_ratio"]
    return m


def fit_ridge(Xtr, Ytr, alpha, prep):
    mu = Xtr.mean(0, keepdims=True)
    Xc = Xtr - mu
    if prep == "zscore":
        sd = Xc.std(0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        Xc = Xc / sd
    else:
        sd = None
    ym = Ytr.mean(0, keepdims=True)
    G = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    W = np.linalg.solve(G, Xc.T @ (Ytr - ym))

    def predict(Xq):
        Zq = Xq - mu
        if sd is not None:
            Zq = Zq / sd
        return Zq @ W + ym
    return predict


def inner_cv(X, Y, k=5, seed=0):
    """Choose (prep, alpha) by 5-fold CV inside the training fold."""
    idx = np.random.default_rng(seed).permutation(len(X))
    best, be = (PREPS[0], ALPHAS[0]), np.inf
    for prep in PREPS:
        for al in ALPHAS:
            errs = []
            for f in np.array_split(idx, k):
                tr = np.setdiff1d(idx, f)
                pred = fit_ridge(X[tr], Y[tr], al, prep)(X[f])
                errs.append(np.abs(pred - Y[f]).mean())
            e = float(np.mean(errs))
            if e < be:
                best, be = (prep, al), e
    return best


@torch.inference_mode()
def encode_raw(encoder, texts, tok, device, bs=128):
    encoder.eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=128,
                return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        st = encoder(**b, return_dict=True).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(dtype=st.dtype)
        out.append(((st * m).sum(1) / m.sum(1).clamp_min(1)).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/rankup_project/data/emobank_processed"))
    ap.add_argument("--pretrain-path", type=Path, default=Path("/root/autodl-tmp/rankup_project/pretrained/roberta-base"))
    ap.add_argument("--p1", type=Path, default=Path("/root/autodl-tmp/rankup_project/Semi-Supervised-Regression/saved_models/vad_research/p1_roberta_lb500_s0/model_best.pth"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/auto_search/features_frozen.npz"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/audit_e0"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(a.cache, allow_pickle=True)
    Xn, Xnd, order = z["Ftr"].astype(np.float64), z["Fdev"].astype(np.float64), [str(x) for x in z["order"]]
    pos = {k: i for i, k in enumerate(order)}
    dv = read_gold(a.data_root / "dev_gold.parquet", expected_n=1000)
    ydev = dv.loc[:, list(NORM_COLUMNS)].to_numpy(float)

    # raw (un-normalised) pooled features, exactly what the MLP head sees
    raw_cache = a.out_dir / "features_raw_pool.npz"
    if raw_cache.exists():
        zr = np.load(raw_cache, allow_pickle=True)
        Xr, Xrd = zr["Ftr"], zr["Fdev"]
        assert [str(x) for x in zr["order"]] == order
    else:
        tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
        model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True)
        ck = torch.load(a.p1, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        del ck
        model.to(dev)
        lb0 = read_gold(a.data_root / "lb500_s0" / "train_lb.parquet", expected_n=500)
        ulb0 = pd.read_parquet(a.data_root / "lb500_s0" / "train_ulb.parquet")
        pool = pd.concat([lb0, ulb0], ignore_index=True)
        assert pool["id"].tolist() == order, "pool order mismatch"
        Xr = encode_raw(model.encoder, pool["text"].tolist(), tok, dev).astype(np.float64)
        Xrd = encode_raw(model.encoder, dv["text"].tolist(), tok, dev).astype(np.float64)
        np.savez_compressed(raw_cache, Ftr=Xr, Fdev=Xrd,
                            order=np.array(order, dtype=object))
    variants = {"l2norm": (Xn, Xnd), "raw": (Xr, Xrd)}

    out = {"protocol": "5-fold OOF, alpha+prep chosen inside the 400 training samples",
           "per_seed": {}, "note": "dev is reported only; it is never used for any choice"}
    for s in (0, 1, 2):
        man_path = a.data_root.parent.parent / "Semi-Supervised-Regression-avrrc" / "outputs" / (
            "p0" if s == 0 else f"p0_s{s}") / "fold_manifest.json"
        man = json.loads(Path(man_path).read_text())
        folds = {int(k): v for k, v in man["fold_ids"].items()}
        lb = read_gold(a.data_root / f"lb500_s{s}" / "train_lb.parquet", expected_n=500)
        Y = lb.loc[:, list(NORM_COLUMNS)].to_numpy(float)
        id2y = {i: y for i, y in zip(lb["id"], Y)}
        union = [i for f in sorted(folds) for i in folds[f]]
        assert len(union) == 500 and len(set(union)) == 500, (s, len(union))
        assert set(union) == set(lb["id"].tolist()), f"manifest/label id mismatch seed {s}"

        for vname, (Xp, Xpd) in variants.items():
            rows, dev_preds = [], []
            for f in sorted(folds):
                hold = folds[f]
                tr_ids = [i for g in sorted(folds) if g != f for i in folds[g]]
                tr_idx = np.array([pos[i] for i in tr_ids])
                ho_idx = np.array([pos[i] for i in hold])
                Ytr = np.stack([id2y[i] for i in tr_ids])
                prep, al = inner_cv(Xp[tr_idx], Ytr)
                pred = fit_ridge(Xp[tr_idx], Ytr, al, prep)
                p_ho = pred(Xp[ho_idx])
                dev_preds.append(pred(Xpd))
                for j, pid in enumerate(hold):
                    rows.append({"id": pid, "fold": f, "prep": prep, "alpha": al,
                                 "y_V": id2y[pid][0], "y_A": id2y[pid][1], "y_D": id2y[pid][2],
                                 "pred_V": float(p_ho[j][0]),
                                 "pred_A": float(p_ho[j][1]),
                                 "pred_D": float(p_ho[j][2])})
            df = pd.DataFrame(rows)
            assert len(df) == 500
            po = df[["pred_V", "pred_A", "pred_D"]].to_numpy()
            yo = df[["y_V", "y_A", "y_D"]].to_numpy()
            m_oof = macro(po, yo)
            m_dev = macro(np.mean(dev_preds, 0), ydev)
            pv = df["pred_V"].to_numpy(); yv = df["y_V"].to_numpy()
            per_dim = {DIMS[i]: metrics(po[:, i], yo[:, i]) for i in range(3)}
            key = f"seed{s}_{vname}"
            out["per_seed"][key] = dict(oof=m_oof, dev=m_dev, per_dim_oof=per_dim,
                                        preps=[str(x) for x in df["prep"].unique()])
            df.to_parquet(a.out_dir / f"ridge_oof_s{s}_{vname}.parquet", index=False)
            print(f"{key:18s} OOF MAE={m_oof['mae']:.5f} r={m_oof['pearson']:.4f} "
                  f"sd={m_oof['sd_ratio']:.4f} CCC={m_oof['ccc']:.4f} R2={m_oof['r2']:.4f} | "
                  f"dev MAE={m_dev['mae']:.5f} CCC={m_dev['ccc']:.4f} R2={m_dev['r2']:.4f}",
                  flush=True)

    print("\n=== 3-seed means (OOF) ===")
    for vname in variants:
        agg = {k: float(np.mean([out["per_seed"][f"seed{s}_{vname}"]["oof"][k] for s in (0, 1, 2)]))
               for k in ("mae", "pearson", "sd_ratio", "ccc", "r2")}
        sd = {k: float(np.std([out["per_seed"][f"seed{s}_{vname}"]["oof"][k] for s in (0, 1, 2)]))
              for k in ("mae", "ccc", "r2")}
        out.setdefault("three_seed", {})[vname] = dict(mean=agg, sd=sd)
        print(f"{vname:8s} MAE={agg['mae']:.5f}±{sd['mae']:.5f} r={agg['pearson']:.4f} "
              f"sd={agg['sd_ratio']:.4f} CCC={agg['ccc']:.4f}±{sd['ccc']:.4f} "
              f"R2={agg['r2']:.4f}±{sd['r2']:.4f}")
    (a.out_dir / "ridge_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", a.out_dir / "ridge_metrics.json")


if __name__ == "__main__":
    main()
