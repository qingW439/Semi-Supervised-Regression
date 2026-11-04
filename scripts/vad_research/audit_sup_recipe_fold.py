#!/usr/bin/env python3
"""One fold of the "train the supervised MLP longer" E0* candidate, strict OOF.

This variant deliberately keeps the ORIGINAL P1 learning rate (2e-5) and batch size (16) and
only increases the number of steps (10 epochs -> 25 epochs).  It is therefore an a-priori
recipe ("just train longer"), not a dev-tuned one: the dev-visible recipe test
(lr 1e-4 / bs 64) is NOT used here because its learning rate was chosen by looking at dev.

No dev data, no unlabeled data, fixed step count, no early stopping.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import (  # noqa: E402
    GoldData, NORM_COLUMNS, TextCollator, read_gold,
)
from scripts.vad_research.train_roberta_vad_supervised import (  # noqa: E402
    RobertaVADRegressor, set_seed,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-seed", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--data-root", type=Path,
                    default=Path("/root/autodl-tmp/rankup_project/data/emobank_processed"))
    ap.add_argument("--pretrain-path", type=Path,
                    default=Path("/root/autodl-tmp/rankup_project/pretrained/roberta-base"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip-grad-norm", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-ckpt", action="store_true",
                    help="also write base.pth so that the semi-supervised branches can start "
                         "from the *same* supervised base (paired comparison)")
    a = ap.parse_args()

    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available (refusing to silently fall back to CPU)")
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = a.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(1000 * a.data_seed + a.fold)

    lb = read_gold(a.data_root / f"lb500_s{a.data_seed}" / "train_lb.parquet", expected_n=500)
    man = json.loads(a.manifest.read_text())
    hold_ids = man["fold_ids"][str(a.fold)]
    hold_set = set(hold_ids)
    train_ids = [i for i in lb["id"] if i not in hold_set]
    if len(hold_ids) != 100 or len(train_ids) != 400:
        raise ValueError("bad fold sizes")
    assert not (set(train_ids) & hold_set)
    lb_train = lb[lb["id"].isin(train_ids)].reset_index(drop=True)
    lb_hold = lb[lb["id"].isin(hold_ids)].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(str(a.pretrain_path), local_files_only=True, use_fast=True)
    model = RobertaVADRegressor(str(a.pretrain_path), dropout=0.1, offline=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    dl = DataLoader(GoldData(lb_train), batch_size=a.batch_size, shuffle=True,
                    collate_fn=TextCollator(tok, 128),
                    generator=torch.Generator().manual_seed(31_000 + 100 * a.data_seed + a.fold))
    losses = []
    it = iter(dl)
    for step in range(a.steps):
        try:
            b, y, _ = next(it)
        except StopIteration:
            it = iter(dl)
            b, y, _ = next(it)
        b = {k: v.to(dev) for k, v in b.items()}
        y = y.to(dev)
        pred = model(**b)
        loss = F.l1_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad_norm)
        opt.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 128 == 0:
            print(f"step {step + 1}/{a.steps} loss={losses[-1]:.6f}", flush=True)

    model.eval()
    hold_dl = DataLoader(GoldData(lb_hold), batch_size=32, shuffle=False,
                         collate_fn=TextCollator(tok, 128))
    ys, ps, ids = [], [], []
    with torch.inference_mode():
        for b, y, bid in hold_dl:
            b = {k: v.to(dev) for k, v in b.items()}
            ps.append(model(**b).float().cpu().numpy())
            ys.append(y.numpy())
            ids.extend(bid)
    Y, P = np.concatenate(ys), np.concatenate(ps)
    df = pd.DataFrame({"id": ids, "fold": a.fold,
                       "y_V": Y[:, 0], "y_A": Y[:, 1], "y_D": Y[:, 2],
                       "p_V": P[:, 0], "p_A": P[:, 1], "p_D": P[:, 2]})
    df.to_parquet(out / "sup_predictions.parquet", index=False)
    if a.save_ckpt:
        torch.save({"model": model.state_dict(), "steps": a.steps, "lr": a.lr,
                    "batch_size": a.batch_size, "fold": a.fold, "data_seed": a.data_seed},
                   out / "base.pth")
    mae = float(np.abs(P - Y).mean() * 4)
    (out / "run_config.json").write_text(json.dumps(vars(a) | {"out_dir": str(out)}, indent=2,
                                                    default=str) + "\n")
    print(f"FOLD {a.fold} seed{a.data_seed} steps={a.steps} OOF MAE(1-5)={mae:.5f}", flush=True)


if __name__ == "__main__":
    main()
