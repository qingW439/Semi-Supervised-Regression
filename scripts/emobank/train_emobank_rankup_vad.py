#!/usr/bin/env python3
"""E2 pilot: dimension-wise RankUp semi-supervised regression on EmoBank.

For qingW439/Semi-Supervised-Regression at commit 543e49f (emobank-vad).
Uses shared BertVADRegressor, RankUpVADNet and RankUpVADLoss without editing
any Yelp implementations. Selects on DEV only; never opens TEST.

Unlabeled weak view: original tokenization.
Unlabeled strong view: conservative [MASK] perturbation of at most 2 nonspecial
content tokens for sentences with >=5 eligible tokens. This is a *different*
input view but emotion preservation is NOT guaranteed; audit before claiming a
faithful, semantic-preserving augmentation of RankUp's original Yelp setup.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.nets.bert.bert_vad import BertVADRegressor
from semilearn.algorithms.rankup_vad.rankup_vad import RankUpVADNet, RankUpVADLoss
from semilearn.datasets.nlp_datasets.emobank import (
    GoldData, UnlabeledData, TextCollator, read_gold, read_unlabeled,
)
from scripts.emobank.train_emobank_supervised import (
    DIMENSIONS, metric_report, predict, set_seed, current_git_commit,
)

DEFAULT_DATA_ROOT = REPO.parent / "data" / "emobank_processed"
DEFAULT_BERT = "google/bert_uncased_L-4_H-512_A-8"


class UnlabeledPairCollator:
    """Weak original tokens and conservative strong masked-token tokens."""
    def __init__(self, tokenizer, max_length=128, mask_prob=0.10, max_mask_tokens=2,
                 min_eligible=5):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.max_mask_tokens = max_mask_tokens
        self.min_eligible = min_eligible
        if tokenizer.mask_token_id is None:
            raise ValueError("Pretrained tokenizer lacks [MASK]; cannot make strong view")
        self.special_ids = set(tokenizer.all_special_ids)

    def __call__(self, rows):
        texts, ids = zip(*rows)
        weak = self.tokenizer(list(texts), truncation=True,
                              max_length=self.max_length, padding=True,
                              return_tensors="pt")
        strong = {k: v.clone() for k, v in weak.items()}
        changed_samples = 0
        changed_tokens = 0
        eligible_tokens = 0
        for i in range(len(ids)):
            candidates = [j for j, tid in enumerate(weak["input_ids"][i].tolist())
                          if int(weak["attention_mask"][i, j]) == 1
                          and int(tid) not in self.special_ids]
            eligible_tokens += len(candidates)
            if len(candidates) < self.min_eligible or self.mask_prob <= 0:
                continue
            count = min(self.max_mask_tokens, len(candidates),
                        max(1, int(round(len(candidates) * self.mask_prob))))
            chosen = random.sample(candidates, count)
            strong["input_ids"][i, chosen] = self.tokenizer.mask_token_id
            changed_samples += 1
            changed_tokens += count
        return weak, strong, list(ids), changed_samples, changed_tokens, eligible_tokens


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--budget", type=int, choices=(250, 500, 1000), default=250)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--pretrain-path", default=DEFAULT_BERT)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--uratio", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--arc-loss-ratio", type=float, default=0.2)
    p.add_argument("--arc-ulb-loss-ratio", type=float, default=1.0)
    p.add_argument("--p-cutoff", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--strong-mask-prob", type=float, default=0.10)
    p.add_argument("--strong-max-masked", type=int, default=2)
    p.add_argument("--strong-min-eligible", type=int, default=5)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.uratio < 1 or args.eval_batch_size < 1:
        p.error("epochs/eval_batch_size/uratio must be positive; batch-size >=2")
    if args.max_length < 8 or args.patience < 1 or args.strong_max_masked < 1:
        p.error("max-length>=8, patience>=1, strong-max-masked>=1 required")
    if not 0 <= args.strong_mask_prob <= 1:
        p.error("strong-mask-prob must be in [0,1]")
    if args.out is None:
        args.out = REPO / "saved_models" / "emobank" / f"rankupvad_lb{args.budget}_s{args.seed}"
    return args


def move(tokens, device):
    return {k:v.to(device) for k,v in tokens.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    out = args.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty experiment directory {out}")
    split = args.data_root / f"lb{args.budget}_s{args.seed}"
    labeled = read_gold(split / "train_lb.parquet", expected_n=args.budget)
    unlabeled = read_unlabeled(split / "train_ulb.parquet", labeled_ids=labeled["id"])
    dev = read_gold(args.data_root / "dev_gold.parquet", expected_n=1000)
    if len(unlabeled) != 8062-args.budget:
        raise RuntimeError("Unexpected unlabeled sample count")
    if set(dev["id"]) & (set(labeled["id"]) | set(unlabeled["id"])):
        raise RuntimeError("DEV and train ID overlap")
    print(f"device={device} budget={args.budget} seed={args.seed} "
          f"labeled={len(labeled)} unlabeled={len(unlabeled)} dev={len(dev)}",flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_path, use_fast=True,
                                                local_files_only=args.offline)
    gold_collate = TextCollator(tokenizer, args.max_length)
    ulb_collate = UnlabeledPairCollator(tokenizer, max_length=args.max_length,
                                       mask_prob=args.strong_mask_prob,
                                       max_mask_tokens=args.strong_max_masked,
                                       min_eligible=args.strong_min_eligible)
    labeled_loader = DataLoader(GoldData(labeled), batch_size=args.batch_size,
                                shuffle=True, num_workers=0, collate_fn=gold_collate,
                                generator=torch.Generator().manual_seed(args.seed))
    unlabeled_loader = DataLoader(UnlabeledData(unlabeled),
                                  batch_size=args.batch_size * args.uratio,
                                  shuffle=True, num_workers=0, collate_fn=ulb_collate,
                                  generator=torch.Generator().manual_seed(args.seed + 10000))
    dev_loader = DataLoader(GoldData(dev), batch_size=args.eval_batch_size,
                            shuffle=False, num_workers=0, collate_fn=gold_collate)
    backbone = BertVADRegressor(args.pretrain_path, dropout=args.dropout,
                                offline=args.offline)
    model = RankUpVADNet(backbone).to(device)
    criterion = RankUpVADLoss(arc_loss_ratio=args.arc_loss_ratio,
                              arc_ulb_loss_ratio=args.arc_ulb_loss_ratio,
                              p_cutoff=args.p_cutoff, temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "data_root": str(args.data_root.resolve()),
              "out": str(out), "repository_commit": current_git_commit(),
              "torch_version": torch.__version__,
              "augmentation": "weak=original, strong=conservative token masking (not verified emotion preserving)",
              "test_evaluated": False}
    config = {k:(str(v) if isinstance(v,Path) else v) for k,v in config.items()}
    (out/"run_config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    train_iter = iter(unlabeled_loader)
    best, best_epoch, stale = float("inf"), 0, 0
    history = []
    unique_unlabeled = set()
    for epoch in range(1, args.epochs+1):
        model.train()
        totals = {k:0.0 for k in ("regression", "supervised_rank", "unsupervised_rank", "loss")}
        accepted = np.zeros(3, dtype=np.int64)
        possible = np.zeros(3, dtype=np.int64)
        changed_samples = changed_tokens = eligible_tokens = seen = 0
        steps = 0
        for lb_tokens, lb_targets, _ in labeled_loader:
            try:
                weak, strong, ulb_ids, nchange, ntok, nelig = next(train_iter)
            except StopIteration:
                train_iter = iter(unlabeled_loader)
                weak, strong, ulb_ids, nchange, ntok, nelig = next(train_iter)
            unique_unlabeled.update(ulb_ids)
            changed_samples += nchange
            changed_tokens += ntok
            eligible_tokens += nelig
            seen += len(ulb_ids)
            lb_tokens, lb_targets = move(lb_tokens, device), lb_targets.to(device)
            weak, strong = move(weak, device), move(strong, device)
            optimizer.zero_grad(set_to_none=True)
            # Stable teacher weak-view prediction; do not train on weak targets.
            model.eval()
            with torch.no_grad():
                teacher_weak = model(**weak)
                pair = RankUpVADNet.pair_logits(teacher_weak["arc_logits"])
                probs = pair.softmax(-1)
                confidence = probs.max(-1).values
                m = confidence.shape[0]
                valid = ~torch.eye(m, dtype=torch.bool,device=device)[:,:,None]
                accepted += ((confidence >= args.p_cutoff) & valid).sum((0,1)).cpu().numpy()
                possible += valid.expand(-1,-1,3).sum((0,1)).cpu().numpy()
            model.train()
            labeled_outputs = model(**lb_tokens)
            strong_outputs = model(**strong)
            losses = criterion(labeled_outputs, lb_targets, weak=teacher_weak,
                                strong=strong_outputs)
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError("Nonfinite E2 loss")
            losses["loss"].backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad_norm)
            optimizer.step()
            for name in totals:
                totals[name] += float(losses[name].detach())
            steps += 1
        y,p,ids = predict(model.backbone, dev_loader, device)
        metrics = metric_report(y,p)
        score = float(metrics["macro_mae_original_scale"])
        acceptance = accepted / np.maximum(possible,1)
        row = {"epoch":epoch,"steps":steps,"dev_macro_mae_original":score,
               "dev_V_mae":metrics["V"]["mae"],"dev_A_mae":metrics["A"]["mae"],
               "dev_D_mae":metrics["D"]["mae"],
               "unlabeled_seen":seen,"unlabeled_unique_cumulative":len(unique_unlabeled),
               "changed_sample_fraction":changed_samples/max(seen,1),
               "masked_token_fraction":changed_tokens/max(eligible_tokens,1)}
        row.update({f"train_{k}":v/max(steps,1) for k,v in totals.items()})
        row.update({f"pseudo_accept_{d}":float(acceptance[i]) for i,d in enumerate(DIMENSIONS)})
        history.append(row)
        print(f"epoch {epoch:03d} dev_MAE={score:.5f} "
              f"V={metrics['V']['mae']:.4f} A={metrics['A']['mae']:.4f} D={metrics['D']['mae']:.4f} "
              f"reg={row['train_regression']:.4f} sup_rank={row['train_supervised_rank']:.4f} "
              f"unsup_rank={row['train_unsupervised_rank']:.4f} "
              f"accept_VAD={[round(v,4) for v in acceptance.tolist()]} "
              f"augmented={row['changed_sample_fraction']:.3f} "
              f"unique_ulb={len(unique_unlabeled)}",flush=True)
        if score < best - args.min_delta:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"model":model.state_dict(),"epoch":epoch,
                        "dev_metrics":metrics,"run_config":config},out/"model_best.pth")
            (out/"dev_metrics_best.json").write_text(
                json.dumps({"epoch":epoch,**metrics},indent=2)+"\n",encoding="utf-8")
            pred_frame=pd.DataFrame({"id":ids})
            for k,d in enumerate(DIMENSIONS):
                pred_frame[f"gold_{d}"]=y[:,k]*4+1
                pred_frame[f"pred_{d}"]=p[:,k]*4+1
            pred_frame.to_csv(out/"dev_predictions_best.csv",index=False)
        else:
            stale += 1
            if stale>=args.patience:
                print(f"early_stop epoch={epoch}, best_epoch={best_epoch}",flush=True)
                break
        pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
    pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
    print(f"BEST epoch={best_epoch} dev_macro_MAE_1to5={best:.5f} "
          f"unique_unlabeled_seen={len(unique_unlabeled)}/{len(unlabeled)}",flush=True)
    print(f"outputs: {out}")
    print("The independent TEST split was never loaded or evaluated.")
    if not any(r["pseudo_accept_V"] > 0 or r["pseudo_accept_A"] > 0 or r["pseudo_accept_D"] > 0 for r in history):
        print("WARNING: NO pseudo-ranked pairs accepted at cutoff; this run does not evidence unlabeled ranking benefits.",flush=True)

if __name__ == "__main__":
    main()
