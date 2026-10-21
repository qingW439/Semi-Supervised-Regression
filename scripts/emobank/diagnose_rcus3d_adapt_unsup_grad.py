#!/usr/bin/env python3
"""Read-only gradient audit of fixed A1/A2 runs; no optimizer is constructed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, UnlabeledData  # noqa: E402
from semilearn.nets.bert.bert_vad import BertVADRegressor  # noqa: E402
from semilearn.nets.bert.bert_vad_rcus_adapt import (  # noqa: E402
    BertVADRCUS3D, supervised_close_mixup_loss, unsupervised_close_mixup_loss,
)
from scripts.emobank.rcus3d_adapt_common import load_splits_no_test, sha256  # noqa: E402
from scripts.emobank.train_emobank_rcus3d_adapt import UnlabeledCollator  # noqa: E402
from scripts.emobank.train_emobank_supervised import set_seed  # noqa: E402

ROOT = REPO / "saved_models/emobank/rcus3d_adapt"
MODEL = "google/bert_uncased_L-4_H-512_A-8"


def norm(grads):
    return float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads if g is not None)))


def cosine(a, b):
    dot = sum((x.detach() * y.detach()).sum() for x, y in zip(a, b)
              if x is not None and y is not None).item()
    na, nb = norm(a), norm(b)
    return dot / (na * nb) if na * nb else None


def one_state(path, cutoffs, temps, active, weight, lb_loader, ulb_loader, device):
    set_seed(2137)
    model = BertVADRCUS3D(BertVADRegressor(MODEL, dropout=.1, offline=True))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"], strict=True)
    model.to(device)
    params = tuple(model.backbone.bert.parameters())
    rng_l = np.random.default_rng(4001)
    rng_u = np.random.default_rng(4002)
    rows = []
    for step, (labeled, unlabeled) in enumerate(zip(lb_loader, ulb_loader)):
        if step >= 4:
            break
        model.train()
        lt, y, _ = labeled
        lt = {k: v.to(device) for k, v in lt.items()}
        y = y.to(device)
        labeled_out = model(**lt)
        l1 = torch.nn.functional.l1_loss(labeled_out["regression"], y)
        lam_l = float(rng_l.beta(.5, .5)); lam_l = max(lam_l, 1-lam_l)
        sup, _ = supervised_close_mixup_loss(model, labeled_out["features"], y, lam_l)
        ut, _ = unlabeled
        ut = {k: v.to(device) for k, v in ut.items()}
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
            model.eval()
            with torch.no_grad():
                teacher = model(**ut)["rank_logits"]
            model.train()
            student = model(**ut)["features"]
        lam_u = float(rng_u.beta(.5, .5)); lam_u = max(lam_u, 1-lam_u)
        unsup, kept, possible = unsupervised_close_mixup_loss(
            model, teacher, student, lam_u, cutoffs, temperature=temps, active=active)
        g_l1 = torch.autograd.grad(l1, params, retain_graph=True, allow_unused=True)
        g_sup = torch.autograd.grad(sup, params, retain_graph=True, allow_unused=True)
        g_unsup = torch.autograd.grad(unsup, params, allow_unused=True)
        n_l1, n_sup, n_unsup = norm(g_l1), norm(g_sup), norm(g_unsup)
        rows.append({"batch": step, "l1": float(l1), "supervised": float(sup),
                     "unsupervised": float(unsup), "kept": kept.cpu().tolist(),
                     "possible": possible.cpu().tolist(), "norm_l1": n_l1,
                     "weighted_norm_supervised": weight * n_sup,
                     "weighted_norm_unsupervised": n_unsup,
                     "ratio_supervised_to_l1": weight*n_sup/n_l1,
                     "ratio_unsupervised_to_l1": n_unsup/n_l1,
                     "cosine_l1_supervised": cosine(g_l1, g_sup),
                     "cosine_l1_unsupervised": cosine(g_l1, g_unsup)})
    return rows


def main():
    path = ROOT / "unsup_gradient_diagnostic.json"
    if path.exists():
        raise FileExistsError(path)
    lb, ulb, _, hashes = load_splits_no_test()
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    lb_loader = DataLoader(GoldData(lb), batch_size=8, shuffle=True,
                           collate_fn=TextCollator(tok, 128),
                           generator=torch.Generator().manual_seed(177))
    ulb_loader = DataLoader(UnlabeledData(ulb), batch_size=8, shuffle=True,
                            collate_fn=UnlabeledCollator(tok, 128),
                            generator=torch.Generator().manual_seed(188))
    weight = json.loads((ROOT / "gradient/gradient_weight.json").read_text())["supervised_mixup_weight"]
    calibration = json.loads((ROOT / "oof/calibration.json").read_text())["dimensions"]
    states = {
        "warmup_fixed95": (ROOT / "warmup/warmup_model.pth", [.95]*3, [1.]*3, [True]*3),
        "a1_best_fixed95": (ROOT / "a1_lb500_s0/model_best.pth", [.95]*3, [1.]*3, [True]*3),
        "a2_best_oof": (ROOT / "a2_lb500_s0/model_best.pth",
                        [calibration[d]["threshold"] for d in "VAD"],
                        [calibration[d]["temperature"] for d in "VAD"],
                        [calibration[d]["enabled"] for d in "VAD"]),
    }
    result = {"data_sha256": hashes, "supervised_weight": weight,
              "optimizer_steps": 0, "states": {}, "test_access": "none"}
    for name, (ckpt, cutoff, temp, active) in states.items():
        lb_loader.generator.manual_seed(177)
        ulb_loader.generator.manual_seed(188)
        rows = one_state(ckpt, cutoff, temp, active, weight,
                         lb_loader, ulb_loader, torch.device("cuda"))
        result["states"][name] = {"checkpoint_sha256": sha256(ckpt),
                                  "cutoffs": cutoff, "temperatures": temp,
                                  "active": active, "rows": rows}
        print(name, "median_unsup_to_l1", float(np.median([r["ratio_unsupervised_to_l1"] for r in rows])),
              flush=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
