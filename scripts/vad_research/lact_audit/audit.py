#!/usr/bin/env python3
"""One-shot, TRAIN-only LACT mechanism audit. No optimizer or TEST/DEV access."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import NORM_COLUMNS, read_gold, read_unlabeled  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor, sha256_file  # noqa: E402
from scripts.vad_research.lact_audit.lact_core import label_subspace, lact_loss, mean_pool  # noqa: E402


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True,
                   help="Original project checkout containing existing saved_models and TRAIN data")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return p.parse_args()


def quant(x):
    x = np.asarray(x, dtype=np.float64)
    return {"min": float(x.min()), "p10": float(np.quantile(x, .1)),
            "median": float(np.median(x)), "p90": float(np.quantile(x, .9)),
            "max": float(x.max()), "mean": float(x.mean())}


def corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


@torch.inference_mode()
def encode(model, frame, tokenizer, device, batch=16):
    model.eval()
    result = []
    for start in range(0, len(frame), batch):
        tok = tokenizer(frame.iloc[start:start + batch].text.tolist(), padding=True,
                        truncation=True, max_length=128, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        result.append(mean_pool(model, tok).float().cpu().numpy())
    return np.concatenate(result)


def load_existing(path, model_path, device, expected_train, expected_held=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if set(checkpoint["train_ids"]) != set(expected_train):
        raise RuntimeError(f"Checkpoint TRAIN IDs mismatch: {path}")
    if expected_held is not None and set(checkpoint["held_out_ids"]) != set(expected_held):
        raise RuntimeError(f"Checkpoint held-out IDs mismatch: {path}")
    if expected_held is not None and set(expected_train) & set(expected_held):
        raise RuntimeError("Checkpoint provenance overlaps")
    model = RobertaVADRegressor(str(model_path), dropout=.1, offline=True).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    return model.eval()


def subspace_report(z_fit, y_fit, z_hold, y_hold, seed):
    zf = torch.as_tensor(z_fit, dtype=torch.float64)
    yf = torch.as_tensor(y_fit, dtype=torch.float64)
    q, singular, rank = label_subspace(zf, yf)
    qn = q.numpy()
    c_train = (z_fit - z_fit.mean(0)).T @ (y_fit - y_fit.mean(0)) / len(z_fit)
    c_hold = (z_hold - z_hold.mean(0)).T @ (y_hold - y_hold.mean(0)) / len(z_hold)

    def alignment(basis):
        numerator = np.linalg.norm(basis.T @ c_hold, "fro") ** 2
        denominator = np.linalg.norm(c_hold, "fro") ** 2
        return float(numerator / denominator) if denominator > 0 else None

    rng = np.random.default_rng(seed)
    random_align = []
    shuffled_align = []
    for _ in range(100):
        qr, _ = np.linalg.qr(rng.standard_normal((z_fit.shape[1], rank)))
        random_align.append(alignment(qr))
        cs = (z_fit - z_fit.mean(0)).T @ (y_fit[rng.permutation(len(y_fit))] - y_fit.mean(0)) / len(z_fit)
        us, _, _ = np.linalg.svd(cs, full_matrices=False)
        shuffled_align.append(alignment(us[:, :rank]))

    per_dim = {}
    for j, name in enumerate(("V", "A", "D")):
        direction = c_train[:, j]
        length = np.linalg.norm(direction)
        score = (z_hold - z_fit.mean(0)) @ (direction / length) if length > 0 else np.zeros(len(z_hold))
        projected = np.linalg.norm(qn.T @ c_hold[:, j]) ** 2
        total = np.linalg.norm(c_hold[:, j]) ** 2
        per_dim[name] = {"heldout_signed_pearson": corr(score, y_hold[:, j]),
                         "heldout_crosscov_capture": float(projected / total) if total > 0 else None,
                         "train_direction_norm": float(length)}
    return {"singular_values": singular.tolist(), "rank": rank,
            "heldout_alignment": alignment(qn),
            "random_same_rank_alignment": quant(random_align),
            "shuffled_train_labels_alignment": quant(shuffled_align),
            "per_dimension": per_dim}, qn


def knn_graph(z, device, k=8):
    zz = F.normalize(torch.as_tensor(z, dtype=torch.float32, device=device), dim=1)
    n = len(z)
    nearest = np.empty((n, k), dtype=np.int32)
    for start in range(0, n, 256):
        sim = zz[start:start + 256] @ zz.T
        row = torch.arange(len(sim), device=device)
        sim[row, torch.arange(start, start + len(sim), device=device)] = -float("inf")
        nearest[start:start + len(sim)] = sim.topk(k, dim=1).indices.cpu().numpy()
    neighbors = [set(row.tolist()) for row in nearest]
    edges = np.array([(i, int(j)) for i in range(n) for j in nearest[i]
                      if i < j and i in neighbors[int(j)]], dtype=np.int32)
    if edges.size == 0:
        raise RuntimeError("Mutual 8-NN graph has no edges")
    return edges, nearest


def random_edges(edges, n, seed):
    rng = np.random.default_rng(seed)
    result = edges.copy()
    targets = edges[:, 1].copy()
    rng.shuffle(targets)
    for i in range(len(result)):
        if targets[i] == result[i, 0]:
            targets[i] = (targets[i] + 1) % n
    result[:, 1] = targets
    return result


def direction_stats(z, edges, q):
    d = z[edges[:, 1]].astype(np.float64) - z[edges[:, 0]].astype(np.float64)
    lengths2 = (d * d).sum(1)
    r = d - (d @ q) @ q.T
    normalized = r / np.sqrt(lengths2[:, None] + 1e-8)
    covariance = normalized.T @ normalized / len(normalized)
    trace = np.trace(covariance)
    effective_rank = float(trace ** 2 / np.square(covariance).sum()) if trace > 0 else 0.0
    return {"raw_distance": quant(np.sqrt(lengths2)),
            "projected_energy_fraction": float(np.mean(np.sum(r * r, 1) / (lengths2 + 1e-8))),
            "direction_cov_effective_rank": effective_rank,
            "direction_cov_trace": float(trace)}, covariance


def head_response(head, z, edges, q, device):
    head.eval()
    response = []
    raw_response = []
    with torch.no_grad():
        for start in range(0, len(edges), 512):
            chunk = edges[start:start + 512]
            u = torch.as_tensor(z[chunk[:, 0]], dtype=torch.float32, device=device)
            v = torch.as_tensor(z[chunk[:, 1]], dtype=torch.float32, device=device)
            d = v - u
            qt = torch.as_tensor(q, dtype=torch.float32, device=device)
            r = d - (d @ qt) @ qt.T
            jr = torch.func.jvp(head, (u,), (r,))[1]
            jd = torch.func.jvp(head, (u,), (d,))[1]
            denominator = d.square().sum(1).clamp_min(1e-8)
            response.append((jr.square() / denominator[:, None]).cpu().numpy())
            raw_response.append((jd.square() / denominator[:, None]).cpu().numpy())
    arr = np.concatenate(response)
    raw = np.concatenate(raw_response)
    return {"projected_loss": float(arr.sum(1).mean()),
            "unprojected_loss": float(raw.sum(1).mean()),
            "projected_per_dimension": arr.mean(0).tolist()}


def flat_gradients(loss, params):
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                      for p, g in zip(params, grads)])


def grad_pair(a, b):
    na = float(a.norm())
    nb = float(b.norm())
    cos = float(F.cosine_similarity(a, b, dim=0)) if na > 0 and nb > 0 else None
    return {"supervised_norm": na, "unlabeled_norm": nb,
            "unlabeled_to_supervised": nb / na if na > 0 else None,
            "cosine": cos}


def gradient_audit(model, fit_frame, ulb_frame, real_edges, random_graph, q, tokenizer, device):
    model.train()
    head_params = list(model.head.parameters())
    encoder_params = list(model.encoder.parameters())
    params = head_params + encoder_params
    fit_batch = fit_frame.iloc[:16]
    tok = tokenizer(fit_batch.text.tolist(), padding=True, truncation=True,
                    max_length=128, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}
    y = torch.as_tensor(fit_batch.loc[:, NORM_COLUMNS].to_numpy(np.float32), device=device)
    # Preserve external RNG. Supervised branch remains in train mode with its P2 dropout.
    with torch.random.fork_rng(devices=[device.index if device.index is not None else 0] if device.type == "cuda" else []):
        torch.manual_seed(20261028)
        supervised = F.l1_loss(model(**tok), y)
    gs = flat_gradients(supervised, params)
    del supervised
    count = min(32, len(real_edges))
    q_t = torch.as_tensor(q, dtype=torch.float32, device=device).detach()

    def one(edges):
        chosen = edges[:count]
        ids = sorted(set(chosen.reshape(-1).tolist()))
        lookup = {idx: pos for pos, idx in enumerate(ids)}
        tok_u = tokenizer(ulb_frame.iloc[ids].text.tolist(), padding=True,
                          truncation=True, max_length=128, return_tensors="pt")
        tok_u = {k: v.to(device) for k, v in tok_u.items()}
        prior = model.encoder.training
        model.encoder.eval()  # deterministic unlabeled encoder; gradients remain enabled
        z = mean_pool(model, tok_u)
        model.encoder.train(prior)
        iu = torch.as_tensor([lookup[int(i)] for i in chosen[:, 0]], device=device)
        iv = torch.as_tensor([lookup[int(i)] for i in chosen[:, 1]], device=device)
        loss = lact_loss(model.head, z[iu], z[iv], q_t)
        gu = flat_gradients(loss, params)
        return float(loss.detach()), gu

    real_loss, gu = one(real_edges)
    rand_loss, gr = one(random_graph)
    nh = sum(p.numel() for p in head_params)
    h = grad_pair(gs[:nh], gu[:nh])
    e = grad_pair(gs[nh:], gu[nh:])
    lambda_initial = .2 * h["supervised_norm"] / (h["unlabeled_norm"] + 1e-12)
    return {"batch_labeled": len(fit_batch), "batch_graph_edges": count,
            "real_loss": real_loss, "random_loss": rand_loss,
            "head": h, "encoder": e,
            "lambda_initial": lambda_initial,
            "lambda_scaled_encoder_ratio": lambda_initial * e["unlabeled_to_supervised"],
            "real_random_head_gradient_cosine": float(F.cosine_similarity(gu[:nh], gr[:nh], dim=0)),
            "real_random_encoder_gradient_cosine": float(F.cosine_similarity(gu[nh:], gr[nh:], dim=0)),
            "real_random_total_gradient_relative_distance": float((gu - gr).norm() / gu.norm())}


def main():
    a = args()
    source = a.source_root.resolve()
    out = a.out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing existing output: {out}")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for requested audit")
    device = torch.device(a.device)
    data_root = source.parent / "data" / "emobank_processed"
    train_path = data_root / "lb500_s0" / "train_lb.parquet"
    ulb_path = data_root / "lb500_s0" / "train_ulb.parquet"
    pretrain = source.parent / "pretrained" / "roberta-base"
    build = source / "saved_models" / "vad_research" / "m1_v3_gpu_seed0_20261028T063714Z_7002" / "build"
    p1 = source / "saved_models" / "vad_research" / "p1_roberta_lb500_s0" / "model_best.pth"
    for p in (train_path, ulb_path, build / "manifest.json", build / "audit_inputs.npz", p1):
        if not p.is_file() or "test" in p.name.lower():
            raise RuntimeError(f"Required TRAIN-only asset unavailable: {p}")
    manifest = json.loads((build / "manifest.json").read_text())
    if sha256_file(train_path) != manifest["train_sha256"]:
        raise RuntimeError("TRAIN label hash mismatch")
    frame = read_gold(train_path, expected_n=500).set_index("id", drop=False)
    ulb = read_unlabeled(ulb_path, labeled_ids=frame.id).reset_index(drop=True)
    if len(ulb) != 7562 or set(ulb.id) & set(frame.id):
        raise RuntimeError("Unlabeled count/overlap mismatch")
    fit_ids, audit_ids = manifest["reference_train_ids"], manifest["audit_ids"]
    if len(fit_ids) != 400 or len(audit_ids) != 100 or set(fit_ids) & set(audit_ids):
        raise RuntimeError("400/100 manifest invalid")
    fit, hold = frame.loc[fit_ids].reset_index(drop=True), frame.loc[audit_ids].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrain, local_files_only=True, use_fast=True)
    result = {"source_commit": manifest["git_commit"], "source_train_sha256": manifest["train_sha256"],
              "p1_checkpoint_sha256": sha256_file(p1), "data_counts": {"fit": 400, "held": 100, "unlabeled": 7562},
              "model": "RoBERTa mean pool -> Dropout(.1) -> Linear(768,768) -> GELU -> Linear(768,3)",
              "optimization_steps": 0, "dev_or_test_access": False}
    # Shared label-blind coordinates isolate Q-estimation stability from drift
    # between independently fine-tuned fold encoders.
    cached = np.load(build / "audit_inputs.npz", allow_pickle=False)
    if list(cached["fit_ids"]) != fit_ids or list(cached["audit_ids"]) != audit_ids:
        raise RuntimeError("Cached label-blind feature IDs mismatch manifest")
    x_shared = cached["fit_features"]
    y_shared = cached["fit_labels"]
    id_pos = {item: i for i, item in enumerate(fit_ids)}
    shared_bases = []
    shared_folds = []
    for f in manifest["folds"]:
        train_pos = np.array([id_pos[item] for item in f["train_ids"]])
        held_pos = np.array([id_pos[item] for item in f["held_out_ids"]])
        report, basis = subspace_report(x_shared[train_pos], y_shared[train_pos],
                                        x_shared[held_pos], y_shared[held_pos], 500 + f["fold"])
        shared_folds.append(report)
        shared_bases.append(basis)
    shared_overlap = [float(np.square(shared_bases[i].T @ shared_bases[j]).sum() / 3)
                      for i in range(5) for j in range(i + 1, 5)]
    result["label_blind_fixed_coordinates"] = {
        "folds": shared_folds, "fold_projector_overlap": quant(shared_overlap),
        "source": "existing audit_inputs.npz; pretrained RoBERTa, no EmoBank labels used to fit encoder"}
    print("SUBSPACE: existing five fold checkpoints", flush=True)
    folds = []
    bases = []
    for f in manifest["folds"]:
        checkpoint_path = build / f["checkpoint"]
        if sha256_file(checkpoint_path) != f["checkpoint_sha256"]:
            raise RuntimeError("Fold checkpoint SHA256 mismatch")
        model = load_existing(checkpoint_path, pretrain, device, f["train_ids"], f["held_out_ids"])
        tr = frame.loc[f["train_ids"]].reset_index(drop=True)
        va = frame.loc[f["held_out_ids"]].reset_index(drop=True)
        z_tr = encode(model, tr, tokenizer, device)
        z_va = encode(model, va, tokenizer, device)
        rep, q = subspace_report(z_tr, tr.loc[:, NORM_COLUMNS].to_numpy(np.float64),
                                 z_va, va.loc[:, NORM_COLUMNS].to_numpy(np.float64), 100 + f["fold"])
        rep["fold"] = f["fold"]
        folds.append(rep)
        bases.append(q)
        del model
        torch.cuda.empty_cache()
        print(f"fold {f['fold']} complete", flush=True)
    overlaps = []
    for i in range(5):
        for j in range(i + 1, 5):
            aij = bases[i].T @ bases[j]
            overlaps.append(float(np.square(aij).sum() / min(bases[i].shape[1], bases[j].shape[1])))
    result["subspace"] = {"folds": folds, "fold_projector_overlap": quant(overlaps)}
    print("SUBSPACE: existing reference_400 and untouched 100", flush=True)
    ref_path = build / "reference_400.pth"
    if sha256_file(ref_path) != manifest["reference_checkpoint_sha256"]:
        raise RuntimeError("Reference checkpoint hash mismatch")
    ref = load_existing(ref_path, pretrain, device, fit_ids)
    z_fit = encode(ref, fit, tokenizer, device)
    z_hold = encode(ref, hold, tokenizer, device)
    ref_report, q_ref = subspace_report(z_fit, fit.loc[:, NORM_COLUMNS].to_numpy(np.float64),
                                         z_hold, hold.loc[:, NORM_COLUMNS].to_numpy(np.float64), 200)
    q_repeat, _, _ = label_subspace(torch.tensor(z_fit, dtype=torch.float64),
                                     torch.tensor(fit.loc[:, NORM_COLUMNS].to_numpy(np.float64)))
    ref_report["repeat_projector_max_abs_difference"] = float(np.max(np.abs(q_ref @ q_ref.T - q_repeat.numpy() @ q_repeat.numpy().T)))
    result["subspace"]["reference_400_to_100"] = ref_report
    print("GRAPH: P1 mutual 8NN on TRAIN unlabeled only", flush=True)
    p1_model = RobertaVADRegressor(str(pretrain), dropout=.1, offline=True).to(device)
    ckpt = torch.load(p1, map_location="cpu", weights_only=False)
    p1_model.load_state_dict(ckpt["model"], strict=True)
    del ckpt
    z_p1 = encode(p1_model, ulb, tokenizer, device, batch=24)
    del p1_model
    torch.cuda.empty_cache()
    graph, nearest = knn_graph(z_p1, device, 8)
    random_graph = random_edges(graph, len(ulb), 20261028)
    degree = np.bincount(graph.reshape(-1), minlength=len(ulb))
    p1_norm = z_p1 / np.linalg.norm(z_p1, axis=1, keepdims=True).clip(1e-8)
    cos_true = np.sum(p1_norm[graph[:, 0]] * p1_norm[graph[:, 1]], 1)
    cos_random = np.sum(p1_norm[random_graph[:, 0]] * p1_norm[random_graph[:, 1]], 1)
    z_ulb_ref = encode(ref, ulb, tokenizer, device, batch=24)
    true_stats, c_true = direction_stats(z_ulb_ref, graph, q_ref)
    rand_stats, c_rand = direction_stats(z_ulb_ref, random_graph, q_ref)
    response_true = head_response(ref.head, z_ulb_ref, graph, q_ref, device)
    response_random = head_response(ref.head, z_ulb_ref, random_graph, q_ref, device)
    covariance_difference = float(np.linalg.norm(c_true - c_rand, "fro") / np.linalg.norm(c_true, "fro"))
    result["graph"] = {"edges": len(graph), "isolated_count": int(np.sum(degree == 0)),
                       "isolated_fraction": float(np.mean(degree == 0)), "degree": quant(degree),
                       "p1_cosine_distance_true": quant(1 - cos_true),
                       "p1_cosine_distance_random": quant(1 - cos_random),
                       "true_direction": true_stats, "random_direction": rand_stats,
                       "true_head_response": response_true, "random_head_response": response_random,
                       "direction_cov_relative_frobenius_difference": covariance_difference,
                       "q_source": "reference_400, 400 TRAIN labels only; P1 graph uses no label for edge selection"}
    print("GRADIENT: fixed TRAIN batches; no optimizer", flush=True)
    result["gradient"] = gradient_audit(ref, fit, ulb, graph, random_graph, q_ref,
                                          tokenizer, device)
    result["source_hashes"] = {"ulb_sha256": sha256_file(ulb_path),
                                "reference_checkpoint_sha256": sha256_file(ref_path)}
    out.mkdir(parents=True)
    (out / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"DONE {out / 'audit.json'}", flush=True)


if __name__ == "__main__":
    main()
