#!/usr/bin/env python3
"""Post-hoc diagnostics for EmoBank RankUp-VAD ARC on held-out DEV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from semilearn.algorithms.rankup_vad.rankup_vad import RankUpVADNet
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, read_gold
from semilearn.nets.bert.bert_vad import BertVADRegressor


def pair_diagnostics(scores: np.ndarray, gold: np.ndarray, *, sample_pairs: int,
                     seed: int, temperature: float) -> dict:
    if scores.shape != gold.shape or scores.ndim != 2 or scores.shape[1] != 3:
        raise ValueError(f"expected matching [N,3] arrays, got {scores.shape}, {gold.shape}")
    rng = np.random.default_rng(seed)
    n = len(scores)
    count = min(sample_pairs, n * (n - 1) // 2)
    if count <= 0:
        raise ValueError("Need at least 2 instances")
    i = rng.integers(0, n, size=count, dtype=np.int64)
    j = rng.integers(0, n - 1, size=count, dtype=np.int64)
    j += j >= i
    score_diff = scores[i, :] - scores[j, :]
    gold_diff = gold[i, :] - gold[j, :]
    result = {"sampling": {"num_draws": int(count), "seed": int(seed),
                            "sampling_with_replacement": True,
                            "test_accessed": False}}
    for k, dim in enumerate(("V", "A", "D")):
        valid = gold_diff[:, k] != 0
        score, target = score_diff[valid, k], gold_diff[valid, k]
        signed_match = score * target > 0
        ties_pred = score == 0
        conf_raw = 1.0 / (1.0 + np.exp(-np.abs(score)))
        conf_temp = 1.0 / (1.0 + np.exp(-np.abs(score) / temperature))

        def summarize(confidence):
            summary = {}
            for threshold in (0.60, 0.70, 0.80, 0.90, 0.95):
                selected = confidence >= threshold
                summary[f"{threshold:.2f}"] = {
                    "coverage": float(selected.mean()) if len(selected) else 0.0,
                    "accepted": int(selected.sum()),
                    "accuracy": float(signed_match[selected].mean()) if selected.any() else None,
                }
            return summary

        result[dim] = {
            "valid_pair_draws": int(valid.sum()),
            "gold_tie_fraction": float(1.0 - valid.mean()),
            "pair_accuracy_all": float(signed_match.mean()) if valid.any() else None,
            "score_tie_fraction": float(ties_pred.mean()) if valid.any() else None,
            "confidence_quantiles_raw": {str(q): float(value) for q, value in zip(
                (0.5, 0.9, 0.95, 0.99), np.quantile(conf_raw, (0.5, 0.9, 0.95, 0.99)))},
            "raw_softmax_thresholds": summarize(conf_raw),
            f"temperature_{temperature:g}_thresholds": summarize(conf_temp),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("saved_models/emobank/rankupvad_smoke_lb250_s0/model_best.pth"))
    parser.add_argument("--data-root", type=Path,
                        default=REPO.parent / "data/emobank_processed")
    parser.add_argument("--pretrain-path", default="google/bert_uncased_L-4_H-512_A-8")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sample-pairs", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose-json", action="store_true")
    args = parser.parse_args()
    if args.sample_pairs <= 0 or args.batch_size <= 0 or args.temperature <= 0:
        parser.error("sample-pairs, batch-size and temperature must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    dev = read_gold(args.data_root / "dev_gold.parquet", expected_n=1000)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrain_path, use_fast=True, local_files_only=args.offline)
    loader = DataLoader(GoldData(dev), batch_size=args.batch_size, shuffle=False,
                        collate_fn=TextCollator(tokenizer, args.max_length))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RankUpVADNet(BertVADRegressor(args.pretrain_path, offline=args.offline)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(checkpoint["model"], strict=True)
    net.eval()
    all_scores, all_gold = [], []
    with torch.inference_mode():
        for tokens, targets, _ in loader:
            outputs = net(**{key: value.to(device) for key, value in tokens.items()})
            logits = outputs["arc_logits"]
            all_scores.append((logits[:, :, 0] - logits[:, :, 1]).cpu().numpy())
            all_gold.append(targets.numpy())
    report = pair_diagnostics(
        np.concatenate(all_scores), np.concatenate(all_gold),
        sample_pairs=args.sample_pairs, seed=args.seed, temperature=args.temperature)
    report["checkpoint"] = str(args.checkpoint)
    report["best_epoch"] = int(checkpoint.get("epoch", -1))
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(f"Checkpoint epoch={report['best_epoch']} | DEV sampled pairs={args.sample_pairs} | TEST untouched")
    print("Thresholds use unsharpened probabilities (temperature=1).")
    for dim in ("V", "A", "D"):
        diagnostics = report[dim]
        print(f"\n{dim}: accuracy_all={diagnostics['pair_accuracy_all']:.4f} "
              f"gold_ties={diagnostics['gold_tie_fraction']:.4f} "
              f"valid_draws={diagnostics['valid_pair_draws']}")
        print("  cutoff   coverage  accepted  precision")
        for cutoff, stats in diagnostics["raw_softmax_thresholds"].items():
            accuracy = stats["accuracy"]
            precision = "NA" if accuracy is None else f"{accuracy:.4f}"
            print(f"  {cutoff:>6s}   {stats['coverage']:>8.4f}  "
                  f"{stats['accepted']:>8d}  {precision:>9s}")
    print("\nThe full report can be saved with --output PATH; no model file is changed.")
    if args.verbose_json:
        print(serialized)
    if args.output is not None:
        if args.output.exists():
            parser.error(f"refusing overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
