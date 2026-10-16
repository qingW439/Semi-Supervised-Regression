"""Re-evaluate the nine Yelp Review 250-label best checkpoints on Test.

Run from the repository root. Evaluation batch size comes from each training
YAML because dynamic padding changes this BERT model's pooled predictions.
"""

import argparse
import csv
import gc
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semilearn.core.utils import (  # noqa: E402
    TorchMinMaxScaler,
    get_data_loader,
    get_dataset,
    get_net_builder,
    over_write_args_from_file,
)


# Values reported by test/best_mae at the end of each archived training run.
# Seed 0 values are in train.log; other values are in final_metrics.txt.
RECORDED_TEST_MAE = {
    ("supervised", 0): 0.698811265317339,
    ("supervised", 1): 0.7617522216305509,
    ("supervised", 2): 0.7252070701186965,
    ("rankup", 0): 0.6622534466865845,
    ("rankup", 1): 0.6794106643372169,
    ("rankup", 2): 0.6848716383365355,
    ("rankuprda", 0): 0.6178077246137987,
    ("rankuprda", 1): 0.6395941696031624,
    ("rankuprda", 2): 0.6438109423285303,
}

ALGORITHMS = ("supervised", "rankup", "rankuprda")
MAE_TOLERANCE = 0.003


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Evaluate one run, e.g. rankup:2")
    parser.add_argument("--checkpoint", choices=("best", "latest"), default="best",
                        help="Use latest only to diagnose archived training metrics")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "repro_records/yelp250")
    return parser.parse_args()


def training_args(config_path):
    # Match eval.py's defaults for fields not explicitly present in the YAML.
    args = argparse.Namespace(
        net="wrn_28_2", net_from_name=False, pretrain_path="",
        eval_batch_size=128, data_dir="./data", dataset="utkface",
        img_size=40, crop_ratio=0.875, max_length=512,
        max_length_seconds=4.0, sample_rate=16000, seed=0, gpu=None,
    )
    over_write_args_from_file(args, str(config_path))
    if args.dataset != "yelp_review" or args.net != "bert_base":
        raise ValueError(f"Unexpected dataset or network in {config_path}")
    if args.eval_batch_size != 8:
        raise ValueError(f"Expected training eval_batch_size=8 in {config_path}")
    return args


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("arc_classifier"):
            continue
        if key.startswith("module.") or key.startswith("backbone."):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def evaluate_run(algorithm, seed, num_workers, checkpoint_kind):
    config = ROOT / f"config/reproduction/{algorithm}_yelp_review_lb250_s{seed}.yaml"
    args = training_args(config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    checkpoint_name = "model_best.pth" if checkpoint_kind == "best" else "latest_model.pth"
    checkpoint_path = ROOT / args.save_dir / args.save_name / checkpoint_name
    print(f"Evaluating {algorithm} seed {seed}: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "ema_model" not in checkpoint or "input_range" not in checkpoint or "output_range" not in checkpoint:
        raise KeyError(f"Missing EMA weights or label scaling range in {checkpoint_path}")

    net_builder = get_net_builder(args.net, args.net_from_name)
    net = net_builder(pretrained=False, pretrained_path=args.pretrain_path)
    net.load_state_dict(clean_state_dict(checkpoint["ema_model"]), strict=True)
    scaler = TorchMinMaxScaler(output_range=checkpoint["output_range"])
    scaler.fit(torch.tensor(checkpoint["input_range"]))
    del checkpoint
    gc.collect()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device).eval()
    datasets = get_dataset(args, "fullysupervised", args.dataset, 0, args.data_dir, False)
    test_set = datasets["test"]
    if test_set is None:
        raise RuntimeError("Yelp Review Test split is unavailable")
    loader = get_data_loader(
        args, test_set, args.eval_batch_size, data_sampler=None,
        num_epochs=0, num_iters=0, num_workers=num_workers, drop_last=False,
    )

    truth, predictions = [], []
    with torch.inference_mode():
        for batch in loader:
            x = batch["x_lb"]
            x = {key: value.to(device) for key, value in x.items()}
            logits = net(x)["logits"]
            predicted = scaler.inverse_transform(logits.cpu().clone())
            truth.extend(batch["y_lb"].cpu().numpy().reshape(-1).tolist())
            predictions.extend(predicted.numpy().reshape(-1).tolist())

    truth = np.asarray(truth, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if len(truth) != len(test_set) or not np.isfinite(predictions).all():
        raise RuntimeError(f"Invalid Test predictions for {algorithm} seed {seed}")

    mae = float(mean_absolute_error(truth, predictions))
    recorded = RECORDED_TEST_MAE[(algorithm, seed)]
    row = {
        "algorithm": algorithm, "seed": seed, "test_count": len(truth),
        "test_mae": mae, "test_r2": float(r2_score(truth, predictions)),
        "test_srcc": float(spearmanr(truth, predictions).statistic),
        "test_lcc": float(pearsonr(truth, predictions).statistic),
        "recorded_test_mae": recorded, "mae_delta": mae - recorded,
        "verification": "PASS" if abs(mae - recorded) <= MAE_TOLERANCE else "CHECK",
    }
    print(
        f"{algorithm}:{seed} Test MAE={mae:.6f} R2={row['test_r2']:.6f} "
        f"SRCC={row['test_srcc']:.6f} recorded={recorded:.6f} "
        f"delta={row['mae_delta']:+.6f} {row['verification']}", flush=True,
    )
    del net, loader, datasets, test_set
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    runs = [(algorithm, seed) for algorithm in ALGORITHMS for seed in range(3)]
    if args.only:
        algorithm, separator, seed_text = args.only.partition(":")
        if not separator or not seed_text.isdigit() or (algorithm, int(seed_text)) not in runs:
            raise ValueError("--only must be algorithm:seed, e.g. rankup:2")
        runs = [(algorithm, int(seed_text))]

    if args.only:
        rows = [evaluate_run(algorithm, seed, args.num_workers, args.checkpoint) for algorithm, seed in runs]
    else:
        if args.checkpoint != "best":
            raise ValueError("Full published summary must use the best checkpoints")
        rows = []
        for algorithm, seed in runs:
            best = evaluate_run(algorithm, seed, args.num_workers, "best")
            latest = evaluate_run(algorithm, seed, args.num_workers, "latest")
            # In the training code, after_run loads model_best.pth but evaluate()
            # applies the EMA shadow left at the final iteration. Therefore the
            # logged test/best_mae measures latest_model.pth, despite its name.
            best["latest_test_mae"] = latest["test_mae"]
            best["latest_mae_delta"] = latest["mae_delta"]
            best["best_vs_logged_mae_delta"] = best.pop("mae_delta")
            best["verification"] = "PASS" if latest["verification"] == "PASS" else "CHECK"
            rows.append(best)
    if args.only:
        return 0 if rows[0]["verification"] == "PASS" else 1

    write_csv(args.output_dir / "test_metrics_by_seed.csv", rows)
    summary = []
    for algorithm in ALGORITHMS:
        group = [row for row in rows if row["algorithm"] == algorithm]
        item = {"algorithm": algorithm, "seeds": len(group)}
        for metric in ("test_mae", "test_r2", "test_srcc", "test_lcc"):
            values = [row[metric] for row in group]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=1))
        item["verification"] = "PASS" if all(row["verification"] == "PASS" for row in group) else "CHECK"
        item["logged_mae_source"] = "latest_model.pth (training EMA shadow)"
        summary.append(item)
    write_csv(args.output_dir / "test_metrics_summary.csv", summary)
    print(f"Wrote {args.output_dir / 'test_metrics_by_seed.csv'}", flush=True)
    print(f"Wrote {args.output_dir / 'test_metrics_summary.csv'}", flush=True)
    return 0 if all(row["verification"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
