"""Shared, Test-free EmoBank lb500/seed0 protocol for RCUS adaptation."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd

from semilearn.datasets.nlp_datasets.emobank import read_gold, read_unlabeled

REPO = Path(__file__).resolve().parents[2]
DATA = REPO.parent / "data" / "emobank_processed"
E1 = REPO.parent / "Semi-Supervised-Regression" / "saved_models/emobank/supervised_lb500_s0/model_best.pth"
B0 = REPO.parent / "Semi-Supervised-Regression-rank3d" / "saved_models/emobank/rank3d_supervised_control_lb500_s0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_splits_no_test():
    paths = {"labeled": DATA / "lb500_s0/train_lb.parquet",
             "unlabeled": DATA / "lb500_s0/train_ulb.parquet",
             "dev": DATA / "dev_gold.parquet"}
    lb = read_gold(paths["labeled"], expected_n=500)
    ulb = read_unlabeled(paths["unlabeled"], labeled_ids=lb.id)
    dev = read_gold(paths["dev"], expected_n=1000)
    if len(ulb) != 7562:
        raise ValueError("Unexpected unlabeled count")
    ids = [set(f.id) for f in (lb, ulb, dev)]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Train/Dev ID overlap")
    record = REPO / "repro_records/emobank/labeled_ids_lb500_s0.csv"
    with record.open(encoding="utf-8", newline="") as stream:
        recorded = {row["id"] for row in csv.DictReader(stream)}
    if recorded != ids[0]:
        raise ValueError("Labeled IDs differ from committed split record")
    return lb, ulb, dev, {str(k): sha256(v) for k, v in paths.items()}
