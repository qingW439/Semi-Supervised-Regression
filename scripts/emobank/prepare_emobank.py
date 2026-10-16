"""Prepare fixed EmoBank semi-supervised VAD splits without label leakage.

Keep official train/dev/test membership. Store original V/A/D and normalized
V_norm/A_norm/D_norm in gold data; unlabeled data has only ID and text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT.parent / "data/emobank_source/EmoBank/corpus/emobank.csv"
DEFAULT_OUTPUT = ROOT.parent / "data/emobank_processed"
DEFAULT_RECORDS = ROOT / "repro_records/emobank"
BUDGETS = (250, 500, 1000)
SEEDS = (0, 1, 2)
LABELS = ("V", "A", "D")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit(source: Path) -> str | None:
    repo = source.parent.parent
    if not (repo / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def load_source(path: Path) -> pd.DataFrame:
    # The official corpus contains a literal text value "None". Pandas would
    # otherwise interpret it as missing and silently discard that sentence.
    frame = pd.read_csv(path, dtype={"id": str, "split": str}, keep_default_na=False)
    expected = ["id", "split", *LABELS, "text"]
    if list(frame.columns) != expected:
        raise ValueError(f"Unexpected EmoBank columns: {list(frame.columns)!r}")
    frame = frame.loc[:, ["id", "split", "text", *LABELS]].copy()
    frame["split"] = frame["split"].str.strip().str.lower()
    if frame["id"].duplicated().any() or frame["id"].str.strip().eq("").any():
        raise ValueError("EmoBank IDs must be nonempty and unique")
    if set(frame["split"]) != {"train", "dev", "test"}:
        raise ValueError(f"Unexpected official splits: {set(frame['split'])!r}")
    if frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("EmoBank contains blank text")
    for label in LABELS:
        frame[label] = pd.to_numeric(frame[label], errors="raise")
        if not np.isfinite(frame[label]).all() or not frame[label].between(1, 5).all():
            raise ValueError(f"{label} must be finite and within [1, 5]")
        frame[f"{label}_norm"] = ((frame[label] - 1.0) / 4.0).astype("float32")
    return frame


def gold_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ["id", "split", "text", *LABELS, *(f"{x}_norm" for x in LABELS)]].reset_index(drop=True)


def write_ids(path: Path, ids: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["id"])
        writer.writerows((item,) for item in ids)


def prepare(source: Path, output: Path, records: Path) -> dict:
    frame = load_source(source)
    splits = {
        name: frame.loc[frame["split"].eq(name)].sort_values("id").reset_index(drop=True)
        for name in ("train", "dev", "test")
    }
    train, dev, test = (splits[name] for name in ("train", "dev", "test"))
    if len(train) < max(BUDGETS):
        raise ValueError(f"Training split has only {len(train)} rows")

    output.mkdir(parents=True, exist_ok=True)
    records.mkdir(parents=True, exist_ok=True)
    gold_columns(dev).to_parquet(output / "dev_gold.parquet", index=False)
    gold_columns(test).to_parquet(output / "test_gold.parquet", index=False)

    manifest = {
        "source_repository": "https://github.com/JULIELab/EmoBank",
        "source_commit": source_commit(source),
        "source_csv_sha256": sha256_file(source),
        "source_rows": len(frame),
        "official_split_counts": {name: int(frame["split"].eq(name).sum()) for name in ("train", "dev", "test")},
        "annotation_perspective": "official combined reader/writer emobank.csv",
        "original_labels": list(LABELS),
        "normalized_labels": [f"{name}_norm" for name in LABELS],
        "label_normalization": "(original_value - 1.0) / 4.0; fixed official [1,5] to [0,1]",
        "selection": "Sort official train by ID; NumPy default_rng(seed).permutation(row positions); first N labeled",
        "unlabeled_policy": "Exclude all labeled IDs for each budget; id and text only",
        "budgets": list(BUDGETS),
        "seeds": list(SEEDS),
        "splits": {},
    }
    for seed in SEEDS:
        permutation = np.random.default_rng(seed).permutation(len(train))
        for budget in BUDGETS:
            key = f"lb{budget}_s{seed}"
            selected = np.zeros(len(train), dtype=bool)
            selected[permutation[:budget]] = True
            lb = train.loc[selected].reset_index(drop=True)
            ulb = train.loc[~selected].reset_index(drop=True)
            assert len(lb) == budget and len(ulb) == len(train) - budget
            assert set(lb["id"]).isdisjoint(set(ulb["id"]))
            split_dir = output / key
            split_dir.mkdir(parents=True, exist_ok=True)
            gold_columns(lb).to_parquet(split_dir / "train_lb.parquet", index=False)
            ulb.loc[:, ["id", "text"]].to_parquet(split_dir / "train_ulb.parquet", index=False)
            ids_path = records / f"labeled_ids_{key}.csv"
            write_ids(ids_path, sorted(lb["id"].tolist()))
            manifest["splits"][key] = {
                "labeled_count": len(lb),
                "unlabeled_count": len(ulb),
                "labeled_ids_file": ids_path.name,
                "labeled_ids_sha256": sha256_file(ids_path),
            }

    if manifest["source_commit"] is not None:
        (records / "source_commit.txt").write_text(manifest["source_commit"] + "\n", encoding="utf-8")
    (records / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output, args.records)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
