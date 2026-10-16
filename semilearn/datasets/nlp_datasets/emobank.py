"""EmoBank VAD Parquet adapter shared by supervised and RankUp experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DIMENSIONS = ("V", "A", "D")
NORM_COLUMNS = tuple(f"{dim}_norm" for dim in DIMENSIONS)


def read_gold(path: Path, *, expected_n: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"id", "text", *DIMENSIONS, *NORM_COLUMNS}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing gold columns in {path}: {sorted(required - set(frame.columns))}")
    if expected_n is not None and len(frame) != expected_n:
        raise ValueError(f"Expected {expected_n} rows in {path}, got {len(frame)}")
    _check_identity(frame, path)
    numeric = frame.loc[:, list(DIMENSIONS) + list(NORM_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Non-finite VAD labels: {path}")
    original = frame.loc[:, list(DIMENSIONS)].to_numpy(dtype=np.float64)
    normalized = frame.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float64)
    if not ((original >= 1).all() and (original <= 5).all()):
        raise ValueError(f"VAD labels outside [1, 5]: {path}")
    if not np.allclose(normalized, (original - 1) / 4, atol=2e-7, rtol=0):
        raise ValueError(f"Inconsistent normalized VAD labels: {path}")
    return frame.reset_index(drop=True)


def read_unlabeled(path: Path, *, labeled_ids=()) -> pd.DataFrame:
    """Read only ID and text; reject gold columns or overlap with labeled data."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if set(frame.columns) != {"id", "text"}:
        raise ValueError(f"Unlabeled data must contain only id and text: {path}")
    _check_identity(frame, path)
    if set(frame["id"]).intersection(labeled_ids):
        raise ValueError(f"Labeled and unlabeled IDs overlap: {path}")
    return frame.reset_index(drop=True)


def _check_identity(frame, path):
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"Missing or duplicate IDs: {path}")
    if frame["text"].isna().any() or not frame["text"].map(lambda x: isinstance(x, str)).all():
        raise ValueError(f"Text must consist of non-null strings: {path}")


class GoldData(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.text = frame["text"].tolist()
        self.y = frame.loc[:, list(NORM_COLUMNS)].to_numpy(dtype=np.float32)
        self.ids = frame["id"].tolist()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.y[idx], self.ids[idx]


class UnlabeledData(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.text = frame["text"].tolist()
        self.ids = frame["id"].tolist()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.ids[idx]


class TextCollator:
    """The exact tokenizer call and output contract used in E1."""
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        texts, values, ids = zip(*rows)
        tokens = self.tokenizer(list(texts), truncation=True,
                                max_length=self.max_length, padding=True,
                                return_tensors="pt")
        return tokens, torch.tensor(np.stack(values), dtype=torch.float32), list(ids)
