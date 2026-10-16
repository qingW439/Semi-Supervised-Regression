#!/usr/bin/env python3
"""Compare the archived E1 implementation and shared VAD backbone."""

import json
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import BertModel


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.nets.bert.bert_vad import BertVADRegressor as SharedModel
from semilearn.datasets.nlp_datasets.emobank import read_gold


class ArchivedE1Model(nn.Module):
    """Frozen copy of the original E1 forward path for parity checking."""
    def __init__(self, pretrained_path, dropout, offline=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_path, local_files_only=offline)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, **tokens):
        hidden = self.bert(**tokens, return_dict=True).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.head(self.dropout(pooled))


def main():
    config = json.loads((REPO / "repro_records/emobank/e1/lb250_s0/run_config.json").read_text())
    path = REPO / "saved_models/emobank/supervised_lb250_s0/model_best.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    old = ArchivedE1Model(config["pretrain_path"], config["dropout"], offline=True)
    new = SharedModel(config["pretrain_path"], config["dropout"], offline=True)
    old.load_state_dict(checkpoint["model"], strict=True)
    new.load_state_dict(checkpoint["model"], strict=True)
    old.eval()
    new.eval()
    tokenizer = AutoTokenizer.from_pretrained(config["pretrain_path"], local_files_only=True)
    frame = read_gold(Path(config["data_root"]) / "dev_gold.parquet")
    tokens = tokenizer(frame["text"].iloc[:8].tolist(), truncation=True,
                       max_length=config["max_length"], padding=True, return_tensors="pt")
    with torch.no_grad():
        expected = old(**tokens)
        actual = new(**tokens)
    error = (expected - actual).abs().max().item()
    print(f"strict_checkpoint_load=OK max_abs_output_difference={error:.9g}")
    if error > 1e-7:
        raise AssertionError("Shared backbone differs from E1")


if __name__ == "__main__":
    main()
