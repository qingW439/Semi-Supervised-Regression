"""Shared BERT backbone for three-dimensional EmoBank VAD regression.

The module and parameter names intentionally match the original E1 model so
its archived ``model_best.pth`` checkpoints load with ``strict=True``.
"""

from torch import nn
from transformers import BertModel


class BertVADRegressor(nn.Module):
    """Mask-aware mean pooled BERT with a V/A/D regression head."""

    def __init__(self, pretrained_path: str, dropout: float = 0.1, offline: bool = False):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_path, local_files_only=offline)
        hidden = self.bert.config.hidden_size
        self.num_features = hidden
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def encode(self, **tokens):
        hidden = self.bert(**tokens, return_dict=True).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.dropout(pooled)

    def regress(self, features):
        return self.head(features)

    def forward(self, **tokens):
        return self.regress(self.encode(**tokens))
