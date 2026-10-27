"""Smoothed, nonnegative VAD weights from a TRAIN-only utility matrix."""
from __future__ import annotations

import torch


class UtilityController:
    def __init__(self, mode, momentum=0.9):
        if mode not in ("diagonal", "crossdim"):
            raise ValueError("Unknown utility mode")
        if not 0 <= momentum < 1:
            raise ValueError("Momentum must lie in [0, 1)")
        self.mode = mode
        self.momentum = momentum
        self.smoothed = None
        self.updates = 0

    @torch.no_grad()
    def update(self, matrix):
        if matrix.shape != (3, 3) or not torch.isfinite(matrix).all():
            raise ValueError("Expected finite 3x3 utility matrix")
        matrix = matrix.detach().double()
        self.smoothed = matrix.clone() if self.smoothed is None else (
            self.momentum * self.smoothed + (1 - self.momentum) * matrix
        )
        self.updates += 1
        return self.weights()

    @torch.no_grad()
    def weights(self):
        if self.smoothed is None:
            return torch.ones(3, dtype=torch.float64)
        raw = self.smoothed.diag() if self.mode == "diagonal" else self.smoothed.sum(dim=0)
        positive = raw.clamp_min(0)
        if not torch.isfinite(positive).all():
            raise FloatingPointError("Nonfinite utility weights")
        if positive.sum() <= 0:
            return torch.ones_like(positive)
        weights = 3.0 * positive / positive.sum()
        if not torch.isfinite(weights).all() or weights.min() < 0 or weights.sum() <= 0:
            raise FloatingPointError("Invalid utility weights")
        return weights

    def state_dict(self):
        return {"mode": self.mode, "momentum": self.momentum,
                "smoothed": self.smoothed, "updates": self.updates}
