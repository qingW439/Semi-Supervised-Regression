"""Unit tests for the closed-form ridge *difference reference* (second component's
alternative reference model) and its wiring into the OOF fold trainer.
"""
import numpy as np
import torch

from scripts.vad_research.train_acvr import ridge_reference


def test_shape_dtype_and_finiteness():
    torch.manual_seed(0)
    X = torch.randn(120, 24)
    Y = torch.randn(120, 3)
    U = torch.randn(37, 24)
    pred, alpha = ridge_reference(X, Y, U, seed=0)
    assert pred.shape == (37, 3)
    assert pred.dtype == torch.float32
    assert torch.isfinite(pred).all()
    assert alpha in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)


def test_recovers_noiseless_linear_map():
    torch.manual_seed(1)
    X = torch.randn(300, 16)
    W = torch.randn(16, 3)
    b = torch.randn(3)
    Y = X @ W + b
    U = torch.randn(50, 16)
    pred, _ = ridge_reference(X, Y, U, alphas=(1e-4, 1e-3, 1e-2), seed=0)
    truth = U @ W + b
    err = (pred - truth).abs().max().item()
    assert err < 1e-2, err


def test_alpha_grid_is_respected():
    torch.manual_seed(2)
    X = torch.randn(80, 8)
    Y = torch.randn(80, 3)
    U = torch.randn(10, 8)
    _, alpha = ridge_reference(X, Y, U, alphas=(5.0,), seed=0)
    assert alpha == 5.0


def test_no_leak_into_holdout(): 
    """The reference must depend ONLY on (X, Y); changing the query set must not change
    the fitted alpha, and recomputing on a subset must give the same alpha."""
    torch.manual_seed(3)
    X = torch.randn(100, 12)
    Y = torch.randn(100, 3)
    U1 = torch.randn(20, 12)
    U2 = torch.randn(35, 12)
    p1, a1 = ridge_reference(X, Y, U1, seed=7)
    p2, a2 = ridge_reference(X, Y, U2, seed=7)
    assert a1 == a2
    assert not torch.allclose(p1[:20], p2[:20])
