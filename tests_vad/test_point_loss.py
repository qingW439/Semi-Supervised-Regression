"""Unit tests for the pluggable point loss in train_acvr.

Guarantees:
1. ``point_loss(..., 'l1')`` reproduces the original objective bit-for-bit
   (unweighted == F.l1_loss; weighted == sum(w*|d|)/sum(w)).
2. ``mse`` / ``huber`` are the expected elementwise losses.
3. gradients flow and are finite for all three kinds.
"""
import torch
import torch.nn.functional as F

from scripts.vad_research.train_acvr import point_loss


def _data(seed=0, n=7):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(n, 3, generator=g)
    target = torch.randn(n, 3, generator=g)
    w = torch.rand(n, 3, generator=g) + 0.1
    return pred, target, w


def test_l1_unweighted_matches_reference():
    pred, target, _ = _data()
    assert torch.allclose(point_loss(pred, target, 'l1'), F.l1_loss(pred, target))


def test_l1_weighted_matches_original_normalisation():
    pred, target, w = _data()
    ref = (w * (pred - target).abs()).sum() / w.sum().clamp_min(1e-6)
    got = point_loss(pred, target, 'l1', weights=w)
    assert torch.allclose(got, ref)


def test_mse_and_huber_values():
    pred, target, _ = _data(1)
    assert torch.allclose(point_loss(pred, target, 'mse'), (pred - target).square().mean())
    d = pred - target
    ref = F.huber_loss(d, torch.zeros_like(d), delta=0.1)
    assert torch.allclose(point_loss(pred, target, 'huber', delta=0.1), ref)


def test_huber_is_quadratic_for_small_residuals():
    pred = torch.full((4, 3), 0.01)
    target = torch.zeros(4, 3)
    # |d| << delta  ->  mean(d^2)/2
    assert torch.allclose(point_loss(pred, target, 'huber', delta=0.5),
                          torch.tensor(1e-4 / 2), atol=1e-8)


def test_unknown_kind_raises():
    pred, target, _ = _data()
    try:
        point_loss(pred, target, 'nope')
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_gradients_finite_for_all_kinds():
    for kind in ('l1', 'mse', 'huber'):
        pred, target, w = _data(2)
        pred = pred.clone().requires_grad_(True)
        out = point_loss(pred, target, kind, weights=w)
        out.backward()
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum() > 0
