import torch

from scripts.vad_research.train_acvr import center_huber, diff_huber


def pairwise_mse(s, p):
    iu = torch.triu_indices(s.shape[0], s.shape[0], offset=1)
    ds = s[:, None, :] - s[None, :, :]
    dp = p[:, None, :] - p[None, :, :]
    return ((ds[iu[0], iu[1]] - dp[iu[0], iu[1]]) ** 2).mean()


def centered_mse(s, p):
    return (((s - s.mean(0, keepdim=True)) - (p - p.mean(0, keepdim=True))) ** 2).mean()


def test_pairwise_mse_centered_identity():
    torch.manual_seed(0)
    s = torch.randn(16, 3, requires_grad=True)
    p = torch.randn(16, 3)
    B = s.shape[0]
    ratio = 2 * B / (B - 1)
    assert torch.allclose(pairwise_mse(s, p), ratio * centered_mse(s, p), atol=1e-6)


def test_huber_gradients_differ():
    torch.manual_seed(0)
    s1 = torch.randn(16, 3, requires_grad=True)
    s2 = s1.detach().clone().requires_grad_(True)
    p = torch.randn(16, 3)
    l1 = diff_huber(s1, p, torch.ones(3))
    l2 = center_huber(s2, p, torch.ones(3))
    l1.backward(); l2.backward()
    assert torch.isfinite(s1.grad).all()
    assert torch.isfinite(s2.grad).all()
    assert not torch.allclose(s1.grad, s2.grad, atol=1e-6)
