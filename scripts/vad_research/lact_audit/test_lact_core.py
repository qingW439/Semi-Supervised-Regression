import torch
from torch import nn

from lact_core import finite_difference_jvp, label_subspace, lact_loss
from audit import knn_graph


def test_true_nonlinear_jvp_matches_finite_difference():
    torch.manual_seed(4)
    head = nn.Sequential(nn.Linear(7, 7), nn.GELU(), nn.Linear(7, 3)).double()
    z = torch.randn(5, 7, dtype=torch.double)
    r = torch.randn_like(z)
    actual = torch.func.jvp(head, (z,), (r,))[1]
    numerical = finite_difference_jvp(head, z, r, 1e-5)
    torch.testing.assert_close(actual, numerical, rtol=1e-6, atol=1e-8)


def test_gradients_and_detaches():
    torch.manual_seed(5)
    head = nn.Sequential(nn.Linear(7, 7), nn.GELU(), nn.Linear(7, 3)).double()
    z_u = torch.randn(5, 7, dtype=torch.double, requires_grad=True)
    z_v = torch.randn(5, 7, dtype=torch.double, requires_grad=True)
    q = torch.linalg.qr(torch.randn(7, 3, dtype=torch.double)).Q.requires_grad_()
    loss = lact_loss(head, z_u, z_v, q)
    grads = torch.autograd.grad(loss, (z_u, z_v, q, *head.parameters()), allow_unused=True)
    # The output bias cancels from every directional derivative by construction.
    assert all(g is not None and torch.isfinite(g).all() and g.norm() > 0
               for g in (*grads[:2], *grads[3:-1]))
    assert grads[-1] is None or grads[-1].norm() == 0
    assert grads[2] is None


def test_denominator_is_stopped_and_q_rank_is_numerical():
    torch.manual_seed(6)
    z = torch.randn(20, 7, dtype=torch.double, requires_grad=True)
    y = torch.randn(20, 3, dtype=torch.double, requires_grad=True)
    q, s, rank = label_subspace(z, y)
    assert rank == 3 and s.shape == (3,) and not q.requires_grad
    head = nn.Sequential(nn.Linear(7, 7), nn.GELU(), nn.Linear(7, 3)).double()
    u = z[:5]
    v = z[5:10]
    delta = v - u
    r = delta - (delta @ q) @ q.T
    j = torch.func.jvp(head, (u,), (r,))[1]
    expected = (j.square().sum(-1) / (delta.detach().square().sum(-1) + 1e-8)).mean()
    torch.testing.assert_close(lact_loss(head, u, v, q), expected)
    # Check the frozen denominator against a deliberately live-denominator alternative.
    live = (j.square().sum(-1) / (delta.square().sum(-1) + 1e-8)).mean()
    g_frozen = torch.autograd.grad(expected, (u, v), retain_graph=True)
    g_live = torch.autograd.grad(live, (u, v))
    assert (g_frozen[0] - g_live[0]).norm() > 1e-8


def test_mutual_graph_indices_are_static_integers():
    torch.manual_seed(7)
    z = torch.randn(30, 7).numpy()
    edges, nearest = knn_graph(z, torch.device("cpu"), k=3)
    assert edges.dtype.kind in "iu" and nearest.dtype.kind in "iu"
    assert all(i in nearest[j] and j in nearest[i] for i, j in edges)


def test_regularizer_does_not_consume_dropout_rng():
    torch.manual_seed(8)
    head = nn.Sequential(nn.Linear(7, 7), nn.GELU(), nn.Linear(7, 3))
    dropout = nn.Dropout(.1).train()
    u, v = torch.randn(5, 7), torch.randn(5, 7)
    q = torch.linalg.qr(torch.randn(7, 3)).Q
    before = torch.random.get_rng_state().clone()
    _ = lact_loss(head, u, v, q)
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    reference = head(dropout(u))
    torch.random.set_rng_state(before)
    after_regularizer = head(dropout(u))
    torch.testing.assert_close(reference, after_regularizer)
