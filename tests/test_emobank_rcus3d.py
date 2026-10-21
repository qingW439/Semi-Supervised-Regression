"""Fast CPU checks for dimensionwise Close Mixup before GPU training."""

import torch
from torch import nn
from torch.nn import functional as F

from semilearn.nets.bert.bert_vad_rcus import (
    BertVADRCUS3D, local_logits, masked_local_loss,
    supervised_close_mixup_loss, unique_pairs, unsupervised_close_mixup_loss,
)


class ToyBackbone(nn.Module):
    num_features = 2

    def encode(self, **tokens):
        return tokens["x"]

    def regress(self, x):
        return torch.cat((x, x[:, :1]), dim=1)


def model():
    m = BertVADRCUS3D(ToyBackbone())
    with torch.no_grad():
        m.rank_head.weight.zero_()
        m.rank_head.bias.zero_()
        for d in range(3):
            m.rank_head.weight[2 * d, 0] = 1.0
    return m


def test_shapes_and_local_directions():
    m = model()
    x = torch.tensor([[2., 0.], [0., 0.], [1., 0.]])
    out = m(x=x)
    assert out["regression"].shape == (3, 3)
    assert out["rank_logits"].shape == (3, 3, 2)
    i, j = unique_pairs(3, x.device)
    assert list(zip(i.tolist(), j.tolist())) == [(0, 1), (0, 2), (1, 2)]
    first, second = local_logits(m, x, i[:1], j[:1], 0.8)
    assert first.shape == second.shape == (1, 3)
    assert torch.allclose(first, torch.full((1, 3), 0.4))
    assert torch.allclose(second, torch.full((1, 3), 0.4))
    reverse_first, reverse_second = local_logits(m, x, j[:1], i[:1], 0.8)
    assert torch.allclose(reverse_first, -first)
    assert torch.allclose(reverse_second, -second)


def test_supervised_vad_labels_and_ties():
    m = model()
    x = torch.tensor([[2., 0.], [0., 0.]], requires_grad=True)
    gold = torch.tensor([[3., 1., 2.], [1., 3., 2.]])
    loss, ties = supervised_close_mixup_loss(m, x, gold, 0.8)
    f, s = local_logits(m, x, *unique_pairs(2, x.device), 0.8)
    expected = (F.binary_cross_entropy_with_logits(f, torch.tensor([[1., 0., .5]]))
                + F.binary_cross_entropy_with_logits(s, torch.tensor([[1., 0., .5]])))
    assert torch.allclose(loss, expected)
    assert ties.tolist() == [0, 0, 1]
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_masked_all_candidates_and_zero_gradient():
    first = torch.zeros((2, 3), requires_grad=True)
    second = torch.zeros((2, 3), requires_grad=True)
    target = torch.ones((2, 3))
    mask = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.bool)
    loss = masked_local_loss(first, second, target, mask)
    assert torch.allclose(loss, 2 * torch.log(torch.tensor(2.)) / 6)
    loss.backward()
    assert torch.allclose(first.grad[0, 0], torch.tensor(-0.5 / 6))
    assert first.grad[0, 1:].abs().sum() == 0
    first.grad.zero_()
    second.grad.zero_()
    zero = masked_local_loss(first, second, target, torch.zeros_like(mask))
    assert zero.item() == 0
    zero.backward()
    assert first.grad.abs().sum() == second.grad.abs().sum() == 0


def test_pseudo_masks_are_dimensionwise_and_no_teacher_gradient():
    m = model()
    x = torch.tensor([[2., 0.], [0., 0.]], requires_grad=True)
    # V accepted, A and D rejected.
    teacher = torch.zeros((2, 3, 2), requires_grad=True)
    with torch.no_grad():
        teacher[0, 0, 0] = 8
    loss, kept, possible = unsupervised_close_mixup_loss(m, teacher, x, .8, .95)
    assert kept.tolist() == [1, 0, 0]
    assert possible.tolist() == [1, 1, 1]
    loss.backward()
    assert teacher.grad is None
    assert x.grad is not None and torch.isfinite(x.grad).all()
    teacher2 = torch.zeros_like(teacher, requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    zero, kept2, _ = unsupervised_close_mixup_loss(m, teacher2, x2, .8, .95)
    assert kept2.tolist() == [0, 0, 0]
    zero.backward()
    assert x2.grad.abs().sum() == 0
    assert teacher2.grad is None
