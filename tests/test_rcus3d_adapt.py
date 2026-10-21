"""Fast CPU checks; callable directly when pytest is unavailable on server."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from semilearn.nets.bert.bert_vad_rcus_adapt import (
    BertVADRCUS3D, local_logits, pairwise_rank_loss, unsupervised_close_mixup_loss,
)
from scripts.emobank.train_emobank_rcus3d_adapt import train_step


class ToyBackbone(nn.Module):
    num_features = 2

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.drop = nn.Dropout(.2)
        self.head = nn.Linear(2, 3)

    def encode(self, **tokens):
        return self.drop(self.linear(tokens["x"]))

    def regress(self, features):
        return self.head(features)


def toy_model():
    torch.manual_seed(22)
    return BertVADRCUS3D(ToyBackbone())


def test_warmup_pair_direction_and_ties():
    logits = torch.zeros((2, 3, 2), requires_grad=True)
    gold = torch.tensor([[.9, .1, .4], [.1, .9, .4]])
    loss, ties = pairwise_rank_loss(logits, gold)
    assert ties.tolist() == [0, 0, 1]
    loss.backward()
    assert logits.grad[0, 0, 0] < 0
    assert logits.grad[0, 1, 0] > 0
    assert abs(float(logits.grad[0, 2, 0])) < 1e-8


def test_linear_head_close_mixup_is_scaled_endpoint_rank():
    model = toy_model()
    x = torch.randn((4, 2))
    i, j = torch.tensor([0, 1]), torch.tensor([2, 3])
    left, right = local_logits(model, x, i, j, .75)
    rank = model.rank(x)
    binary = rank[..., 0] - rank[..., 1]
    endpoint = binary[i] - binary[j]
    assert torch.allclose(left, .25 * endpoint, atol=1e-6)
    assert torch.allclose(right, .25 * endpoint, atol=1e-6)


def test_calibrated_dimensionwise_mask_and_zero_gradient():
    model = toy_model()
    x = torch.randn((2, 2), requires_grad=True)
    teacher = torch.zeros((2, 3, 2), requires_grad=True)
    with torch.no_grad():
        teacher[0, :, 0] = 5
    loss, accepted, possible = unsupervised_close_mixup_loss(
        model, teacher, x, .8, [.9, .99, .5],
        temperature=[1., 1., 1.], active=[True, False, True])
    assert accepted.tolist() == [1, 0, 1]
    assert possible.tolist() == [1, 1, 1]
    loss.backward()
    assert teacher.grad is None
    x2 = x.detach().clone().requires_grad_()
    zero, accepted, _ = unsupervised_close_mixup_loss(
        model, teacher, x2, .8, [1., 1., 1.],
        temperature=[1., 1., 1.], active=[False, False, False])
    assert accepted.tolist() == [0, 0, 0]
    zero.backward()
    assert x2.grad.abs().sum() == 0


def test_a0_a1_same_labeled_rng_when_no_pseudo_pairs():
    base = toy_model()
    a0, a1 = deepcopy(base), deepcopy(base)
    opt0 = torch.optim.AdamW(a0.parameters(), lr=1e-3)
    opt1 = torch.optim.AdamW(a1.parameters(), lr=1e-3)
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
    y = torch.tensor([[.7, .4, .5], [.2, .5, .6], [.4, .2, .8], [.5, .8, .2]])
    batch = ({"x": x}, y, ["a", "b", "c", "d"])
    unlabeled = ({"x": x}, ["u", "v", "w", "z"])
    args = dict(mixup_alpha=.5, supervised_mixup_weight=.1,
                cutoffs=[1., 1., 1.], temperatures=[1., 1., 1.],
                active=[True] * 3, clip_grad_norm=1.)
    torch.manual_seed(123)
    r0 = train_step(a0, opt0, batch, None,
                    SimpleNamespace(**args, mode="a0", unsupervised_mixup_weight=0.),
                    torch.device("cpu"), np.random.default_rng(55), np.random.default_rng(66))
    torch.manual_seed(123)
    r1 = train_step(a1, opt1, batch, unlabeled,
                    SimpleNamespace(**args, mode="a1", unsupervised_mixup_weight=1.),
                    torch.device("cpu"), np.random.default_rng(55), np.random.default_rng(66))
    assert r1["accepted"] == [0, 0, 0]
    assert r0["regression"] == r1["regression"]
    assert r0["supervised_mixup"] == r1["supervised_mixup"]
    assert all(torch.equal(a0.state_dict()[k], a1.state_dict()[k]) for k in a0.state_dict())
