"""CPU checks for M1 gradient surgery and TRAIN-only utility logic."""
import copy

import pytest
import torch
from torch import nn

from scripts.vad_research.m1.dimensional_utility import (
    disjoint_meta_batch, per_dimension_unlabeled, virtual_utility_matrix,
)
from scripts.vad_research.m1.ggr_vlr import merge_head_gradients, vlr_project
from scripts.vad_research.m1.utility_controller import UtilityController


def test_vlr_matches_official_halfspace_projection():
    sup = torch.tensor([2.0, 0.0])
    aux = torch.tensor([-3.0, 4.0])
    projected = vlr_project(sup, aux)
    assert torch.allclose(projected, torch.tensor([0.0, 4.0]))
    assert torch.dot(projected, sup) >= -1e-7
    assert torch.equal(vlr_project(sup, torch.tensor([1.0, 4.0])), torch.tensor([1.0, 4.0]))
    assert torch.equal(vlr_project(torch.zeros(2), aux), aux)


def test_gradient_merge_counts_supervision_once():
    p = nn.Parameter(torch.zeros(2))
    merged, stats = merge_head_gradients([p], [torch.tensor([2.0, 0.0])],
                                         [torch.tensor([-3.0, 4.0])])
    assert torch.allclose(merged[0], torch.tensor([2.0, 4.0]))
    assert stats['ggr_projection_applied'] == 1
    assert stats['post_projection_dot'] >= -1e-7


def test_virtual_matrix_is_3x3_and_does_not_mutate_head_or_optimizer():
    torch.manual_seed(9)
    head = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3))
    optimizer = torch.optim.Adam(head.parameters(), lr=5e-6)
    original = copy.deepcopy(head.state_dict())
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    x = torch.randn(6, 4)
    strong = head(x)
    weak = torch.zeros_like(strong)
    losses = per_dimension_unlabeled(strong, weak, torch.zeros_like(strong))
    matrix = virtual_utility_matrix(head, losses, torch.randn(5, 4),
                                    torch.randn(5, 3), 5e-6)
    assert matrix.shape == (3, 3)
    assert torch.isfinite(matrix).all()
    assert torch.count_nonzero(matrix) > 0
    assert all(torch.equal(value, original[key]) for key, value in head.state_dict().items())
    assert optimizer.state_dict() == original_optimizer


def test_meta_sample_isolation():
    def batches():
        while True:
            yield {'input_ids': torch.arange(4).reshape(4, 1)}, torch.zeros(4, 3), ['a', 'b', 'c', 'd']
    tokens, targets, ids = disjoint_meta_batch(iter(batches()), ['a', 'c'], desired_size=2)
    assert ids == ['b', 'd']
    assert tokens['input_ids'].flatten().tolist() == [1, 3]
    assert targets.shape == (2, 3)
    with pytest.raises(RuntimeError):
        disjoint_meta_batch(iter(batches()), ['a', 'b', 'c', 'd'], desired_size=2, max_tries=2)


def test_meta_batch_refills_to_full_size_without_duplicate_ids():
    def batches():
        yield {'input_ids': torch.tensor([[1, 2], [3, 4]])}, torch.zeros(2, 3), ['a', 'b']
        yield {'input_ids': torch.tensor([[5, 6, 7], [8, 9, 10]])}, torch.ones(2, 3), ['b', 'c']
        yield {'input_ids': torch.tensor([[11]])}, torch.ones(1, 3), ['d']
    tokens, targets, ids = disjoint_meta_batch(iter(batches()), ['a'], desired_size=3)
    assert ids == ['b', 'c', 'd']
    assert tokens['input_ids'].shape == (3, 3)
    assert targets.shape == (3, 3)


@pytest.mark.parametrize('mode', ['diagonal', 'crossdim'])
def test_controller_is_finite_nonnegative_and_avoids_total_collapse(mode):
    controller = UtilityController(mode)
    weights = controller.update(-torch.ones(3, 3))
    assert torch.equal(weights, torch.ones(3, dtype=torch.float64))
    weights = controller.update(torch.tensor([[10., -1., -1.],
                                              [-1., 2., -1.],
                                              [-1., -1., 1.]]))
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    assert weights.sum() > 0


def test_zero_unlabeled_weights_are_pure_supervision():
    p = nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    supervised = p.square().mean()
    unsupervised = (p - 3).square()
    expected = torch.autograd.grad(supervised, p, retain_graph=True)[0]
    actual = torch.autograd.grad(supervised + (unsupervised * torch.zeros(3)).mean(), p)[0]
    assert torch.equal(actual, expected)


def test_crossdim_uses_offdiagonal_effects():
    matrix = torch.tensor([[2., 8., 0.], [0., 2., 0.], [0., 0., 2.]])
    diagonal = UtilityController('diagonal', momentum=0).update(matrix)
    crossdim = UtilityController('crossdim', momentum=0).update(matrix)
    assert torch.allclose(diagonal, torch.ones(3, dtype=torch.float64))
    assert crossdim[1] > crossdim[0]
    assert crossdim[0] == crossdim[2]


def test_baseline_unlabeled_formula_matches_three_dimension_average():
    torch.manual_seed(3)
    strong, weak, logvar = torch.randn(5, 3), torch.randn(5, 3), torch.randn(5, 3)
    dimension_losses = per_dimension_unlabeled(strong, weak, logvar)
    baseline = (0.5 * torch.exp(-logvar) * (strong - weak).square()).mean()
    assert torch.allclose(dimension_losses.mean(), baseline)
