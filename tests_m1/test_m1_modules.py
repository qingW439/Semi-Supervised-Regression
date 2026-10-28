"""CPU checks for M1 gradient surgery and TRAIN-only utility logic."""
import copy

import pytest
import torch
from torch import nn

from scripts.vad_research.m1.dimensional_utility import (
    disjoint_meta_batch, per_dimension_unlabeled, virtual_utility_matrix,
    virtual_utility_probe, weighted_dimensional_pseudo_mse,
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
    meta_x, meta_y = torch.randn(5, 4), torch.randn(5, 3)
    matrix = virtual_utility_matrix(head, losses, meta_x, meta_y, 5e-6)
    probe = virtual_utility_probe(head, losses, meta_x, meta_y, 5e-6,
                                  include_float32=True)
    assert matrix.shape == (3, 3)
    assert torch.isfinite(matrix).all()
    assert torch.count_nonzero(matrix) > 0
    assert torch.equal(matrix, probe['raw_matrix'])
    assert probe['normalized_matrix'].shape == (3, 3)
    assert (probe['normalized_matrix'].abs() <= 1 + 1e-8).all()
    assert probe['parameter_norm'] > 0
    assert torch.isfinite(probe['virtual_parameter_norm_after']).all()
    assert (probe['relative_virtual_change'] > 0).all()
    assert torch.isfinite(probe['float32_matrix']).all()
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


def test_utility_sampler_does_not_advance_hpl_meta_sampler():
    data = list(range(500))
    def loader(seed):
        return torch.utils.data.DataLoader(data, batch_size=16, shuffle=True,
                                           generator=torch.Generator().manual_seed(seed))
    reference = iter(loader(23_000))
    expected_first, expected_second = next(reference), next(reference)
    hpl_meta = iter(loader(23_000))
    utility_meta = iter(loader(29_000))
    assert torch.equal(next(hpl_meta), expected_first)
    next(utility_meta)
    next(utility_meta)
    assert torch.equal(next(hpl_meta), expected_second)


@pytest.mark.parametrize('mode', ['diagonal', 'crossdim'])
def test_controller_is_finite_nonnegative_and_avoids_total_collapse(mode):
    controller = UtilityController(mode)
    weights = controller.update(-torch.ones(3, 3))
    assert torch.equal(weights, torch.ones(3, dtype=torch.float64))
    weights = controller.update(torch.tensor([[0.8, -0.1, -0.1],
                                              [0.1, 0.2, -0.1],
                                              [-0.1, -0.1, 0.1]]))
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
    matrix = torch.tensor([[0.2, 0.8, -0.1],
                           [0.1, 0.2, 0.1],
                           [-0.3, -0.3, 0.2]])
    diagonal = UtilityController('diagonal', momentum=0).update(matrix)
    crossdim = UtilityController('crossdim', momentum=0).update(matrix)
    assert torch.allclose(diagonal, torch.ones(3, dtype=torch.float64))
    assert crossdim[0] > crossdim[1] > crossdim[2]
    assert crossdim[2] == 0


def test_baseline_unlabeled_formula_matches_three_dimension_average():
    torch.manual_seed(3)
    strong, weak, logvar = torch.randn(5, 3), torch.randn(5, 3), torch.randn(5, 3)
    dimension_losses = per_dimension_unlabeled(strong, weak, logvar)
    baseline = (0.5 * torch.exp(-logvar) * (strong - weak).square()).mean()
    assert torch.allclose(dimension_losses.mean(), baseline)
    strong.requires_grad_()
    baseline = (0.5 * torch.exp(-logvar) * (strong - weak).square()).mean()
    weighted = weighted_dimensional_pseudo_mse(strong, weak, logvar, torch.ones(3))
    assert torch.equal(weighted, baseline)
    grad_baseline = torch.autograd.grad(baseline, strong, retain_graph=True)[0]
    grad_weighted = torch.autograd.grad(weighted, strong)[0]
    assert torch.equal(grad_weighted, grad_baseline)


def test_float64_difference_tracks_first_order_without_rescaling():
    torch.manual_seed(31)
    head = nn.Sequential(nn.Linear(4, 7), nn.GELU(), nn.Linear(7, 3))
    x = torch.randn(8, 4)
    losses = per_dimension_unlabeled(head(x), torch.randn(8, 3), torch.zeros(8, 3))
    meta_x, meta_y = torch.randn(6, 4), torch.randn(6, 3)
    small = virtual_utility_probe(head, losses, meta_x, meta_y, 5e-6, include_float32=True)
    larger = virtual_utility_probe(head, losses, meta_x, meta_y, 5e-5)
    first = small['first_order_matrix']
    error = (small['raw_matrix'] - first).abs().max()
    assert error < first.abs().max() * 0.01
    stable = first.abs() > first.abs().max() * 0.05
    assert torch.equal(torch.sign(small['raw_matrix'][stable]), torch.sign(larger['raw_matrix'][stable]))
    assert torch.isfinite(small['float32_matrix']).all()


def test_distinct_controller_weights_produce_distinct_adam_updates():
    torch.manual_seed(35)
    matrix = torch.tensor([[0.2, 0.8, -0.1], [0.1, 0.2, 0.1], [-0.3, -0.3, 0.2]])
    diagonal = UtilityController('diagonal', momentum=0).update(matrix).float()
    crossdim = UtilityController('crossdim', momentum=0).update(matrix).float()
    base = nn.Sequential(nn.Linear(4, 6), nn.GELU(), nn.Linear(6, 3))
    x, y = torch.randn(5, 4), torch.randn(5, 3)
    warm = torch.optim.Adam(base.parameters(), lr=0.01)
    warm.zero_grad()
    (base(x) - y).square().mean().backward()
    warm.step()
    states = []
    for weights in (diagonal, crossdim):
        head = copy.deepcopy(base)
        opt = torch.optim.Adam(head.parameters(), lr=0.01)
        opt.load_state_dict(copy.deepcopy(warm.state_dict()))
        opt.zero_grad()
        weighted_dimensional_pseudo_mse(head(x), torch.zeros(5, 3),
                                        torch.zeros(5, 3), weights).backward()
        opt.step()
        states.append(head.state_dict())
    assert any(not torch.equal(states[0][key], states[1][key]) for key in states[0])
