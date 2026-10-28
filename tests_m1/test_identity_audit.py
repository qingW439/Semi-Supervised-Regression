"""Fixed TRAIN-like batch counterfactual for original HPL versus unit utility."""
import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from scripts.vad_research.m1.dimensional_utility import (
    per_dimension_unlabeled, virtual_utility_probe,
    weighted_dimensional_pseudo_mse,
)


class TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Dropout(0.3))
        self.head = nn.Sequential(nn.Linear(4, 7), nn.GELU(), nn.Linear(7, 3))

    def forward(self, x):
        return self.head(self.encoder(x))


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_fixed_batch_hpl_identity_including_adam_and_rng():
    pytest.importorskip('higher')
    from scripts.vad_research.m0.hpl_core import (
        VADUncertainty, bilevel_head_step, weighted_pseudo_mse,
    )

    torch.manual_seed(73)
    labeled_x, labeled_y = torch.randn(6, 4), torch.randn(6, 3)
    hpl_meta_x, hpl_meta_y = torch.randn(6, 4), torch.randn(6, 3)
    utility_meta_x, utility_meta_y = torch.randn(6, 4), torch.randn(6, 3)
    unlabeled_x = torch.randn(9, 4)
    base, base_uncertainty = TinyRegressor(), VADUncertainty(hidden=8)
    base_encoder_opt = torch.optim.Adam(base.encoder.parameters(), lr=5e-4)
    base_head_opt = torch.optim.Adam(base.head.parameters(), lr=5e-4)
    base_unc_opt = torch.optim.Adam(base_uncertainty.parameters(), lr=1e-4)
    # Populate every Adam state before cloning the checkpoint.
    F.mse_loss(base(labeled_x), labeled_y).backward()
    base_encoder_opt.step()
    base_head_opt.step()
    base_uncertainty(torch.randn(4, 3), torch.randn(4, 3)).mean().backward()
    base_unc_opt.step()
    checkpoint = (copy.deepcopy(base.state_dict()), copy.deepcopy(base_uncertainty.state_dict()),
                  copy.deepcopy(base_encoder_opt.state_dict()), copy.deepcopy(base_head_opt.state_dict()),
                  copy.deepcopy(base_unc_opt.state_dict()))

    def one_update(with_unit_utility):
        model, uncertainty = TinyRegressor(), VADUncertainty(hidden=8)
        model.load_state_dict(checkpoint[0])
        uncertainty.load_state_dict(checkpoint[1])
        encoder_opt = torch.optim.Adam(model.encoder.parameters(), lr=5e-4)
        head_opt = torch.optim.Adam(model.head.parameters(), lr=5e-4)
        unc_opt = torch.optim.Adam(uncertainty.parameters(), lr=1e-4)
        for opt, state in zip((encoder_opt, head_opt, unc_opt), checkpoint[2:]):
            opt.load_state_dict(copy.deepcopy(state))
        model.train()
        encoder_opt.zero_grad(set_to_none=True)
        head_opt.zero_grad(set_to_none=True)
        torch.manual_seed(1001)
        supervised = F.mse_loss(model(labeled_x), labeled_y)
        with torch.random.fork_rng():
            torch.manual_seed(1002)
            with torch.no_grad():
                model.train()
                labeled_features = model.encoder(labeled_x)
                hpl_meta_features = model.encoder(hpl_meta_x)
                model.eval()
                weak_features = model.encoder(unlabeled_x)
                model.train()
                strong_features = model.encoder(unlabeled_x)
            bilevel_head_step(model.head, head_opt, uncertainty, unc_opt,
                              labeled_features, labeled_y, weak_features,
                              strong_features, hpl_meta_features, hpl_meta_y)
            encoder_opt.zero_grad(set_to_none=True)
            head_opt.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad():
                weak = model(unlabeled_x).detach()
            model.train()
            strong = model(unlabeled_x)
            logvar = uncertainty(strong - weak, strong)
            if with_unit_utility:
                losses = per_dimension_unlabeled(strong, weak, logvar.detach())
                rng_before = torch.get_rng_state().clone()
                optimizer_before = copy.deepcopy(head_opt.state_dict())
                with torch.random.fork_rng():
                    with torch.no_grad():
                        utility_features = model.encoder(utility_meta_x)
                assert torch.equal(torch.get_rng_state(), rng_before)
                assert model.training
                virtual_utility_probe(model.head, losses, utility_features,
                                      utility_meta_y, head_opt.param_groups[0]['lr'])
                assert all(parameter.grad is None for parameter in model.head.parameters())
                assert_nested_equal(head_opt.state_dict(), optimizer_before)
                unlabeled = weighted_dimensional_pseudo_mse(
                    strong, weak, logvar.detach(), torch.ones(3))
            else:
                unlabeled = weighted_pseudo_mse(strong, weak, logvar.detach())
            total = supervised + unlabeled
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        encoder_opt.step()
        head_opt.step()
        return (model.state_dict(), uncertainty.state_dict(),
                encoder_opt.state_dict(), head_opt.state_dict(), unc_opt.state_dict(),
                torch.get_rng_state().clone(), model.training)

    original = one_update(False)
    utility = one_update(True)
    assert_nested_equal(original, utility)
