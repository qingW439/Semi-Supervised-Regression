import copy

import torch
from torch import nn

from scripts.vad_research.gradient_audit import project_conflict


def test_projection_removes_negative_inner_product():
    gs = [torch.tensor([1.0, 0.0])]
    gu = [torch.tensor([-1.0, 1.0])]
    proj = project_conflict(gs, gu)
    assert torch.allclose((gs[0] * proj[0]).sum(), torch.tensor(0.0), atol=1e-6)


def test_non_conflicting_gradient_unchanged():
    gs = [torch.tensor([1.0, 0.0])]
    gu = [torch.tensor([1.0, 1.0])]
    proj = project_conflict(gs, gu)
    assert torch.allclose(proj[0], gu[0])


def test_none_gradients_pass_through():
    gs = [None, torch.tensor([1.0])]
    gu = [torch.tensor([1.0]), torch.tensor([-1.0])]
    proj = project_conflict(gs, gu)
    assert proj[0] is not None
    assert torch.allclose((gs[1] * proj[1]).sum(), torch.tensor(0.0), atol=1e-6)


def test_model_and_optimizer_state_restore():
    torch.manual_seed(0)
    m1 = nn.Linear(3, 2)
    m2 = copy.deepcopy(m1)
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-3, weight_decay=0.01)
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=0.01)
    x = torch.randn(4, 3)
    m1(x).sum().backward()
    o1.step()
    sd = copy.deepcopy(m1.state_dict())
    od = copy.deepcopy(o1.state_dict())
    m2.load_state_dict(sd)
    o2.load_state_dict(od)
    for p, q in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p, q)
    for key in o1.state_dict()['state']:
        for k in o1.state_dict()['state'][key]:
            a = o1.state_dict()['state'][key][k]
            b = o2.state_dict()['state'][key][k]
            if torch.is_tensor(a):
                assert torch.allclose(a, b)
            else:
                assert a == b
