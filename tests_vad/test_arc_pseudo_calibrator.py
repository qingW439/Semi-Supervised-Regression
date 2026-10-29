import torch
from torch import nn

from scripts.vad_research.arc_pseudo_calibrator import (
    mc_predict, calibrated_pseudo_targets, build_calibrated_pseudo_ranks,
)


class DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(self.drop(x)), None


def test_mc_predict_shape_and_dropout():
    torch.manual_seed(0)
    model = DropoutModel()
    model.eval()
    x = torch.randn(8, 4)
    mc = mc_predict(model, {'x': x}, passes=5)
    assert mc.shape == (5, 8, 3)
    assert not model.training
    assert not torch.allclose(mc[0], mc[1])


def test_calibrated_targets_shapes_and_range():
    torch.manual_seed(0)
    mc = torch.rand(5, 6, 3)
    u = torch.randn(6, 4)
    l = torch.randn(10, 4)
    ly = torch.rand(10, 3)
    calib, reliable, thr, new_thr, unc = calibrated_pseudo_targets(mc, u, l, ly)
    assert calib.shape == (6, 3)
    assert reliable.shape == (6,)
    assert reliable.dtype == torch.bool
    assert unc.shape == (6,)
    assert (calib >= 0).all() and (calib <= 1).all()
    assert isinstance(thr, float) and isinstance(new_thr, float)


def test_rank_direction_and_mask():
    calib = torch.tensor([[0.2, 0.5, 0.8],
                          [0.7, 0.4, 0.1],
                          [0.2, 0.5, 0.8]])
    reliable = torch.tensor([True, True, True])
    target, mask = build_calibrated_pseudo_ranks(calib, reliable)
    assert target.shape == (3, 3, 3)
    assert mask.shape == (3, 3, 3)
    expected = (calib.unsqueeze(1) - calib.unsqueeze(0) > 0).long()
    assert torch.equal(target, expected)
    # diagonal always excluded
    for d in range(3):
        assert not mask.diagonal(dim1=0, dim2=1)[:, d].any()
    # rows 0 and 2 are identical -> all their pairs are ties and masked
    assert not mask[0, 2].any() and not mask[2, 0].any()


def test_unreliable_samples_removed():
    calib = torch.rand(4, 3)
    reliable = torch.tensor([True, False, True, False])
    _, mask = build_calibrated_pseudo_ranks(calib, reliable)
    assert not mask[1].any() and not mask[:, 1].any()
    assert not mask[3].any() and not mask[:, 3].any()


def test_zero_reliable_gives_empty_mask():
    calib = torch.rand(4, 3)
    _, mask = build_calibrated_pseudo_ranks(calib, torch.zeros(4, dtype=torch.bool))
    assert mask.sum() == 0


def test_outputs_are_detached():
    calib = torch.rand(4, 3, requires_grad=True)
    reliable = torch.tensor([True, True, True, True])
    target, mask = build_calibrated_pseudo_ranks(calib, reliable)
    assert not target.requires_grad and not mask.requires_grad


def test_ce_gradient_reaches_encoder():
    torch.manual_seed(0)
    enc = nn.Linear(4, 4)
    arc = nn.Linear(4, 6)
    x = torch.randn(6, 4)
    h = enc(x)
    scores = arc(h).view(-1, 3, 2)
    calib = torch.rand(6, 3)
    target, mask = build_calibrated_pseudo_ranks(calib, torch.ones(6, dtype=torch.bool))
    loss = torch.zeros(())
    for d in range(3):
        m = mask[:, :, d]
        a = scores[:, d, :]
        pair = a.unsqueeze(0) - a.unsqueeze(1)
        onehot = torch.nn.functional.one_hot(target[:, :, d], 2).float()
        ce = -(onehot * torch.log_softmax(pair, -1)).sum(-1)
        loss = loss + (ce * m).sum() / m.sum().clamp_min(1)
    loss.backward()
    assert enc.weight.grad is not None and enc.weight.grad.abs().sum() > 0
