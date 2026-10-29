import torch

from scripts.vad_research.jacf import jacf_fuse


def _inputs(b=4, k=10, h=8):
    torch.manual_seed(0)
    return (torch.rand(5, b, 3),
            torch.randn(b, h),
            torch.randn(k, h),
            torch.rand(k, 3))


def test_shapes_and_ranges():
    mc, u, l, ly = _inputs()
    y, r, d = jacf_fuse(mc, u, l, ly)
    assert y.shape == (4, 3) and r.shape == (4, 3)
    assert (y >= 0).all() and (y <= 1).all()
    assert (r > 0).all() and (r <= 1).all()
    assert (d['alpha'] >= 0).all() and (d['alpha'] <= 1).all()


def test_outputs_detached():
    mc, u, l, ly = _inputs()
    y, r, _ = jacf_fuse(mc, u, l, ly)
    assert not y.requires_grad and not r.requires_grad


def test_unstable_teacher_relies_on_anchor():
    # teacher views disagree strongly within each sample -> high v_T -> alpha ~ 1
    mc = torch.zeros(5, 2, 3)
    mc[0] = 0.0
    mc[1] = 1.0
    mc[2] = 0.5
    mc[3] = 0.0
    mc[4] = 1.0
    u = torch.randn(2, 8)
    l = torch.randn(10, 8)
    ly = torch.full((10, 3), 0.5)
    _, _, d = jacf_fuse(mc, u, l, ly, eps_t=1e-4)
    assert (d['alpha'] > 0.99).all()


def test_dispersed_anchors_favor_teacher():
    # neighbors with very different labels -> high v_A -> alpha close to 0
    mc = torch.full((5, 2, 3), 0.5)
    u = torch.zeros(2, 8)
    l = torch.zeros(10, 8)
    ly = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]] * 5)
    _, _, d = jacf_fuse(mc, u, l, ly, k=10, eps_a=1e-4)
    assert (d['alpha'] < 0.05).all()


def test_conflict_lowers_reliability():
    mc = torch.full((5, 2, 3), 0.2)
    u = torch.zeros(2, 8)
    l = torch.zeros(10, 8)
    agree = torch.full((10, 3), 0.2)
    disagree = torch.full((10, 3), 0.9)
    _, r_agree, _ = jacf_fuse(mc, u, l, agree)
    _, r_dis, _ = jacf_fuse(mc, u, l, disagree)
    assert (r_agree > r_dis).all()


def test_bad_shapes_rejected():
    for bad in [
        (torch.rand(2, 3), torch.randn(2, 8), torch.randn(4, 8), torch.rand(4, 3)),
        (torch.rand(3, 2, 2), torch.randn(2, 8), torch.randn(4, 8), torch.rand(4, 3)),
    ]:
        try:
            jacf_fuse(*bad)
        except ValueError:
            continue
        raise AssertionError('expected ValueError')
