import torch

from scripts.vad_research.jacf_modes import evidence, build_objective
from scripts.vad_research.jacf import jacf_fuse


def _ev(b=4, k=10, h=8):
    torch.manual_seed(0)
    mc = torch.rand(5, b, 3)
    u = torch.randn(b, h)
    l = torch.randn(k, h)
    ly = torch.rand(k, 3)
    return evidence(mc, u, l, ly), (mc, u, l, ly)


def test_modes_shapes_and_finiteness():
    ev, _ = _ev()
    for mode in ('multi', 'anchor', 'seq', 'jacf', 'no_rel'):
        target, rel = build_objective(mode, ev)
        assert target.shape == (4, 3) and rel.shape == (4, 3)
        assert torch.isfinite(target).all() and torch.isfinite(rel).all()
        assert (rel > 0).all()


def test_multi_uses_teacher_only():
    ev, _ = _ev()
    target, rel = build_objective('multi', ev)
    mu_t, v_t, mu_a, v_a, conf = ev
    assert torch.allclose(target, mu_t)
    assert torch.allclose(rel, 1e-3 / (1e-3 + v_t))


def test_anchor_uses_anchor_only():
    ev, _ = _ev()
    target, rel = build_objective('anchor', ev)
    mu_t, v_t, mu_a, v_a, conf = ev
    assert torch.allclose(target, mu_a)
    assert torch.allclose(rel, 1e-3 / (1e-3 + v_a))


def test_seq_is_fixed_alpha_blend():
    ev, _ = _ev()
    target, rel = build_objective('seq', ev, alpha=0.4, c0=0.3, gcf_tau=0.05)
    mu_t, v_t, mu_a, v_a, conf = ev
    gate = ((conf - 0.3) / 0.7).clamp(0, 1)
    a = (0.4 * gate).unsqueeze(-1)
    assert torch.allclose(target, (1 - a) * mu_t + a * mu_a)
    assert torch.allclose(rel, torch.exp(-v_t.clamp_min(0).sqrt() / 0.05))


def test_jacf_matches_reference_fuse():
    ev, (mc, u, l, ly) = _ev()
    target, rel = build_objective('jacf', ev)
    ref_target, ref_rel, _ = jacf_fuse(mc, u, l, ly)
    assert torch.allclose(target, ref_target, atol=1e-6)
    assert torch.allclose(rel, ref_rel, atol=1e-6)


def test_no_rel_is_unit_weight():
    ev, _ = _ev()
    target, rel = build_objective('no_rel', ev)
    jt, jr = build_objective('jacf', ev)
    assert torch.allclose(target, jt)
    assert torch.allclose(rel, torch.ones_like(rel))


def test_outputs_detached():
    ev, _ = _ev()
    target, rel = build_objective('jacf', ev)
    assert not target.requires_grad and not rel.requires_grad


def test_unknown_mode_rejected():
    ev, _ = _ev()
    try:
        build_objective('nope', ev)
    except ValueError:
        return
    raise AssertionError('expected ValueError')
