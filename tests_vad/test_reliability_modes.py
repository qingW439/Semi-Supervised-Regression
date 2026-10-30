import torch

from scripts.vad_research.jacf_modes import evidence, fixed_alpha_objective


def _ev(b=6, k=10, h=8):
    torch.manual_seed(0)
    mc = torch.rand(5, b, 3)
    u = torch.randn(b, h)
    l = torch.randn(k, h)
    ly = torch.rand(k, 3)
    return evidence(mc, u, l, ly)


def test_modes_shapes_and_finiteness():
    ev = _ev()
    for mode in ('none', 'anchor_var', 'prop_var', 'max_var', 'anchor_var_conf',
                 'anchor_filter', 'jacf'):
        target, rel = fixed_alpha_objective(ev, alpha=0.75, rel_mode=mode, filter_q=0.5)
        assert target.shape == (6, 3) and rel.shape == (6, 3)
        assert torch.isfinite(target).all() and torch.isfinite(rel).all()
        assert (rel >= 0).all()
        assert rel.sum() > 0


def test_none_is_unit_weight_and_fixed_target():
    ev = _ev()
    mu_t, v_t, mu_a, v_a, conf = ev
    target, rel = fixed_alpha_objective(ev, alpha=0.75, rel_mode='none')
    assert torch.allclose(target, 0.25 * mu_t + 0.75 * mu_a)
    assert torch.allclose(rel, torch.ones_like(rel))


def test_anchor_var_decreases_with_anchor_variance():
    ev = _ev()
    _, v_t, _, v_a, _ = ev
    _, rel = fixed_alpha_objective(ev, alpha=0.75, rel_mode='anchor_var')
    order = torch.argsort(v_a.flatten())
    assert (rel.flatten()[order][1:] <= rel.flatten()[order][:-1] + 1e-7).all()


def test_prop_var_uses_actual_fusion_weights():
    ev = _ev()
    _, v_t, _, v_a, _ = ev
    _, rel = fixed_alpha_objective(ev, alpha=0.75, rel_mode='prop_var')
    expected = 1e-3 / (1e-3 + 0.75 ** 2 * v_a + 0.25 ** 2 * v_t)
    assert torch.allclose(rel, expected)


def test_filter_keeps_fraction():
    ev = _ev(b=20)
    _, rel = fixed_alpha_objective(ev, alpha=0.75, rel_mode='anchor_filter', filter_q=0.5)
    assert set(rel.unique().tolist()) <= {0.0, 1.0}
    assert 0.3 < float(rel.mean()) < 0.7


def test_dimnorm_equalises_dimension_means():
    ev = _ev()
    _, rel = fixed_alpha_objective(ev, alpha=0.75, rel_mode='anchor_var', dimnorm=True)
    means = rel.mean(dim=0)
    assert torch.allclose(means, torch.ones_like(means), atol=1e-5)
