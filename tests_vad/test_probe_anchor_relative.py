import torch

from scripts.vad_research.probe_anchor_relative import boot_macro, report


def test_report_shapes_and_keys():
    y = torch.rand(12, 3)
    p = (y + 0.05 * torch.randn(12, 3)).clamp(0, 1)
    m, macro = report(y, p)
    assert set(macro) == {'mae', 'rmse', 'pearson', 'srcc', 'ccc', 'r2'}
    assert 'macro_mae_original_scale' in m
    assert macro['mae'] > 0


def test_bootstrap_positive_when_candidate_better():
    torch.manual_seed(0)
    y = torch.rand(50, 3)
    pa = y + 0.10 * torch.randn(50, 3)
    pb = y + 0.01 * torch.randn(50, 3)
    lo, hi = boot_macro(y, pa, pb, n=200, seed=0)
    assert lo > 0 and hi > 0


def test_bootstrap_negative_when_candidate_worse():
    torch.manual_seed(0)
    y = torch.rand(50, 3)
    pa = y + 0.01 * torch.randn(50, 3)
    pb = y + 0.10 * torch.randn(50, 3)
    lo, hi = boot_macro(y, pa, pb, n=200, seed=0)
    assert hi < 0 and lo < 0
