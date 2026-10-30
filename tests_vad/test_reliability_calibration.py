import torch

from scripts.vad_research.reliability_calibration import predictor, dim_metrics, macro_metrics


def test_predictor_positive_and_shape():
    torch.manual_seed(0)
    X = torch.randn(10, 3, 3)
    w = torch.randn(3)
    b = torch.randn(3)
    y = predictor(X, w, b)
    assert y.shape == (10, 3)
    assert (y > 0).all()


def test_dim_metrics_detects_perfect_ordering():
    torch.manual_seed(0)
    e = torch.rand(50)
    r = -e  # higher reliability <-> lower error
    m = dim_metrics(r, e)
    assert m['spearman'] < -0.9
    assert m['aurc'] < float(e.mean())


def test_macro_metrics_shapes():
    torch.manual_seed(0)
    s = torch.rand(30, 3)
    e = torch.rand(30, 3)
    macro, per = macro_metrics(s, e)
    assert set(per) == {'V', 'A', 'D'}
    assert set(macro) == {'spearman', 'aurc', 'quartile_mean_error',
                          'high_r_high_error_fraction'}
    assert len(macro['quartile_mean_error']) == 4
