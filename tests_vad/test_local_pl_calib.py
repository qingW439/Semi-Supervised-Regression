import torch

from scripts.vad_research.retrieval_bank import RetrievalBank
from scripts.vad_research.local_pl_calib import local_calibrate


def _bank(n=8, h=16):
    torch.manual_seed(0)
    return RetrievalBank(torch.randn(n, h), torch.rand(n, 3))


def test_shape_and_finiteness():
    bank = _bank()
    pred = torch.rand(5, 3)
    out = local_calibrate(pred, torch.randn(5, 16), bank)
    assert out.shape == (5, 3)
    assert torch.isfinite(out).all()


def test_alpha_zero_is_identity():
    bank = _bank()
    pred = torch.rand(4, 3)
    out = local_calibrate(pred, torch.randn(4, 16), bank, alpha=0.0)
    assert torch.allclose(out, pred, atol=1e-6)


def test_k_larger_than_bank_is_clamped():
    bank = _bank(n=5)
    out = local_calibrate(torch.rand(3, 3), torch.randn(3, 16), bank, k=50)
    assert out.shape == (3, 3)


def test_query_near_cluster_moves_to_cluster_label():
    # two labeled clusters in feature space; query is close to the high-V cluster
    h = 8
    hi = torch.zeros(4, h); hi[:, 0] = 1.0
    lo = torch.zeros(4, h); lo[:, 0] = -1.0
    feat = torch.cat([hi, lo])
    labels = torch.tensor([[1.0, 0.5, 0.5]] * 4 + [[0.0, 0.5, 0.5]] * 4)
    bank = RetrievalBank(feat, labels)
    query = torch.zeros(1, h); query[0, 0] = 1.0
    pred = torch.tensor([[0.5, 0.5, 0.5]])
    out = local_calibrate(pred, query, bank, k=4, tau=0.1, alpha=0.8)
    assert out[0, 0] > pred[0, 0] + 0.2


def test_low_similarity_gate_keeps_prediction():
    h = 8
    feat = torch.zeros(4, h); feat[:, 0] = 1.0
    labels = torch.zeros(4, 3)
    bank = RetrievalBank(feat, labels)
    query = torch.zeros(1, h); query[0, 0] = -1.0  # cosine = -1
    pred = torch.tensor([[0.4, 0.6, 0.8]])
    out = local_calibrate(pred, query, bank, c0=0.3, alpha=1.0)
    assert torch.allclose(out, pred, atol=1e-6)


def test_rejects_wrong_shapes():
    bank = _bank()
    for pred, q in [(torch.rand(3, 2), torch.randn(3, 16)),
                    (torch.rand(3, 3), torch.randn(4, 16))]:
        try:
            local_calibrate(pred, q, bank)
        except ValueError:
            continue
        raise AssertionError('expected ValueError')
