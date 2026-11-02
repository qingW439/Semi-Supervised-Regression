import numpy as np

from scripts.vad_research.m1.relations.graph_ops import cosine_graph, fiedler, gclss, laplacian
from scripts.vad_research.m1.relations.metrics import cr_ir


def _graph(n=12, d=8, seed=0):
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(n, d))
    S = cosine_graph(H, nonnegative=True)
    return S, laplacian(S)


def test_gclss_three_dimensions_differ():
    S, L = _graph()
    rng = np.random.default_rng(1)
    rl = rng.normal(size=(4, 3))
    outs = [gclss(L, 8, rl[:, k]) for k in range(3)]
    assert not np.allclose(outs[0], outs[1])
    assert not np.allclose(outs[1], outs[2])


def test_gclss_equals_harmonic_extension():
    S, L = _graph()
    rl = np.arange(4, dtype=float)
    ru = gclss(L, 8, rl)
    Luu, Lul = L[:8, :8], L[:8, 8:]
    ref = -np.linalg.pinv(Luu) @ (Lul @ rl)
    assert np.allclose(ru, ref, atol=1e-10)


def test_fiedler_sign_invariance_pairwise_accuracy():
    S, _ = _graph()
    r = fiedler(S)
    true = np.arange(len(r), dtype=float)
    def acc(x):
        ii, jj = np.triu_indices(len(x), 1)
        return np.mean(np.sign(x[ii] - x[jj]) == np.sign(true[ii] - true[jj]))
    assert abs(max(acc(r), 1 - acc(r)) - max(acc(-r), 1 - acc(-r))) < 1e-12


def test_cr_ir_counts():
    true = np.array([0, 1, 2, 3.0])
    teacher = np.array([0, 2, 1, 3.0])   # two wrong pairs? signs: (1,2) wrong
    new = np.array([0, 1, 2, 3.0])
    out = cr_ir(teacher, new, true)
    assert out["n_corrected"] >= 1
    assert out["net"] == out["n_corrected"] - out["n_introduced"]
