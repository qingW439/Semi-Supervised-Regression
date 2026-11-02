"""Graph construction and spectral seriation for M1-R0."""
from __future__ import annotations

import numpy as np


def cosine_graph(H: np.ndarray, nonnegative: bool = True, knn: int = 0) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    H = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    S = H @ H.T
    if nonnegative:
        S = (1.0 + S) / 2.0
    if knn and knn < len(S):
        S = S.copy()
        np.fill_diagonal(S, -np.inf)
        keep = np.argsort(-S, axis=1)[:, :knn]
        mask = np.zeros_like(S, dtype=bool)
        rows = np.arange(len(S))[:, None]
        mask[rows, keep] = True
        S = np.where(mask, S, 0.0)
        S = np.maximum(S, S.T)
        np.fill_diagonal(S, 0.0)
    return S


def laplacian(S: np.ndarray, normalized: bool = False) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    d = S.sum(axis=1)
    if normalized:
        dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        return np.eye(len(S)) - (dinv[:, None] * S * dinv[None, :])
    return np.diag(d) - S


def fiedler(S: np.ndarray, normalized: bool = False) -> np.ndarray:
    L = laplacian(S, normalized=normalized)
    w, V = np.linalg.eigh(L)
    order = np.argsort(w)
    w, V = w[order], V[:, order]
    for i, val in enumerate(w):
        if val > 1e-8:
            return V[:, i]
    return V[:, -1]


def gclss(L: np.ndarray, n_u: int, r_l: np.ndarray) -> np.ndarray:
    """Generalized spectral seriation: r_U = -L_UU^+ L_UL r_L."""
    L = np.asarray(L, dtype=np.float64)
    Luu = L[:n_u, :n_u]
    Lul = L[:n_u, n_u:]
    r_l = np.asarray(r_l, dtype=np.float64).reshape(-1, 1)
    r_u = -np.linalg.pinv(Luu, rcond=1e-10) @ (Lul @ r_l)
    return r_u.reshape(-1)


def sign_matrix(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    return np.sign(r[:, None] - r[None, :])


def graph_diagnostics(S: np.ndarray) -> dict:
    L = laplacian(S)
    w = np.linalg.eigvalsh(L)
    nz = w[w > 1e-8]
    cond = float(np.linalg.cond(L + 1e-6 * np.eye(len(L))))
    return {
        "negative_similarity_fraction": float(np.mean(S < 0)),
        "degree_min": float(S.sum(1).min()),
        "degree_max": float(S.sum(1).max()),
        "eig_min": float(w.min()),
        "eig_second": float(nz[0]) if len(nz) else float("nan"),
        "cond_L": cond,
    }
