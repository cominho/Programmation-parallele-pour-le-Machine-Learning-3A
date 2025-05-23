import numpy as np
from numba import njit, prange

@njit
def _softmax_1d(x):
    """
    Numerically stable softmax for a 1-D array.
    """
    # find max
    m = x[0]
    for i in range(1, x.shape[0]):
        if x[i] > m:
            m = x[i]
    # exponentiate and sum
    total = 0.0
    exps = np.empty_like(x)
    for i in range(x.shape[0]):
        exps[i] = np.exp(x[i] - m)
        total += exps[i]
    # normalize
    for i in range(x.shape[0]):
        exps[i] /= total
    return exps

@njit(parallel=True)
def compute_attention_numba(Q, K, V):
    """
    Numba-accelerated self-attention with explicit loops for all steps.
    Q: (n, d), K: (n, d), V: (n, dv)
    """
    n, d = Q.shape
    dv = V.shape[1]

    # compute scores = Q @ K.T / sqrt(d)
    scores = np.empty((n, n), dtype=Q.dtype)
    scale = 1.0 / np.sqrt(d)
    for i in prange(n):
        for j in range(n):
            tmp = 0.0
            for k in range(d):
                tmp += Q[i, k] * K[j, k]
            scores[i, j] = tmp * scale

    # softmax row-wise
    weights = np.empty_like(scores)
    for i in prange(n):
        weights[i] = _softmax_1d(scores[i])

    # weighted sum weights @ V
    out = np.empty((n, dv), dtype=Q.dtype)
    for i in prange(n):
        for j in range(dv):
            acc = 0.0
            for k in range(n):
                acc += weights[i, k] * V[k, j]
            out[i, j] = acc

    return out
