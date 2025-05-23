# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: language=c++

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp

ctypedef fused float_t:
    np.float32_t
    np.float64_t

def compute_attention_cython(
    np.ndarray[float_t, ndim=2] Q,
    np.ndarray[float_t, ndim=2] K,
    np.ndarray[float_t, ndim=2] V,
    int n_threads=1,
    int block_size=32
):
    """Cython/C++ self-attention."""
    cdef int n_q = Q.shape[0], n_k = K.shape[0], d = Q.shape[1]
    cdef float_t[:, :] matQ = Q
    cdef float_t[:, :] matK = K
    cdef float_t[:, :] matV = V

    # compute scaled dot product
    cdef np.ndarray[float_t, ndim=2] scores = np.dot(matQ, matK.T) / sqrt(d)
    cdef float_t[:, :] s = scores
    cdef np.ndarray[float_t, ndim=2] weights = np.empty((n_q, n_k), dtype=scores.dtype)
    cdef float_t[:, :] w = weights

    cdef int i, j
    cdef float_t row_max, sum_exp, val

    # row-wise softmax
    for i in range(n_q):
        row_max = s[i, 0]
        for j in range(1, n_k):
            if s[i, j] > row_max:
                row_max = s[i, j]
        sum_exp = 0
        for j in range(n_k):
            val = exp(s[i, j] - row_max)
            w[i, j] = val
            sum_exp += val
        for j in range(n_k):
            w[i, j] /= sum_exp

    return np.dot(weights, matV)
