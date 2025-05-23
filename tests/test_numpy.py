import numpy as np
from att_bench_lib.numpy_att import compute_attention_numpy

def test_numpy_identity():
    Q = K = np.eye(4, dtype=float)
    V = np.arange(16, dtype=float).reshape(4,4)
    out = compute_attention_numpy(Q, K, V)
    assert np.allclose(out, V)
