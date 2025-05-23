import numpy as np
from att_bench_lib.numba_att import compute_attention_numba
from att_bench_lib.numpy_att import compute_attention_numpy

def test_numba_matches_numpy():
    np.random.seed(0)
    Q = K = np.random.randn(8,8).astype(np.float64)
    V = np.random.randn(8,8).astype(np.float64)
    out_nb = compute_attention_numba(Q, K, V)
    out_np = compute_attention_numpy(Q, K, V)
    assert np.allclose(out_nb, out_np, atol=1e-6)
