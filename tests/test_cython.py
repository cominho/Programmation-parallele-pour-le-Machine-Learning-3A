import numpy as np
from att_bench_lib.cython_att import compute_attention_cython
from att_bench_lib.numpy_att import compute_attention_numpy

def test_cython_matches_numpy():
    np.random.seed(1)
    Q = K = np.random.randn(6,6).astype(np.float32)
    V = np.random.randn(6,6).astype(np.float32)
    out_cy = compute_attention_cython(Q, K, V, 1, 32)
    out_np = compute_attention_numpy(Q, K, V)
    assert np.allclose(out_cy, out_np, atol=1e-5)
