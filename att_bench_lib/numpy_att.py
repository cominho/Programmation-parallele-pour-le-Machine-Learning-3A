import numpy as np

def compute_attention_numpy(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """NumPy scaled-dot self-attention."""
    dk = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(dk)
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return weights @ V
