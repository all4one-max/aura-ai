"""
Utility functions for computing similarity between vectors.
"""

import numpy as np


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes cosine similarity between two vectors.
    Returns a value between -1 and 1.

    Args:
        vec_a: First vector
        vec_b: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return np.dot(vec_a, vec_b) / (norm_a * norm_b)


