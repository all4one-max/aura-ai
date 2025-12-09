"""
Beauty standard embedding configuration.

This module stores the reference vector for "beauty" used in product ranking.
The embedding should be a 768-dimensional vector representing the ideal beauty standard.
"""

import numpy as np
import os
from typing import Optional


def get_beauty_standard_embedding() -> np.ndarray:
    """
    Returns the beauty standard embedding vector.
    
    Currently returns a placeholder zero vector.
    TODO: Replace with actual beauty standard embedding loaded from:
    - Environment variable (base64 encoded)
    - Database
    - File system (numpy file, pickle, etc.)
    - External API
    
    Returns:
        numpy.ndarray: 768-dimensional beauty standard embedding vector
    """
    # Try to load from environment variable first
    beauty_embedding_env = os.getenv("BEAUTY_STANDARD_EMBEDDING")
    if beauty_embedding_env:
        # If stored as comma-separated values or base64, parse it here
        # For now, placeholder - implement parsing based on your storage format
        pass
    
    # Try to load from file if exists
    beauty_embedding_path = os.getenv("BEAUTY_STANDARD_EMBEDDING_PATH", "data/beauty_standard_embedding.npy")
    if os.path.exists(beauty_embedding_path):
        try:
            return np.load(beauty_embedding_path)
        except Exception as e:
            print(f"Warning: Could not load beauty standard embedding from {beauty_embedding_path}: {e}")
    
    # Default: return placeholder zero vector
    # TODO: Replace with actual beauty standard embedding
    print("Warning: Using placeholder beauty standard embedding (zero vector). Please configure BEAUTY_STANDARD_EMBEDDING_PATH or set BEAUTY_STANDARD_EMBEDDING environment variable.")
    return np.zeros(768)


def set_beauty_standard_embedding(embedding: np.ndarray, save_path: Optional[str] = None) -> None:
    """
    Saves the beauty standard embedding to a file.
    
    Args:
        embedding: The beauty standard embedding vector (768-dim)
        save_path: Optional path to save the embedding. Defaults to data/beauty_standard_embedding.npy
    """
    if save_path is None:
        save_path = os.getenv("BEAUTY_STANDARD_EMBEDDING_PATH", "data/beauty_standard_embedding.npy")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as numpy file
    np.save(save_path, embedding)
    print(f"Beauty standard embedding saved to {save_path}")


