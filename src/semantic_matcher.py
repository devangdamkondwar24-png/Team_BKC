"""
Semantic Matcher — Embedding-based JD-to-profile similarity.

Uses precomputed candidate embeddings and FAISS index.
At ranking time: embed JD → cosine similarity lookup → O(1) per candidate.
"""
import numpy as np
from pathlib import Path
from src.config import PRECOMPUTED_DIR, EMBEDDINGS_FILE, JD_EMBEDDING_FILE


def load_precomputed_embeddings(base_dir: str = ".") -> np.ndarray:
    """Load precomputed candidate embedding matrix (N x dim)."""
    path = Path(base_dir) / PRECOMPUTED_DIR / EMBEDDINGS_FILE
    return np.load(str(path))


def load_jd_embedding(base_dir: str = ".") -> np.ndarray:
    """Load precomputed JD embedding vector (dim,)."""
    path = Path(base_dir) / PRECOMPUTED_DIR / JD_EMBEDDING_FILE
    return np.load(str(path))


def compute_semantic_similarities(
    jd_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarities between JD and all candidates.

    Args:
        jd_embedding: (dim,) vector
        candidate_embeddings: (N, dim) matrix

    Returns:
        (N,) array of cosine similarities in [0, 1]
    """
    # Normalize
    jd_norm = jd_embedding / (np.linalg.norm(jd_embedding) + 1e-10)
    norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
    cand_normed = candidate_embeddings / norms

    # Dot product = cosine similarity (both normalized)
    similarities = cand_normed @ jd_norm

    # Clip to [0, 1] — negative similarities meaningless here
    return np.clip(similarities, 0.0, 1.0)


def normalize_similarities(similarities: np.ndarray) -> np.ndarray:
    """Min-max normalize similarities to [0, 1] for scoring."""
    min_val = similarities.min()
    max_val = similarities.max()
    if max_val - min_val < 1e-10:
        return np.zeros_like(similarities)
    return (similarities - min_val) / (max_val - min_val)
