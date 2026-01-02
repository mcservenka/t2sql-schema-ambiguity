import numpy as np
import faiss
from typing import List, Tuple


class EmbeddingIndex:
    """
    FAISS index wrapper for cosine similarity search
    over dictionary embeddings
    """

    def __init__(self, dict_words: List[str], dict_vectors: np.ndarray):
        # dict_words: list of vocabulary words
        # dict_vectors: (N, D) float32 numpy matrix
        
        self.dict_words = dict_words
        self.dim = dict_vectors.shape[1]

        # normalize dictionary vectors
        self.dict_vectors_norm = normalize_vectors(dict_vectors.astype("float32"))

        # build FAISS index (inner product)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.dict_vectors_norm)  # add all dictionary embeddings

    def nearest_neighbor(self, query_vec: np.ndarray, k: int = 1):
        """
        Return (distances, indices) of the k nearest dictionary words
        distances = cosine similarities because vectors are normalized
        """
        if query_vec.ndim == 1:
            query_vec = query_vec[np.newaxis, :]

        # Normalize the query vector
        q = normalize_vectors(query_vec.astype("float32"))

        # Search the top-k nearest neighbors
        distances, indices = self.index.search(q, k)
        return distances, indices

    def nearest_word(self, query_vec: np.ndarray) -> Tuple[str, float]:
        """
        Return (nearest_word, similarity) for one vector
        """
        distances, indices = self.nearest_neighbor(query_vec, k=1)
        idx = int(indices[0][0])
        sim = float(distances[0][0])
        word = self.dict_words[idx]
        return word, sim

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors along axis 1
    Input: (N, D)
    Output: (N, D) normalized
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms