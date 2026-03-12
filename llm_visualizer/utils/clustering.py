"""
Level 4: Clustering — find groups of similar tokens/representations.
"""
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from typing import Dict, List, Optional
from utils.model_loader import load_tokenizer


def cluster_kmeans(data: np.ndarray, n_clusters: int = 8,
                   random_state: int = 42) -> Dict:
    """
    KMeans clustering on embedding data.
    data shape: (n_points, d)
    Returns labels, centroids, and silhouette score.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(data)
    sil_score = float(silhouette_score(data, labels)) if n_clusters > 1 else 0.0
    return {
        "labels": labels.tolist(),
        "centroids": km.cluster_centers_.tolist(),
        "silhouette_score": sil_score,
        "n_clusters": n_clusters,
    }


def cluster_dbscan(data: np.ndarray, eps: float = 0.5,
                   min_samples: int = 5) -> Dict:
    """
    DBSCAN clustering — automatically finds number of clusters.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    sil_score = float(silhouette_score(data, labels)) if n_clusters > 1 else 0.0
    return {
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "n_noise_points": int(np.sum(labels == -1)),
        "silhouette_score": sil_score,
    }


def cluster_vocab_embeddings(model_key: str, token_ids: Optional[List[int]] = None,
                              n_clusters: int = 10, sample_size: int = 5000) -> Dict:
    """
    Cluster tokens from the vocabulary embedding matrix.
    If token_ids is None, randomly sample `sample_size` tokens.
    """
    from utils.embedding_utils import get_token_embedding_matrix
    tokenizer = load_tokenizer(model_key)
    emb_matrix = get_token_embedding_matrix(model_key)

    if token_ids is None:
        rng = np.random.RandomState(42)
        token_ids = rng.choice(emb_matrix.shape[0], size=min(sample_size, emb_matrix.shape[0]),
                               replace=False).tolist()

    embeddings = emb_matrix[token_ids]
    cluster_result = cluster_kmeans(embeddings, n_clusters=n_clusters)

    # Add token labels
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    cluster_result["token_ids"] = token_ids
    cluster_result["tokens"] = tokens

    return cluster_result


def find_neighbors_in_embedding(target_ids: List[int], model_key: str,
                                 top_k: int = 20) -> Dict:
    """
    For each target token, find its nearest neighbors in embedding space.
    """
    from utils.embedding_utils import get_token_embedding_matrix
    tokenizer = load_tokenizer(model_key)
    emb_matrix = get_token_embedding_matrix(model_key)

    # Normalize for cosine similarity
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9
    emb_normed = emb_matrix / norms

    results = {}
    for tid in target_ids:
        vec = emb_normed[tid]
        sims = emb_normed @ vec
        top_indices = np.argsort(sims)[-top_k:][::-1]
        neighbors = []
        for idx in top_indices:
            neighbors.append({
                "token_id": int(idx),
                "token": tokenizer.decode([int(idx)]),
                "similarity": float(sims[idx]),
            })
        results[tokenizer.decode([tid])] = neighbors

    return results

