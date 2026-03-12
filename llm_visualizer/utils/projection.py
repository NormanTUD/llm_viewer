"""
Level 3: Project high-dimensional embeddings to 2D/3D for visualization.
Supports PCA, t-SNE, and UMAP.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional


def project_to_2d(data: np.ndarray, method: str = "pca",
                  perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
    """
    Project data from (n_points, d) to (n_points, 2).
    Methods: 'pca', 'tsne', 'umap'
    """
    if data.shape[0] < 2:
        return np.zeros((data.shape[0], 2))

    if method == "pca":
        pca = PCA(n_components=2, random_state=random_state)
        return pca.fit_transform(data)
    elif method == "tsne":
        perp = min(perplexity, data.shape[0] - 1)
        tsne = TSNE(n_components=2, perplexity=max(1, perp), random_state=random_state)
        return tsne.fit_transform(data)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=random_state)
            return reducer.fit_transform(data)
        except ImportError:
            # Fallback to PCA
            pca = PCA(n_components=2, random_state=random_state)
            return pca.fit_transform(data)
    else:
        raise ValueError(f"Unknown method: {method}")


def project_to_3d(data: np.ndarray, method: str = "pca",
                  perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
    """
    Project data from (n_points, d) to (n_points, 3).
    """
    if data.shape[0] < 2:
        return np.zeros((data.shape[0], 3))

    if method == "pca":
        pca = PCA(n_components=3, random_state=random_state)
        return pca.fit_transform(data)
    elif method == "tsne":
        perp = min(perplexity, data.shape[0] - 1)
        tsne = TSNE(n_components=3, perplexity=max(1, perp), random_state=random_state)
        return tsne.fit_transform(data)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=3, random_state=random_state)
            return reducer.fit_transform(data)
        except ImportError:
            pca = PCA(n_components=3, random_state=random_state)
            return pca.fit_transform(data)
    else:
        raise ValueError(f"Unknown method: {method}")


def project_direct_dims(data: np.ndarray, dim_x: int, dim_y: int,
                         dim_z: Optional[int] = None) -> np.ndarray:
    """
    Instead of PCA/UMAP, directly use specific embedding dimensions as axes.
    Returns (n_points, 2) or (n_points, 3).
    """
    if dim_z is not None:
        return data[:, [dim_x, dim_y, dim_z]]
    return data[:, [dim_x, dim_y]]


def get_pca_explained_variance(data: np.ndarray, n_components: int = 10) -> list:
    """Return explained variance ratio for the top N principal components."""
    n_components = min(n_components, data.shape[0], data.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return [float(v) for v in pca.explained_variance_ratio_]

