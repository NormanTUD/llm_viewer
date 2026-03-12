"""
Level 4: Pattern finder — generate mathematical patterns (spirals, circles, 
lines, etc.) and search through all layers for closest matches.
"""
import numpy as np
from typing import Dict, List, Callable, Optional
from scipy.spatial.distance import cdist
from utils.projection import project_to_2d, project_to_3d


# ─── Pattern Generators ───────────────────────────────────────────

def generate_spiral(n_points: int = 100, turns: float = 3.0,
                    radius_growth: float = 1.0) -> np.ndarray:
    """Generate 3D spiral. Returns (n_points, 3)."""
    t = np.linspace(0, turns * 2 * np.pi, n_points)
    r = radius_growth * t / (turns * 2 * np.pi)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = t / (turns * 2 * np.pi)
    return np.stack([x, y, z], axis=1)


def generate_circle(n_points: int = 100, radius: float = 1.0) -> np.ndarray:
    """Generate 2D circle embedded in 3D. Returns (n_points, 3)."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros(n_points)
    return np.stack([x, y, z], axis=1)


def generate_line(n_points: int = 100) -> np.ndarray:
    """Generate a straight line in 3D. Returns (n_points, 3)."""
    t = np.linspace(0, 1, n_points)
    return np.stack([t, t, t], axis=1)


def generate_helix(n_points: int = 100, turns: float = 3.0,
                   radius: float = 1.0) -> np.ndarray:
    """Generate a helix (constant radius spiral). Returns (n_points, 3)."""
    t = np.linspace(0, turns * 2 * np.pi, n_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = t / (turns * 2 * np.pi)
    return np.stack([x, y, z], axis=1)


def generate_sphere_surface(n_points: int = 200, radius: float = 1.0) -> np.ndarray:
    """Generate points on sphere surface using Fibonacci lattice."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_points)
    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_points)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def generate_grid(n_side: int = 10) -> np.ndarray:
    """Generate a flat grid in 3D. Returns (n_side^2, 3)."""
    x = np.linspace(0, 1, n_side)
    y = np.linspace(0, 1, n_side)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def generate_clusters_pattern(n_clusters: int = 5, points_per_cluster: int = 20,
                               spread: float = 0.1) -> np.ndarray:
    """Generate clustered points in 3D."""
    rng = np.random.RandomState(42)
    points = []
    for _ in range(n_clusters):
        center = rng.randn(3)
        cluster_points = center + spread * rng.randn(points_per_cluster, 3)
        points.append(cluster_points)
    return np.vstack(points)


PATTERN_GENERATORS = {
    "spiral": generate_spiral,
    "circle": generate_circle,
    "line": generate_line,
    "helix": generate_helix,
    "sphere": generate_sphere_surface,
    "grid": generate_grid,
    "clusters": generate_clusters_pattern,
}


# ─── Pattern Matching ─────────────────────────────────────────────

def normalize_shape(points: np.ndarray) -> np.ndarray:
    """Normalize points to zero mean, unit variance."""
    centered = points - points.mean(axis=0)
    scale = np.std(centered) + 1e-9
    return centered / scale


def shape_similarity(pattern: np.ndarray, data: np.ndarray) -> float:
    """
    Compute similarity between a pattern and data points (both in 3D).
    Uses minimum average distance after normalization.
    Lower = more similar. We return negative so higher = more similar.
    """
    p = normalize_shape(pattern)
    d = normalize_shape(data)

    # Subsample if sizes differ
    if p.shape[0] > d.shape[0]:
        indices = np.linspace(0, p.shape[0] - 1, d.shape[0]).astype(int)
        p = p[indices]
    elif d.shape[0] > p.shape[0]:
        indices = np.linspace(0, d.shape[0] - 1, p.shape[0]).astype(int)
        d = d[indices]

    # Compute pairwise distances and use Hungarian-like matching
    dists = cdist(p, d, metric="euclidean")
    # Average nearest-neighbor distance (both directions)
    avg_dist_p = np.mean(np.min(dists, axis=1))
    avg_dist_d = np.mean(np.min(dists, axis=0))
    avg_dist = (avg_dist_p + avg_dist_d) / 2

    # Convert to similarity score (0 to 1, higher is more similar)
    similarity = 1.0 / (1.0 + avg_dist)
    return float(similarity)


def scan_all_layers_for_pattern(pattern_name: str, trace: Dict,
                                 projection_method: str = "pca",
                                 pattern_params: Optional[Dict] = None) -> List[Dict]:
    """
    Scan all layers' hidden states, project to 3D, and compare against a pattern.
    Returns a sorted list of (layer, similarity_score).
    """
    if pattern_params is None:
        pattern_params = {}

    generator = PATTERN_GENERATORS.get(pattern_name)
    if generator is None:
        raise ValueError(f"Unknown pattern: {pattern_name}. Available: {list(PATTERN_GENERATORS.keys())}")

    n_points = trace["hidden_states"][0].shape[0]
    pattern = generator(n_points=max(n_points, 20), **pattern_params)

    results = []
    for layer_idx, hs in enumerate(trace["hidden_states"]):
        # Project hidden state to 3D
        if hs.shape[0] < 3:
            projected = np.zeros((hs.shape[0], 3))
        else:
            projected = project_to_3d(hs, method=projection_method)

        sim = shape_similarity(pattern, projected)
        results.append({
            "layer": layer_idx,
            "similarity": sim,
            "is_input": layer_idx == 0,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


def scan_all_patterns_in_layer(trace: Dict, layer_idx: int,
                                projection_method: str = "pca") -> List[Dict]:
    """
    For a specific layer, check all known patterns and rank them.
    """
    hs = trace["hidden_states"][layer_idx]
    if hs.shape[0] < 3:
        projected = np.zeros((hs.shape[0], 3))
    else:
        projected = project_to_3d(hs, method=projection_method)

    n_points = hs.shape[0]
    results = []
    for name, generator in PATTERN_GENERATORS.items():
        try:
            pattern = generator(n_points=max(n_points, 20))
            sim = shape_similarity(pattern, projected)
            results.append({"pattern": name, "similarity": sim})
        except Exception:
            pass

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results

