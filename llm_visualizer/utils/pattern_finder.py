"""
Pattern Finder — Search the real embedding/feature space for tokens
whose positions match geometric templates (spirals, helices, etc.).

The idea: given a pattern like "helix", find the actual tokens in the
vocabulary (or in a layer's hidden states) that best form that shape.
e.g., discover that year tokens 1990..2024 sit on a helix, or that
number tokens form a spiral.

Strategies:
1. ICP (Iterative Closest Point) template matching with multi-start
2. Greedy curve following from seed tokens
3. Subspace search — find the best 3D slice where the pattern appears
4. Filtered search — user constrains to token subsets (numbers, words, etc.)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
import re

from utils.model_loader import load_tokenizer
from utils.embedding_utils import get_token_embedding_matrix


# ═══════════════════════════════════════════════════════════════════
# PATTERN GENERATORS
# ═══════════════════════════════════════════════════════════════════

def generate_spiral(n_points=100, turns=3.0, radius_growth=1.0, **kw):
    """Expanding spiral in 3D."""
    t = np.linspace(0, turns * 2 * np.pi, n_points)
    r = radius_growth * t / (turns * 2 * np.pi)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = t / (turns * 2 * np.pi)
    return np.stack([x, y, z], axis=1)


def generate_circle(n_points=100, radius=1.0, **kw):
    """Flat circle in 3D."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.zeros(n_points)
    return np.stack([x, y, z], axis=1)


def generate_line(n_points=100, **kw):
    """Straight line."""
    t = np.linspace(0, 1, n_points)
    return np.stack([t, t * 0, t * 0], axis=1)


def generate_helix(n_points=100, turns=3.0, radius=1.0, **kw):
    """Constant-radius helix — the classic "years" pattern."""
    t = np.linspace(0, turns * 2 * np.pi, n_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = t / (turns * 2 * np.pi)
    return np.stack([x, y, z], axis=1)


def generate_sphere_surface(n_points=200, radius=1.0, **kw):
    """Fibonacci lattice on sphere."""
    golden = (1 + np.sqrt(5)) / 2
    idx = np.arange(n_points)
    theta = 2 * np.pi * idx / golden
    phi = np.arccos(1 - 2 * (idx + 0.5) / n_points)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def generate_grid(n_points=100, **kw):
    """Flat 2D grid in 3D."""
    n_side = int(np.sqrt(n_points))
    x = np.linspace(0, 1, n_side)
    y = np.linspace(0, 1, n_side)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def generate_clusters_pattern(n_points=100, n_clusters=5, spread=0.15, **kw):
    """Gaussian clusters in 3D."""
    rng = np.random.RandomState(42)
    ppg = max(1, n_points // n_clusters)
    pts = []
    for _ in range(n_clusters):
        center = rng.randn(3) * 2
        pts.append(center + spread * rng.randn(ppg, 3))
    return np.vstack(pts)[:n_points]


def generate_figure_eight(n_points=100, **kw):
    """Figure-eight / lemniscate in 3D."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = np.sin(t)
    y = np.sin(t) * np.cos(t)
    z = np.cos(t) * 0.3
    return np.stack([x, y, z], axis=1)


def generate_torus_knot(n_points=200, p=2, q=3, **kw):
    """Torus knot — more complex curved manifold."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r = np.cos(q * t) + 2
    x = r * np.cos(p * t)
    y = r * np.sin(p * t)
    z = -np.sin(q * t)
    return np.stack([x, y, z], axis=1)


PATTERN_GENERATORS = {
    "spiral": generate_spiral,
    "circle": generate_circle,
    "line": generate_line,
    "helix": generate_helix,
    "sphere": generate_sphere_surface,
    "grid": generate_grid,
    "clusters": generate_clusters_pattern,
    "figure_eight": generate_figure_eight,
    "torus_knot": generate_torus_knot,
}


# ═══════════════════════════════════════════════════════════════════
# VOCABULARY FILTERING — select subsets of tokens to search
# ═══════════════════════════════════════════════════════════════════

def filter_vocab_tokens(model_key: str, filter_type: str = "all",
                         custom_regex: str = None,
                         max_tokens: int = 30000) -> List[Tuple[int, str]]:
    """
    Get a filtered subset of vocabulary tokens.

    Filters:
      "all"      — all tokens (sampled if too many)
      "numbers"  — tokens that are digits
      "years"    — tokens that look like years (1000–2100)
      "words"    — alphabetic tokens >= 2 chars
      "letters"  — single letter tokens
      "months"   — month names/abbreviations
      "colors"   — common color words
      "regex"    — user-provided regex on the token string

    Returns list of (token_id, token_string).
    """
    tokenizer = load_tokenizer(model_key)
    vocab_size = tokenizer.vocab_size

    MONTH_WORDS = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    }
    COLOR_WORDS = {
        "red", "orange", "yellow", "green", "blue", "purple", "violet",
        "pink", "brown", "black", "white", "gray", "grey", "cyan",
        "magenta", "indigo", "teal", "crimson", "scarlet", "gold",
    }

    results = []
    for i in range(vocab_size):
        tok = tokenizer.decode([i])
        stripped = tok.strip().lower()

        if filter_type == "all":
            results.append((i, tok))
        elif filter_type == "numbers":
            if stripped.lstrip("-").replace(".", "", 1).isdigit():
                results.append((i, tok))
        elif filter_type == "years":
            if stripped.isdigit() and 1000 <= int(stripped) <= 2100:
                results.append((i, tok))
        elif filter_type == "words":
            if stripped.isalpha() and len(stripped) >= 2:
                results.append((i, tok))
        elif filter_type == "letters":
            if stripped.isalpha() and len(stripped) == 1:
                results.append((i, tok))
        elif filter_type == "months":
            if stripped in MONTH_WORDS:
                results.append((i, tok))
        elif filter_type == "colors":
            if stripped in COLOR_WORDS:
                results.append((i, tok))
        elif filter_type == "regex" and custom_regex:
            try:
                if re.search(custom_regex, tok):
                    results.append((i, tok))
            except re.error:
                pass

        if len(results) >= max_tokens:
            break

    return results


# ═══════════════════════════════════════════════════════════════════
# NORMALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════

def normalize_points(points: np.ndarray):
    """Zero-center and unit-scale. Returns (normalized, mean, scale)."""
    mean = points.mean(axis=0)
    centered = points - mean
    scale = np.std(centered) + 1e-9
    return centered / scale, mean, scale


def shape_similarity_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Similarity between two point sets after normalization.
    Uses bidirectional nearest-neighbor average distance.
    Returns 0..1, higher = more similar.
    """
    if a.shape[0] < 2 or b.shape[0] < 2:
        return 0.0
    an, _, _ = normalize_points(a)
    bn, _, _ = normalize_points(b)
    dists = cdist(an, bn)
    avg_a = np.mean(np.min(dists, axis=1))
    avg_b = np.mean(np.min(dists, axis=0))
    return float(1.0 / (1.0 + (avg_a + avg_b) / 2))


# ═══════════════════════════════════════════════════════════════════
# ICP (Iterative Closest Point) TEMPLATE MATCHING
# ═══════════════════════════════════════════════════════════════════

def icp_match(pattern_3d: np.ndarray, target_3d: np.ndarray,
              max_iter: int = 40, tolerance: float = 1e-6) -> Tuple:
    """
    ICP: iteratively find the rigid transform of `pattern_3d`
    that aligns it best to the nearest points in `target_3d`.

    Returns:
      matched_indices — index into target_3d for each pattern point
      transformed     — the final transformed pattern positions
      error           — mean nearest-neighbor distance
    """
    tree = KDTree(target_3d)
    current = pattern_3d.copy()
    prev_error = float("inf")

    for _ in range(max_iter):
        dists, indices = tree.query(current)

        matched = target_3d[indices]
        centroid_c = current.mean(axis=0)
        centroid_m = matched.mean(axis=0)
        c_cen = current - centroid_c
        m_cen = matched - centroid_m

        H = c_cen.T @ m_cen
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        scale = np.sum(S) / (np.sum(c_cen ** 2) + 1e-9)
        current = scale * (c_cen @ R.T) + centroid_m

        error = float(np.mean(dists))
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error

    dists, indices = tree.query(current)
    return indices, current, float(np.mean(dists))


def optimal_assignment(pattern_transformed: np.ndarray,
                        target_3d: np.ndarray,
                        candidate_indices: np.ndarray) -> np.ndarray:
    """
    Given the ICP result where multiple pattern points may map to the
    same target point, use the Hungarian algorithm for optimal 1-to-1
    assignment within the candidate set.
    """
    unique_candidates = np.unique(candidate_indices)
    if len(unique_candidates) >= len(pattern_transformed):
        candidate_pts = target_3d[unique_candidates]
        cost = cdist(pattern_transformed, candidate_pts)
        row_ind, col_ind = linear_sum_assignment(cost)
        return unique_candidates[col_ind]

    # If not enough unique candidates, expand: include neighbors
    tree = KDTree(target_3d)
    k = min(len(pattern_transformed) * 3, target_3d.shape[0])
    center = pattern_transformed.mean(axis=0)
    _, expanded = tree.query(center, k=k)
    candidate_pts = target_3d[expanded]
    cost = cdist(pattern_transformed, candidate_pts)
    row_ind, col_ind = linear_sum_assignment(cost)
    return expanded[col_ind]


def multi_start_icp(pattern_3d: np.ndarray, target_3d: np.ndarray,
                     n_starts: int = 60) -> Tuple:
    """
    Run ICP from many random rotations/scales/positions.
    Returns the best match: (matched_indices, transformed_pattern, error).
    """
    pn, _, _ = normalize_points(pattern_3d)
    tn, t_mean, t_scale = normalize_points(target_3d)

    best_indices = None
    best_error = float("inf")
    best_transformed = None

    for trial in range(n_starts):
        rng = np.random.RandomState(trial)
        rot = Rotation.random(random_state=trial).as_matrix()
        scale = rng.uniform(0.2, 3.0)
        translation = rng.uniform(tn.min(axis=0) - 1.0,
                                   tn.max(axis=0) + 1.0)

        init = scale * (pn @ rot.T) + translation
        raw_indices, transformed, error = icp_match(init, tn)

        if error < best_error:
            best_error = error
            # Optimal 1-to-1 assignment
            best_indices = optimal_assignment(transformed, tn, raw_indices)
            best_transformed = transformed * t_scale + t_mean

    return best_indices, best_transformed, best_error


# ═══════════════════════════════════════════════════════════════════
# GREEDY CURVE FOLLOWING — grow a chain of tokens along the pattern
# ═══════════════════════════════════════════════════════════════════

def greedy_curve_match(pattern_3d: np.ndarray, target_3d: np.ndarray,
                        n_seeds: int = 300,
                        direction_w: float = 0.6,
                        distance_w: float = 0.4) -> Tuple:
    """
    From many seed tokens, greedily build a chain that follows the
    pattern's curvature direction. Return the best chain.
    """
    pn, _, _ = normalize_points(pattern_3d)
    tn, _, _ = normalize_points(target_3d)
    n_pat = pn.shape[0]

    # Precompute pattern step directions
    pat_dirs = np.diff(pn, axis=0)
    pat_dirs /= (np.linalg.norm(pat_dirs, axis=1, keepdims=True) + 1e-9)

    # Precompute target step lengths from pattern
    pat_step_lens = np.linalg.norm(np.diff(pn, axis=0), axis=1)
    typical_step = float(np.median(pat_step_lens)) if len(pat_step_lens) > 0 else 0.1

    tree = KDTree(tn)
    rng = np.random.RandomState(42)
    n_seeds = min(n_seeds, target_3d.shape[0])
    seeds = rng.choice(target_3d.shape[0], size=n_seeds, replace=False)

    best_chain = []
    best_score = -1.0

    for seed_idx in seeds:
        chain = [int(seed_idx)]
        used = {int(seed_idx)}

        for step in range(min(n_pat - 1, 80)):
            cur_pos = tn[chain[-1]]
            target_dir = pat_dirs[min(step, len(pat_dirs) - 1)]

            # Search in a neighborhood of ~3x typical step length
            radius = typical_step * 4.0
            nearby = tree.query_ball_point(cur_pos, radius)
            if not nearby:
                _, nearby_arr = tree.query(cur_pos, k=min(100, tn.shape[0]))
                nearby = nearby_arr.tolist() if hasattr(nearby_arr, 'tolist') else [nearby_arr]

            best_next = None
            best_next_score = -float("inf")

            for idx in nearby:
                if idx in used:
                    continue
                direction = tn[idx] - cur_pos
                dist = np.linalg.norm(direction)
                if dist < 1e-9:
                    continue
                direction_normed = direction / dist

                alignment = float(np.dot(direction_normed, target_dir))
                # Prefer steps that are similar length to pattern steps
                step_match = 1.0 / (1.0 + abs(dist - typical_step) / (typical_step + 1e-9))
                score = direction_w * alignment + distance_w * step_match

                if score > best_next_score:
                    best_next_score = score
                    best_next = int(idx)

            if best_next is None:
                break
            chain.append(best_next)
            used.add(best_next)

        # Score chain by shape similarity
        if len(chain) >= 5:
            chain_pts = tn[chain]
            sub_pat = pn[:len(chain)]
            score = shape_similarity_score(chain_pts, sub_pat)
            if score > best_score:
                best_score = score
                best_chain = chain

    return best_chain, best_score


# ═══════════════════════════════════════════════════════════════════
# SUBSPACE SEARCH — find the best 3D projection for the pattern
# ═══════════════════════════════════════════════════════════════════

def search_best_subspace(embeddings: np.ndarray,
                          pattern_name: str,
                          n_pattern_pts: int = 50,
                          n_random: int = 40,
                          n_icp_starts: int = 15) -> Dict:
    """
    Search for the 3D subspace (projection) of the high-D embedding
    space where the target pattern is most visible.

    Strategies tried:
      - PCA top-3
      - PCA components 2-4, 3-5, etc.
      - Random orthogonal projections
      - Top-variance dimension triplets
    """
    d = embeddings.shape[1]
    gen = PATTERN_GENERATORS[pattern_name]
    pattern = gen(n_points=n_pattern_pts)

    best = {"error": float("inf"), "indices": None, "desc": "", "proj": None}

    def try_projection(proj_3d, desc):
        nonlocal best
        indices, _, error = multi_start_icp(pattern, proj_3d, n_starts=n_icp_starts)
        if error < best["error"]:
            best["error"] = error
            best["indices"] = indices
            best["desc"] = desc
            best["proj_3d"] = proj_3d

    # Strategy 1: PCA various component windows
    max_comp = min(20, d, embeddings.shape[0])
    pca = PCA(n_components=max_comp)
    all_pcs = pca.fit_transform(embeddings)  # (N, max_comp)

    for start in range(0, min(12, max_comp - 2)):
        proj = all_pcs[:, start:start + 3]
        try_projection(proj, f"PCA components {start + 1}–{start + 3}")

    # Strategy 2: Random orthogonal projections
    rng = np.random.RandomState(42)
    for i in range(n_random):
        mat = rng.randn(d, 3)
        Q, _ = np.linalg.qr(mat)
        proj = embeddings @ Q[:, :3]
        try_projection(proj, f"Random projection #{i + 1}")

    # Strategy 3: Top-variance dimension triplets
    variances = np.var(embeddings, axis=0)
    top_dims = np.argsort(variances)[-30:][::-1]
    tried = 0
    for i in range(min(15, len(top_dims))):
        for j in range(i + 1, min(15, len(top_dims))):
            for k in range(j + 1, min(8, len(top_dims))):
                dims = [top_dims[i], top_dims[j], top_dims[k]]
                proj = embeddings[:, dims]
                try_projection(proj, f"Dims [{dims[0]}, {dims[1]}, {dims[2]}]")
                tried += 1
                if tried > 60:
                    break
            if tried > 60:
                break
        if tried > 60:
            break

    return best


# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — orchestrates the full search
# ═══════════════════════════════════════════════════════════════════

def find_pattern_in_embeddings(
    model_key: str,
    pattern_name: str,
    n_pattern_points: int = 50,
    filter_type: str = "all",
    custom_regex: str = None,
    sample_size: int = 10000,
    search_method: str = "both",       # "icp", "greedy", "both"
    projection_method: str = "pca",    # for initial 3D projection
    search_subspaces: bool = False,    # try many 3D subspaces
    layer_hidden_states: np.ndarray = None,  # if searching a layer
    layer_tokens: List[str] = None,          # token strings for the layer
) -> Dict:
    """
    Find real tokens in the embedding/feature space whose positions
    form a specific geometric pattern.

    Returns a dict with:
      matched_tokens, matched_positions_3d, pattern_template,
      context cloud for background, similarity score, etc.
    """
    # ── Get embeddings & token labels ──────────────────────
    if layer_hidden_states is not None:
        embeddings = layer_hidden_states
        token_ids = list(range(embeddings.shape[0]))
        tokens = layer_tokens or [f"pos_{i}" for i in token_ids]
    else:
        filtered = filter_vocab_tokens(model_key, filter_type, custom_regex, sample_size)
        if len(filtered) < max(n_pattern_points, 10):
            return {"error": f"Only {len(filtered)} tokens match filter '{filter_type}', "
                             f"need ≥ {n_pattern_points}. Try a broader filter."}
        token_ids = [f[0] for f in filtered]
        tokens = [f[1] for f in filtered]
        emb_matrix = get_token_embedding_matrix(model_key)
        embeddings = emb_matrix[token_ids]

    # ── Generate pattern ──────────────────────────────────
    gen = PATTERN_GENERATORS.get(pattern_name)
    if gen is None:
        return {"error": f"Unknown pattern '{pattern_name}'. "
                         f"Available: {list(PATTERN_GENERATORS.keys())}"}
    pattern = gen(n_points=n_pattern_points)

    # ── Project / search ──────────────────────────────────
    search_info = {}
    if search_subspaces:
        result = search_best_subspace(embeddings, pattern_name,
                                       n_pattern_pts=n_pattern_points)
        emb_3d = result["proj_3d"]
        matched_indices = result["indices"]
        match_error = result["error"]
        search_info = {"strategy": "subspace_search",
                       "best_subspace": result["desc"]}
    else:
        from utils.projection import project_to_3d
        emb_3d = project_to_3d(embeddings, method=projection_method)

        icp_idx, icp_err = None, float("inf")
        greedy_chain, greedy_score = None, -1.0

        if search_method in ("icp", "both"):
            icp_idx, _, icp_err = multi_start_icp(pattern, emb_3d, n_starts=60)

        if search_method in ("greedy", "both"):
            greedy_chain, greedy_score = greedy_curve_match(pattern, emb_3d)

        # Pick winner
        greedy_err = 1.0 - greedy_score if greedy_chain else float("inf")
        if icp_err <= greedy_err and icp_idx is not None:
            matched_indices = icp_idx
            match_error = icp_err
            search_info["strategy"] = "icp"
        elif greedy_chain:
            matched_indices = np.array(greedy_chain)
            match_error = greedy_err
            search_info["strategy"] = "greedy"
        else:
            return {"error": "No matches found. Try a different filter or pattern."}

    # ── Build response ────────────────────────────────────
    valid = matched_indices[matched_indices >= 0] if matched_indices is not None else []
    valid = [int(v) for v in valid if 0 <= v < len(token_ids)]

    # Context cloud — random sample of all tokens for background
    rng = np.random.RandomState(42)
    ctx_n = min(3000, len(token_ids))
    ctx_idx = rng.choice(len(token_ids), size=ctx_n, replace=False)

    # Normalized pattern template for overlay
    pat_norm, _, _ = normalize_points(pattern)
    # Scale/translate to match the matched points in 3D for visual overlay
    if len(valid) >= 2:
        matched_pts = emb_3d[valid]
        m_mean = matched_pts.mean(axis=0)
        m_scale = np.std(matched_pts) + 1e-9
        pattern_overlay = pat_norm[:len(valid)] * m_scale + m_mean
    else:
        pattern_overlay = pat_norm

    return {
        # The discovered tokens
        "matched_token_ids": [int(token_ids[i]) for i in valid],
        "matched_tokens": [tokens[i] for i in valid],
        "matched_positions_3d": emb_3d[valid].tolist() if len(valid) > 0 else [],
        "n_matched": len(valid),

        # Pattern template (aligned to match for overlay)
        "pattern_overlay_3d": pattern_overlay.tolist(),
        "pattern_raw_3d": pat_norm.tolist(),

        # Background context cloud
        "context_positions_3d": emb_3d[ctx_idx].tolist(),
        "context_tokens": [tokens[i] for i in ctx_idx],
        "context_token_ids": [int(token_ids[i]) for i in ctx_idx],

        # Scores
        "match_error": float(match_error),
        "match_similarity": float(1.0 / (1.0 + match_error)),

        # Metadata
        "search_info": search_info,
        "n_searched": len(token_ids),
        "filter_type": filter_type,
        "pattern_name": pattern_name,
    }


# ═══════════════════════════════════════════════════════════════════
# SCAN: run pattern search for ALL patterns against a token set
# ═══════════════════════════════════════════════════════════════════

def scan_all_patterns(model_key: str,
                       filter_type: str = "all",
                       sample_size: int = 5000,
                       n_pattern_points: int = 40,
                       projection_method: str = "pca") -> List[Dict]:
    """
    For each known pattern, search the embedding space and rank
    by match quality. Answers: "what geometric structure does this
    token set most resemble?"
    """
    results = []
    for name in PATTERN_GENERATORS:
        try:
            out = find_pattern_in_embeddings(
                model_key=model_key,
                pattern_name=name,
                n_pattern_points=n_pattern_points,
                filter_type=filter_type,
                sample_size=sample_size,
                search_method="icp",
                projection_method=projection_method,
                search_subspaces=False,
            )
            if "error" not in out:
                results.append({
                    "pattern": name,
                    "similarity": out["match_similarity"],
                    "error": out["match_error"],
                    "n_matched": out["n_matched"],
                    "sample_tokens": out["matched_tokens"][:10],
                })
        except Exception as e:
            results.append({"pattern": name, "similarity": 0, "error": str(e)})

    results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return results
