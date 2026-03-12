"""
Pattern Finder v3 — CURVE-FITTING with inlier selection.

Key insight from v2's failure: ICP matches 1-to-1 with template POINTS,
so matched tokens can be far from the actual curve. 

v3 approach:
  1. Align a continuous curve to the token cloud (multi-start ICP)
  2. Compute distance from EVERY token to the continuous POLYLINE
  3. Select only tokens within a tight distance threshold (INLIERS)
  4. Iteratively refine: re-align curve to inliers, recompute, repeat
  5. Order inliers by their parameter along the curve
  6. Prune isolated tokens that don't have curve-neighbors

This gives tokens that truly LIE ON the manifold.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Generator
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

from utils.model_loader import load_tokenizer
from utils.embedding_utils import get_token_embedding_matrix


# ═══════════════════════════════════════════════════════════════════
# CACHING
# ═══════════════════════════════════════════════════════════════════

_filter_cache = {}
_projection_cache = {}


def clear_caches():
    _filter_cache.clear()
    _projection_cache.clear()


# ═══════════════════════════════════════════════════════════════════
# PATTERN GENERATORS
# ═══════════════════════════════════════════════════════════════════

def generate_spiral(n_points=100, **kw):
    t = np.linspace(0, 3 * 2 * np.pi, n_points)
    r = t / (3 * 2 * np.pi)
    return np.stack([r * np.cos(t), r * np.sin(t), t / (3 * 2 * np.pi)], axis=1)

def generate_circle(n_points=100, **kw):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.stack([np.cos(t), np.sin(t), np.zeros(n_points)], axis=1)

def generate_line(n_points=100, **kw):
    t = np.linspace(0, 1, n_points)
    return np.stack([t, np.zeros(n_points), np.zeros(n_points)], axis=1)

def generate_helix(n_points=100, turns=3.0, **kw):
    t = np.linspace(0, turns * 2 * np.pi, n_points)
    return np.stack([np.cos(t), np.sin(t), t / (turns * 2 * np.pi)], axis=1)

def generate_sphere(n_points=200, **kw):
    golden = (1 + np.sqrt(5)) / 2
    idx = np.arange(n_points)
    theta = 2 * np.pi * idx / golden
    phi = np.arccos(1 - 2 * (idx + 0.5) / n_points)
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)

def generate_grid(n_points=100, **kw):
    n = int(np.sqrt(n_points))
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    return np.stack([x.ravel(), y.ravel(), np.zeros(n * n)], axis=1)

def generate_clusters(n_points=100, **kw):
    rng = np.random.RandomState(42)
    nc, ppg = 5, max(1, n_points // 5)
    pts = [rng.randn(3) * 2 + 0.15 * rng.randn(ppg, 3) for _ in range(nc)]
    return np.vstack(pts)[:n_points]

def generate_figure_eight(n_points=100, **kw):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.stack([np.sin(t), np.sin(t) * np.cos(t), 0.3 * np.cos(t)], axis=1)

def generate_torus_knot(n_points=200, **kw):
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r = np.cos(3 * t) + 2
    return np.stack([r * np.cos(2 * t), r * np.sin(2 * t), -np.sin(3 * t)], axis=1)

PATTERN_GENERATORS = {
    "spiral": generate_spiral, "circle": generate_circle, "line": generate_line,
    "helix": generate_helix, "sphere": generate_sphere, "grid": generate_grid,
    "clusters": generate_clusters, "figure_eight": generate_figure_eight,
    "torus_knot": generate_torus_knot,
}

# Patterns that are curves (use polyline distance) vs point clouds
CURVE_PATTERNS = {"spiral", "circle", "line", "helix", "figure_eight", "torus_knot"}


# ═══════════════════════════════════════════════════════════════════
# CORE GEOMETRY: Polyline distance (the key improvement)
# ═══════════════════════════════════════════════════════════════════

def polyline_dists(points, curve):
    """
    For each point, compute distance to the NEAREST point on the
    continuous polyline defined by `curve`.

    This is the KEY DIFFERENCE from v2: we measure distance to the
    CURVE, not to discrete template points.

    points: (N, D)
    curve:  (M, D) — vertices of polyline

    Returns:
      dists:  (N,) — distance from each point to nearest curve segment
      params: (N,) — parameter along curve [0, M-1] of nearest point
      projs:  (N, D) — the actual nearest point on the curve (projection)
    """
    N = points.shape[0]
    M = curve.shape[0]
    D = points.shape[1]

    best_dists = np.full(N, np.inf)
    best_params = np.zeros(N)
    best_projs = np.zeros((N, D))

    for i in range(M - 1):
        a = curve[i]                           # (D,)
        ab = curve[i + 1] - a                  # (D,)
        ab_sq = np.dot(ab, ab) + 1e-12

        ap = points - a                        # (N, D)
        t = np.clip((ap @ ab) / ab_sq, 0, 1)  # (N,)
        proj = a + np.outer(t, ab)             # (N, D)
        d = np.linalg.norm(points - proj, axis=1)  # (N,)

        mask = d < best_dists
        best_dists[mask] = d[mask]
        best_params[mask] = i + t[mask]
        best_projs[mask] = proj[mask]

    return best_dists, best_params, best_projs


def point_cloud_dists(points, template):
    """
    For non-curve patterns (sphere, grid, clusters):
    distance from each point to nearest TEMPLATE POINT.

    Returns: dists (N,), nearest_idx (N,)
    """
    tree = KDTree(template)
    dists, idx = tree.query(points)
    return dists, idx


# ═══════════════════════════════════════════════════════════════════
# NORMALIZATION & TRANSFORMS
# ═══════════════════════════════════════════════════════════════════

def normalize(pts):
    """Zero-center, unit-scale."""
    m = pts.mean(axis=0)
    c = pts - m
    s = np.std(c) + 1e-9
    return c / s, m, s


def derive_transform(source, aligned):
    """
    Derive the rigid+scale transform that maps source -> aligned.
    Returns (R, scale, src_center, tgt_center).
    """
    mu_s = source.mean(axis=0)
    mu_a = aligned.mean(axis=0)
    sc = source - mu_s
    ac = aligned - mu_a
    H = sc.T @ ac
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    scale = np.sum(S) / (np.sum(sc ** 2) + 1e-9)
    return R, scale, mu_s, mu_a


def apply_transform(points, R, scale, src_center, tgt_center):
    """Apply rigid+scale transform to points."""
    return scale * ((points - src_center) @ R.T) + tgt_center


# ═══════════════════════════════════════════════════════════════════
# ICP (unchanged, but now returns aligned pattern for transform derivation)
# ═══════════════════════════════════════════════════════════════════

def fast_icp(pattern, target, tree, max_iter=25):
    """
    Standard ICP: align pattern to target using nearest neighbors.
    Returns: (aligned_pattern, mean_error)
    """
    cur = pattern.copy()
    for _ in range(max_iter):
        dists, idx = tree.query(cur)
        matched = target[idx]
        mu_c = cur.mean(axis=0)
        mu_m = matched.mean(axis=0)
        cc = cur - mu_c
        mc = matched - mu_m
        H = cc.T @ mc
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        scale = np.sum(S) / (np.sum(cc ** 2) + 1e-9)
        cur = scale * (cc @ R.T) + mu_m
    dists, _ = tree.query(cur)
    return cur, float(np.mean(dists))


# ═══════════════════════════════════════════════════════════════════
# SCORING — count inliers on the actual curve
# ═══════════════════════════════════════════════════════════════════

def compute_threshold(all_dists, n_pattern_pts, strictness=0.5):
    """
    Adaptive distance threshold for inlier selection.

    strictness 0.0 → keep ~3x pattern points (loose)
    strictness 0.5 → keep ~2x pattern points (balanced)
    strictness 1.0 → keep ~1x pattern points (strict)
    """
    n = len(all_dists)
    target = int(n_pattern_pts * (3.0 - 2.0 * strictness))
    target = max(n_pattern_pts // 2, min(target, n - 1))
    sorted_d = np.sort(all_dists)
    return float(sorted_d[target])


def score_curve_alignment(aligned_dense_curve, all_targets_n, n_pat, strictness):
    """
    Score an alignment by:
      1. Distance from ALL tokens to the continuous curve
      2. Number of inliers within threshold
      3. How well inliers COVER the full curve length
      4. Mean distance of inliers (tightness)

    Returns: (score, inlier_indices, dists, params, projections, threshold)
    """
    dists, params, projs = polyline_dists(all_targets_n, aligned_dense_curve)
    threshold = compute_threshold(dists, n_pat, strictness)

    inlier_mask = dists <= threshold
    inlier_idx = np.where(inlier_mask)[0]
    n_inliers = len(inlier_idx)

    if n_inliers < 3:
        return 0.0, inlier_idx, dists, params, projs, threshold

    # Coverage: what fraction of curve length do inliers span?
    max_param = len(aligned_dense_curve) - 1
    inlier_params = params[inlier_mask]
    coverage = (inlier_params.max() - inlier_params.min()) / (max_param + 1e-9)

    # Tightness: how close are inliers to the curve?
    mean_dist = np.mean(dists[inlier_mask])

    # Uniformity: are inliers evenly spread along the curve?
    sorted_ip = np.sort(inlier_params)
    if len(sorted_ip) > 1:
        gaps = np.diff(sorted_ip)
        mean_gap = np.mean(gaps)
        gap_cv = np.std(gaps) / (mean_gap + 1e-9)  # coefficient of variation
        uniformity = 1.0 / (1.0 + gap_cv)
    else:
        uniformity = 0.0

    # Combined score (higher = better)
    score = n_inliers * coverage * uniformity / (1.0 + mean_dist * 20.0)

    return score, inlier_idx, dists, params, projs, threshold


def score_pointcloud_alignment(aligned_template, all_targets_n, n_pat, strictness):
    """Score for non-curve patterns (sphere, grid, clusters)."""
    dists, _ = point_cloud_dists(all_targets_n, aligned_template)
    threshold = compute_threshold(dists, n_pat, strictness)
    inlier_mask = dists <= threshold
    inlier_idx = np.where(inlier_mask)[0]
    n_inliers = len(inlier_idx)
    if n_inliers < 3:
        return 0.0, inlier_idx, dists, np.zeros(len(dists)), all_targets_n, threshold
    mean_dist = np.mean(dists[inlier_mask])
    score = n_inliers / (1.0 + mean_dist * 20.0)
    return score, inlier_idx, dists, np.zeros(len(dists)), all_targets_n, threshold


# ═══════════════════════════════════════════════════════════════════
# VOCABULARY FILTERING (cached, fast)
# ═══════════════════════════════════════════════════════════════════

import re

def filter_vocab_tokens(model_key, filter_type="all", custom_regex=None,
                         max_tokens=30000):
    """Returns (token_ids, token_strings). Cached."""
    key = (model_key, filter_type, custom_regex or "", max_tokens)
    if key in _filter_cache:
        return _filter_cache[key]

    tokenizer = load_tokenizer(model_key)
    all_toks = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]

    ids, toks = [], []
    for i, tok in enumerate(all_toks):
        s = tok.strip().lower()
        keep = False
        if filter_type == "all":
            keep = True
        elif filter_type == "numbers":
            try:
                float(s)
                keep = all(c in '0123456789.-' for c in s)
            except (ValueError, TypeError):
                keep = False
        elif filter_type == "years":
            if len(s) == 4 and all(c in '0123456789' for c in s):
                try:
                    keep = 1000 <= int(s) <= 2100
                except ValueError:
                    keep = False
            keep = s.isalpha() and len(s) >= 2
        elif filter_type == "letters":
            keep = s.isalpha() and len(s) == 1
        elif filter_type == "months":
            keep = s in {"january","february","march","april","may","june","july",
                         "august","september","october","november","december",
                         "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec"}
        elif filter_type == "colors":
            keep = s in {"red","orange","yellow","green","blue","purple","violet","pink",
                         "brown","black","white","gray","grey","cyan","magenta","indigo",
                         "teal","crimson","scarlet","gold"}
        elif filter_type == "regex" and custom_regex:
            try:
                keep = bool(re.search(custom_regex, tok))
            except re.error:
                pass
        if keep:
            ids.append(i)
            toks.append(tok)
            if len(ids) >= max_tokens:
                break

    if filter_type == "all" and len(ids) > max_tokens:
        rng = np.random.RandomState(42)
        sel = rng.choice(len(ids), size=max_tokens, replace=False)
        sel.sort()
        ids = [ids[j] for j in sel]
        toks = [toks[j] for j in sel]

    _filter_cache[key] = (ids, toks)
    return ids, toks


def get_filtered_3d(model_key, filter_type, custom_regex=None,
                     max_tokens=30000, projection="pca"):
    """Get filtered tokens with 3D PCA projection. Cached."""
    ck = (model_key, filter_type, custom_regex or "", max_tokens, projection)
    if ck in _projection_cache:
        return _projection_cache[ck]

    ids, toks = filter_vocab_tokens(model_key, filter_type, custom_regex, max_tokens)
    emb = get_token_embedding_matrix(model_key)[ids]

    if emb.shape[0] >= 3:
        pca = PCA(n_components=3)
        emb_3d = pca.fit_transform(emb)
    else:
        emb_3d = np.zeros((emb.shape[0], 3))

    result = (ids, toks, emb, emb_3d)
    _projection_cache[ck] = result
    return result


# ═══════════════════════════════════════════════════════════════════
# ITERATIVE REFINEMENT — the quality step
# ═══════════════════════════════════════════════════════════════════

def refine_alignment(pattern_coarse_n, pattern_dense_n,
                      target_all_n, inlier_idx, tree_all,
                      n_pat, strictness, is_curve, max_iter=30):
    """
    Refinement step:
      1. Build KDTree of just the inlier tokens
      2. Re-run ICP against inliers only → curve moves toward the inlier manifold
      3. Derive transform, apply to dense curve
      4. Re-score against ALL tokens
      5. Return new score + inliers
    """
    if len(inlier_idx) < 3:
        return None, 0, inlier_idx

    inlier_pts = target_all_n[inlier_idx]
    inlier_tree = KDTree(inlier_pts)

    aligned_coarse, _ = fast_icp(pattern_coarse_n, inlier_pts, inlier_tree, max_iter)
    R, scale, src_c, tgt_c = derive_transform(pattern_coarse_n, aligned_coarse)
    aligned_dense = apply_transform(pattern_dense_n, R, scale, src_c, tgt_c)

    if is_curve:
        score, new_inliers, dists, params, projs, thresh = \
            score_curve_alignment(aligned_dense, target_all_n, n_pat, strictness)
    else:
        score, new_inliers, dists, params, projs, thresh = \
            score_pointcloud_alignment(aligned_dense, target_all_n, n_pat, strictness)

    return aligned_dense, score, new_inliers, dists, params, projs, thresh


# ═══════════════════════════════════════════════════════════════════
# INLIER ORDERING & PRUNING
# ═══════════════════════════════════════════════════════════════════

def order_and_prune_inliers(inlier_idx, params, dists, max_gap_factor=3.0):
    """
    Order inlier tokens along the curve parameter.
    Prune isolated tokens that have huge gaps to their neighbors.

    max_gap_factor: a token is isolated if the gap to its neighbor
    is > max_gap_factor * median_gap.
    """
    if len(inlier_idx) < 2:
        return inlier_idx, params[inlier_idx], dists[inlier_idx]

    inlier_params = params[inlier_idx]
    order = np.argsort(inlier_params)
    ordered_idx = inlier_idx[order]
    ordered_params = inlier_params[order]
    ordered_dists = dists[inlier_idx[order]]

    if len(ordered_idx) < 4:
        return ordered_idx, ordered_params, ordered_dists

    # Compute gaps
    gaps = np.diff(ordered_params)
    median_gap = np.median(gaps)
    max_gap = median_gap * max_gap_factor

    # Find the longest contiguous run without huge gaps
    runs = []
    run_start = 0
    for i, g in enumerate(gaps):
        if g > max_gap:
            if i - run_start >= 2:
                runs.append((run_start, i + 1))
            run_start = i + 1
    if len(ordered_idx) - run_start >= 2:
        runs.append((run_start, len(ordered_idx)))

    if not runs:
        return ordered_idx, ordered_params, ordered_dists

    # Pick longest run
    best_run = max(runs, key=lambda r: r[1] - r[0])
    s, e = best_run

    return ordered_idx[s:e], ordered_params[s:e], ordered_dists[s:e]


# ═══════════════════════════════════════════════════════════════════
# PROGRESSIVE SEARCH GENERATOR
# ═══════════════════════════════════════════════════════════════════

def progressive_search(
    model_key: str,
    pattern_name: str,
    n_pattern_points: int = 40,
    filter_type: str = "all",
    custom_regex: str = None,
    sample_size: int = 10000,
    projection: str = "pca",
    search_subspaces: bool = False,
    strictness: float = 0.5,
    # For layer search:
    layer_hidden_states: np.ndarray = None,
    layer_tokens: List[str] = None,
) -> Generator[Dict, None, None]:
    """
    Generator that yields live updates as the search runs.
    """
    t0 = time.time()
    is_curve = pattern_name in CURVE_PATTERNS

    # ── 1. Load data ──────────────────────────────────
    yield {"status": "progress", "phase": "loading", "message": "Loading embeddings..."}

    if layer_hidden_states is not None:
        emb_hd = layer_hidden_states
        token_ids = list(range(emb_hd.shape[0]))
        tokens = layer_tokens or [f"pos_{i}" for i in token_ids]
        if emb_hd.shape[0] >= 3:
            emb_3d = PCA(n_components=3).fit_transform(emb_hd)
        else:
            emb_3d = np.zeros((emb_hd.shape[0], 3))
    else:
        token_ids, tokens, emb_hd, emb_3d = get_filtered_3d(
            model_key, filter_type, custom_regex, sample_size, projection
        )

    n_tokens = len(token_ids)
    if n_tokens < max(n_pattern_points, 5):
        yield {"status": "error",
               "message": f"Only {n_tokens} tokens match. Need ≥ {n_pattern_points}."}
        return

    yield {"status": "progress", "phase": "loading",
           "message": f"Loaded {n_tokens} tokens."}

    # ── 2. Generate pattern ───────────────────────────
    gen = PATTERN_GENERATORS.get(pattern_name)
    if gen is None:
        yield {"status": "error", "message": f"Unknown pattern: {pattern_name}"}
        return

    # Coarse pattern for ICP, dense pattern for distance computation
    pat_coarse = gen(n_points=n_pattern_points)
    pat_dense = gen(n_points=n_pattern_points * 5)  # Smooth curve for distance

    # Normalize everything
    pat_coarse_n, pc_m, pc_s = normalize(pat_coarse)
    pat_dense_n = (pat_dense - pc_m) / pc_s  # Same transform as coarse
    tn, t_mean, t_scale = normalize(emb_3d)
    tree_n = KDTree(tn)

    # ── 3. Context cloud for background ───────────────
    rng = np.random.RandomState(42)
    ctx_n = min(3000, n_tokens)
    ctx_idx = rng.choice(n_tokens, size=ctx_n, replace=False)
    context = {
        "positions_3d": emb_3d[ctx_idx].tolist(),
        "tokens": [tokens[i] for i in ctx_idx],
        "token_ids": [int(token_ids[i]) for i in ctx_idx],
    }

    yield {"status": "progress", "phase": "searching",
           "message": "Starting search...", "context": context}

    # ── Helper: build result dict ─────────────────────
    def build_result(inlier_idx, aligned_dense_n, all_dists, all_params,
                     all_projs, threshold, strategy, is_final=False):
        # Un-normalize curve and projections
        curve_real = aligned_dense_n * t_scale + t_mean

        # Order and prune inliers
        ordered_idx, ordered_params, ordered_dists = \
            order_and_prune_inliers(inlier_idx, all_params, all_dists)

        valid = [int(v) for v in ordered_idx]
        matched_pts = emb_3d[valid] if valid else np.zeros((0, 3))

        # Projection points (nearest point on curve per inlier)
        if len(valid) > 0:
            proj_pts = (all_projs[valid] * t_scale + t_mean).tolist()
        else:
            proj_pts = []

        return {
            "status": "best_update" if not is_final else "done",
            "matched_token_ids": [int(token_ids[i]) for i in valid],
            "matched_tokens": [tokens[i] for i in valid],
            "matched_positions_3d": matched_pts.tolist(),
            "n_matched": len(valid),
            "aligned_curve_3d": curve_real.tolist(),
            "projection_points_3d": proj_pts,
            "inlier_distances": ordered_dists.tolist(),
            "curve_parameters": ordered_params.tolist(),
            "inlier_threshold": float(threshold),
            "match_error": float(np.mean(ordered_dists)) if len(ordered_dists) > 0 else 999,
            "match_similarity": float(1.0 / (1.0 + np.mean(ordered_dists))) if len(ordered_dists) > 0 else 0,
            "search_info": {"strategy": strategy},
            "elapsed": round(time.time() - t0, 2),
            "context": context,
            "n_searched": n_tokens,
            "filter_type": filter_type,
            "pattern_name": pattern_name,
            "strictness": strictness,
        }

    # ── 4. COARSE ICP SCAN ────────────────────────────
    n_coarse = 100
    best_score = -1
    best_aligned_dense = None
    best_inliers = np.array([], dtype=int)
    best_dists = np.zeros(n_tokens)
    best_params = np.zeros(n_tokens)
    best_projs = np.zeros((n_tokens, 3))
    best_thresh = 0
    best_strategy = ""
    coarse_results = []

    last_yield_time = time.time()

    for trial in range(n_coarse):
        r = np.random.RandomState(trial)
        # Random rotation via QR
        Q, _ = np.linalg.qr(r.randn(3, 3))
        scale = r.uniform(0.3, 2.5)
        offset = r.uniform(tn.min(axis=0) - 0.5, tn.max(axis=0) + 0.5)
        init = scale * (pat_coarse_n @ Q.T) + offset

        aligned_coarse, icp_err = fast_icp(init, tn, tree_n, max_iter=12)

        # Derive transform, apply to dense curve
        R, sc, src_c, tgt_c = derive_transform(pat_coarse_n, aligned_coarse)
        aligned_dense = apply_transform(pat_dense_n, R, sc, src_c, tgt_c)

        # Score by inlier count on continuous curve
        if is_curve:
            score, inliers, dists, params, projs, thresh = \
                score_curve_alignment(aligned_dense, tn, n_pattern_points, strictness)
        else:
            score, inliers, dists, params, projs, thresh = \
                score_pointcloud_alignment(aligned_dense, tn, n_pattern_points, strictness)

        coarse_results.append((trial, score, aligned_coarse, aligned_dense,
                                inliers, dists, params, projs, thresh))

        if score > best_score:
            best_score = score
            best_aligned_dense = aligned_dense
            best_inliers = inliers
            best_dists = dists
            best_params = params
            best_projs = projs
            best_thresh = thresh
            best_strategy = f"ICP coarse #{trial}"

            # Live update (throttled)
            now = time.time()
            if now - last_yield_time > 0.3 and len(best_inliers) >= 3:
                last_yield_time = now
                yield build_result(best_inliers, best_aligned_dense,
                                    best_dists, best_params, best_projs,
                                    best_thresh, best_strategy)

        if (trial + 1) % 10 == 0:
            yield {"status": "progress", "phase": "coarse",
                   "message": f"Coarse: {trial+1}/{n_coarse} | "
                              f"best: {len(best_inliers)} inliers (score {best_score:.1f})",
                   "trial": trial + 1, "total": n_coarse}

    # ── 5. REFINE TOP CANDIDATES ──────────────────────
    coarse_results.sort(key=lambda x: x[1], reverse=True)
    top_k = min(12, len(coarse_results))

    yield {"status": "progress", "phase": "fine",
           "message": f"Refining top {top_k} candidates...",
           "trial": 0, "total": top_k}

    for rank in range(top_k):
        trial, _, aligned_coarse, aligned_dense, _, _, _, _, _ = coarse_results[rank]

        # Re-run ICP with more iterations
        R, sc, src_c, tgt_c = derive_transform(pat_coarse_n, aligned_coarse)
        reinit = apply_transform(pat_coarse_n, R, sc, src_c, tgt_c)
        refined_coarse, _ = fast_icp(reinit, tn, tree_n, max_iter=50)

        R2, sc2, src_c2, tgt_c2 = derive_transform(pat_coarse_n, refined_coarse)
        refined_dense = apply_transform(pat_dense_n, R2, sc2, src_c2, tgt_c2)

        if is_curve:
            score, inliers, dists, params, projs, thresh = \
                score_curve_alignment(refined_dense, tn, n_pattern_points, strictness)
        else:
            score, inliers, dists, params, projs, thresh = \
                score_pointcloud_alignment(refined_dense, tn, n_pattern_points, strictness)

        if score > best_score:
            best_score = score
            best_aligned_dense = refined_dense
            best_inliers = inliers
            best_dists = dists
            best_params = params
            best_projs = projs
            best_thresh = thresh
            best_strategy = f"Refined coarse #{trial}"

            yield build_result(best_inliers, best_aligned_dense,
                                best_dists, best_params, best_projs,
                                best_thresh, best_strategy)

        yield {"status": "progress", "phase": "fine",
               "message": f"Refining: {rank+1}/{top_k} | "
                          f"best: {len(best_inliers)} inliers",
               "trial": rank + 1, "total": top_k}

    # ── 6. ITERATIVE INLIER REFINEMENT ────────────────
    # The key quality step: re-align curve to just the inliers,
    # then recompute inliers against all tokens. Repeat.
    yield {"status": "progress", "phase": "refine",
           "message": "Iterative inlier refinement..."}

    for rnd in range(5):
        if len(best_inliers) < 3:
            break

        result = refine_alignment(
            pat_coarse_n, pat_dense_n, tn, best_inliers, tree_n,
            n_pattern_points, strictness, is_curve, max_iter=40
        )
        if result[0] is None:
            break

        new_dense, new_score, new_inliers, new_dists, new_params, new_projs, new_thresh = result

        if new_score > best_score:
            best_score = new_score
            best_aligned_dense = new_dense
            best_inliers = new_inliers
            best_dists = new_dists
            best_params = new_params
            best_projs = new_projs
            best_thresh = new_thresh
            best_strategy = f"Inlier refinement round {rnd + 1}"

            yield build_result(best_inliers, best_aligned_dense,
                                best_dists, best_params, best_projs,
                                best_thresh, best_strategy)
        else:
            break  # No improvement, stop

        yield {"status": "progress", "phase": "refine",
               "message": f"Refinement round {rnd+1}/5 | "
                          f"{len(best_inliers)} inliers | score {best_score:.1f}"}

    # ── 7. OPTIONAL: SUBSPACE SEARCH ─────────────────
    if search_subspaces and emb_hd.shape[1] > 3:
        yield {"status": "progress", "phase": "subspace",
               "message": "Trying alternate 3D subspaces..."}

        max_comp = min(15, emb_hd.shape[1], emb_hd.shape[0])
        pca_full = PCA(n_components=max_comp)
        all_pcs = pca_full.fit_transform(emb_hd)

        for start in range(0, min(10, max_comp - 2)):
            sub_3d = all_pcs[:, start:start + 3]
            sub_n, sub_m, sub_s = normalize(sub_3d)
            sub_tree = KDTree(sub_n)

            for trial in range(8):
                r = np.random.RandomState(2000 + start * 100 + trial)
                Q, _ = np.linalg.qr(r.randn(3, 3))
                sc = r.uniform(0.5, 2.0)
                off = r.uniform(sub_n.min(0) - 0.5, sub_n.max(0) + 0.5)
                init = sc * (pat_coarse_n @ Q.T) + off

                aligned_c, _ = fast_icp(init, sub_n, sub_tree, 20)
                R, s, sc2, tc2 = derive_transform(pat_coarse_n, aligned_c)
                aligned_d = apply_transform(pat_dense_n, R, s, sc2, tc2)

                if is_curve:
                    score, inliers, dists, params, projs, thresh = \
                        score_curve_alignment(aligned_d, sub_n, n_pattern_points, strictness)
                else:
                    score, inliers, dists, params, projs, thresh = \
                        score_pointcloud_alignment(aligned_d, sub_n, n_pattern_points, strictness)

                if score > best_score:
                    best_score = score
                    # Project back to original 3D for visualization
                    best_aligned_dense = aligned_d * sub_s + sub_m
                    # Also re-project the embedding space
                    tn_disp = sub_n  # display coords
                    t_mean_disp = sub_m
                    t_scale_disp = sub_s
                    best_inliers = inliers
                    best_dists = dists
                    best_params = params
                    best_projs = projs
                    best_thresh = thresh
                    best_strategy = f"Subspace PCA[{start+1}:{start+4}] trial {trial}"

                    # Rebuild context in this subspace
                    context["positions_3d"] = sub_3d[ctx_idx].tolist()
                    emb_3d = sub_3d
                    tn = sub_n
                    t_mean = sub_m
                    t_scale = sub_s

                    yield build_result(best_inliers, best_aligned_dense,
                                        best_dists, best_params, best_projs,
                                        best_thresh, best_strategy)

            yield {"status": "progress", "phase": "subspace",
                   "message": f"Subspace PCA[{start+1}:{start+4}] done"}

    # ── 8. DONE ──────────────────────────────────────
    if len(best_inliers) >= 3:
        yield build_result(best_inliers, best_aligned_dense,
                            best_dists, best_params, best_projs,
                            best_thresh, best_strategy, is_final=True)
    else:
        yield {"status": "done", "match_similarity": 0, "n_matched": 0,
               "message": "No good match found. Try adjusting strictness or filter."}


# ═══════════════════════════════════════════════════════════════════
# SYNC WRAPPERS
# ═══════════════════════════════════════════════════════════════════

def find_pattern_sync(model_key, pattern_name, **kwargs) -> Dict:
    last = {"status": "error", "message": "No result"}
    for update in progressive_search(model_key, pattern_name, **kwargs):
        if update.get("status") in ("best_update", "done"):
            last = update
    return last


def scan_all_patterns(model_key, filter_type="all", sample_size=5000,
                       n_pattern_points=30, projection="pca",
                       strictness=0.5) -> List[Dict]:
    results = []
    for name in PATTERN_GENERATORS:
        try:
            out = find_pattern_sync(
                model_key, name,
                n_pattern_points=n_pattern_points,
                filter_type=filter_type,
                sample_size=sample_size,
                projection=projection,
                strictness=strictness,
            )
            if out.get("status") != "error":
                results.append({
                    "pattern": name, "similarity": out.get("match_similarity", 0),
                    "error": out.get("match_error", 999),
                    "n_matched": out.get("n_matched", 0),
                    "sample_tokens": out.get("matched_tokens", [])[:10],
                })
        except Exception as e:
            results.append({"pattern": name, "similarity": 0, "error": str(e)})
    results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return results

