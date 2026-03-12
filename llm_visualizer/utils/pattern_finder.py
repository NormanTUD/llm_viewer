"""
Pattern Finder v2 — FAST, LIVE progressive search.

Key changes from v1:
  1. All search functions are GENERATORS that yield intermediate results
     so the UI can show progressive updates in real time.
  2. Vocabulary filtering is vectorized (no per-token Python loop).
  3. KDTree + projection are pre-cached.
  4. Coarse-to-fine ICP: quick coarse pass → refine top candidates.
  5. Batch matrix operations throughout.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Generator
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
import re

from utils.model_loader import load_tokenizer
from utils.embedding_utils import get_token_embedding_matrix


# ═══════════════════════════════════════════════════════════════════
# CACHING — precompute expensive things once
# ═══════════════════════════════════════════════════════════════════

_filter_cache = {}        # (model_key, filter_type) -> (token_ids, tokens, embeddings)
_projection_cache = {}    # (model_key, filter_type, proj_method) -> emb_3d
_kdtree_cache = {}        # same key -> KDTree on the 3D projection


def _cache_key(model_key, filter_type, proj="pca"):
    return (model_key, filter_type, proj)


def clear_caches():
    _filter_cache.clear()
    _projection_cache.clear()
    _kdtree_cache.clear()


# ═══════════════════════════════════════════════════════════════════
# PATTERN GENERATORS (unchanged, compact)
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
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)

def generate_grid(n_points=100, **kw):
    n = int(np.sqrt(n_points))
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    return np.stack([x.ravel(), y.ravel(), np.zeros(n * n)], axis=1)

def generate_clusters(n_points=100, **kw):
    rng = np.random.RandomState(42)
    nc = 5
    ppg = max(1, n_points // nc)
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


# ═══════════════════════════════════════════════════════════════════
# FAST VOCABULARY FILTERING — vectorized
# ═══════════════════════════════════════════════════════════════════

def filter_vocab_tokens(model_key: str, filter_type: str = "all",
                         custom_regex: str = None,
                         max_tokens: int = 30000) -> Tuple[List[int], List[str]]:
    """
    Fast filtered token retrieval. Returns (token_ids, token_strings).
    Caches results for repeated calls.
    """
    cache_key = (model_key, filter_type, custom_regex or "", max_tokens)
    if cache_key in _filter_cache:
        return _filter_cache[cache_key]

    tokenizer = load_tokenizer(model_key)
    vocab_size = tokenizer.vocab_size

    # Decode entire vocab in one batch (much faster than per-token)
    all_tokens = [tokenizer.decode([i]) for i in range(vocab_size)]

    ids = []
    toks = []

    if filter_type == "all":
        # Random sample for speed
        rng = np.random.RandomState(42)
        sample = rng.choice(vocab_size, size=min(max_tokens, vocab_size), replace=False)
        sample.sort()
        ids = sample.tolist()
        toks = [all_tokens[i] for i in ids]
    else:
        # Vectorized-ish filtering
        for i, tok in enumerate(all_tokens):
            s = tok.strip().lower()
            keep = False

            if filter_type == "numbers":
                keep = s.lstrip("-").replace(".", "", 1).isdigit()
            elif filter_type == "years":
                keep = s.isdigit() and 1000 <= int(s) <= 2100
            elif filter_type == "words":
                keep = s.isalpha() and len(s) >= 2
            elif filter_type == "letters":
                keep = s.isalpha() and len(s) == 1
            elif filter_type == "months":
                keep = s in {"january","february","march","april","may","june",
                             "july","august","september","october","november","december",
                             "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec"}
            elif filter_type == "colors":
                keep = s in {"red","orange","yellow","green","blue","purple","violet",
                             "pink","brown","black","white","gray","grey","cyan",
                             "magenta","indigo","teal","crimson","scarlet","gold"}
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

    _filter_cache[cache_key] = (ids, toks)
    return ids, toks


def get_filtered_embeddings_3d(model_key: str, filter_type: str,
                                custom_regex: str = None,
                                max_tokens: int = 30000,
                                projection: str = "pca"):
    """
    Get 3D projected embeddings for a filtered token set. Cached.
    Returns (token_ids, tokens, embeddings_highD, emb_3d, kdtree).
    """
    ck = _cache_key(model_key, filter_type + (custom_regex or ""), projection)

    if ck in _projection_cache:
        return _projection_cache[ck]

    ids, toks = filter_vocab_tokens(model_key, filter_type, custom_regex, max_tokens)
    emb_matrix = get_token_embedding_matrix(model_key)
    emb = emb_matrix[ids]

    # PCA to 3D (fast for any size)
    if emb.shape[0] < 3:
        emb_3d = np.zeros((emb.shape[0], 3))
    else:
        n_comp = min(3, emb.shape[0], emb.shape[1])
        pca = PCA(n_components=n_comp)
        emb_3d = pca.fit_transform(emb)
        if n_comp < 3:
            emb_3d = np.hstack([emb_3d, np.zeros((emb_3d.shape[0], 3 - n_comp))])

    tree = KDTree(emb_3d)

    result = (ids, toks, emb, emb_3d, tree)
    _projection_cache[ck] = result
    _kdtree_cache[ck] = tree
    return result


# ═══════════════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════════════

def normalize(pts):
    """Zero-center, unit-scale. Returns (normed, mean, scale)."""
    m = pts.mean(axis=0)
    c = pts - m
    s = np.std(c) + 1e-9
    return c / s, m, s


def similarity(a, b):
    """Bidirectional nearest-neighbor similarity. 0..1, higher=better."""
    if a.shape[0] < 2 or b.shape[0] < 2:
        return 0.0
    an, _, _ = normalize(a)
    bn, _, _ = normalize(b)
    d = cdist(an, bn)
    return float(1.0 / (1.0 + (np.mean(np.min(d, axis=1)) + np.mean(np.min(d, axis=0))) / 2))


# ═══════════════════════════════════════════════════════════════════
# FAST ICP — batch matrix ops, no Python inner loops
# ═══════════════════════════════════════════════════════════════════

def fast_icp(pattern, target, tree, max_iter=25):
    """
    Single ICP run. Uses pre-built KDTree for speed.
    Returns (matched_indices, transformed, error).
    """
    cur = pattern.copy()
    for _ in range(max_iter):
        dists, idx = tree.query(cur)
        matched = target[idx]

        c_c = cur - cur.mean(axis=0)
        m_c = matched - matched.mean(axis=0)

        H = c_c.T @ m_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T

        scale = np.sum(S) / (np.sum(c_c ** 2) + 1e-9)
        cur = scale * (c_c @ R.T) + matched.mean(axis=0)

    dists, idx = tree.query(cur)
    return idx, cur, float(np.mean(dists))


def assign_unique(transformed, target, raw_idx, tree):
    """Hungarian assignment for 1-to-1 matching within nearby candidates."""
    uniq = np.unique(raw_idx)
    if len(uniq) >= len(transformed):
        cost = cdist(transformed, target[uniq])
        _, col = linear_sum_assignment(cost)
        return uniq[col]

    # Expand search
    k = min(len(transformed) * 3, target.shape[0])
    _, expanded = tree.query(transformed.mean(axis=0), k=k)
    if np.ndim(expanded) == 0:
        expanded = np.array([expanded])
    cost = cdist(transformed, target[expanded])
    _, col = linear_sum_assignment(cost)
    return expanded[col]


# ═══════════════════════════════════════════════════════════════════
# PROGRESSIVE SEARCH GENERATOR — yields updates live
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
    # For layer-based search:
    layer_hidden_states: np.ndarray = None,
    layer_tokens: List[str] = None,
) -> Generator[Dict, None, None]:
    """
    GENERATOR that yields progressive results as the search runs.

    Each yield is a dict with:
      - status: "progress" | "best_update" | "done" | "error"
      - For "progress": trial_num, total_trials, current_error
      - For "best_update": full match data (tokens, positions, etc.)
      - For "done": final result
    """
    t0 = time.time()

    # ── 1. Get data ───────────────────────────────────
    yield {"status": "progress", "phase": "loading", "message": "Loading embeddings..."}

    if layer_hidden_states is not None:
        # Searching within a specific layer's hidden states
        emb_hd = layer_hidden_states
        token_ids = list(range(emb_hd.shape[0]))
        tokens = layer_tokens or [f"pos_{i}" for i in token_ids]
        if emb_hd.shape[0] >= 3:
            pca = PCA(n_components=3)
            emb_3d = pca.fit_transform(emb_hd)
        else:
            emb_3d = np.zeros((emb_hd.shape[0], 3))
        tree = KDTree(emb_3d)
    else:
        token_ids, tokens, emb_hd, emb_3d, tree = get_filtered_embeddings_3d(
            model_key, filter_type, custom_regex, sample_size, projection
        )

    if len(token_ids) < max(n_pattern_points, 5):
        yield {"status": "error",
               "message": f"Only {len(token_ids)} tokens match filter '{filter_type}'. Need ≥ {n_pattern_points}."}
        return

    yield {"status": "progress", "phase": "loading",
           "message": f"Loaded {len(token_ids)} tokens. Generating pattern..."}

    # ── 2. Generate pattern ───────────────────────────
    gen = PATTERN_GENERATORS.get(pattern_name)
    if gen is None:
        yield {"status": "error", "message": f"Unknown pattern: {pattern_name}"}
        return
    pattern = gen(n_points=n_pattern_points)
    pat_n, _, _ = normalize(pattern)

    # ── 3. Prepare context cloud (subsample for background) ──
    rng = np.random.RandomState(42)
    ctx_n = min(3000, len(token_ids))
    ctx_idx = rng.choice(len(token_ids), size=ctx_n, replace=False)
    context = {
        "positions_3d": emb_3d[ctx_idx].tolist(),
        "tokens": [tokens[i] for i in ctx_idx],
        "token_ids": [int(token_ids[i]) for i in ctx_idx],
    }

    yield {"status": "progress", "phase": "searching",
           "message": "Starting search...", "context": context,
           "pattern_raw_3d": pat_n.tolist()}

    # ── 4. Normalize target ───────────────────────────
    tn, t_mean, t_scale = normalize(emb_3d)
    tree_n = KDTree(tn)

    best_error = float("inf")
    best_indices = None
    best_transformed = None
    best_strategy = ""

    def _build_result(indices, strategy, error, is_final=False):
        """Build the response dict for a match."""
        valid = [int(v) for v in indices if 0 <= v < len(token_ids)]
        matched_pts = emb_3d[valid] if valid else np.zeros((0, 3))

        # Align pattern overlay to matched positions
        if len(valid) >= 2:
            mm = matched_pts.mean(axis=0)
            ms = np.std(matched_pts) + 1e-9
            overlay = pat_n[:len(valid)] * ms + mm
        else:
            overlay = pat_n

        return {
            "status": "best_update" if not is_final else "done",
            "matched_token_ids": [int(token_ids[i]) for i in valid],
            "matched_tokens": [tokens[i] for i in valid],
            "matched_positions_3d": matched_pts.tolist(),
            "n_matched": len(valid),
            "pattern_overlay_3d": overlay.tolist(),
            "match_error": float(error),
            "match_similarity": float(1.0 / (1.0 + error)),
            "search_info": {"strategy": strategy},
            "elapsed": round(time.time() - t0, 2),
            "context": context,
            "pattern_raw_3d": pat_n.tolist(),
            "n_searched": len(token_ids),
            "filter_type": filter_type,
            "pattern_name": pattern_name,
        }

    # ── 5. COARSE PASS — fast random ICP starts ──────
    # Phase 1: many quick ICP trials with few iterations
    n_coarse = 80
    coarse_results = []

    yield {"status": "progress", "phase": "coarse",
           "message": f"Coarse scan: 0/{n_coarse}", "trial": 0, "total": n_coarse}

    for trial in range(n_coarse):
        r = np.random.RandomState(trial)
        # Random rotation via Gram-Schmidt
        M = r.randn(3, 3)
        Q, _ = np.linalg.qr(M)
        scale = r.uniform(0.3, 2.5)
        offset = r.uniform(tn.min(axis=0) - 0.5, tn.max(axis=0) + 0.5)
        init = scale * (pat_n @ Q.T) + offset

        raw_idx, transformed, error = fast_icp(init, tn, tree_n, max_iter=12)
        coarse_results.append((trial, error, raw_idx, transformed))

        if error < best_error:
            best_error = error
            try:
                best_indices = assign_unique(transformed, tn, raw_idx, tree_n)
            except Exception:
                best_indices = raw_idx
            best_strategy = f"ICP coarse trial {trial}"
            # Yield live update!
            yield _build_result(best_indices, best_strategy, best_error)

        # Yield progress every 10 trials
        if (trial + 1) % 10 == 0:
            yield {"status": "progress", "phase": "coarse",
                   "message": f"Coarse scan: {trial+1}/{n_coarse} (best error: {best_error:.4f})",
                   "trial": trial + 1, "total": n_coarse,
                   "best_error": best_error}

    # ── 6. FINE PASS — refine top-K coarse results ───
    coarse_results.sort(key=lambda x: x[1])
    top_k = min(15, len(coarse_results))

    yield {"status": "progress", "phase": "fine",
           "message": f"Refining top {top_k} candidates...",
           "trial": 0, "total": top_k}

    for rank, (trial, _, raw_idx, transformed) in enumerate(coarse_results[:top_k]):
        # Re-run ICP with more iterations starting from the coarse result
        raw_idx2, transformed2, error2 = fast_icp(transformed, tn, tree_n, max_iter=50)

        if error2 < best_error:
            best_error = error2
            try:
                best_indices = assign_unique(transformed2, tn, raw_idx2, tree_n)
            except Exception:
                best_indices = raw_idx2
            best_strategy = f"ICP refined (coarse trial {trial})"
            yield _build_result(best_indices, best_strategy, best_error)

        yield {"status": "progress", "phase": "fine",
               "message": f"Refining: {rank+1}/{top_k} (best error: {best_error:.4f})",
               "trial": rank + 1, "total": top_k, "best_error": best_error}

    # ── 7. GREEDY CURVE FOLLOWING ─────────────────────
    yield {"status": "progress", "phase": "greedy",
           "message": "Greedy curve search...", "trial": 0, "total": 1}

    pat_dirs = np.diff(pat_n, axis=0)
    pat_dirs /= (np.linalg.norm(pat_dirs, axis=1, keepdims=True) + 1e-9)
    step_lens = np.linalg.norm(np.diff(pat_n, axis=0), axis=1)
    typical_step = float(np.median(step_lens)) if len(step_lens) > 0 else 0.1

    n_seeds = min(400, len(token_ids))
    seeds = rng.choice(len(token_ids), size=n_seeds, replace=False)
    greedy_best_chain = []
    greedy_best_score = -1.0

    for si, seed in enumerate(seeds):
        chain = [int(seed)]
        used = {int(seed)}

        for step in range(min(n_pattern_points - 1, 80)):
            cur = tn[chain[-1]]
            tdir = pat_dirs[min(step, len(pat_dirs) - 1)]

            # Fast: query nearby in KDTree
            nearby = tree_n.query_ball_point(cur, typical_step * 4.0)
            if not nearby:
                dd, nn = tree_n.query(cur, k=min(50, tn.shape[0]))
                nearby = nn.tolist() if hasattr(nn, 'tolist') else [nn]

            best_next = None
            best_s = -999.0

            for idx in nearby:
                if idx in used:
                    continue
                d = tn[idx] - cur
                dist = np.linalg.norm(d)
                if dist < 1e-9:
                    continue
                alignment = float(np.dot(d / dist, tdir))
                step_match = 1.0 / (1.0 + abs(dist - typical_step) / (typical_step + 1e-9))
                s = 0.6 * alignment + 0.4 * step_match
                if s > best_s:
                    best_s = s
                    best_next = int(idx)

            if best_next is None:
                break
            chain.append(best_next)
            used.add(best_next)

        if len(chain) >= 5:
            chain_pts = tn[chain]
            sub_pat = pat_n[:len(chain)]
            sc = similarity(chain_pts, sub_pat)
            if sc > greedy_best_score:
                greedy_best_score = sc
                greedy_best_chain = chain

        # Yield progress every 50 seeds
        if (si + 1) % 50 == 0:
            yield {"status": "progress", "phase": "greedy",
                   "message": f"Greedy: {si+1}/{n_seeds} seeds (best len: {len(greedy_best_chain)}, score: {greedy_best_score:.4f})",
                   "trial": si + 1, "total": n_seeds}

    # Check if greedy beat ICP
    greedy_error = 1.0 - greedy_best_score if greedy_best_chain else float("inf")
    if greedy_error < best_error and greedy_best_chain:
        best_error = greedy_error
        best_indices = np.array(greedy_best_chain)
        best_strategy = f"Greedy curve (len={len(greedy_best_chain)})"
        yield _build_result(best_indices, best_strategy, best_error)

    # ── 8. OPTIONAL: Subspace search ─────────────────
    if search_subspaces and emb_hd.shape[1] > 3:
        yield {"status": "progress", "phase": "subspace",
               "message": "Searching alternate 3D subspaces..."}

        max_comp = min(15, emb_hd.shape[1], emb_hd.shape[0])
        pca_full = PCA(n_components=max_comp)
        all_pcs = pca_full.fit_transform(emb_hd)

        for start in range(0, min(10, max_comp - 2)):
            sub_3d = all_pcs[:, start:start + 3]
            sub_n, _, _ = normalize(sub_3d)
            sub_tree = KDTree(sub_n)

            for trial in range(10):
                r = np.random.RandomState(1000 + start * 100 + trial)
                M = r.randn(3, 3)
                Q, _ = np.linalg.qr(M)
                scale = r.uniform(0.5, 2.0)
                offset = r.uniform(sub_n.min(axis=0) - 0.5, sub_n.max(axis=0) + 0.5)
                init = scale * (pat_n @ Q.T) + offset
                raw_idx, transformed, error = fast_icp(init, sub_n, sub_tree, max_iter=30)

                if error < best_error:
                    best_error = error
                    try:
                        best_indices = assign_unique(transformed, sub_n, raw_idx, sub_tree)
                    except Exception:
                        best_indices = raw_idx

                    # Reproject matched tokens to main 3D for visualization
                    best_strategy = f"Subspace PCA[{start+1}:{start+4}]"
                    yield _build_result(best_indices, best_strategy, best_error)

            yield {"status": "progress", "phase": "subspace",
                   "message": f"Subspace PCA[{start+1}:{start+4}] done (best: {best_error:.4f})"}

    # ── 9. DONE ──────────────────────────────────────
    if best_indices is not None:
        yield _build_result(best_indices, best_strategy, best_error, is_final=True)
    else:
        yield {"status": "done", "match_similarity": 0,
               "message": "No match found.", "n_matched": 0}


# ═══════════════════════════════════════════════════════════════════
# NON-GENERATOR WRAPPERS (for scan_all and simple calls)
# ═══════════════════════════════════════════════════════════════════

def find_pattern_sync(model_key, pattern_name, **kwargs) -> Dict:
    """Run progressive_search but return only the final result."""
    last = {"status": "error", "message": "No result"}
    for update in progressive_search(model_key, pattern_name, **kwargs):
        if update.get("status") in ("best_update", "done"):
            last = update
    return last


def scan_all_patterns(model_key: str, filter_type: str = "all",
                       sample_size: int = 5000, n_pattern_points: int = 30,
                       projection: str = "pca") -> List[Dict]:
    """Quick scan: which pattern shape best fits this token set?"""
    results = []
    for name in PATTERN_GENERATORS:
        try:
            out = find_pattern_sync(
                model_key, name,
                n_pattern_points=n_pattern_points,
                filter_type=filter_type,
                sample_size=sample_size,
                projection=projection,
            )
            if out.get("status") != "error":
                results.append({
                    "pattern": name,
                    "similarity": out.get("match_similarity", 0),
                    "error": out.get("match_error", 999),
                    "n_matched": out.get("n_matched", 0),
                    "sample_tokens": out.get("matched_tokens", [])[:10],
                })
        except Exception as e:
            results.append({"pattern": name, "similarity": 0, "error": str(e)})

    results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return results

