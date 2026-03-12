"""
Level 5: Flask application — ties everything together with API endpoints
and serves the frontend.
"""
from flask import Flask, render_template, request, jsonify
from flask import Response, stream_with_context
import numpy as np
import json

from utils.model_loader import list_available_models, get_model_config, load_model
from utils.tokenizer_utils import tokenize_text, get_vocab_tokens, decode_token_id
from utils.embedding_utils import (
    get_token_embeddings, get_position_embeddings,
    get_combined_input_embeddings, get_token_embedding_matrix,
    get_nearest_tokens_to_vector,
)
from utils.layer_trace import (
    forward_full_trace, get_layer_output, get_layer_delta,
    get_token_trajectory, compute_layer_norms,
)
from utils.attention_utils import (
    get_attention_weights, get_attention_from_token,
    get_attention_to_token, get_attention_rollout, get_head_importance,
)
from utils.dimension_slicer import parse_dim_spec, slice_dimensions, get_dimension_stats
from utils.projection import (
    project_to_2d, project_to_3d, project_direct_dims, get_pca_explained_variance,
)
from utils.clustering import (
    cluster_kmeans, cluster_vocab_embeddings, find_neighbors_in_embedding,
)

from utils.pattern_finder import (
    PATTERN_GENERATORS,
    filter_vocab_tokens,
    progressive_search,
    find_pattern_sync,
    scan_all_patterns,
    clear_caches,
)



app = Flask(__name__)

# Store traces in memory for the session (simple approach)
_trace_cache = {}


# ─── Pages ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    models = list_available_models()
    return render_template("index.html", models=models)


# ─── Model Info API ────────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def api_models():
    return jsonify(list_available_models())


@app.route("/api/model_config/<model_key>", methods=["GET"])
def api_model_config(model_key):
    try:
        config = get_model_config(model_key)
        return jsonify(config)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─── Tokenization API ─────────────────────────────────────────────

@app.route("/api/tokenize", methods=["POST"])
def api_tokenize():
    data = request.json
    text = data.get("text", "")
    model_key = data.get("model", "gpt2")
    result = tokenize_text(text, model_key)
    # Remove tensor from JSON response
    result.pop("input_ids_tensor", None)
    return jsonify(result)


@app.route("/api/vocab", methods=["GET"])
def api_vocab():
    model_key = request.args.get("model", "gpt2")
    start = int(request.args.get("start", 0))
    end = int(request.args.get("end", 500))
    tokens = get_vocab_tokens(model_key, start, end)
    return jsonify(tokens)


# ─── Embedding API ─────────────────────────────────────────────────

@app.route("/api/embeddings", methods=["POST"])
def api_embeddings():
    """Get token + position embeddings for input text."""
    data = request.json
    text = data.get("text", "")
    model_key = data.get("model", "gpt2")

    tok = tokenize_text(text, model_key)
    token_emb = get_token_embeddings(tok["token_ids"], model_key)
    pos_emb = get_position_embeddings(tok["num_tokens"], model_key)
    combined_emb = get_combined_input_embeddings(tok["token_ids"], model_key)

    return jsonify({
        "tokens": tok["tokens"],
        "token_ids": tok["token_ids"],
        "token_embeddings": token_emb.tolist(),
        "position_embeddings": pos_emb.tolist(),
        "combined_embeddings": combined_emb.tolist(),
        "d_model": token_emb.shape[1],
    })


@app.route("/api/nearest_tokens", methods=["POST"])
def api_nearest_tokens():
    """Find nearest tokens to a given vector in embedding space."""
    data = request.json
    vector = np.array(data.get("vector", []))
    model_key = data.get("model", "gpt2")
    top_k = data.get("top_k", 10)
    results = get_nearest_tokens_to_vector(vector, model_key, top_k)
    return jsonify(results)


# ─── Full Trace API ───────────────────────────────────────────────

@app.route("/api/trace", methods=["POST"])
def api_trace():
    """Run a full forward pass and cache the trace."""
    data = request.json
    text = data.get("text", "")
    model_key = data.get("model", "gpt2")

    tok = tokenize_text(text, model_key)
    trace = forward_full_trace(tok["token_ids"], model_key)

    # Cache with a simple key
    trace_key = f"{model_key}:{text}"
    _trace_cache[trace_key] = trace

    layer_norms = compute_layer_norms(trace)
    config = get_model_config(model_key)

    return jsonify({
        "trace_key": trace_key,
        "tokens": tok["tokens"],
        "token_ids": tok["token_ids"],
        "n_layers": config["n_layers"],
        "n_heads": config["n_heads"],
        "d_model": config["d_model"],
        "layer_norms": layer_norms,
        "num_hidden_states": len(trace["hidden_states"]),
    })


# ─── Layer Exploration API ─────────────────────────────────────────

@app.route("/api/layer_output", methods=["POST"])
def api_layer_output():
    """Get the hidden state output of a specific layer."""
    data = request.json
    trace_key = data.get("trace_key", "")
    layer_idx = data.get("layer", 0)
    dim_spec = data.get("dim_spec", "all")
    projection = data.get("projection", "pca")  # pca, tsne, umap, direct
    dims_3d = data.get("dims_3d", None)  # for direct projection: [dimX, dimY, dimZ]

    trace = _trace_cache.get(trace_key)
    if trace is None:
        return jsonify({"error": "Trace not found. Run /api/trace first."}), 404

    hs = get_layer_output(trace, layer_idx)
    config_text = trace_key.split(":")[0]
    d_model = hs.shape[1]

    # Slice dimensions if specified
    if dim_spec != "all":
        hs_sliced = slice_dimensions(hs, dim_spec)
    else:
        hs_sliced = hs

    # Project for visualization
    if projection == "direct" and dims_3d:
        projected_3d = project_direct_dims(hs_sliced, *dims_3d[:3])
        projected_2d = project_direct_dims(hs_sliced, dims_3d[0], dims_3d[1])
    else:
        projected_3d = project_to_3d(hs_sliced, method=projection).tolist() if hs_sliced.shape[0] >= 2 else np.zeros((hs_sliced.shape[0], 3)).tolist()
        projected_2d = project_to_2d(hs_sliced, method=projection).tolist() if hs_sliced.shape[0] >= 2 else np.zeros((hs_sliced.shape[0], 2)).tolist()

    # For each token position, find what tokens the vector is closest to
    from utils.embedding_utils import get_nearest_tokens_to_vector
    hover_info = []
    model_key = trace_key.split(":")[0]
    for i in range(hs.shape[0]):
        nearest = get_nearest_tokens_to_vector(hs[i], model_key, top_k=5)
        hover_info.append(nearest)

    return jsonify({
        "layer": layer_idx,
        "shape": list(hs.shape),
        "sliced_shape": list(hs_sliced.shape),
        "projected_3d": projected_3d if isinstance(projected_3d, list) else projected_3d.tolist(),
        "projected_2d": projected_2d if isinstance(projected_2d, list) else projected_2d.tolist(),
        "raw_data": hs_sliced.tolist(),
        "hover_info": hover_info,
    })


@app.route("/api/layer_delta", methods=["POST"])
def api_layer_delta():
    """Get the change made by a specific layer."""
    data = request.json
    trace_key = data.get("trace_key", "")
    layer_idx = data.get("layer", 0)

    trace = _trace_cache.get(trace_key)
    if trace is None:
        return jsonify({"error": "Trace not found."}), 404

    delta = get_layer_delta(trace, layer_idx)
    return jsonify({
        "layer": layer_idx,
        "delta_norm_per_token": np.linalg.norm(delta, axis=1).tolist(),
        "delta_mean": float(np.mean(np.abs(delta))),
    })


@app.route("/api/token_trajectory", methods=["POST"])
def api_token_trajectory():
    """Trace a single token through all layers (its embedding evolving)."""
    data = request.json
    trace_key = data.get("trace_key", "")
    token_idx = data.get("token_idx", 0)
    projection = data.get("projection", "pca")

    trace = _trace_cache.get(trace_key)
    if trace is None:
        return jsonify({"error": "Trace not found."}), 404

    trajectory = get_token_trajectory(trace, token_idx)
    projected_3d = project_to_3d(trajectory, method=projection)

    # Hover: nearest tokens at each layer
    model_key = trace_key.split(":")[0]
    hover_info = []
    for i in range(trajectory.shape[0]):
        nearest = get_nearest_tokens_to_vector(trajectory[i], model_key, top_k=5)
        hover_info.append({
            "layer": i if i > 0 else "input",
            "nearest_tokens": nearest,
        })

    return jsonify({
        "token_idx": token_idx,
        "n_layers": trajectory.shape[0],
        "projected_3d": projected_3d.tolist(),
        "hover_info": hover_info,
    })


# ─── Attention API ─────────────────────────────────────────────────

@app.route("/api/attention", methods=["POST"])
def api_attention():
    """Get attention weights for a layer (optionally specific head)."""
    data = request.json
    trace_key = data.get("trace_key", "")
    layer_idx = data.get("layer", 0)
    head_idx = data.get("head", None)

    trace = _trace_cache.get(trace_key)
    if trace is None:
        return jsonify({"error": "Trace not found."}), 404

    attn = get_attention_weights(trace, layer_idx, head_idx)
    head_stats = get_head_importance(trace, layer_idx)

    return jsonify({
        "layer": layer_idx,
        "head": head_idx,
        "attention": attn.tolist(),
        "shape": list(attn.shape),
        "head_importance": head_stats,
    })


@app.route("/api/attention_rollout", methods=["POST"])
def api_attention_rollout():
    """Compute attention rollout up to a specific layer."""
    data = request.json
    trace_key = data.get("trace_key", "")
    max_layer = data.get("max_layer", None)

    trace = _trace_cache.get(trace_key)
    if trace is None:
        return jsonify({"error": "Trace not found."}), 404

    rollout = get_attention_rollout(trace, max_layer)
    return jsonify({
        "rollout": rollout.tolist(),
        "shape": list(rollout.shape),
    })


# ─── Dimension Slicer API ─────────────────────────────────────────

@app.route("/api/slice", methods=["POST"])
def api_slice():
    """Slice specific dimensions and return stats + visualization data."""
    data = request.json
    trace_key = data.get("trace_key", "")
    layer_idx = data.get("layer", 0)
    dim_spec = data.get("dim_spec", "0:10")
    projection = data.get("projection", "pca")

    trace = _trace_cache.get(trace_key)
    if trace is None:
        return jsonify({"error": "Trace not found."}), 404

    hs = get_layer_output(trace, layer_idx)
    d_model = hs.shape[1]
    dims = parse_dim_spec(dim_spec, d_model)
    sliced = slice_dimensions(hs, dim_spec)
    stats = get_dimension_stats(hs, dims)

    # Project the sliced data
    if sliced.shape[1] >= 3 and sliced.shape[0] >= 2:
        proj_3d = project_to_3d(sliced, method=projection).tolist()
    else:
        proj_3d = sliced.tolist() if sliced.shape[1] <= 3 else [[0, 0, 0]] * sliced.shape[0]

    if sliced.shape[1] >= 2 and sliced.shape[0] >= 2:
        proj_2d = project_to_2d(sliced, method=projection).tolist()
    else:
        proj_2d = sliced.tolist() if sliced.shape[1] <= 2 else [[0, 0]] * sliced.shape[0]

    # PCA explained variance on the slice
    variance = get_pca_explained_variance(sliced) if sliced.shape[0] >= 2 else []

    return jsonify({
        "layer": layer_idx,
        "dim_spec": dim_spec,
        "selected_dims": dims,
        "n_dims_selected": len(dims),
        "dim_stats": stats,
        "projected_3d": proj_3d,
        "projected_2d": proj_2d,
        "pca_variance": variance,
    })


# ─── Clustering API ───────────────────────────────────────────────

@app.route("/api/cluster_layer", methods=["POST"])
def api_cluster_layer():
    """Cluster hidden states at a specific layer."""
    data = request.json
    trace_key = data.get("trace_key", "")
    layer_idx = data.get("layer", 0)
    n_clusters = data.get("n_clusters", 4)
    projection = data.get("projection", "pca")

    trace = _trace_cache.get(trace_key)
    if trace is None:
        return jsonify({"error": "Trace not found."}), 404

    hs = get_layer_output(trace, layer_idx)
    if hs.shape[0] < n_clusters:
        n_clusters = max(2, hs.shape[0] // 2)

    cluster_result = cluster_kmeans(hs, n_clusters=n_clusters)
    proj_3d = project_to_3d(hs, method=projection).tolist() if hs.shape[0] >= 2 else [[0, 0, 0]]

    return jsonify({
        "layer": layer_idx,
        "cluster_labels": cluster_result["labels"],
        "n_clusters": cluster_result["n_clusters"],
        "silhouette_score": cluster_result["silhouette_score"],
        "projected_3d": proj_3d,
    })


@app.route("/api/cluster_vocab", methods=["POST"])
def api_cluster_vocab():
    """Cluster tokens from the vocabulary embedding space."""
    data = request.json
    model_key = data.get("model", "gpt2")
    n_clusters = data.get("n_clusters", 10)
    sample_size = data.get("sample_size", 3000)
    projection = data.get("projection", "pca")

    result = cluster_vocab_embeddings(model_key, n_clusters=n_clusters, sample_size=sample_size)

    # Project for visualization
    from utils.embedding_utils import get_token_embedding_matrix
    emb_matrix = get_token_embedding_matrix(model_key)
    embeddings = emb_matrix[result["token_ids"]]
    proj_3d = project_to_3d(embeddings, method=projection).tolist()

    return jsonify({
        "tokens": result["tokens"],
        "token_ids": result["token_ids"],
        "cluster_labels": result["labels"],
        "n_clusters": result["n_clusters"],
        "silhouette_score": result["silhouette_score"],
        "projected_3d": proj_3d,
    })


@app.route("/api/token_neighbors", methods=["POST"])
def api_token_neighbors():
    """Find nearest neighbor tokens in embedding space."""
    data = request.json
    model_key = data.get("model", "gpt2")
    text = data.get("text", "hello")
    top_k = data.get("top_k", 20)

    from utils.tokenizer_utils import find_token_id
    token_ids = find_token_id(text, model_key)
    result = find_neighbors_in_embedding(token_ids, model_key, top_k=top_k)
    return jsonify(result)


# ─── Pattern Finding API ──────────────────────────────────────────

@app.route("/api/patterns/available", methods=["GET"])
def api_available_patterns():
    return jsonify({
        "patterns": list(PATTERN_GENERATORS.keys()),
        "filters": ["all", "numbers", "years", "words", "letters",
                     "months", "colors", "regex"],
    })

@app.route("/api/patterns/scan", methods=["POST"])
def api_pattern_scan():
    """Scan all layers for a specific pattern."""
    data = request.json
    trace_key = data.get("trace_key", "")
    pattern_name = data.get("pattern", "spiral")
    projection = data.get("projection", "pca")

    trace = _trace_cache.get(trace_key)
    if trace is None:
        return jsonify({"error": "Trace not found."}), 404

    results = scan_all_layers_for_pattern(pattern_name, trace, projection_method=projection)
    return jsonify({
        "pattern": pattern_name,
        "results": results,
    })

@app.route("/api/patterns/clear_cache", methods=["POST"])
def api_clear_cache():
    clear_caches()
    return jsonify({"status": "ok"})

@app.route("/api/patterns/scan_all", methods=["POST"])
def api_pattern_scan_all():
    data = request.json
    results = scan_all_patterns(
        model_key=data.get("model", "gpt2"),
        filter_type=data.get("filter", "all"),
        sample_size=data.get("sample_size", 5000),
        projection=data.get("projection", "pca"),
    )
    return jsonify({"results": results, "filter": data.get("filter", "all")})

@app.route("/api/patterns/generate", methods=["POST"])
def api_pattern_generate():
    data = request.json
    name = data.get("pattern", "spiral")
    n = data.get("n_points", 200)
    gen = PATTERN_GENERATORS.get(name)
    if not gen:
        return jsonify({"error": f"Unknown: {name}"}), 400
    try:
        pts = gen(n_points=n)
    except TypeError:
        pts = gen()
    return jsonify({"pattern": name, "points": pts.tolist()})

@app.route("/api/patterns/filter_preview", methods=["POST"])
def api_filter_preview():
    data = request.json
    ids, toks = filter_vocab_tokens(
        data.get("model", "gpt2"),
        data.get("filter", "all"),
        data.get("regex", None),
        max_tokens=200,
    )
    return jsonify({
        "filter": data.get("filter", "all"),
        "count": len(ids),
        "tokens": [{"id": ids[i], "text": toks[i]} for i in range(len(ids))],
    })

# Keep non-streaming version for scan_all and simple calls
@app.route("/api/patterns/search", methods=["POST"])
def api_pattern_search():
    data = request.json
    result = find_pattern_sync(
        model_key=data.get("model", "gpt2"),
        pattern_name=data.get("pattern", "helix"),
        n_pattern_points=data.get("n_pattern_points", 40),
        filter_type=data.get("filter", "all"),
        custom_regex=data.get("regex", None),
        sample_size=data.get("sample_size", 10000),
        projection=data.get("projection", "pca"),
        search_subspaces=data.get("search_subspaces", False),
    )
    return jsonify(result)

@app.route("/api/patterns/search_stream", methods=["POST"])
def api_pattern_search_stream():
    data = request.json
    model_key = data.get("model", "gpt2")
    pattern_name = data.get("pattern", "helix")
    n_points = data.get("n_pattern_points", 40)
    filter_type = data.get("filter", "all")
    custom_regex = data.get("regex", None)
    sample_size = data.get("sample_size", 10000)
    projection = data.get("projection", "pca")
    search_subspaces = data.get("search_subspaces", False)
    strictness = float(data.get("strictness", 0.5))

    trace_key = data.get("trace_key", None)
    layer_idx = data.get("layer", None)
    layer_hs = None
    layer_toks = None
    if trace_key and layer_idx is not None:
        trace = _trace_cache.get(trace_key)
        if trace:
            layer_hs = trace["hidden_states"][layer_idx]
            text = ":".join(trace_key.split(":")[1:])
            tok_data = tokenize_text(text, model_key)
            layer_toks = tok_data["tokens"]

    def generate():
        for update in progressive_search(
            model_key=model_key, pattern_name=pattern_name,
            n_pattern_points=n_points, filter_type=filter_type,
            custom_regex=custom_regex, sample_size=sample_size,
            projection=projection, search_subspaces=search_subspaces,
            strictness=strictness,
            layer_hidden_states=layer_hs, layer_tokens=layer_toks,
        ):
            yield f"data: {json.dumps(update)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Run ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)

