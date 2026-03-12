/**
 * embedding_viz.js — Visualize token, position, and combined embeddings in 3D
 */

document.getElementById('btn-show-emb')?.addEventListener('click', showEmbeddings);

async function showEmbeddings() {
    if (!APP_STATE.text) return;

    const projection = document.getElementById('emb-projection').value;
    const embType = document.getElementById('emb-type').value;

    const data = await apiPost('/api/embeddings', {
        text: APP_STATE.text,
        model: APP_STATE.model,
    });

    let embeddings;
    if (embType === 'token') embeddings = data.token_embeddings;
    else if (embType === 'position') embeddings = data.position_embeddings;
    else embeddings = data.combined_embeddings;

    // Project
    const layerData = await apiPost('/api/layer_output', {
        trace_key: APP_STATE.traceKey,
        layer: 0,
        projection: projection,
    });

    const pts3d = layerData.projected_3d;
    const tokens = data.tokens;
    const hoverInfo = layerData.hover_info;

    // Build hover text
    const hoverTexts = tokens.map((tok, i) => {
        let text = `<b>Token ${i}: "${tok}"</b><br>ID: ${data.token_ids[i]}`;
        if (hoverInfo && hoverInfo[i]) {
            text += '<br><br>Nearest vocab tokens:';
            hoverInfo[i].slice(0, 5).forEach(n => {
                text += `<br>  "${n.token}" (sim: ${n.similarity.toFixed(3)})`;
            });
        }
        return text;
    });

    // Colors by position
    const colors = tokens.map((_, i) => i);

    const trace = {
        x: pts3d.map(p => p[0]),
        y: pts3d.map(p => p[1]),
        z: pts3d.map(p => p[2]),
        mode: 'markers+text',
        type: 'scatter3d',
        text: tokens,
        textposition: 'top center',
        textfont: { size: 10, color: '#c9d1d9' },
        hovertext: hoverTexts,
        hoverinfo: 'text',
        marker: {
            size: 8,
            color: colors,
            colorscale: 'Viridis',
            opacity: 0.9,
            line: { width: 1, color: '#fff' },
        },
    };

    const layout = {
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        scene: {
            xaxis: { gridcolor: '#30363d', color: '#8b949e' },
            yaxis: { gridcolor: '#30363d', color: '#8b949e' },
            zaxis: { gridcolor: '#30363d', color: '#8b949e' },
            bgcolor: '#0d1117',
        },
        title: `${embType.charAt(0).toUpperCase() + embType.slice(1)} Embeddings (${projection.toUpperCase()})`,
        margin: { l: 0, r: 0, t: 40, b: 0 },
    };

    Plotly.newPlot('emb-plot-3d', [trace], layout, { responsive: true });

    // Info panel
    const infoPanel = document.getElementById('emb-info');
    infoPanel.innerHTML = `
        <h3>Embedding Info</h3>
        <p><b>Tokens:</b> ${tokens.length}</p>
        <p><b>Dimensions:</b> ${data.d_model}</p>
        <p><b>Type:</b> ${embType}</p>
        <p><b>Projection:</b> ${projection}</p>
        <hr style="border-color:#30363d; margin:10px 0">
        <h4>Tokens:</h4>
        ${tokens.map((t, i) => `<p>[${i}] <b>${escapeHtml(t)}</b> — ID ${data.token_ids[i]}</p>`).join('')}
    `;
}

