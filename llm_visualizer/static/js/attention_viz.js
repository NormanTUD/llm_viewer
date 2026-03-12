/**
 * attention_viz.js — Attention heatmap and rollout visualization
 */

document.getElementById('btn-show-attn')?.addEventListener('click', showAttention);

// Layer slider for layer trace tab
const layerSlider = document.getElementById('layer-slider');
if (layerSlider) {
    layerSlider.addEventListener('input', async () => {
        const layer = parseInt(layerSlider.value);
        document.getElementById('layer-label').textContent =
            layer === 0 ? 'Layer 0 (input)' : `Layer ${layer}`;
        await showLayerOutput(layer);
    });
}

document.getElementById('btn-animate-layers')?.addEventListener('click', animateLayers);

async function showLayerOutput(layer) {
    if (!APP_STATE.traceKey) return;

    const projection = document.getElementById('layer-projection').value;
    const data = await apiPost('/api/layer_output', {
        trace_key: APP_STATE.traceKey,
        layer: layer,
        projection: projection,
    });

    const pts3d = data.projected_3d;
    const hoverTexts = APP_STATE.tokens.map((tok, i) => {
        let text = `<b>[${i}] "${tok}"</b>`;
        if (data.hover_info && data.hover_info[i]) {
            text += '<br>Nearest:';
            data.hover_info[i].slice(0, 5).forEach(n => {
                text += `<br>  "${n.token}" (${n.similarity.toFixed(3)})`;
            });
        }
        return text;
    });

    const trace = {
        x: pts3d.map(p => p[0]),
        y: pts3d.map(p => p[1]),
        z: pts3d.map(p => p[2]),
        mode: 'markers+text',
        type: 'scatter3d',
        text: APP_STATE.tokens,
        textposition: 'top center',
        textfont: { size: 10, color: '#c9d1d9' },
        hovertext: hoverTexts,
        hoverinfo: 'text',
        marker: {
            size: 8,
            color: APP_STATE.tokens.map((_, i) => i),
            colorscale: 'Portland',
            opacity: 0.9,
        },
    };

    const layout = {
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        scene: {
            xaxis: { gridcolor: '#30363d' },
            yaxis: { gridcolor: '#30363d' },
            zaxis: { gridcolor: '#30363d' },
            bgcolor: '#0d1117',
        },
        title: layer === 0 ? 'Input Embeddings' : `Layer ${layer} Output`,
        margin: { l: 0, r: 0, t: 40, b: 0 },
    };

    Plotly.newPlot('layer-plot-3d', [trace], layout, { responsive: true });

    // Also show delta norms
    const deltaData = await apiPost('/api/layer_delta', {
        trace_key: APP_STATE.traceKey,
        layer: Math.max(0, layer - 1),
    });

    if (deltaData.delta_norm_per_token) {
        const deltaTrace = {
            x: APP_STATE.tokens,
            y: deltaData.delta_norm_per_token,
            type: 'bar',
            marker: { color: '#f97316' },
        };
        const deltaLayout = {
            paper_bgcolor: '#161b22',
            plot_bgcolor: '#0d1117',
            font: { color: '#c9d1d9', size: 10 },
            title: `Layer ${Math.max(0, layer - 1)} Delta Norms`,
            margin: { l: 40, r: 10, t: 40, b: 60 },
            xaxis: { tickangle: -45 },
        };
        Plotly.newPlot('layer-delta-chart', [deltaTrace], deltaLayout, { responsive: true });
    }
}

async function animateLayers() {
    if (!APP_STATE.traceKey) return;
    for (let i = 0; i <= APP_STATE.nLayers; i++) {
        layerSlider.value = i;
        document.getElementById('layer-label').textContent =
            i === 0 ? 'Layer 0 (input)' : `Layer ${i}`;
        await showLayerOutput(i);
        await new Promise(resolve => setTimeout(resolve, 600));
    }
}

async function showAttention() {
    if (!APP_STATE.traceKey) return;

    const view = document.getElementById('attn-view').value;

    if (view === 'rollout') {
        return showAttentionRollout();
    }

    const layer = parseInt(document.getElementById('attn-layer-select').value);
    const headVal = document.getElementById('attn-head-select').value;
    const head = headVal === 'avg' ? null : parseInt(headVal);

    const data = await apiPost('/api/attention', {
        trace_key: APP_STATE.traceKey,
        layer: layer,
        head: head,
    });

    let attnMatrix = data.attention;
    // If all heads returned and no specific head, average them
    if (head === null && attnMatrix.length > 0 && Array.isArray(attnMatrix[0][0])) {
        // Shape: (n_heads, seq, seq) -> average to (seq, seq)
        const nHeads = attnMatrix.length;
        const seqLen = attnMatrix[0].length;
        const avg = Array.from({ length: seqLen }, () => Array(seqLen).fill(0));
        for (let h = 0; h < nHeads; h++) {
            for (let i = 0; i < seqLen; i++) {
                for (let j = 0; j < seqLen; j++) {
                    avg[i][j] += attnMatrix[h][i][j] / nHeads;
                }
            }
        }
        attnMatrix = avg;
    }

    // Plot heatmap
    const heatmapTrace = {
        z: attnMatrix,
        x: APP_STATE.tokens,
        y: APP_STATE.tokens,
        type: 'heatmap',
        colorscale: 'Hot',
        hovertemplate: 'From: %{y}<br>To: %{x}<br>Weight: %{z:.4f}<extra></extra>',
    };

    const layout = {
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        title: `Attention — Layer ${layer}${head !== null ? ', Head ' + head : ' (avg)'}`,
        xaxis: { title: 'Key (attending to)', tickangle: -45, color: '#8b949e' },
        yaxis: { title: 'Query (from)', color: '#8b949e', autorange: 'reversed' },
        margin: { l: 100, r: 20, t: 50, b: 100 },
    };

    Plotly.newPlot('attn-heatmap', [heatmapTrace], layout, { responsive: true });

    // Head stats
    const statsPanel = document.getElementById('attn-head-stats');
    if (data.head_importance) {
        statsPanel.innerHTML = `
            <h3>Head Importance (Entropy)</h3>
            ${data.head_importance.map(h =>
                `<p>Head ${h.head}: entropy=${h.avg_entropy.toFixed(3)}, max_attn=${h.max_attn.toFixed(3)}</p>`
            ).join('')}
        `;
    }
}

async function showAttentionRollout() {
    const data = await apiPost('/api/attention_rollout', {
        trace_key: APP_STATE.traceKey,
    });

    const heatmapTrace = {
        z: data.rollout,
        x: APP_STATE.tokens,
        y: APP_STATE.tokens,
        type: 'heatmap',
        colorscale: 'Viridis',
        hovertemplate: 'From: %{y}<br>To: %{x}<br>Flow: %{z:.4f}<extra></extra>',
    };

    const layout = {
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        title: 'Attention Rollout (cumulative flow)',
        xaxis: { title: 'Key', tickangle: -45, color: '#8b949e' },
        yaxis: { title: 'Query', color: '#8b949e', autorange: 'reversed' },
        margin: { l: 100, r: 20, t: 50, b: 100 },
    };

    Plotly.newPlot('attn-heatmap', [heatmapTrace], layout, { responsive: true });
}

