/**
 * pattern_viz.js — Pattern generation, scanning, and visualization
 */

document.getElementById('btn-scan-pattern')?.addEventListener('click', scanPattern);
document.getElementById('btn-scan-all-patterns')?.addEventListener('click', scanAllPatterns);

// Preview pattern when selection changes
document.getElementById('pattern-select')?.addEventListener('change', previewPattern);

async function previewPattern() {
    const pattern = document.getElementById('pattern-select').value;
    const data = await apiPost('/api/patterns/generate', {
        pattern: pattern,
        n_points: 200,
    });

    const pts = data.points;
    const trace = {
        x: pts.map(p => p[0]),
        y: pts.map(p => p[1]),
        z: pts.map(p => p[2]),
        mode: 'markers+lines',
        type: 'scatter3d',
        marker: {
            size: 3,
            color: pts.map((_, i) => i),
            colorscale: 'Turbo',
        },
        line: { width: 2, color: '#58a6ff' },
        hoverinfo: 'text',
        hovertext: pts.map((p, i) => `Point ${i}<br>x=${p[0].toFixed(3)}<br>y=${p[1].toFixed(3)}<br>z=${p[2].toFixed(3)}`),
    };

    Plotly.newPlot('pattern-preview', [trace], {
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        scene: {
            bgcolor: '#0d1117',
            xaxis: { gridcolor: '#30363d' },
            yaxis: { gridcolor: '#30363d' },
            zaxis: { gridcolor: '#30363d' },
        },
        title: `Pattern Preview: ${pattern}`,
        margin: { l: 0, r: 0, t: 40, b: 0 },
    }, { responsive: true });
}

async function scanPattern() {
    if (!APP_STATE.traceKey) {
        alert('Run analysis first!');
        return;
    }

    const pattern = document.getElementById('pattern-select').value;
    const rankingDiv = document.getElementById('pattern-ranking');
    rankingDiv.innerHTML = '<p>🔎 Scanning all layers... <span class="loading"></span></p>';

    const data = await apiPost('/api/patterns/scan', {
        trace_key: APP_STATE.traceKey,
        pattern: pattern,
        projection: 'pca',
    });

    // Show ranking
    rankingDiv.innerHTML = `
        <h3>Pattern Match: "${pattern}" across all layers</h3>
        <p>Sorted by similarity (higher = better match):</p>
        <table style="width:100%; border-collapse:collapse; margin-top:8px;">
            <tr style="border-bottom:1px solid #30363d;">
                <th style="text-align:left; padding:6px;">Rank</th>
                <th style="text-align:left; padding:6px;">Layer</th>
                <th style="text-align:left; padding:6px;">Similarity</th>
                <th style="text-align:left; padding:6px;">Bar</th>
            </tr>
            ${data.results.map((r, i) => `
                <tr style="border-bottom:1px solid #21262d; cursor:pointer;"
                    onclick="showPatternLayerComparison(${r.layer}, '${pattern}')"
                    onmouseover="this.style.background='#21262d'"
                    onmouseout="this.style.background='transparent'">
                    <td style="padding:6px;">${i + 1}</td>
                    <td style="padding:6px;">${r.is_input ? 'Input' : 'Layer ' + r.layer}</td>
                    <td style="padding:6px;">${r.similarity.toFixed(4)}</td>
                    <td style="padding:6px;">
                        <div style="background:#1e3a5f; height:16px; width:${(r.similarity * 300).toFixed(0)}px; border-radius:3px;
                            background: linear-gradient(90deg, #1e3a5f, ${r.similarity > 0.5 ? '#238636' : r.similarity > 0.3 ? '#d29922' : '#da3633'});"></div>
                    </td>
                </tr>
            `).join('')}
        </table>
    `;

    // Also draw a bar chart of similarities
    const barTrace = {
        x: data.results.map(r => r.is_input ? 'Input' : `L${r.layer}`),
        y: data.results.map(r => r.similarity),
        type: 'bar',
        marker: {
            color: data.results.map(r => r.similarity),
            colorscale: [
                [0, '#da3633'],
                [0.5, '#d29922'],
                [1, '#238636'],
            ],
        },
    };

    Plotly.newPlot('pattern-results', [barTrace], {
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        title: `"${pattern}" similarity by layer`,
        xaxis: { title: 'Layer', color: '#8b949e', tickangle: -45 },
        yaxis: { title: 'Similarity', color: '#8b949e' },
        margin: { l: 50, r: 20, t: 50, b: 60 },
    }, { responsive: true });
}

async function scanAllPatterns() {
    if (!APP_STATE.traceKey) {
        alert('Run analysis first!');
        return;
    }

    const layerSlider = document.getElementById('layer-slider');
    const layer = layerSlider ? parseInt(layerSlider.value) : 0;
    const rankingDiv = document.getElementById('pattern-ranking');
    rankingDiv.innerHTML = `<p>🔎 Scanning all patterns for layer ${layer}... <span class="loading"></span></p>`;

    const data = await apiPost('/api/patterns/scan_all', {
        trace_key: APP_STATE.traceKey,
        layer: layer,
        projection: 'pca',
    });

    rankingDiv.innerHTML = `
        <h3>All Patterns — Layer ${layer}</h3>
        <p>Which geometric patterns best match this layer's hidden state geometry?</p>
        <table style="width:100%; border-collapse:collapse; margin-top:8px;">
            <tr style="border-bottom:1px solid #30363d;">
                <th style="text-align:left; padding:6px;">Rank</th>
                <th style="text-align:left; padding:6px;">Pattern</th>
                <th style="text-align:left; padding:6px;">Similarity</th>
                <th style="text-align:left; padding:6px;">Bar</th>
            </tr>
            ${data.pattern_matches.map((r, i) => `
                <tr style="border-bottom:1px solid #21262d; cursor:pointer;"
                    onclick="previewAndCompare('${r.pattern}', ${layer})"
                    onmouseover="this.style.background='#21262d'"
                    onmouseout="this.style.background='transparent'">
                    <td style="padding:6px;">${i + 1}</td>
                    <td style="padding:6px; text-transform:capitalize;">${r.pattern}</td>
                    <td style="padding:6px;">${r.similarity.toFixed(4)}</td>
                    <td style="padding:6px;">
                        <div style="height:16px; width:${(r.similarity * 300).toFixed(0)}px; border-radius:3px;
                            background: linear-gradient(90deg, #1e3a5f, ${r.similarity > 0.5 ? '#238636' : r.similarity > 0.3 ? '#d29922' : '#da3633'});"></div>
                    </td>
                </tr>
            `).join('')}
        </table>
    `;

    // Bar chart
    const barTrace = {
        x: data.pattern_matches.map(r => r.pattern),
        y: data.pattern_matches.map(r => r.similarity),
        type: 'bar',
        marker: {
            color: data.pattern_matches.map(r => r.similarity),
            colorscale: [
                [0, '#da3633'],
                [0.5, '#d29922'],
                [1, '#238636'],
            ],
        },
    };

    Plotly.newPlot('pattern-results', [barTrace], {
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        title: `Pattern matches for Layer ${layer}`,
        xaxis: { title: 'Pattern', color: '#8b949e' },
        yaxis: { title: 'Similarity', color: '#8b949e' },
        margin: { l: 50, r: 20, t: 50, b: 60 },
    }, { responsive: true });
}

async function showPatternLayerComparison(layer, patternName) {
    /**
     * Show the actual layer data projected to 3D alongside the pattern template.
     * Called when user clicks a row in the pattern ranking table.
     */
    if (!APP_STATE.traceKey) return;

    // Get layer output projected to 3D
    const layerData = await apiPost('/api/layer_output', {
        trace_key: APP_STATE.traceKey,
        layer: layer,
        projection: 'pca',
    });

    // Get pattern
    const patternData = await apiPost('/api/patterns/generate', {
        pattern: patternName,
        n_points: Math.max(APP_STATE.tokens.length, 50),
    });

    const layerPts = layerData.projected_3d;
    const patPts = patternData.points;

    // Normalize both for visual comparison
    function normalize(pts) {
        const xs = pts.map(p => p[0]);
        const ys = pts.map(p => p[1]);
        const zs = pts.map(p => p[2]);
        const mean = [avg(xs), avg(ys), avg(zs)];
        const std = Math.max(stdDev(xs), stdDev(ys), stdDev(zs), 0.001);
        return pts.map(p => [(p[0] - mean[0]) / std, (p[1] - mean[1]) / std, (p[2] - mean[2]) / std]);
    }

    function avg(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
    function stdDev(arr) {
        const m = avg(arr);
        return Math.sqrt(arr.reduce((sum, v) => sum + (v - m) ** 2, 0) / arr.length);
    }

    const normLayer = normalize(layerPts);
    const normPattern = normalize(patPts);

    // Layer trace
    const layerTrace = {
        x: normLayer.map(p => p[0]),
        y: normLayer.map(p => p[1]),
        z: normLayer.map(p => p[2]),
        mode: 'markers+text',
        type: 'scatter3d',
        text: APP_STATE.tokens,
        textfont: { size: 9, color: '#a5f3fc' },
        hovertext: APP_STATE.tokens.map((t, i) => {
            let text = `<b>"${t}"</b><br>Layer ${layer}`;
            if (layerData.hover_info && layerData.hover_info[i]) {
                text += '<br>Nearest:';
                layerData.hover_info[i].slice(0, 3).forEach(n => {
                    text += `<br>  "${n.token}" (${n.similarity.toFixed(3)})`;
                });
            }
            return text;
        }),
        hoverinfo: 'text',
        marker: { size: 7, color: '#58a6ff', opacity: 0.9 },
        name: `Layer ${layer} data`,
    };

    // Pattern trace
    const patternTrace = {
        x: normPattern.map(p => p[0]),
        y: normPattern.map(p => p[1]),
        z: normPattern.map(p => p[2]),
        mode: 'markers+lines',
        type: 'scatter3d',
        marker: { size: 2, color: '#f97316', opacity: 0.5 },
        line: { width: 1, color: '#f97316' },
        hoverinfo: 'skip',
        name: `${patternName} template`,
    };

    Plotly.newPlot('pattern-preview', [patternTrace, layerTrace], {
        paper_bgcolor: '#161b22',
        plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        scene: {
            bgcolor: '#0d1117',
            xaxis: { gridcolor: '#30363d' },
            yaxis: { gridcolor: '#30363d' },
            zaxis: { gridcolor: '#30363d' },
        },
        title: `Layer ${layer} vs "${patternName}" pattern (normalized overlay)`,
        margin: { l: 0, r: 0, t: 40, b: 0 },
        legend: {
            bgcolor: '#161b22',
            bordercolor: '#30363d',
            font: { color: '#c9d1d9' },
        },
    }, { responsive: true });
}

async function previewAndCompare(patternName, layer) {
    /**
     * Called from the "scan all patterns" table rows.
     * Shows the pattern preview and overlays with layer data.
     */
    document.getElementById('pattern-select').value = patternName;
    await showPatternLayerComparison(layer, patternName);
}

// Auto-preview on page load
setTimeout(() => {
    previewPattern();
}, 500);

