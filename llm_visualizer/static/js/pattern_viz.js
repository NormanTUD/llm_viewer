/**
 * pattern_viz.js v2 — Live progressive search with SSE.
 *
 * The search runs on the server and streams updates via Server-Sent Events.
 * Each "best_update" event re-renders the 3D plot immediately, so you
 * watch the best match improve in real time.
 */

let _searchAbort = null;  // To cancel ongoing searches
let _latestContext = null; // Cached background cloud
let _searchRunning = false;

// ─── Bindings ────────────────────────────────────────────

document.getElementById('btn-preview-pattern')?.addEventListener('click', previewPattern);
document.getElementById('btn-preview-filter')?.addEventListener('click', previewFilter);
document.getElementById('btn-search-pattern')?.addEventListener('click', searchPatternLive);
document.getElementById('btn-scan-all-patterns')?.addEventListener('click', scanAllPatterns);

document.getElementById('pattern-filter')?.addEventListener('change', () => {
    const el = document.getElementById('pattern-regex');
    el.style.display = document.getElementById('pattern-filter').value === 'regex' ? 'inline-block' : 'none';
});


// ─── Pattern Preview ─────────────────────────────────────

async function previewPattern() {
    const pattern = document.getElementById('pattern-select').value;
    const n = parseInt(document.getElementById('pattern-n-points').value) || 50;

    const data = await apiPost('/api/patterns/generate', { pattern, n_points: n * 3 });
    const pts = data.points;

    Plotly.newPlot('pattern-main-plot', [{
        x: pts.map(p => p[0]), y: pts.map(p => p[1]), z: pts.map(p => p[2]),
        mode: 'markers+lines', type: 'scatter3d',
        marker: { size: 3, color: pts.map((_, i) => i), colorscale: 'Turbo', opacity: 0.8 },
        line: { width: 2, color: '#58a6ff' },
        hovertext: pts.map((p, i) => `Point ${i}<br>${p.map(v => v.toFixed(3)).join(', ')}`),
        hoverinfo: 'text',
    }], darkLayout(`Pattern Preview: ${pattern}`), { responsive: true });
}


// ─── Filter Preview ──────────────────────────────────────

async function previewFilter() {
    const data = await apiPost('/api/patterns/filter_preview', {
        model: APP_STATE.model,
        filter: document.getElementById('pattern-filter').value,
        regex: document.getElementById('pattern-regex').value || null,
        max_show: 200,
    });
    document.getElementById('filter-preview').innerHTML =
        `<b>${data.count} tokens match</b>` +
        (data.count > 200 ? ' (showing 200)' : '') + '<br>' +
        `<span style="color:#a5f3fc">${data.tokens.map(t => `"${escapeHtml(t.text)}"`).join(', ')}</span>`;
}


// ─── LIVE SEARCH with SSE ────────────────────────────────

async function searchPatternLive() {
    if (_searchRunning) {
        // Abort previous search
        if (_searchAbort) _searchAbort.abort();
        _searchRunning = false;
        updateSearchButton(false);
        return;
    }

    const model = APP_STATE.model;
    const pattern = document.getElementById('pattern-select').value;
    const filter = document.getElementById('pattern-filter').value;
    const regex = document.getElementById('pattern-regex').value || null;
    const nPoints = parseInt(document.getElementById('pattern-n-points').value) || 40;
    const sampleSize = parseInt(document.getElementById('pattern-sample-size').value) || 10000;
    const searchSub = document.getElementById('pattern-search-subspaces')?.checked || false;

    const infoPanel = document.getElementById('pattern-info');
    const progressBar = document.getElementById('pattern-ranking');

    _searchRunning = true;
    updateSearchButton(true);

    infoPanel.innerHTML = `
        <h3>🔎 Searching live...</h3>
        <p>Pattern: <b>${pattern}</b> | Filter: <b>${filter}</b></p>
        <p><span class="loading"></span> Watch the 3D plot update as better matches are found!</p>
    `;
    progressBar.innerHTML = '';

    // We POST the search params, then read the SSE stream
    const body = JSON.stringify({
        model, pattern, n_pattern_points: nPoints,
        filter, regex: filter === 'regex' ? regex : null,
        sample_size: sampleSize, projection: 'pca',
        search_subspaces: searchSub,
        trace_key: APP_STATE.traceKey || null,
    });

    // Use fetch + ReadableStream to read SSE from POST
    const controller = new AbortController();
    _searchAbort = controller;

    try {
        const response = await fetch('/api/patterns/search_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: body,
            signal: controller.signal,
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let updateCount = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Parse SSE messages (each is "data: {...}\n\n")
            const messages = buffer.split('\n\n');
            buffer = messages.pop(); // Keep incomplete message in buffer

            for (const msg of messages) {
                const line = msg.trim();
                if (!line.startsWith('data: ')) continue;
                const jsonStr = line.substring(6);

                let data;
                try {
                    data = JSON.parse(jsonStr);
                } catch (e) {
                    continue;
                }

                if (data.status === 'progress') {
                    handleProgress(data, progressBar);
                } else if (data.status === 'best_update') {
                    updateCount++;
                    handleBestUpdate(data, infoPanel, updateCount);
                } else if (data.status === 'done') {
                    updateCount++;
                    handleBestUpdate(data, infoPanel, updateCount, true);
                } else if (data.status === 'error') {
                    infoPanel.innerHTML = `<h3>❌ Error</h3><p>${data.message}</p>`;
                }
            }
        }
    } catch (e) {
        if (e.name !== 'AbortError') {
            infoPanel.innerHTML += `<p style="color:#da3633;">Error: ${e.message}</p>`;
        }
    }

    _searchRunning = false;
    updateSearchButton(false);
}

function updateSearchButton(running) {
    const btn = document.getElementById('btn-search-pattern');
    if (running) {
        btn.innerHTML = '⏹ Stop Search';
        btn.style.background = '#da3633';
    } else {
        btn.innerHTML = '🔎 Find Pattern in Embeddings';
        btn.style.background = '#238636';
    }
}


// ─── Handle progress updates ─────────────────────────────

function handleProgress(data, progressBar) {
    const phase = data.phase || '';
    const msg = data.message || '';
    const trial = data.trial || 0;
    const total = data.total || 1;
    const pct = total > 0 ? Math.round(trial / total * 100) : 0;

    const phaseColors = {
        loading: '#58a6ff', coarse: '#f97316',
        fine: '#a855f7', greedy: '#22c55e', subspace: '#ec4899',
    };
    const color = phaseColors[phase] || '#8b949e';

    progressBar.innerHTML = `
        <div style="display:flex; align-items:center; gap:12px; padding:8px;">
            <span class="loading"></span>
            <span style="color:${color}; font-weight:bold; text-transform:uppercase;">${phase}</span>
            <span>${msg}</span>
            <div style="flex:1; background:#21262d; border-radius:4px; height:8px; min-width:100px;">
                <div style="background:${color}; width:${pct}%; height:100%; border-radius:4px;
                    transition: width 0.2s;"></div>
            </div>
            <span style="color:#6b7280;">${pct}%</span>
        </div>
    `;
}


// ─── Handle new best match — re-render 3D plot ──────────

function handleBestUpdate(data, infoPanel, updateCount, isFinal = false) {
    // ── Rebuild 3D plot ──────────────────────────────
    const traces = [];

    // 1. Background cloud
    const ctx = data.context;
    if (ctx && ctx.positions_3d && ctx.positions_3d.length > 0) {
        traces.push({
            x: ctx.positions_3d.map(p => p[0]),
            y: ctx.positions_3d.map(p => p[1]),
            z: ctx.positions_3d.map(p => p[2]),
            mode: 'markers', type: 'scatter3d',
            marker: { size: 2, color: '#2d333b', opacity: 0.35 },
            hovertext: ctx.tokens.map((t, i) =>
                `<b>"${t}"</b><br>ID: ${ctx.token_ids[i]}`
            ),
            hoverinfo: 'text',
            name: `Background (${ctx.tokens.length})`,
        });
    }

    // 2. Pattern template ghost
    if (data.pattern_overlay_3d && data.pattern_overlay_3d.length > 1) {
        traces.push({
            x: data.pattern_overlay_3d.map(p => p[0]),
            y: data.pattern_overlay_3d.map(p => p[1]),
            z: data.pattern_overlay_3d.map(p => p[2]),
            mode: 'lines', type: 'scatter3d',
            line: { width: 3, color: 'rgba(255,255,255,0.12)', dash: 'dot' },
            hoverinfo: 'skip',
            name: 'Pattern template',
        });
    }

    // 3. Matched tokens — path
    if (data.matched_positions_3d && data.matched_positions_3d.length > 1) {
        traces.push({
            x: data.matched_positions_3d.map(p => p[0]),
            y: data.matched_positions_3d.map(p => p[1]),
            z: data.matched_positions_3d.map(p => p[2]),
            mode: 'lines', type: 'scatter3d',
            line: { width: 4, color: '#f97316' },
            hoverinfo: 'skip',
            name: 'Match path',
        });
    }

    // 4. Matched tokens — points with hover
    if (data.matched_positions_3d && data.matched_positions_3d.length > 0) {
        const shortLabels = data.matched_tokens.map(t =>
            t.trim().length <= 10 ? t.trim() : ''
        );
        traces.push({
            x: data.matched_positions_3d.map(p => p[0]),
            y: data.matched_positions_3d.map(p => p[1]),
            z: data.matched_positions_3d.map(p => p[2]),
            mode: 'markers+text', type: 'scatter3d',
            text: shortLabels,
            textposition: 'top center',
            textfont: { size: 9, color: '#fbbf24' },
            hovertext: data.matched_tokens.map((t, i) =>
                `<b>"${t}"</b><br>` +
                `ID: ${data.matched_token_ids[i]}<br>` +
                `#${i + 1} of ${data.n_matched}<br>` +
                `<span style="color:#f97316">★ MATCHED</span>`
            ),
            hoverinfo: 'text',
            marker: {
                size: 7,
                color: data.matched_tokens.map((_, i) => i),
                colorscale: 'Hot', opacity: 1.0,
                line: { width: 1, color: '#fff' },
                colorbar: { title: 'Order', len: 0.4, x: 1.02 },
            },
            name: `Matched (${data.n_matched})`,
        });
    }

    const sim = (data.match_similarity || 0).toFixed(3);
    const statusIcon = isFinal ? '✅' : '🔄';
    const title = `${statusIcon} "${data.pattern_name}" in ${data.filter_type} — ` +
                  `${data.n_matched} tokens (sim: ${sim}) — update #${updateCount}`;

    // Use Plotly.react for FAST re-renders (no flicker)
    Plotly.react('pattern-main-plot', traces, darkLayout(title), { responsive: true });

    // ── Update info panel ────────────────────────────
    const simColor = data.match_similarity > 0.5 ? '#238636' :
                     data.match_similarity > 0.3 ? '#d29922' : '#da3633';

    infoPanel.innerHTML = `
        <h3>${isFinal ? '✅ Search Complete!' : '🔄 Best Match So Far (live)'}</h3>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px 16px; margin:8px 0;">
            <span style="color:#8b949e;">Pattern:</span> <b>${data.pattern_name}</b>
            <span style="color:#8b949e;">Filter:</span> <b>${data.filter_type}</b>
            <span style="color:#8b949e;">Searched:</span> <b>${(data.n_searched || 0).toLocaleString()} tokens</b>
            <span style="color:#8b949e;">Matched:</span> <b>${data.n_matched} tokens</b>
            <span style="color:#8b949e;">Similarity:</span>
                <b style="color:${simColor}; font-size:1.1em;">${sim}</b>
            <span style="color:#8b949e;">Strategy:</span>
                <b>${data.search_info?.strategy || 'N/A'}</b>
            <span style="color:#8b949e;">Elapsed:</span> <b>${data.elapsed || '?'}s</b>
            <span style="color:#8b949e;">Updates:</span> <b>${updateCount}</b>
        </div>

        <hr style="border-color:#30363d; margin:12px 0">
        <h4>Matched Tokens (in pattern order):</h4>
        <div style="max-height:280px; overflow-y:auto;">
            <table style="width:100%; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #30363d; position:sticky; top:0; background:#161b22;">
                    <th style="padding:3px; text-align:left;">#</th>
                    <th style="padding:3px; text-align:left;">Token</th>
                    <th style="padding:3px; text-align:left;">ID</th>
                </tr>
                ${(data.matched_tokens || []).map((t, i) => `
                    <tr style="border-bottom:1px solid #21262d;">
                        <td style="padding:3px; color:#6b7280;">${i + 1}</td>
                        <td style="padding:3px; color:#fbbf24; font-weight:bold;">"${escapeHtml(t)}"</td>
                        <td style="padding:3px; color:#6b7280;">${data.matched_token_ids[i]}</td>
                    </tr>
                `).join('')}
            </table>
        </div>
    `;
}


// ─── Scan all patterns ───────────────────────────────────

async function scanAllPatterns() {
    const filter = document.getElementById('pattern-filter').value;
    const sampleSize = parseInt(document.getElementById('pattern-sample-size').value) || 5000;
    const ranking = document.getElementById('pattern-ranking');

    ranking.innerHTML = `<p>📊 Scanning all patterns against <b>${filter}</b>...
        <span class="loading"></span> (this runs each pattern — may take a minute)</p>`;

    const data = await apiPost('/api/patterns/scan_all', {
        model: APP_STATE.model, filter, sample_size: sampleSize, projection: 'pca',
    });

    if (!data.results?.length) {
        ranking.innerHTML = '<p>No results.</p>';
        return;
    }

    ranking.innerHTML = `
        <h3>All Patterns — "${filter}" tokens</h3>
        <table style="width:100%; border-collapse:collapse; margin-top:8px;">
            <tr style="border-bottom:1px solid #30363d;">
                <th style="padding:6px; text-align:left;">#</th>
                <th style="padding:6px; text-align:left;">Pattern</th>
                <th style="padding:6px; text-align:left;">Similarity</th>
                <th style="padding:6px; text-align:left;">Matched</th>
                <th style="padding:6px; text-align:left;">Samples</th>
                <th style="padding:6px; text-align:left;"></th>
            </tr>
            ${data.results.map((r, i) => {
                const sim = (r.similarity || 0);
                const col = sim > 0.5 ? '#238636' : sim > 0.3 ? '#d29922' : '#da3633';
                return `
                <tr style="border-bottom:1px solid #21262d; cursor:pointer;"
                    onclick="quickSearch('${r.pattern}')"
                    onmouseover="this.style.background='#21262d'"
                    onmouseout="this.style.background='transparent'">
                    <td style="padding:6px;">${i + 1}</td>
                    <td style="padding:6px; font-weight:bold;">${r.pattern}</td>
                    <td style="padding:6px; color:${col};">${sim.toFixed(4)}</td>
                    <td style="padding:6px;">${r.n_matched || '?'}</td>
                    <td style="padding:6px; color:#8b949e; font-size:0.8rem;">
                        ${(r.sample_tokens || []).slice(0, 5).map(t => `"${t.trim()}"`).join(', ')}
                    </td>
                    <td style="padding:6px;">
                        <div style="height:12px; width:${(sim * 200).toFixed(0)}px;
                            border-radius:3px; background:linear-gradient(90deg,#1e3a5f,${col});"></div>
                    </td>
                </tr>`;
            }).join('')}
        </table>
        <p class="hint">Click a row to run a full live search for that pattern.</p>
    `;

    // Bar chart
    Plotly.newPlot('pattern-bar-chart', [{
        x: data.results.map(r => r.pattern),
        y: data.results.map(r => r.similarity || 0),
        type: 'bar',
        marker: {
            color: data.results.map(r => r.similarity || 0),
            colorscale: [[0, '#da3633'], [0.5, '#d29922'], [1, '#238636']],
        },
    }], {
        paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117', font: { color: '#c9d1d9' },
        title: `Pattern similarity — "${filter}" tokens`,
        margin: { l: 50, r: 20, t: 50, b: 80 },
    }, { responsive: true });
}

function quickSearch(patternName) {
    document.getElementById('pattern-select').value = patternName;
    searchPatternLive();
}


// ─── Layout helper ───────────────────────────────────────

function darkLayout(title) {
    return {
        paper_bgcolor: '#161b22', plot_bgcolor: '#0d1117',
        font: { color: '#c9d1d9' },
        scene: {
            bgcolor: '#0d1117',
            xaxis: { gridcolor: '#30363d', color: '#6b7280' },
            yaxis: { gridcolor: '#30363d', color: '#6b7280' },
            zaxis: { gridcolor: '#30363d', color: '#6b7280' },
        },
        title: { text: title, font: { size: 13 } },
        margin: { l: 0, r: 0, t: 50, b: 0 },
        legend: { bgcolor: 'rgba(22,27,34,0.8)', bordercolor: '#30363d',
                  font: { color: '#c9d1d9', size: 10 } },
    };
}

// Auto-preview
setTimeout(() => { previewPattern(); }, 400);
