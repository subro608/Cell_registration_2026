#!/usr/bin/env python3
"""
Generate dino_repr_interactive_v2.html from dino_repr_interactive_v2_data.json.

Panels
------
§1  Three-panel scatter (PCA 2-D):
      Left   = DINO space (1152-d multi-scale CLS, what feeds the MLP)
      Centre = MLP mid-layer (256-d ReLU, intermediate representation)
      Right  = MLP output (64-d L2-norm, final representation)
    Both IV (●) and EX (◆) shown by default, same color per cell.
    Small translucent aug dots (size 4) cluster around large base dots (size 12).

§2  Discriminability matrix (cosine similarity 115×115).

§3  EX→IV contact sheet: EX cell as query, top-5 most similar IV cells shown.
    Consistent with the EX→IV retrieval direction used everywhere else.
"""
from __future__ import annotations
import json
from pathlib import Path

VIZ_DIR  = Path(__file__).resolve().parent
DATA_IN  = VIZ_DIR / "dino_repr_data.json"
HTML_OUT = VIZ_DIR / "dino_repr_interactive.html"


def build_html(data: dict) -> str:
    meta    = data["meta"]
    metrics = data["metrics"]
    N       = meta["n_cells"]
    K       = meta.get("n_aug_embed", 0)
    tag     = meta["tag"]

    # Panel titles
    dino_dim = meta.get("dino_dim", 1152)
    mid_dim  = meta.get("mlp_mid_dim", 256)
    out_dim  = meta.get("mlp_out_dim", 64)

    has_augs = (K > 0
                and "aug_dino_pca" in data.get("iv", {})
                and "aug_mid_pca"  in data.get("iv", {}))

    top1_pct = f"{metrics['ex_to_iv_top1']*100:.1f}%"
    top5_pct = f"{metrics['ex_to_iv_top5_cell_proto']*100:.1f}%"
    mrr_val  = f"{metrics['ex_to_iv_mrr']:.3f}"
    bal_val  = f"{metrics['best_balanced_val_knn']*100:.1f}%"
    best_ep  = metrics['best_epoch']
    ms_xy    = meta['ms_xy_um']
    ms_z     = meta['ms_z_um']
    thumb_xy = meta['thumb_xy_um']
    thumb_z  = meta['thumb_z_um']

    data_js = json.dumps(data, separators=(",", ":"))

    aug_note = (f"large dot=base patch · small dot=aug ({K}/cell)"
                if has_augs else "base patches only")
    aug_controls = ""
    if has_augs:
        aug_controls = """
    <div class="muted mini" style="margin-left:16px">Augs:</div>
    <div class="btn-group">
      <button id="btnAugOn" class="active" onclick="toggleAugs(true)">On</button>
      <button id="btnAugOff" onclick="toggleAugs(false)">Off</button>
    </div>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>DINO → MLP Representation Explorer v2</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{{box-sizing:border-box;}}
body{{margin:0;font-family:-apple-system,Segoe UI,sans-serif;background:#0f1116;color:#e6e6e6;}}
.wrap{{max-width:1800px;margin:0 auto;padding:16px;}}
h2{{margin:0 0 4px;font-size:1.4rem;color:#c8d8f8;}}
h3{{margin:16px 0 8px;font-size:1.05rem;color:#a8c0e8;border-bottom:1px solid #2b3242;padding-bottom:6px;}}
.badge{{display:inline-block;background:#1e2d50;border:1px solid #3a5080;border-radius:6px;
        padding:4px 10px;font-size:12px;color:#9ec6ff;margin:2px;}}
.badge b{{color:#6cf;}}
.card{{background:#171b24;border:1px solid #2b3242;border-radius:12px;padding:12px;}}
.muted{{color:#a7b0c0;font-size:13px;}}
.mini{{font-size:12px;color:#9fb0cc;}}
select{{background:#111520;color:#e6e6e6;border:1px solid #2f3748;border-radius:8px;padding:5px 8px;}}
button{{background:#1f2a44;color:#e6e6e6;border:1px solid #2f3748;border-radius:8px;
        padding:6px 12px;cursor:pointer;font-size:12px;}}
button:hover{{background:#2a3a5a;}}
button.active{{background:#1a3a6a;border-color:#4a80c0;color:#9ec6ff;}}
.contact-sheet{{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-top:8px;}}
.cs-cell{{background:#12161e;border:1px solid #2b3242;border-radius:8px;padding:6px;
          text-align:center;cursor:pointer;transition:border 0.15s;}}
.cs-cell:hover{{border-color:#4a80c0;}}
.cs-cell.correct{{border-color:#22cc66;box-shadow:0 0 6px #22cc6640;}}
.cs-cell img{{width:100%;border-radius:6px;image-rendering:pixelated;}}
.cs-cell .label{{font-size:10px;color:#9fb0cc;margin-top:4px;}}
.cs-cell .sim{{font-size:11px;color:#6cf;font-weight:600;}}
.cs-query{{grid-column:1;border:2px solid #5080c0;background:#141928;}}
.matrix-controls,.scatter-controls{{display:flex;gap:10px;align-items:center;
                                     flex-wrap:wrap;margin-bottom:8px;}}
.btn-group{{display:flex;gap:4px;}}
#scatterRow{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;}}
.panel-label{{font-size:11px;color:#7090b0;margin-bottom:3px;}}
.dir-badge{{display:inline-block;background:#1e3020;border:1px solid #2a5030;
            border-radius:5px;padding:2px 8px;font-size:11px;color:#5dc078;margin-left:8px;}}
</style>
</head>
<body>
<div class="wrap">
  <h2>DINO → MLP Representation Explorer v2</h2>
  <p class="muted">Run: <code style="color:#9ec6ff">{tag}</code> &nbsp;·&nbsp;
  DINOv2-small frozen · multiscale {ms_xy}µm × {ms_z}µm ·
  MLP {dino_dim}d→{mid_dim}d→{out_dim}d · Sinkhorn λ=0.7 · {N} cells</p>
  <div style="margin:8px 0">
    <span class="badge">EX→IV top-1 <b>{top1_pct}</b></span>
    <span class="badge">EX→IV proto top-5 <b>{top5_pct}</b></span>
    <span class="badge">MRR <b>{mrr_val}</b></span>
    <span class="badge">Bal. val kNN <b>{bal_val}</b></span>
    <span class="badge">Best epoch <b>{best_ep}</b></span>
    <span class="badge" style="color:#aaa">No IV↔EX pair labels in training</span>
  </div>
</div>

<!-- §1 Scatter ─────────────────────────────────────────────────────────── -->
<div class="wrap" style="padding-top:0">
  <h3>§1 — Embedding space (PCA 2-D) &nbsp; <span style="font-weight:400;font-size:12px;color:#7090b0">{aug_note}</span></h3>
  <div class="scatter-controls">
    <div class="muted mini">Modality:</div>
    <div class="btn-group">
      <button id="btnIV"   onclick="toggleModality('iv')">IV only</button>
      <button id="btnEX"   onclick="toggleModality('ex')">EX only</button>
      <button id="btnBoth" class="active" onclick="toggleModality('both')">Both ● IV  ◆ EX</button>
    </div>{aug_controls}
    <div class="mini" style="margin-left:auto;color:#6080a0">
      ● circle=IV &nbsp; ◆ diamond=EX &nbsp; same color=same cell &nbsp; large=base · small=aug
    </div>
  </div>
  <div id="scatterRow">
    <div class="card">
      <div class="panel-label">① DINO space — {dino_dim}-d multi-scale CLS  (input to MLP)</div>
      <div id="scatterDino" style="height:440px;"></div>
    </div>
    <div class="card">
      <div class="panel-label">② MLP mid-layer — {mid_dim}-d ReLU  (intermediate representation)</div>
      <div id="scatterMid" style="height:440px;"></div>
    </div>
    <div class="card">
      <div class="panel-label">③ MLP output — {out_dim}-d L2-norm  (final representation)</div>
      <div id="scatterMlp" style="height:440px;"></div>
    </div>
  </div>
  <div class="card" style="margin-top:10px">
    <div class="mini" id="scatterDetail">Click any point to see cell details.</div>
    <div id="cellDetailRow" style="display:none;grid-template-columns:120px 120px 1fr;gap:12px;margin-top:8px;">
      <div>
        <div class="muted mini">IV patch</div>
        <img id="detailIvDisp" style="width:100%;border-radius:8px;border:1px solid #2f3748;image-rendering:pixelated;"/>
      </div>
      <div>
        <div class="muted mini">EX patch</div>
        <img id="detailExDisp" style="width:100%;border-radius:8px;border:1px solid #2f3748;image-rendering:pixelated;"/>
      </div>
      <div id="detailStats" class="mini"></div>
    </div>
  </div>
</div>

<!-- §2 Matrix ──────────────────────────────────────────────────────────── -->
<div class="wrap" style="padding-top:0">
  <h3>§2 — Discriminability matrix (cosine similarity)</h3>
  <div class="card">
    <div class="matrix-controls">
      <div class="muted mini">Space:</div>
      <div class="btn-group">
        <button id="btnMatDino" onclick="setMatSpace('dino')">DINO</button>
        <button id="btnMatMlp"  class="active" onclick="setMatSpace('mlp')">MLP out</button>
      </div>
      <div class="muted mini" style="margin-left:12px">Query:</div>
      <div class="btn-group">
        <button id="btnMatIV" onclick="setMatQuery('iv')">IV × IV</button>
        <button id="btnMatEX" class="active" onclick="setMatQuery('ex')">EX × IV (retrieval)</button>
      </div>
      <div class="mini" id="matrixInfo" style="margin-left:auto;color:#9ec6ff"></div>
    </div>
    <div id="discMatrix" style="height:560px;"></div>
    <div class="mini" id="matrixCellInfo" style="margin-top:6px">Click a cell for details.</div>
  </div>
</div>

<!-- §3 Contact sheet ───────────────────────────────────────────────────── -->
<div class="wrap" style="padding-top:0">
  <h3>§3 — EX→IV contact sheet
    <span class="dir-badge">EX query → top-5 IV retrieved</span>
    <span style="font-size:11px;color:#7090b0;font-weight:400;margin-left:8px">{thumb_xy}µm/{thumb_z}µm crops · MLP cosine</span>
  </h3>
  <div class="card">
    <div style="display:flex;gap:10px;align-items:center;margin-bottom:8px;flex-wrap:wrap">
      <div class="muted mini">Select EX cell (query):</div>
      <select id="csExSelect" style="width:200px;"></select>
      <div class="mini" id="csCorrectBadge"></div>
      <div class="mini" style="margin-left:auto;color:#888">Green border = true IV match · Rank 1 first</div>
    </div>
    <div id="contactSheet" class="contact-sheet"></div>
    <div class="mini" style="margin-top:8px;color:#5a6a7a">
      Left (blue border) = EX query cell · Right 5 = top-5 IV cells ranked by MLP cosine similarity
    </div>
  </div>
  <div class="card" style="margin-top:12px">
    <div class="muted mini" style="margin-bottom:8px">
      EX→IV retrieval summary — all {N} cells (MLP space)
      <span class="dir-badge" style="margin-left:6px">EX→IV</span>
    </div>
    <div id="retrievalSummary"
         style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px;max-height:520px;overflow-y:auto;">
    </div>
  </div>
</div>

<div class="wrap">
  <p class="mini" style="color:#555;margin-top:24px">autoresearch · dino_repr_interactive_v2 · {tag}</p>
</div>

<script>
const D = {data_js};
const N = D.meta.n_cells;
const K = D.meta.n_aug_embed || 0;
const HAS_AUGS = K > 0 && D.iv.aug_dino_pca !== undefined && D.iv.aug_mid_pca !== undefined;
const cellIds = D.cell_ids;

let selectedCell  = 0;
let showModality  = 'both';
let showAugs      = true;
let matSpace      = 'mlp';
let matQuery      = 'ex';

// ── Color ────────────────────────────────────────────────────────────────
function cellColor(i) {{
  const h = (i * 137.508) % 360;
  return `hsl(${{h}},65%,58%)`;
}}

// ── Build scatter traces ──────────────────────────────────────────────────
// Returns Plotly trace array for a scatter panel.
// pca_iv/pca_ex: (N,2)  aug_pca_iv/aug_pca_ex: (N,K,2) or null
function buildTraces(pca_iv, pca_ex, aug_pca_iv, aug_pca_ex, modality, augsOn) {{
  const traces = [];
  const showIV = modality === 'iv' || modality === 'both';
  const showEX = modality === 'ex' || modality === 'both';

  // Aug dots first (rendered behind base dots)
  if (showIV && augsOn && HAS_AUGS && aug_pca_iv) {{
    for (let i = 0; i < N; i++) {{
      const xs = aug_pca_iv[i].map(p => p[0]);
      const ys = aug_pca_iv[i].map(p => p[1]);
      traces.push({{
        x: xs, y: ys, mode: 'markers',
        marker: {{ color: cellColor(i), size: 4, symbol: 'circle',
                   opacity: 0.28, line: {{ width: 0 }} }},
        hoverinfo: 'skip', showlegend: false,
        customdata: Array(K).fill(null),
      }});
    }}
  }}
  if (showEX && augsOn && HAS_AUGS && aug_pca_ex) {{
    for (let i = 0; i < N; i++) {{
      const xs = aug_pca_ex[i].map(p => p[0]);
      const ys = aug_pca_ex[i].map(p => p[1]);
      traces.push({{
        x: xs, y: ys, mode: 'markers',
        marker: {{ color: cellColor(i), size: 4, symbol: 'diamond',
                   opacity: 0.28, line: {{ width: 0 }} }},
        hoverinfo: 'skip', showlegend: false,
        customdata: Array(K).fill(null),
      }});
    }}
  }}

  // Base dots on top (large, clickable)
  if (showIV) {{
    for (let i = 0; i < N; i++) {{
      const sel = i === selectedCell;
      traces.push({{
        x: [pca_iv[i][0]], y: [pca_iv[i][1]], mode: 'markers',
        marker: {{ color: cellColor(i), size: sel ? 16 : 10, symbol: 'circle',
                   line: {{ width: sel ? 2.5 : 0.6, color: '#fff' }} }},
        text: [cellIds[i] + ' (IV)'], hoverinfo: 'text', showlegend: false,
        customdata: [{{ cellIdx: i, modality: 'iv' }}],
      }});
    }}
  }}
  if (showEX) {{
    for (let i = 0; i < N; i++) {{
      const sel = i === selectedCell;
      traces.push({{
        x: [pca_ex[i][0]], y: [pca_ex[i][1]], mode: 'markers',
        marker: {{ color: cellColor(i), size: sel ? 16 : 10, symbol: 'diamond',
                   opacity: 0.85, line: {{ width: sel ? 2.5 : 0.6, color: '#fff' }} }},
        text: [cellIds[i] + ' (EX)'], hoverinfo: 'text', showlegend: false,
        customdata: [{{ cellIdx: i, modality: 'ex' }}],
      }});
    }}
  }}
  return traces;
}}

const scatterLayout = (title) => ({{
  paper_bgcolor: '#171b24', plot_bgcolor: '#0f1318',
  margin: {{ l: 30, r: 10, t: 30, b: 30 }},
  title: {{ text: title, font: {{ size: 10, color: '#7090b0' }}, x: 0.02, xanchor: 'left' }},
  xaxis: {{ color: '#555', gridcolor: '#1c2030', zeroline: false, title: 'PC1' }},
  yaxis: {{ color: '#555', gridcolor: '#1c2030', zeroline: false, title: 'PC2' }},
  hovermode: 'closest', showlegend: false,
}});

function renderScatters() {{
  const aug_d_iv  = HAS_AUGS ? D.iv.aug_dino_pca : null;
  const aug_d_ex  = HAS_AUGS ? D.ex.aug_dino_pca : null;
  const aug_m_iv  = HAS_AUGS ? D.iv.aug_mid_pca  : null;
  const aug_m_ex  = HAS_AUGS ? D.ex.aug_mid_pca  : null;
  const aug_o_iv  = HAS_AUGS ? D.iv.aug_mlp_pca  : null;
  const aug_o_ex  = HAS_AUGS ? D.ex.aug_mlp_pca  : null;

  const augsOn = showAugs;

  Plotly.react('scatterDino',
    buildTraces(D.iv.dino_pca, D.ex.dino_pca, aug_d_iv, aug_d_ex, showModality, augsOn),
    scatterLayout('DINO PCA · ● IV  ◆ EX'));

  Plotly.react('scatterMid',
    buildTraces(D.iv.mid_pca, D.ex.mid_pca, aug_m_iv, aug_m_ex, showModality, augsOn),
    scatterLayout('MLP mid-layer PCA · ● IV  ◆ EX'));

  Plotly.react('scatterMlp',
    buildTraces(D.iv.mlp_pca, D.ex.mlp_pca, aug_o_iv, aug_o_ex, showModality, augsOn),
    scatterLayout('MLP output PCA · ● IV  ◆ EX'));
}}

function attachClick(divId) {{
  document.getElementById(divId).on('plotly_click', ev => {{
    const cd = ev.points[0].customdata;
    if (cd === null || cd === undefined) return;   // aug point → skip
    if (cd.cellIdx === undefined) return;
    selectCell(cd.cellIdx);
  }});
}}

// ── Cell detail ───────────────────────────────────────────────────────────
function dotNorm(a, b) {{
  let d = 0, na = 0, nb = 0;
  for (let k = 0; k < a.length; k++) {{ d += a[k]*b[k]; na += a[k]*a[k]; nb += b[k]*b[k]; }}
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}}

function selectCell(idx) {{
  selectedCell = idx;
  renderScatters();
  updateDetail(idx);
  buildContactSheet(idx);
  document.getElementById('csExSelect').value = idx;
}}

function updateDetail(idx) {{
  document.getElementById('scatterDetail').textContent = 'Cell ' + cellIds[idx];
  document.getElementById('detailIvDisp').src = 'data:image/png;base64,' + D.iv.disp_b64[idx];
  document.getElementById('detailExDisp').src = 'data:image/png;base64,' + D.ex.disp_b64[idx];
  const cosim  = dotNorm(D.iv.mlp_emb[idx], D.ex.mlp_emb[idx]);
  const top5iv = D.retrieval.ex_top5_iv[idx];
  const rank   = top5iv.indexOf(idx) + 1;
  const correct = D.retrieval.correct_in_top5[idx];
  document.getElementById('detailStats').innerHTML =
    `<b>${{cellIds[idx]}}</b><br>` +
    `IV↔EX cosine (true pair): <b style="color:#6cf">${{cosim.toFixed(4)}}</b><br>` +
    `EX→IV rank: <b style="color:${{rank<=5?'#2c6':'#f66'}}">${{rank > 0 ? rank : '>5'}}</b> ` +
    `<b style="color:${{correct?'#2c6':'#f66'}}">${{correct ? '✓ top-5' : '✗ not top-5'}}</b><br>` +
    `<span class="mini" style="color:#888">EX→IV top-5: ${{top5iv.map(j=>cellIds[j]).join(', ')}}</span>`;
  document.getElementById('cellDetailRow').style.display = 'grid';
}}

// ── Modality / aug toggles ────────────────────────────────────────────────
function toggleModality(m) {{
  showModality = m;
  ['btnIV','btnEX','btnBoth'].forEach(id => document.getElementById(id).classList.remove('active'));
  document.getElementById(m==='iv'?'btnIV': m==='ex'?'btnEX':'btnBoth').classList.add('active');
  renderScatters();
}}

function toggleAugs(on) {{
  showAugs = on;
  const a = document.getElementById('btnAugOn');
  const b = document.getElementById('btnAugOff');
  if (a) a.classList.toggle('active',  on);
  if (b) b.classList.toggle('active', !on);
  renderScatters();
}}

// ── Discriminability matrix ───────────────────────────────────────────────
function getMatrix() {{
  if (matSpace==='dino' && matQuery==='iv') return D.matrices.iv_iv_dino_cosim;
  if (matSpace==='dino' && matQuery==='ex') return D.matrices.ex_iv_dino_cosim;
  if (matSpace==='mlp'  && matQuery==='iv') return D.matrices.iv_iv_mlp_cosim;
  return D.matrices.ex_iv_mlp_cosim;
}}

function renderMatrix() {{
  const mat    = getMatrix();
  const diag   = mat.map((r,i) => r[i]);
  const offMean = mat.map((r,i) => {{
    const v = r.filter((_,j) => j!==i);
    return v.reduce((a,b) => a+b, 0) / v.length;
  }});
  const disc   = diag.map((d,i) => d - offMean[i]);
  const meanD  = disc.reduce((a,b) => a+b, 0) / N;
  document.getElementById('matrixInfo').textContent =
    `${{matSpace.toUpperCase()}} · Row=${{matQuery==='iv'?'IV':'EX'}}, Col=IV · mean disc=${{meanD.toFixed(4)}}`;
  Plotly.react('discMatrix', [{{
    z: mat, x: cellIds, y: cellIds, type: 'heatmap',
    colorscale: 'RdBu', zmid: 0,
    hovertemplate: 'Row %{{y}} → %{{x}}<br>cosim=%{{z:.4f}}<extra></extra>',
  }}], {{
    paper_bgcolor: '#171b24', plot_bgcolor: '#171b24',
    margin: {{ l:60, r:20, t:10, b:60 }},
    xaxis: {{ color:'#666', tickfont:{{size:8,color:'#888'}}, title:'IV gallery' }},
    yaxis: {{ color:'#666', tickfont:{{size:8,color:'#888'}},
              title: matQuery==='iv' ? 'IV query' : 'EX query', autorange:'reversed' }},
  }});
  document.getElementById('discMatrix').on('plotly_click', ev => {{
    const pt = ev.points[0];
    const ri = cellIds.indexOf(pt.y), ci = cellIds.indexOf(pt.x);
    if (ri < 0 || ci < 0) return;
    document.getElementById('matrixCellInfo').innerHTML =
      `Row <b style="color:#9ec6ff">${{pt.y}}</b> → Col <b style="color:#9ec6ff">${{pt.x}}</b> ` +
      `cosim=<b style="color:#6cf">${{mat[ri][ci].toFixed(4)}}</b>  disc=${{disc[ri].toFixed(4)}}`;
    selectCell(ri);
  }});
}}

function setMatSpace(s) {{
  matSpace = s;
  ['btnMatDino','btnMatMlp'].forEach(id => document.getElementById(id).classList.remove('active'));
  document.getElementById(s==='dino'?'btnMatDino':'btnMatMlp').classList.add('active');
  renderMatrix();
}}
function setMatQuery(q) {{
  matQuery = q;
  ['btnMatIV','btnMatEX'].forEach(id => document.getElementById(id).classList.remove('active'));
  document.getElementById(q==='iv'?'btnMatIV':'btnMatEX').classList.add('active');
  renderMatrix();
}}

// ── §3 Contact sheet — EX→IV direction ───────────────────────────────────
// exIdx = the EX cell used as query
// Shows top-5 IV cells retrieved
function buildContactSheet(exIdx) {{
  const top5iv   = D.retrieval.ex_top5_iv[exIdx];     // top-5 IV indices for this EX query
  const exIvMat  = D.matrices.ex_iv_mlp_cosim;
  const correct  = D.retrieval.correct_in_top5[exIdx];

  document.getElementById('csCorrectBadge').innerHTML = correct
    ? `<span style="color:#2c6;font-weight:600">✓ True IV match in top-5  (rank ${{top5iv.indexOf(exIdx)+1}})</span>`
    : `<span style="color:#f66">✗ True IV match not in top-5</span>`;

  // Query: EX cell (blue border)
  let html = `<div class="cs-cell cs-query">
    <img src="data:image/png;base64,${{D.ex.thumb_b64[exIdx]}}"/>
    <div class="label">${{cellIds[exIdx]}}</div>
    <div class="sim" style="color:#5b9cf6">EX query</div>
  </div>`;

  // Top-5 IV cells
  for (let r = 0; r < 5; r++) {{
    const ivIdx  = top5iv[r];
    const sim    = exIvMat[exIdx][ivIdx];
    const isTrue = ivIdx === exIdx;
    html += `<div class="cs-cell ${{isTrue ? 'correct' : ''}}"
        onclick="selectCell(${{ivIdx}})">
      <img src="data:image/png;base64,${{D.iv.thumb_b64[ivIdx]}}"/>
      <div class="label">${{cellIds[ivIdx]}}${{isTrue ? ' ✓' : ''}}</div>
      <div class="sim">${{sim.toFixed(3)}}</div>
    </div>`;
  }}
  document.getElementById('contactSheet').innerHTML = html;
}}

// EX select dropdown
function buildExSelect() {{
  const sel = document.getElementById('csExSelect');
  sel.innerHTML = '';
  for (let i = 0; i < N; i++) {{
    const o = document.createElement('option');
    o.value = i;
    const c = D.retrieval.correct_in_top5[i];
    o.textContent = cellIds[i] + (c ? ' ✓' : '');
    if (c) o.style.color = '#2c6';
    sel.appendChild(o);
  }}
  sel.addEventListener('change', e => {{
    const idx = parseInt(e.target.value);
    selectCell(idx);
    buildContactSheet(idx);
  }});
}}

// ── Retrieval summary ─────────────────────────────────────────────────────
// Each card shows an EX cell (query) and its EX→IV result.
// Clicking loads the EX→IV contact sheet for that cell.
function buildRetrievalSummary() {{
  const el = document.getElementById('retrievalSummary');
  const exIvMat = D.matrices.ex_iv_mlp_cosim;
  let html = '';
  for (let i = 0; i < N; i++) {{
    const top5iv = D.retrieval.ex_top5_iv[i];
    const c      = D.retrieval.correct_in_top5[i];
    const rank   = top5iv.indexOf(i) + 1;
    const sim    = exIvMat[i][i];   // true-pair cosine: EX_i vs IV_i
    html += `<div class="cs-cell" style="background:#12161e;padding:8px;${{c?'border-color:#22cc66':''}}"
      onclick="selectCell(${{i}})">
      <div style="display:flex;gap:6px;align-items:center">
        <img src="data:image/png;base64,${{D.ex.thumb_b64[i]}}"
          style="width:36px;height:36px;object-fit:cover;border-radius:4px;image-rendering:pixelated;
                 border:1px solid #5080c0"/>
        <div>
          <div class="mini" style="color:#9ec6ff;font-weight:600">${{cellIds[i]}} <span style="color:#5b9cf6;font-weight:400">EX</span></div>
          <div class="mini">true-pair sim ${{sim.toFixed(3)}}</div>
          <div class="mini" style="color:${{c?'#2c6':'#f66'}}">${{c ? `rank ${{rank}} ✓` : '✗ >5'}}</div>
        </div>
        <div style="margin-left:auto;font-size:10px;color:#555;max-width:90px;overflow:hidden;text-align:right">
          IV: ${{top5iv.map(j=>cellIds[j]).join(' ')}}
        </div>
      </div>
    </div>`;
  }}
  el.innerHTML = html;
}}

// ── Init ──────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {{
  renderScatters();
  ['scatterDino','scatterMid','scatterMlp'].forEach(attachClick);
  setMatSpace('mlp');
  setMatQuery('ex');
  buildExSelect();
  buildContactSheet(0);
  buildRetrievalSummary();
  selectCell(0);
}});
</script>
</body>
</html>"""


def main() -> None:
    print(f"Loading {DATA_IN} ...", flush=True)
    with open(DATA_IN) as f:
        data = json.load(f)
    n   = data['meta']['n_cells']
    k   = data['meta'].get('n_aug_embed', 0)
    has = (k > 0
           and 'aug_dino_pca' in data.get('iv', {})
           and 'aug_mid_pca'  in data.get('iv', {}))
    print(f"  {n} cells  n_aug={k}  has_mid_layer={'mid_pca' in data.get('iv',{})}  has_augs={has}",
          flush=True)
    html = build_html(data)
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {HTML_OUT} ({HTML_OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
