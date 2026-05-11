#!/usr/bin/env python3
"""
Build viewer_rbf.html — exact clone of viewer.html + RBF kernel interpolation.
Same volume, same 4x downsample, same rendering. Only addition: RBF sliders.
"""
import numpy as np
import tifffile
from PIL import Image
import io, base64

# --- Load ORIGINAL volume (same source as viewer.html) ---
print("Loading original stitched volume...")
vol = tifffile.imread("/Users/neurolab/neuroinformatics/margaret/registration_video/stitched/stitched_gfp_elastix_1um_isotropic.tif")
print(f"  Shape: {vol.shape}, dtype: {vol.dtype}, range: {vol.min()}-{vol.max()}")

NZ_orig, NY_orig, NX_orig = vol.shape  # 516, 2748, 2748

# --- 4x downsample ALL axes (exactly like viewer.html) ---
DS = 4
nz = NZ_orig // DS  # 129
ny = NY_orig // DS  # 687
nx = NX_orig // DS  # 687
print(f"4x downsample all axes: ({nz}, {ny}, {nx})")

# Pure subsampling (every 4th voxel — matches viewer.html)
print("Subsampling (every 4th voxel)...")
vol_ds = vol[::DS, ::DS, ::DS].astype(np.float32)
vol_ds = vol_ds[:nz, :ny, :nx]
print(f"  Result: {vol_ds.shape}")

# --- Normalize by /4000 (matches viewer.html) ---
NORM = 4000
print(f"  Normalizing by {NORM}")
vol_u8 = np.clip(vol_ds / NORM * 255, 0, 255).astype(np.uint8)

# --- Encode slices as base64 PNGs ---
print("Encoding slices as base64 PNG...")
b64_slices = []
for z in range(nz):
    img = Image.fromarray(vol_u8[z])
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    b64_slices.append(base64.b64encode(buf.getvalue()).decode('ascii'))

total_mb = sum(len(s) for s in b64_slices) / 1e6
print(f"  {nz} slices, total base64: {total_mb:.1f} MB")

# Aspect ratio: same as viewer.html
aspect_z = nz / max(ny, nx)  # 129/687 = 0.187773
print(f"  ASPECT_Z = {aspect_z:.6f}")

# --- Build HTML (exact clone of viewer.html + RBF) ---
print("Generating HTML...")

# Slice data array
slice_js = ",".join(f'"{s}"' for s in b64_slices)

html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Ex-vivo Hippocampus 3D — RBF Kernel</title>
<style>
  body {{ margin:0; overflow:hidden; background:#000; color:#ddd; font-family:sans-serif; font-size:12px; }}
  #info {{ position:absolute; top:10px; left:10px; z-index:10; background:rgba(0,0,0,0.7); padding:8px; border-radius:4px; }}
  #controls {{ position:absolute; top:10px; right:10px; z-index:10; background:rgba(0,0,0,0.7);
               padding:8px 12px; border-radius:4px; }}
  #controls label {{ display:block; margin:4px 0; }}
</style>
</head><body>
<div id="info">
  <b>Ex-vivo Hippocampus — RBF Kernel Interpolation</b><br>
  Display: {nz}&times;{ny}&times;{nx} (4x downsampled)<br>
  Drag: rotate | Scroll: zoom | Shift+drag: pan
</div>
<div id="controls">
  <label>Threshold: <input type="range" id="thresh" min="0" max="255" value="15" style="width:120px"><span id="threshVal">15</span></label>
  <label>Opacity: <input type="range" id="opacity" min="1" max="100" value="30" style="width:120px"><span id="opVal">30</span></label>
  <label>Point size: <input type="range" id="psize" min="1" max="10" value="2" style="width:120px"><span id="psVal">2</span></label>
  <label><input type="checkbox" id="autorot" checked> Auto-rotate</label>
  <label>Colormap: <select id="cmap"><option value="green">Green</option><option value="hot">Hot</option><option value="gray">Gray</option><option value="cyan">Cyan</option></select></label>
  <hr style="border-color:#555;margin:6px 0">
  <label style="color:#0f0;font-weight:bold">RBF Lengthscale: <input type="range" id="rbf_l" min="0" max="50" value="0" step="1" style="width:120px"><span id="rbfVal">0.0</span></label>
  <label style="color:#0f0">Spread radius: <input type="range" id="rbf_r" min="0" max="10" value="3" step="1" style="width:120px"><span id="rbfRVal">3</span></label>
  <div style="font-size:10px;color:#888;margin-top:4px">k(d)=exp(-d&sup2;/2l&sup2;) | l=0: no spread<br><span id="ptCount"></span></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const NZ={nz}, NY={ny}, NX={nx};
const ASPECT_Z = {aspect_z:.6f};
const sliceData = [{slice_js}];
let scene, camera, renderer, points, rotY=0, rotX=-0.3;
let dragging=false, lastX=0, lastY=0, zoom=2.5, panX=0, panY=0;
let autoRotate=true;

function colormap(v, name) {{
  if(name==='green') return [0, v, 0];
  if(name==='hot') return [Math.min(v*2,1), Math.max(v*2-1,0)*0.8, Math.max(v*3-2,0)];
  if(name==='cyan') return [0, v*0.8, v];
  return [v, v, v];
}}

async function loadSlices() {{
  const voxels = [];
  for(let z=0; z<NZ; z++) {{
    const img = new Image();
    await new Promise(r => {{ img.onload=r; img.src='data:image/png;base64,'+sliceData[z]; }});
    const c = document.createElement('canvas');
    c.width=img.width; c.height=img.height;
    const ctx = c.getContext('2d');
    ctx.drawImage(img,0,0);
    const d = ctx.getImageData(0,0,img.width,img.height).data;
    for(let y=0; y<NY; y++)
      for(let x=0; x<NX; x++)
        voxels.push({{x, y, z, val: d[(y*NX+x)*4]}});
  }}
  return voxels;
}}

async function init() {{
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 100);
  camera.position.z = zoom;
  renderer = new THREE.WebGLRenderer({{antialias:true}});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x000000);
  document.body.appendChild(renderer.domElement);
  const voxels = await loadSlices();
  buildPoints(voxels);
  animate();
}}

function buildPoints(voxels) {{
  const thresh = +document.getElementById('thresh').value;
  const opac = +document.getElementById('opacity').value / 100;
  const ps = +document.getElementById('psize').value;
  const cmapName = document.getElementById('cmap').value;
  const sigma = +document.getElementById('rbf_l').value / 10;  // 0-5.0
  const spreadR = +document.getElementById('rbf_r').value;      // 0-10

  if(points) scene.remove(points);
  const filtered = voxels.filter(v => v.val > thresh);

  // Pre-compute RBF weights
  const sigma2 = sigma * sigma;
  const weights = [];
  for(let d = -spreadR; d <= spreadR; d++) {{
    if(sigma < 0.01) {{ weights.push(d === 0 ? 1.0 : 0.0); }}
    else {{ weights.push(Math.exp(-(d*d)/(2*sigma2))); }}
  }}

  // Count total points (original + spread copies)
  let totalPts = filtered.length;  // originals
  if(sigma >= 0.01) {{
    for(let i = 0; i < filtered.length; i++) {{
      const z = filtered[i].z;
      for(let d = -spreadR; d <= spreadR; d++) {{
        if(d === 0) continue;
        const zz = z + d;
        if(zz >= 0 && zz < NZ && weights[d + spreadR] > 0.05) totalPts++;
      }}
    }}
  }}

  const n = totalPts;
  document.getElementById('ptCount').textContent = n.toLocaleString() + ' points';
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  let idx = 0;

  for(let i = 0; i < filtered.length; i++) {{
    const v = filtered[i];
    const px = (v.x/NX - 0.5) * 2;
    const py = -(v.y/NY - 0.5) * 2;
    const pz = (v.z/NZ - 0.5) * 2 * ASPECT_Z;
    const nv = v.val / 255;
    const [r,g,b] = colormap(nv, cmapName);

    // Original point
    pos[idx*3] = px; pos[idx*3+1] = py; pos[idx*3+2] = pz;
    col[idx*3] = r; col[idx*3+1] = g; col[idx*3+2] = b;
    idx++;

    // RBF spread copies
    if(sigma >= 0.01) {{
      for(let d = -spreadR; d <= spreadR; d++) {{
        if(d === 0) continue;
        const zz = v.z + d;
        if(zz < 0 || zz >= NZ) continue;
        const w = weights[d + spreadR];
        if(w < 0.05) continue;
        const nv_w = nv * w;
        const [rw,gw,bw] = colormap(nv_w, cmapName);
        pos[idx*3] = px;
        pos[idx*3+1] = py;
        pos[idx*3+2] = (zz/NZ - 0.5) * 2 * ASPECT_Z;
        col[idx*3] = rw; col[idx*3+1] = gw; col[idx*3+2] = bw;
        idx++;
      }}
    }}
  }}

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos.subarray(0, idx*3), 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col.subarray(0, idx*3), 3));
  const mat = new THREE.PointsMaterial({{
    size: ps * 0.01, vertexColors: true, transparent: true, opacity: opac,
    blending: THREE.AdditiveBlending, depthWrite: false
  }});
  points = new THREE.Points(geo, mat);
  scene.add(points);
  window._voxels = voxels;
}}

function animate() {{
  requestAnimationFrame(animate);
  if(autoRotate && !dragging) rotY += 0.003;
  if(points) {{
    points.rotation.y = rotY;
    points.rotation.x = rotX;
    points.position.x = panX;
    points.position.y = panY;
  }}
  camera.position.z = zoom;
  renderer.render(scene, camera);
}}

document.addEventListener('mousedown', e => {{ dragging=true; lastX=e.clientX; lastY=e.clientY; }});
document.addEventListener('mouseup', () => dragging=false);
document.addEventListener('mousemove', e => {{
  if(!dragging) return;
  const dx = e.clientX-lastX, dy = e.clientY-lastY;
  if(e.shiftKey) {{ panX+=dx*0.002; panY-=dy*0.002; }}
  else {{ rotY+=dx*0.005; rotX+=dy*0.005; }}
  lastX=e.clientX; lastY=e.clientY;
}});
document.addEventListener('wheel', e => {{ zoom=Math.max(0.5, Math.min(10, zoom+e.deltaY*0.002)); }});
window.addEventListener('resize', () => {{
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

// Slider updates — debounced rebuild
let rebuildTimer = null;
['thresh','opacity','psize','cmap','rbf_l','rbf_r'].forEach(id => {{
  document.getElementById(id).addEventListener('input', () => {{
    document.getElementById('threshVal').textContent=document.getElementById('thresh').value;
    document.getElementById('opVal').textContent=document.getElementById('opacity').value;
    document.getElementById('psVal').textContent=document.getElementById('psize').value;
    document.getElementById('rbfVal').textContent=(+document.getElementById('rbf_l').value/10).toFixed(1);
    document.getElementById('rbfRVal').textContent=document.getElementById('rbf_r').value;
    clearTimeout(rebuildTimer);
    rebuildTimer = setTimeout(() => {{ if(window._voxels) buildPoints(window._voxels); }}, 200);
  }});
}});
document.getElementById('autorot').addEventListener('change', e => autoRotate=e.target.checked);

init();
</script></body></html>
'''

out_path = "/Users/neurolab/neuroinformatics/margaret/3d_viewer/viewer_rbf.html"
with open(out_path, 'w') as f:
    f.write(html)
print(f"\nDone! {out_path} ({len(html)/1e6:.1f} MB)")
