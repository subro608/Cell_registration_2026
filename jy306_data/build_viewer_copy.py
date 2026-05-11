#!/usr/bin/env python3
"""
Exact recreation of viewer.html from the stitched volume.
4x downsample all axes → 129×687×687. Same rendering code.
"""
import numpy as np
import tifffile
from PIL import Image
import io, base64

print("Loading volume...")
vol = tifffile.imread("/Users/neurolab/neuroinformatics/margaret/registration_video/stitched/stitched_gfp_elastix_1um_isotropic.tif")
print(f"  Shape: {vol.shape}, dtype: {vol.dtype}")

NZ_orig, NY_orig, NX_orig = vol.shape
DS = 4
nz = NZ_orig // DS
ny = NY_orig // DS
nx = NX_orig // DS
print(f"4x DS: ({nz}, {ny}, {nx})")

# Pure subsampling: take every 4th voxel
print("Subsampling (every 4th voxel)...")
vol_ds = vol[::DS, ::DS, ::DS].astype(np.float32)
# Trim to exact shape
vol_ds = vol_ds[:nz, :ny, :nx]
print(f"  Result: {vol_ds.shape}")

# Normalize by ~4000 (matches original viewer.html — keeps most pixels dark,
# only brightest cells punch through)
NORM = 4000
print(f"  Normalizing by {NORM}")
vol_u8 = np.clip(vol_ds / NORM * 255, 0, 255).astype(np.uint8)

print("Encoding PNGs...")
b64 = []
for z in range(nz):
    img = Image.fromarray(vol_u8[z])
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    b64.append(base64.b64encode(buf.getvalue()).decode('ascii'))

aspect_z = nz / max(ny, nx)
slice_js = ",".join(f'"{s}"' for s in b64)

html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Ex-vivo Hippocampus 3D (copy)</title>
<style>
  body {{ margin:0; overflow:hidden; background:#000; color:#ddd; font-family:sans-serif; font-size:12px; }}
  #info {{ position:absolute; top:10px; left:10px; z-index:10; background:rgba(0,0,0,0.7); padding:8px; border-radius:4px; }}
  #controls {{ position:absolute; top:10px; right:10px; z-index:10; background:rgba(0,0,0,0.7);
               padding:8px 12px; border-radius:4px; }}
  #controls label {{ display:block; margin:4px 0; }}
</style>
</head><body>
<div id="info">
  <b>Ex-vivo Hippocampus 3D</b><br>
  Display: {nz}&times;{ny}&times;{nx} (4x downsampled)<br>
  Drag: rotate | Scroll: zoom | Shift+drag: pan
</div>
<div id="controls">
  <label>Threshold: <input type="range" id="thresh" min="0" max="255" value="15" style="width:120px"><span id="threshVal">15</span></label>
  <label>Opacity: <input type="range" id="opacity" min="1" max="100" value="30" style="width:120px"><span id="opVal">30</span></label>
  <label>Point size: <input type="range" id="psize" min="1" max="10" value="2" style="width:120px"><span id="psVal">2</span></label>
  <label><input type="checkbox" id="autorot" checked> Auto-rotate</label>
  <label>Colormap: <select id="cmap"><option value="green">Green</option><option value="hot">Hot</option><option value="gray">Gray</option><option value="cyan">Cyan</option></select></label>
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
  if(points) scene.remove(points);
  const filtered = voxels.filter(v => v.val > thresh);
  const n = filtered.length;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for(let i=0; i<n; i++) {{
    const v = filtered[i];
    pos[i*3]   = (v.x/NX - 0.5) * 2;
    pos[i*3+1] = -(v.y/NY - 0.5) * 2;
    pos[i*3+2] = (v.z/NZ - 0.5) * 2 * ASPECT_Z;
    const nv = v.val / 255;
    const [r,g,b] = colormap(nv, cmapName);
    col[i*3]=r; col[i*3+1]=g; col[i*3+2]=b;
  }}
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
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
['thresh','opacity','psize','cmap'].forEach(id => {{
  document.getElementById(id).addEventListener('input', () => {{
    document.getElementById('threshVal').textContent=document.getElementById('thresh').value;
    document.getElementById('opVal').textContent=document.getElementById('opacity').value;
    document.getElementById('psVal').textContent=document.getElementById('psize').value;
    if(window._voxels) buildPoints(window._voxels);
  }});
}});
document.getElementById('autorot').addEventListener('change', e => autoRotate=e.target.checked);
init();
</script></body></html>
'''

out = "/Users/neurolab/neuroinformatics/margaret/3d_viewer/viewer_copy.html"
with open(out, 'w') as f:
    f.write(html)
print(f"Done! {out} ({len(html)/1e6:.1f} MB)")
