#!/usr/bin/env python3
"""
Dual 3D viewer: IOU-only stitched ex-vivo + JY306 in-vivo
with cyan arrows connecting matching landmarks.
"""
import numpy as np
import tifffile
import cv2
from PIL import Image
from scipy.ndimage import median_filter
import io, base64, json

BASE = '/Users/neurolab/neuroinformatics/margaret'

# ============================================================
# Load ex-vivo volume
# ============================================================
print("Loading IOU-only stitched 1µm isotropic volume...")
vol_path = f'{BASE}/registration_video/stitched/stitched_gfp_iou_only_1um_isotropic.tif'
with tifffile.TiffFile(vol_path) as tif:
    n_pages = len(tif.pages)
    h, w = tif.pages[0].shape
    ev_vol = np.zeros((n_pages, h, w), dtype=np.uint16)
    for i, page in enumerate(tif.pages):
        ev_vol[i] = page.asarray()
print(f"  Ex-vivo shape: {ev_vol.shape}")

# Load in-vivo volume
print("Loading JY306 in-vivo volume...")
iv_vol = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
print(f"  In-vivo shape: {iv_vol.shape}")

# Load landmarks
print("Loading landmarks...")
lm = np.load(f'{BASE}/registration_video/affine_3d_iou_results.npz', allow_pickle=True)
ev_um = lm['ev_stitched_um']  # (N, 3) x,y,z in µm
iv_um = lm['iv_um']           # (N, 3) x,y,z in µm
N_LM = ev_um.shape[0]
print(f"  {N_LM} landmarks")

# ============================================================
# Process ex-vivo: DS4, equalize, /4000
# ============================================================
DS_EX = 4
nz_ex = ev_vol.shape[0] // DS_EX
ny_ex = ev_vol.shape[1] // DS_EX
nx_ex = ev_vol.shape[2] // DS_EX
print(f"\nEx-vivo DS{DS_EX}: ({nz_ex}, {ny_ex}, {nx_ex})")

ev_ds = ev_vol[::DS_EX, ::DS_EX, ::DS_EX].astype(np.float32)[:nz_ex, :ny_ex, :nx_ex]
del ev_vol

# Per-slice equalization
nz_vals = ev_ds[ev_ds > 0]
if len(nz_vals) > 0:
    gmean = nz_vals.mean()
    for z in range(nz_ex):
        sl = ev_ds[z]
        mask = sl > 0
        if mask.sum() > 100:
            sm = sl[mask].mean()
            if sm > 1:
                ev_ds[z][mask] *= (gmean / sm)

ev_u8 = np.clip(ev_ds / 4000 * 255, 0, 255).astype(np.uint8)
del ev_ds

print("Encoding ex-vivo PNGs...")
ev_b64 = []
for z in range(nz_ex):
    buf = io.BytesIO()
    Image.fromarray(ev_u8[z]).save(buf, format='PNG', optimize=True)
    ev_b64.append(base64.b64encode(buf.getvalue()).decode('ascii'))
del ev_u8

# ============================================================
# Process in-vivo: median filter BG sub, normalize
# ============================================================
nz_iv, ny_iv, nx_iv = iv_vol.shape
print(f"\nIn-vivo: ({nz_iv}, {ny_iv}, {nx_iv})")

# Median filter background subtraction
print("Median filter BG subtraction...")
iv_proc = np.zeros_like(iv_vol)
for z in range(nz_iv):
    bg = median_filter(iv_vol[z], size=15)
    iv_proc[z] = np.clip(iv_vol[z] - bg, 0, None)
del iv_vol

# Normalize
vals = iv_proc[iv_proc > 0]
if len(vals) > 0:
    p995 = np.percentile(vals, 99.5)
    iv_u8 = np.clip(iv_proc / max(p995, 1) * 255, 0, 255).astype(np.uint8)
else:
    iv_u8 = np.zeros_like(iv_proc, dtype=np.uint8)
del iv_proc

print("Encoding in-vivo PNGs...")
iv_b64 = []
for z in range(nz_iv):
    buf = io.BytesIO()
    Image.fromarray(iv_u8[z]).save(buf, format='PNG', optimize=True)
    iv_b64.append(base64.b64encode(buf.getvalue()).decode('ascii'))
del iv_u8

# ============================================================
# Compute landmark positions in viewer coordinates
# ============================================================
# Ex-vivo: µm → DS4 pixel → normalized [-1,1]
# Stitched volume is 1µm/px, so µm = pixel
ev_lm_px = ev_um.copy()
ev_lm_px[:, 0] /= DS_EX  # x
ev_lm_px[:, 1] /= DS_EX  # y
ev_lm_px[:, 2] /= DS_EX  # z (1µm iso, so z_µm = z_px)

az_ex = nz_ex / max(ny_ex, nx_ex)

ev_lm_norm = np.zeros((N_LM, 3))
ev_lm_norm[:, 0] = (ev_lm_px[:, 0] / nx_ex - 0.5) * 2
ev_lm_norm[:, 1] = -(ev_lm_px[:, 1] / ny_ex - 0.5) * 2
ev_lm_norm[:, 2] = (ev_lm_px[:, 2] / nz_ex - 0.5) * 2 * az_ex

# In-vivo: µm → JY306 pixel → normalized [-1,1]
IV_XY_UM = 0.6835
IV_Z_UM = 3.0
iv_lm_px = np.zeros((N_LM, 3))
iv_lm_px[:, 0] = iv_um[:, 0] / IV_XY_UM  # x in JY306 px
iv_lm_px[:, 1] = iv_um[:, 1] / IV_XY_UM  # y
iv_lm_px[:, 2] = iv_um[:, 2] / IV_Z_UM   # z

az_iv = nz_iv / max(ny_iv, nx_iv)

iv_lm_norm = np.zeros((N_LM, 3))
iv_lm_norm[:, 0] = (iv_lm_px[:, 0] / nx_iv - 0.5) * 2
iv_lm_norm[:, 1] = -(iv_lm_px[:, 1] / ny_iv - 0.5) * 2
iv_lm_norm[:, 2] = (iv_lm_px[:, 2] / nz_iv - 0.5) * 2 * az_iv

# Spacing between volumes
SPACING = 3.0

# Shift: ex-vivo left, in-vivo right
ev_lm_shifted = ev_lm_norm.copy()
ev_lm_shifted[:, 0] -= SPACING / 2

iv_lm_shifted = iv_lm_norm.copy()
iv_lm_shifted[:, 0] += SPACING / 2

# Format for JS
lines_js = []
for i in range(N_LM):
    lines_js.append(f"[{ev_lm_shifted[i,0]:.4f},{ev_lm_shifted[i,1]:.4f},{ev_lm_shifted[i,2]:.4f},"
                    f"{iv_lm_shifted[i,0]:.4f},{iv_lm_shifted[i,1]:.4f},{iv_lm_shifted[i,2]:.4f}]")
lines_js_str = ",".join(lines_js)

ev_slice_js = ",".join(f'"{s}"' for s in ev_b64)
iv_slice_js = ",".join(f'"{s}"' for s in iv_b64)

# ============================================================
# Build HTML
# ============================================================
print("\nBuilding HTML...")

js_code = f"""
const EV_NZ={nz_ex}, EV_NY={ny_ex}, EV_NX={nx_ex}, EV_AZ={az_ex:.6f};
const IV_NZ={nz_iv}, IV_NY={ny_iv}, IV_NX={nx_iv}, IV_AZ={az_iv:.6f};
const SPACING={SPACING};
const evSlices=[{ev_slice_js}];
const ivSlices=[{iv_slice_js}];
const landmarks=[{lines_js_str}];

let scene, camera, renderer;
let evPoints, ivPoints, lineGroup;
let rotY=0, rotX=-0.3, zoom=4.0, panX=0, panY=0;
let dragging=false, lastX=0, lastY=0, autoRotate=false;
let evRaw=null, ivRaw=null;

function colormap(v, name) {{
  if(name==='green') return [0,v,0];
  if(name==='cyan') return [0,v*0.8,v];
  if(name==='hot') return [Math.min(v*2,1),Math.max(v*2-1,0)*0.8,Math.max(v*3-2,0)];
  if(name==='magenta') return [v,0,v*0.8];
  return [v,v,v];
}}

async function loadVol(slices, NZ, NY, NX) {{
  const raw = new Uint8Array(NZ*NY*NX);
  for(let z=0;z<NZ;z++) {{
    const img=new Image();
    await new Promise(r=>{{img.onload=r;img.src='data:image/png;base64,'+slices[z];}});
    const c=document.createElement('canvas');c.width=img.width;c.height=img.height;
    const ctx=c.getContext('2d');ctx.drawImage(img,0,0);
    const d=ctx.getImageData(0,0,img.width,img.height).data;
    const off=z*NY*NX;
    for(let y=0;y<NY;y++) for(let x=0;x<NX;x++) raw[off+y*NX+x]=d[(y*NX+x)*4];
    if(z%10===0) document.getElementById('status').textContent='Loading z='+z;
  }}
  return raw;
}}

function buildCloud(raw, NZ, NY, NX, AZ, thresh, opac, ps, cmap, offsetX) {{
  const ptsX=[],ptsY=[],ptsZ=[],ptsV=[];
  for(let z=0;z<NZ;z++) for(let y=0;y<NY;y++) for(let x=0;x<NX;x++) {{
    const v=raw[z*NY*NX+y*NX+x];
    if(v>thresh) {{ ptsX.push(x);ptsY.push(y);ptsZ.push(z);ptsV.push(v); }}
  }}
  const n=ptsX.length;
  const pos=new Float32Array(n*3), col=new Float32Array(n*3);
  for(let i=0;i<n;i++) {{
    pos[i*3]=(ptsX[i]/NX-0.5)*2+offsetX;
    pos[i*3+1]=-(ptsY[i]/NY-0.5)*2;
    pos[i*3+2]=(ptsZ[i]/NZ-0.5)*2*AZ;
    const nv=ptsV[i]/255;
    const [r,g,b]=colormap(nv,cmap);
    col[i*3]=r;col[i*3+1]=g;col[i*3+2]=b;
  }}
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color',new THREE.BufferAttribute(col,3));
  const mat=new THREE.PointsMaterial({{
    size:ps*0.01,vertexColors:true,transparent:true,opacity:opac,
    blending:THREE.AdditiveBlending,depthWrite:false
  }});
  return {{pts:new THREE.Points(geo,mat),count:n}};
}}

function buildLines(opac) {{
  if(lineGroup) scene.remove(lineGroup);
  lineGroup=new THREE.Group();
  const mat=new THREE.LineBasicMaterial({{color:0x00ffff,transparent:true,opacity:opac,linewidth:1}});
  for(const lm of landmarks) {{
    const geo=new THREE.BufferGeometry();
    const pos=new Float32Array([lm[0],lm[1],lm[2],lm[3],lm[4],lm[5]]);
    geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
    lineGroup.add(new THREE.Line(geo,mat));
  }}
  scene.add(lineGroup);
}}

function rebuild() {{
  const et=+document.getElementById('ev_thresh').value;
  const eo=+document.getElementById('ev_opac').value/100;
  const ep=+document.getElementById('ev_ps').value;
  const it=+document.getElementById('iv_thresh').value;
  const io_=+document.getElementById('iv_opac').value/100;
  const ip=+document.getElementById('iv_ps').value;
  const lo=+document.getElementById('line_opac').value/100;

  if(evPoints) scene.remove(evPoints);
  if(ivPoints) scene.remove(ivPoints);

  document.getElementById('status').textContent='Building...';
  setTimeout(()=>{{
    const ev=buildCloud(evRaw,EV_NZ,EV_NY,EV_NX,EV_AZ,et,eo,ep,'green',-SPACING/2);
    evPoints=ev.pts; scene.add(evPoints);
    const iv=buildCloud(ivRaw,IV_NZ,IV_NY,IV_NX,IV_AZ,it,io_,ip,'magenta',SPACING/2);
    ivPoints=iv.pts; scene.add(ivPoints);
    buildLines(lo);
    document.getElementById('status').textContent=
      'EV: '+ev.count.toLocaleString()+' | IV: '+iv.count.toLocaleString()+' | Lines: '+landmarks.length;
  }},30);
}}

function animate() {{
  requestAnimationFrame(animate);
  if(autoRotate&&!dragging) rotY+=0.003;
  const grp=[evPoints,ivPoints,lineGroup].filter(Boolean);
  grp.forEach(g=>{{g.rotation.y=rotY;g.rotation.x=rotX;g.position.x=panX;g.position.y=panY;}});
  camera.position.z=zoom;
  renderer.render(scene,camera);
}}

async function init() {{
  scene=new THREE.Scene();
  camera=new THREE.PerspectiveCamera(50,window.innerWidth/window.innerHeight,0.1,100);
  camera.position.z=zoom;
  renderer=new THREE.WebGLRenderer({{antialias:true}});
  renderer.setSize(window.innerWidth,window.innerHeight);
  renderer.setClearColor(0x000000);
  document.body.appendChild(renderer.domElement);

  document.getElementById('status').textContent='Loading ex-vivo...';
  evRaw=await loadVol(evSlices,EV_NZ,EV_NY,EV_NX);
  document.getElementById('status').textContent='Loading in-vivo...';
  ivRaw=await loadVol(ivSlices,IV_NZ,IV_NY,IV_NX);
  document.getElementById('status').textContent='Building point clouds...';
  setTimeout(()=>{{rebuild();animate();}},50);
}}

document.addEventListener('mousedown',e=>{{dragging=true;lastX=e.clientX;lastY=e.clientY;}});
document.addEventListener('mouseup',()=>dragging=false);
document.addEventListener('mousemove',e=>{{
  if(!dragging) return;
  const dx=e.clientX-lastX,dy=e.clientY-lastY;
  if(e.shiftKey){{panX+=dx*0.002;panY-=dy*0.002;}}
  else{{rotY+=dx*0.005;rotX+=dy*0.005;}}
  lastX=e.clientX;lastY=e.clientY;
}});
document.addEventListener('wheel',e=>{{zoom=Math.max(0.5,Math.min(20,zoom+e.deltaY*0.003));}});
window.addEventListener('resize',()=>{{
  camera.aspect=window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth,window.innerHeight);
}});

let timer=null;
document.querySelectorAll('.ctrl').forEach(el=>{{
  el.addEventListener('input',()=>{{
    document.getElementById('ev_thresh_v').textContent=document.getElementById('ev_thresh').value;
    document.getElementById('ev_opac_v').textContent=document.getElementById('ev_opac').value;
    document.getElementById('iv_thresh_v').textContent=document.getElementById('iv_thresh').value;
    document.getElementById('iv_opac_v').textContent=document.getElementById('iv_opac').value;
    document.getElementById('line_opac_v').textContent=document.getElementById('line_opac').value;
    clearTimeout(timer);
    timer=setTimeout(()=>{{if(evRaw&&ivRaw) rebuild();}},400);
  }});
}});
document.getElementById('autorot').addEventListener('change',e=>autoRotate=e.target.checked);
init();
"""

html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Dual 3D: IOU Stitched Ex-vivo + In-vivo — Landmark Arrows</title>
<style>
  body {{ margin:0; overflow:hidden; background:#000; color:#ddd; font-family:sans-serif; font-size:12px; }}
  #info {{ position:absolute; top:10px; left:10px; z-index:10; background:rgba(0,0,0,0.8); padding:10px; border-radius:6px; max-width:260px; }}
  #controls {{ position:absolute; top:10px; right:10px; z-index:10; background:rgba(0,0,0,0.8); padding:10px 14px; border-radius:6px; }}
  #controls label {{ display:block; margin:3px 0; }}
  .section {{ color:#aaa; font-weight:bold; margin-top:8px; }}
  hr {{ border-color:#444; margin:6px 0; }}
</style>
</head><body>
<div id="info">
  <b>Dual 3D Viewer — IOU-only Stitched</b><br>
  <span style="color:#0f0">Green</span>: Ex-vivo ({nz_ex}&times;{ny_ex}&times;{nx_ex})<br>
  <span style="color:#f0f">Magenta</span>: In-vivo ({nz_iv}&times;{ny_iv}&times;{nx_iv})<br>
  <span style="color:#0ff">Cyan lines</span>: {N_LM} matched landmarks<br>
  <br>Drag: rotate | Scroll: zoom | Shift+drag: pan<br>
  <span id="status">Initializing...</span>
</div>
<div id="controls">
  <div class="section" style="color:#0f0">Ex-vivo</div>
  <label>Threshold: <input type="range" class="ctrl" id="ev_thresh" min="0" max="30" value="8" style="width:100px"><span id="ev_thresh_v">8</span></label>
  <label>Opacity: <input type="range" class="ctrl" id="ev_opac" min="1" max="100" value="50" style="width:100px"><span id="ev_opac_v">50</span></label>
  <label>Pt size: <input type="range" class="ctrl" id="ev_ps" min="1" max="10" value="2" style="width:100px"></label>
  <hr>
  <div class="section" style="color:#f0f">In-vivo</div>
  <label>Threshold: <input type="range" class="ctrl" id="iv_thresh" min="0" max="100" value="30" style="width:100px"><span id="iv_thresh_v">30</span></label>
  <label>Opacity: <input type="range" class="ctrl" id="iv_opac" min="1" max="100" value="70" style="width:100px"><span id="iv_opac_v">70</span></label>
  <label>Pt size: <input type="range" class="ctrl" id="iv_ps" min="1" max="10" value="2" style="width:100px"></label>
  <hr>
  <div class="section" style="color:#0ff">Landmark lines</div>
  <label>Opacity: <input type="range" class="ctrl" id="line_opac" min="1" max="100" value="40" style="width:100px"><span id="line_opac_v">40</span></label>
  <hr>
  <label><input type="checkbox" id="autorot"> Auto-rotate</label>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
{js_code}
</script></body></html>
'''

out = f'{BASE}/3d_viewer/viewer_dual_iou_arrows.html'
with open(out, 'w') as f:
    f.write(html)
print(f"\nDone! {out} ({len(html)/1e6:.1f} MB)")
