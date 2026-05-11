#!/usr/bin/env python3
"""
3D viewer: IOU-only stitched ex-vivo + in-vivo TRANSFORMED into ex-vivo space
using the 3D affine. Both volumes overlaid in the same coordinate system.
Cyan arrows show matching landmark pairs (ex-vivo actual vs in-vivo predicted).
"""
import numpy as np
import tifffile
import cv2
from PIL import Image
from scipy.ndimage import median_filter, affine_transform
import io, base64

BASE = '/Users/neurolab/neuroinformatics/margaret'

IV_XY_UM = 0.6835
IV_Z_UM = 3.0

# ============================================================
# Load data
# ============================================================
print("Loading IOU-only stitched 1µm isotropic volume...")
vol_path = f'{BASE}/registration_video/stitched/stitched_gfp_iou_only_1um_isotropic.tif'
with tifffile.TiffFile(vol_path) as tif:
    n_pages = len(tif.pages)
    h, w = tif.pages[0].shape
    ev_vol = np.zeros((n_pages, h, w), dtype=np.uint16)
    for i, page in enumerate(tif.pages):
        ev_vol[i] = page.asarray()
print(f"  Ex-vivo shape: {ev_vol.shape}")  # (516, 3554, 3545) in 1µm

print("Loading JY306 in-vivo volume...")
iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
print(f"  In-vivo shape: {iv_vol_raw.shape}")  # (16, 658, 629)

print("Loading landmarks and affine...")
lm = np.load(f'{BASE}/registration_video/affine_3d_iou_results.npz', allow_pickle=True)
ev_um = lm['ev_stitched_um']  # (N, 3) x,y,z in µm
iv_um = lm['iv_um']           # (N, 3) x,y,z in µm
A = lm['affine_3x4']          # (3, 4) in-vivo µm → stitched ex-vivo µm
N_LM = ev_um.shape[0]
print(f"  {N_LM} landmarks, affine loaded")

# ============================================================
# Median filter BG subtraction on in-vivo
# ============================================================
print("\nMedian filter BG subtraction on in-vivo...")
iv_vol = np.zeros_like(iv_vol_raw)
for z in range(iv_vol_raw.shape[0]):
    bg = median_filter(iv_vol_raw[z], size=15)
    iv_vol[z] = np.clip(iv_vol_raw[z] - bg, 0, None)
del iv_vol_raw

# ============================================================
# Transform in-vivo volume into stitched ex-vivo space using 3D affine
# ============================================================
# The affine maps: in-vivo µm → stitched µm
# In-vivo µm from pixel: x_um = col * IV_XY_UM, y_um = row * IV_XY_UM, z_um = z * IV_Z_UM
# Stitched is 1µm/px, so stitched_px = stitched_µm
#
# Combined: stitched_px(z,y,x) = A @ [col*IV_XY_UM, row*IV_XY_UM, z*IV_Z_UM, 1]
# For scipy affine_transform (backward mapping): for each output voxel (z_out, y_out, x_out),
# find the input voxel (z_in, y_in, x_in).
#
# Forward: p_out = A @ [x_in*sx, y_in*sy, z_in*sz, 1]  (x,y,z order)
# But output is indexed as (z,y,x) and input as (z,y,x).
#
# Let's build the full transform in (z,y,x) convention.
# Forward (input px → output px):
#   x_out = A[0,0]*x_in*sx + A[0,1]*y_in*sy + A[0,2]*z_in*sz + A[0,3]
#   y_out = A[1,0]*x_in*sx + A[1,1]*y_in*sy + A[1,2]*z_in*sz + A[1,3]
#   z_out = A[2,0]*x_in*sx + A[2,1]*y_in*sy + A[2,2]*z_in*sz + A[2,3]
#
# In matrix form with (z,y,x) ordering:
#   [z_out]   [A[2,2]*sz  A[2,1]*sy  A[2,0]*sx] [z_in]   [A[2,3]]
#   [y_out] = [A[1,2]*sz  A[1,1]*sy  A[1,0]*sx] [y_in] + [A[1,3]]
#   [x_out]   [A[0,2]*sz  A[0,1]*sy  A[0,0]*sx] [x_in]   [A[0,3]]

sx, sy, sz = IV_XY_UM, IV_XY_UM, IV_Z_UM

# Forward matrix in (z,y,x) order
M_fwd = np.array([
    [A[2,2]*sz, A[2,1]*sy, A[2,0]*sx],
    [A[1,2]*sz, A[1,1]*sy, A[1,0]*sx],
    [A[0,2]*sz, A[0,1]*sy, A[0,0]*sx],
])
t_fwd = np.array([A[2,3], A[1,3], A[0,3]])

# scipy affine_transform uses backward mapping: input_coord = M_inv @ output_coord + offset
M_inv = np.linalg.inv(M_fwd)
offset_inv = -M_inv @ t_fwd

# Output shape = ex-vivo shape (but we can crop to the region where in-vivo data lands)
# For efficiency, compute bounding box of transformed in-vivo in stitched space
nz_iv, ny_iv, nx_iv = iv_vol.shape
corners_iv = np.array([
    [0, 0, 0], [0, 0, nx_iv-1], [0, ny_iv-1, 0], [0, ny_iv-1, nx_iv-1],
    [nz_iv-1, 0, 0], [nz_iv-1, 0, nx_iv-1], [nz_iv-1, ny_iv-1, 0], [nz_iv-1, ny_iv-1, nx_iv-1]
], dtype=np.float64)

corners_out = (M_fwd @ corners_iv.T).T + t_fwd
z_lo = max(0, int(np.floor(corners_out[:,0].min())))
z_hi = min(ev_vol.shape[0], int(np.ceil(corners_out[:,0].max())) + 1)
y_lo = max(0, int(np.floor(corners_out[:,1].min())))
y_hi = min(ev_vol.shape[1], int(np.ceil(corners_out[:,1].max())) + 1)
x_lo = max(0, int(np.floor(corners_out[:,2].min())))
x_hi = min(ev_vol.shape[2], int(np.ceil(corners_out[:,2].max())) + 1)

print(f"\nTransformed in-vivo bounding box in stitched space:")
print(f"  z: [{z_lo}, {z_hi}], y: [{y_lo}, {y_hi}], x: [{x_lo}, {x_hi}]")

# We'll render the overlapping region only
# Crop ex-vivo to this region for the viewer
out_shape = (z_hi - z_lo, y_hi - y_lo, x_hi - x_lo)
print(f"  Cropped region shape: {out_shape}")

# Adjust offset for the crop
offset_crop = offset_inv + M_inv @ np.array([z_lo, y_lo, x_lo], dtype=np.float64)

print("Transforming in-vivo volume into stitched space...")
iv_warped = affine_transform(iv_vol, M_inv, offset=offset_crop,
                             output_shape=out_shape, order=1, mode='constant', cval=0)
iv_warped = np.clip(iv_warped, 0, None).astype(np.float32)
print(f"  Warped in-vivo shape: {iv_warped.shape}, max={iv_warped.max():.0f}")
del iv_vol

# Crop ex-vivo to same region
ev_crop = ev_vol[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi].astype(np.float32)
del ev_vol
print(f"  Cropped ex-vivo shape: {ev_crop.shape}")

# ============================================================
# Downsample both for viewer
# ============================================================
DS = 4
nz = out_shape[0] // DS
ny = out_shape[1] // DS
nx = out_shape[2] // DS
print(f"\nDS{DS}: ({nz}, {ny}, {nx})")

ev_ds = ev_crop[::DS, ::DS, ::DS][:nz, :ny, :nx].astype(np.float32)
iv_ds = iv_warped[::DS, ::DS, ::DS][:nz, :ny, :nx].astype(np.float32)
del ev_crop, iv_warped

# Per-slice equalize ex-vivo
nz_vals = ev_ds[ev_ds > 0]
if len(nz_vals) > 0:
    gmean = nz_vals.mean()
    for z in range(nz):
        sl = ev_ds[z]
        mask = sl > 0
        if mask.sum() > 100:
            sm = sl[mask].mean()
            if sm > 1:
                ev_ds[z][mask] *= (gmean / sm)

ev_u8 = np.clip(ev_ds / 4000 * 255, 0, 255).astype(np.uint8)
del ev_ds

# Normalize in-vivo
vals = iv_ds[iv_ds > 0]
if len(vals) > 0:
    p995 = np.percentile(vals, 99.5)
    iv_u8 = np.clip(iv_ds / max(p995, 1) * 255, 0, 255).astype(np.uint8)
else:
    iv_u8 = np.zeros_like(iv_ds, dtype=np.uint8)
del iv_ds

# Encode PNGs — interleave both into one array (ev then iv)
print("Encoding PNGs...")
ev_b64 = []
for z in range(nz):
    buf = io.BytesIO()
    Image.fromarray(ev_u8[z]).save(buf, format='PNG', optimize=True)
    ev_b64.append(base64.b64encode(buf.getvalue()).decode('ascii'))

iv_b64 = []
for z in range(nz):
    buf = io.BytesIO()
    Image.fromarray(iv_u8[z]).save(buf, format='PNG', optimize=True)
    iv_b64.append(base64.b64encode(buf.getvalue()).decode('ascii'))
del ev_u8, iv_u8

# ============================================================
# Landmark positions in cropped DS4 viewer space
# ============================================================
az = nz / max(ny, nx)

# Ex-vivo landmarks: µm → cropped pixel → DS4 → normalized
ev_lm = np.zeros((N_LM, 3))
ev_lm[:, 0] = (ev_um[:, 0] - x_lo) / DS / nx  # normalized x
ev_lm[:, 1] = (ev_um[:, 1] - y_lo) / DS / ny   # normalized y
ev_lm[:, 2] = (ev_um[:, 2] - z_lo) / DS / nz   # normalized z

# In-vivo landmarks: apply affine to get predicted position in stitched space
iv_h = np.hstack([iv_um, np.ones((N_LM, 1))])
iv_predicted = (iv_h @ A.T)  # (N, 3) x,y,z in stitched µm

iv_lm = np.zeros((N_LM, 3))
iv_lm[:, 0] = (iv_predicted[:, 0] - x_lo) / DS / nx
iv_lm[:, 1] = (iv_predicted[:, 1] - y_lo) / DS / ny
iv_lm[:, 2] = (iv_predicted[:, 2] - z_lo) / DS / nz

# Convert to viewer coords: [-1,1] range
def to_viewer(lm_norm, az):
    out = np.zeros_like(lm_norm)
    out[:, 0] = (lm_norm[:, 0] - 0.5) * 2
    out[:, 1] = -(lm_norm[:, 1] - 0.5) * 2
    out[:, 2] = (lm_norm[:, 2] - 0.5) * 2 * az
    return out

ev_view = to_viewer(ev_lm, az)
iv_view = to_viewer(iv_lm, az)

# Filter landmarks that are inside the cropped volume
valid = ((ev_lm >= 0) & (ev_lm <= 1)).all(axis=1)
print(f"\n{valid.sum()}/{N_LM} landmarks inside cropped region")

lines_js = []
for i in range(N_LM):
    if not valid[i]:
        continue
    lines_js.append(f"[{ev_view[i,0]:.4f},{ev_view[i,1]:.4f},{ev_view[i,2]:.4f},"
                    f"{iv_view[i,0]:.4f},{iv_view[i,1]:.4f},{iv_view[i,2]:.4f}]")
lines_js_str = ",".join(lines_js)

# Compute error stats for valid landmarks
errors = np.sqrt(np.sum((iv_predicted[valid] - ev_um[valid])**2, axis=1))
print(f"  Reprojection error (µm): mean={errors.mean():.1f}, median={np.median(errors):.1f}, max={errors.max():.1f}")

ev_slice_js = ",".join(f'"{s}"' for s in ev_b64)
iv_slice_js = ",".join(f'"{s}"' for s in iv_b64)

# ============================================================
# Build HTML
# ============================================================
print("\nBuilding HTML...")

js_code = f"""
const NZ={nz}, NY={ny}, NX={nx}, AZ={az:.6f};
const evSlices=[{ev_slice_js}];
const ivSlices=[{iv_slice_js}];
const landmarks=[{lines_js_str}];

let scene, camera, renderer;
let evPoints, ivPoints, lineGroup, pivotGroup;
let rotY=0, rotX=-0.3, zoom=3.0, panX=0, panY=0;
let dragging=false, lastX=0, lastY=0, autoRotate=false;
let evRaw=null, ivRaw=null;

function colormap(v, name) {{
  if(name==='green') return [0,v,0];
  if(name==='magenta') return [v*0.9,0,v];
  if(name==='cyan') return [0,v*0.8,v];
  if(name==='hot') return [Math.min(v*2,1),Math.max(v*2-1,0)*0.8,Math.max(v*3-2,0)];
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
    if(z%10===0) document.getElementById('status').textContent='Loading z='+z+'/'+NZ;
  }}
  return raw;
}}

function buildCloud(raw, NZ, NY, NX, AZ, thresh, opac, ps, cmap) {{
  const ptsX=[],ptsY=[],ptsZ=[],ptsV=[];
  for(let z=0;z<NZ;z++) for(let y=0;y<NY;y++) for(let x=0;x<NX;x++) {{
    const v=raw[z*NY*NX+y*NX+x];
    if(v>thresh) {{ ptsX.push(x);ptsY.push(y);ptsZ.push(z);ptsV.push(v); }}
  }}
  const n=ptsX.length;
  const pos=new Float32Array(n*3), col=new Float32Array(n*3);
  for(let i=0;i<n;i++) {{
    pos[i*3]=(ptsX[i]/NX-0.5)*2;
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
  if(lineGroup) pivotGroup.remove(lineGroup);
  lineGroup=new THREE.Group();
  const tubeMat=new THREE.MeshBasicMaterial({{color:0x00ffff,transparent:true,opacity:opac}});
  const evSphereMat=new THREE.MeshBasicMaterial({{color:0x00ff00,transparent:true,opacity:Math.min(1,opac*1.5)}});
  const ivSphereMat=new THREE.MeshBasicMaterial({{color:0xffff00,transparent:true,opacity:Math.min(1,opac*1.5)}});
  const sphereGeo=new THREE.SphereGeometry(0.015,6,6);
  for(const lm of landmarks) {{
    const a=new THREE.Vector3(lm[0],lm[1],lm[2]);
    const b=new THREE.Vector3(lm[3],lm[4],lm[5]);
    const dir=new THREE.Vector3().subVectors(b,a);
    const len=dir.length();
    if(len<0.0001) continue;
    const mid=new THREE.Vector3().addVectors(a,b).multiplyScalar(0.5);
    const cyl=new THREE.CylinderGeometry(0.004,0.004,len,4,1);
    const mesh=new THREE.Mesh(cyl,tubeMat);
    mesh.position.copy(mid);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir.normalize());
    lineGroup.add(mesh);
    const s1=new THREE.Mesh(sphereGeo,evSphereMat);s1.position.copy(a);lineGroup.add(s1);
    const s2=new THREE.Mesh(sphereGeo,ivSphereMat);s2.position.copy(b);lineGroup.add(s2);
  }}
  pivotGroup.add(lineGroup);
}}

function rebuild() {{
  const et=+document.getElementById('ev_thresh').value;
  const eo=+document.getElementById('ev_opac').value/100;
  const ep=+document.getElementById('ev_ps').value;
  const it=+document.getElementById('iv_thresh').value;
  const io_=+document.getElementById('iv_opac').value/100;
  const ip=+document.getElementById('iv_ps').value;
  const lo=+document.getElementById('line_opac').value/100;

  if(evPoints) pivotGroup.remove(evPoints);
  if(ivPoints) pivotGroup.remove(ivPoints);

  document.getElementById('status').textContent='Building...';
  setTimeout(()=>{{
    const ev=buildCloud(evRaw,NZ,NY,NX,AZ,et,eo,ep,'green');
    evPoints=ev.pts; pivotGroup.add(evPoints);
    const iv=buildCloud(ivRaw,NZ,NY,NX,AZ,it,io_,ip,'magenta');
    ivPoints=iv.pts; pivotGroup.add(ivPoints);
    buildLines(lo);
    document.getElementById('status').textContent=
      'EV:'+ev.count.toLocaleString()+' | IV:'+iv.count.toLocaleString()+' | Arrows:'+landmarks.length;
  }},30);
}}

function animate() {{
  requestAnimationFrame(animate);
  if(autoRotate&&!dragging) rotY+=0.003;
  pivotGroup.rotation.y=rotY;
  pivotGroup.rotation.x=rotX;
  pivotGroup.position.x=panX;
  pivotGroup.position.y=panY;
  camera.position.z=zoom;
  renderer.render(scene,camera);
}}

async function init() {{
  scene=new THREE.Scene();
  pivotGroup=new THREE.Group();
  scene.add(pivotGroup);
  camera=new THREE.PerspectiveCamera(50,window.innerWidth/window.innerHeight,0.1,100);
  camera.position.z=zoom;
  renderer=new THREE.WebGLRenderer({{antialias:true}});
  renderer.setSize(window.innerWidth,window.innerHeight);
  renderer.setClearColor(0x000000);
  document.body.appendChild(renderer.domElement);

  document.getElementById('status').textContent='Loading ex-vivo...';
  evRaw=await loadVol(evSlices,NZ,NY,NX);
  document.getElementById('status').textContent='Loading in-vivo (transformed)...';
  ivRaw=await loadVol(ivSlices,NZ,NY,NX);
  document.getElementById('status').textContent='Building...';
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

n_valid = int(valid.sum())
html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>3D Overlay: Ex-vivo + In-vivo (affine-transformed) — Landmark Arrows</title>
<style>
  body {{ margin:0; overflow:hidden; background:#000; color:#ddd; font-family:sans-serif; font-size:12px; }}
  #info {{ position:absolute; top:10px; left:10px; z-index:10; background:rgba(0,0,0,0.8); padding:10px; border-radius:6px; max-width:320px; }}
  #controls {{ position:absolute; top:10px; right:10px; z-index:10; background:rgba(0,0,0,0.8); padding:10px 14px; border-radius:6px; }}
  #controls label {{ display:block; margin:3px 0; }}
  .section {{ color:#aaa; font-weight:bold; margin-top:8px; }}
  hr {{ border-color:#444; margin:6px 0; }}
</style>
</head><body>
<div id="info">
  <b>3D Overlay — In-vivo Affine-Transformed into Ex-vivo Space</b><br>
  <span style="color:#0f0">Green</span>: Ex-vivo IOU-only stitched ({nz}&times;{ny}&times;{nx})<br>
  <span style="color:#f0f">Magenta</span>: In-vivo (affine-warped into ex-vivo space)<br>
  <span style="color:#0ff">Cyan tubes</span>: {n_valid} landmark arrows<br>
  <span style="color:#0f0">Green spheres</span>: ex-vivo landmark | <span style="color:#ff0">Yellow spheres</span>: in-vivo predicted<br>
  <br>Mean error: {errors.mean():.1f} µm | Median: {np.median(errors):.1f} µm<br>
  <br>Drag: rotate | Scroll: zoom | Shift+drag: pan<br>
  <span id="status">Initializing...</span>
</div>
<div id="controls">
  <div class="section" style="color:#0f0">Ex-vivo</div>
  <label>Threshold: <input type="range" class="ctrl" id="ev_thresh" min="0" max="30" value="8" style="width:100px"><span id="ev_thresh_v">8</span></label>
  <label>Opacity: <input type="range" class="ctrl" id="ev_opac" min="1" max="100" value="40" style="width:100px"><span id="ev_opac_v">40</span></label>
  <label>Pt size: <input type="range" class="ctrl" id="ev_ps" min="1" max="10" value="2" style="width:100px"></label>
  <hr>
  <div class="section" style="color:#f0f">In-vivo (transformed)</div>
  <label>Threshold: <input type="range" class="ctrl" id="iv_thresh" min="0" max="100" value="20" style="width:100px"><span id="iv_thresh_v">20</span></label>
  <label>Opacity: <input type="range" class="ctrl" id="iv_opac" min="1" max="100" value="60" style="width:100px"><span id="iv_opac_v">60</span></label>
  <label>Pt size: <input type="range" class="ctrl" id="iv_ps" min="1" max="10" value="3" style="width:100px"></label>
  <hr>
  <div class="section" style="color:#0ff">Landmark arrows</div>
  <label>Opacity: <input type="range" class="ctrl" id="line_opac" min="0" max="100" value="50" style="width:100px"><span id="line_opac_v">50</span></label>
  <hr>
  <label><input type="checkbox" id="autorot"> Auto-rotate</label>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
{js_code}
</script></body></html>
'''

out = f'{BASE}/3d_viewer/viewer_aligned_overlay.html'
with open(out, 'w') as f:
    f.write(html)
print(f"\nDone! {out} ({len(html)/1e6:.1f} MB)")
