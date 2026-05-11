#!/usr/bin/env python3
"""
Build a 3D HTML viewer showing:
  - Stitched ex-vivo volume (green)
  - In-vivo warped into ex-vivo stitched space via per-tile 3D affine (magenta)
Both overlapping in the same coordinate system.
"""
import numpy as np
import cv2
import os
import base64
import json
import glob
import tifffile
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT = f'{BASE}/3d_viewer/viewer_warped_invivo_3d.html'
os.makedirs(f'{BASE}/3d_viewer', exist_ok=True)

IV_XY_UM = 0.6835
IV_Z_UM = 3.0
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0

DS = 4  # downsample factor for viewer
VOXEL_THRESH_EX = 8
VOXEL_THRESH_IV = 3

# ============================================================
# Helpers
# ============================================================
def gauss(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def find_z_gaussian(intensities):
    zs = np.arange(len(intensities), dtype=np.float64)
    vals = np.array(intensities, dtype=np.float64)
    vals = vals - vals.min()
    total = vals.sum()
    if total < 1e-6:
        return float(np.argmax(intensities))
    centroid = float(np.sum(zs * vals) / total)
    peak_z = np.argmax(vals)
    try:
        p0 = [vals[peak_z], float(peak_z), 2.0]
        popt, _ = curve_fit(gauss, zs, vals, p0=p0,
                            bounds=([0, -1, 0.3], [vals.max() * 3, 12, 8]),
                            maxfev=1000)
        mu = popt[1]
        if 0 <= mu <= 11:
            return mu
    except (RuntimeError, ValueError):
        pass
    return centroid

def norm8(img):
    vals = img[img > 0]
    if len(vals) < 100:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

def encode_f32(arr):
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()

# ============================================================
# 1. Load stitch params
# ============================================================
print("Loading stitch params...")
with open(f'{BASE}/registration_video/stitch_iou_only_params.json') as f:
    params = json.load(f)
TILE_ORDER = params['tile_order']
tile_z_offsets = params['tile_z_offsets']
canvas_w = params['canvas_w']
canvas_h = params['canvas_h']
cum_iou = params['cumulative_iou']

total_z_native = max(tile_z_offsets.values()) + 12  # 258
print(f"  Canvas: {canvas_w}x{canvas_h}, {total_z_native} z-slices")

# ============================================================
# 2. Load in-vivo
# ============================================================
print("Loading JY306 in-vivo...")
iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = iv_vol_raw.shape
print(f"  In-vivo: {iv_vol_raw.shape}")

# ============================================================
# 3. Load landmark files
# ============================================================
print("Finding landmark files...")
lm_files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz'))
legacy = f'{BASE}/registration_video/landmarks_27_nd2_native.npz'
if os.path.exists(legacy):
    lm_files.append(legacy)

tile_lm_files = {}
for lm_file in lm_files:
    bn = os.path.basename(lm_file)
    if 'landmarks_27_nd2_native' in bn:
        tile = 'row2_1'
    else:
        tile = bn.replace('landmarks_nd2_native_', '').replace('.npz', '')
    if tile in TILE_ORDER:
        tile_lm_files[tile] = lm_file
print(f"  {len(tile_lm_files)} tiles with landmarks")

# ============================================================
# 4. Build warped in-vivo volume in stitched canvas space
# ============================================================
# Work at native nd2 resolution (0.645µm XY, 2µm Z), then downsample
# The stitched volume canvas is in nd2 pixel space
print(f"\nBuilding warped in-vivo volume ({canvas_w}x{canvas_h}x{total_z_native})...")
print("  (Downsampled {0}x during processing)".format(DS))

ds_w = canvas_w // DS
ds_h = canvas_h // DS
warped_vol = np.zeros((total_z_native, ds_h, ds_w), dtype=np.float32)

for tile in sorted(tile_lm_files.keys()):
    print(f"\n  {tile}:")

    # Load nd2 slices for z-fit
    img_dir = f'{BASE}/png_exports/registration_video/{tile}'
    nd2_slices = []
    for zi in range(12):
        img = cv2.imread(f'{img_dir}/GFP_z{zi:03d}.png', cv2.IMREAD_UNCHANGED)
        if img is None:
            nd2_slices.append(np.zeros((4200, 4200), dtype=np.float32))
        else:
            nd2_slices.append(img.astype(np.float32))
    nd2_slices = np.array(nd2_slices)
    nd2_h, nd2_w = nd2_slices.shape[1], nd2_slices.shape[2]

    # Load landmarks
    d = np.load(tile_lm_files[tile])
    ev_nd2 = d['ev_nd2']
    pcd_iv = d['pcd_invivo_jy306']
    N_LM = ev_nd2.shape[0]
    if N_LM < 4:
        print(f"    SKIP: only {N_LM} landmarks")
        continue

    # Gaussian z
    nd2_z_vals = []
    for i in range(N_LM):
        x, y = ev_nd2[i, 0], ev_nd2[i, 1]
        c = int(round(np.clip(x, 10, nd2_h - 11)))
        r = int(round(np.clip(y, 10, nd2_h - 11)))
        intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]
        nd2_z_vals.append(find_z_gaussian(intensities))

    # 3D affine
    src = np.column_stack([pcd_iv[:, 2] * IV_XY_UM, pcd_iv[:, 1] * IV_XY_UM, pcd_iv[:, 0] * IV_Z_UM])
    dst = np.column_stack([ev_nd2[:, 0] * ND2_XY_UM, ev_nd2[:, 1] * ND2_XY_UM, np.array(nd2_z_vals) * ND2_Z_UM])
    src_h = np.hstack([src, np.ones((N_LM, 1))])
    A_T, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)
    A = A_T.T
    errors = np.sqrt(np.sum((src_h @ A_T - dst) ** 2, axis=1))
    print(f"    {N_LM} lm | affine err: {errors.mean():.1f}µm")

    # Pixel-space transforms
    sx, sy, sz = IV_XY_UM, IV_XY_UM, IV_Z_UM
    ex, ey, ez = ND2_XY_UM, ND2_XY_UM, ND2_Z_UM
    M_fwd = np.array([
        [A[2,2]*sz/ez, A[2,1]*sy/ez, A[2,0]*sx/ez],
        [A[1,2]*sz/ey, A[1,1]*sy/ey, A[1,0]*sx/ey],
        [A[0,2]*sz/ex, A[0,1]*sy/ex, A[0,0]*sx/ex],
    ])
    t_fwd = np.array([A[2,3]/ez, A[1,3]/ey, A[0,3]/ex])
    M_inv = np.linalg.inv(M_fwd)
    offset_inv = -M_inv @ t_fwd

    # Get cumulative IOU for this tile (3x3 -> 2x3)
    M_stitch = np.array(cum_iou[tile])[:2, :]  # 2x3 for warpAffine

    # For each nd2 z-slice
    z_offset = tile_z_offsets[tile]
    for z_nd2 in range(12):
        # Map nd2 z back to in-vivo z
        center_nd2 = np.array([z_nd2, nd2_h / 2, nd2_w / 2])
        center_iv = M_inv @ center_nd2 + offset_inv
        z_iv = int(round(center_iv[0]))
        if z_iv < 0 or z_iv >= nz_iv:
            continue

        # 2D inverse affine: nd2 -> in-vivo
        M2d = np.array([
            [M_inv[2, 2], M_inv[2, 1], M_inv[2, 0] * z_nd2 + offset_inv[2]],
            [M_inv[1, 2], M_inv[1, 1], M_inv[1, 0] * z_nd2 + offset_inv[1]],
        ], dtype=np.float64)

        # Warp in-vivo into nd2 tile space (4200x4200)
        iv_warped = cv2.warpAffine(iv_vol_raw[z_iv], M2d, (nd2_w, nd2_h),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)

        # Apply stitch transform to place on canvas, then downsample
        iv_stitched = cv2.warpAffine(iv_warped, M_stitch, (canvas_w, canvas_h),
                                      flags=cv2.INTER_LINEAR, borderValue=0)

        # Downsample and accumulate (max blend to handle overlaps)
        iv_ds = cv2.resize(iv_stitched, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        z_out = z_offset + z_nd2
        warped_vol[z_out] = np.maximum(warped_vol[z_out], iv_ds)

    print(f"    Placed at z={z_offset}-{z_offset+11}")
    del nd2_slices

print(f"\nWarped volume built: {warped_vol.shape}")
print(f"  Non-zero slices: {np.sum(warped_vol.max(axis=(1,2)) > 0)}")

# ============================================================
# 5. Load stitched ex-vivo (downsampled)
# ============================================================
print("\nLoading stitched ex-vivo...")
EX_TIFF = f"{BASE}/registration_video/stitched/stitched_gfp_fullres_v5_1um_isotropic.tif"
with tifffile.TiffFile(EX_TIFF) as tif:
    ex_nz_full = len(tif.pages)
    ex_h_full, ex_w_full = tif.pages[0].shape
    # The stitched tif is at 1µm iso, but native canvas is 0.645µm
    # At 1µm iso: canvas is canvas_w * 0.645 = 3545 wide
    # We'll downsample the 1µm iso volume by DS_EX to match
    DS_EX = 4
    ex_nz = ex_nz_full // DS_EX
    ex_ny = ex_h_full // DS_EX
    ex_nx = ex_w_full // DS_EX
    print(f"  Full: ({ex_nz_full}, {ex_h_full}, {ex_w_full}) -> DS{DS_EX}: ({ex_nz}, {ex_ny}, {ex_nx})")

    ex_vol = np.zeros((ex_nz, ex_ny, ex_nx), dtype=np.float32)
    for zi in range(ex_nz):
        sl = tif.pages[zi * DS_EX].asarray().astype(np.float32)
        ex_vol[zi] = sl[::DS_EX, ::DS_EX][:ex_ny, :ex_nx]

# Normalize
ex_u8 = np.clip(ex_vol / 4000 * 255, 0, 255).astype(np.uint8)
del ex_vol

# ============================================================
# 6. Normalize warped in-vivo volume
# ============================================================
print("Normalizing warped in-vivo volume...")
# The warped volume is in native nd2 space (0.645µm XY, 2µm Z)
# Need to resample to match the 1µm iso ex-vivo coordinate system
# Native canvas: (total_z_native, canvas_h, canvas_w) @ 0.645µm XY, 2µm Z
# At DS4 of 1µm iso: effectively same as DS ~6.2 of native XY, DS 2 of native Z

# Approach: the warped_vol is already DS4 of native canvas.
# DS4 of native 0.645µm = 2.58µm per pixel
# The ex-vivo 1µm iso at DS4 = 4µm per pixel
# So the warped vol has ~1.55x more resolution. We need to resample to match.

# Scale factor: warped is (total_z_native, ds_h, ds_w) at (2µm Z, 2.58µm XY)
# Ex-vivo is (ex_nz, ex_ny, ex_nx) at (4µm Z, 4µm XY)
# To match: resize warped to same pixel grid

# Physical size of warped canvas: ds_w * DS * 0.645 µm = canvas_w * 0.645 µm
# Physical size of ex-vivo: ex_nx * DS_EX * 1.0 µm = ex_w_full µm

# The ex-vivo 1µm iso canvas is slightly different from native canvas because
# the stitching was done at native res then resampled. Let's just resize to match dims.

# Target: same dims as ex-vivo volume for overlay
iv_resized = np.zeros((ex_nz, ex_ny, ex_nx), dtype=np.float32)

# Z: warped has total_z_native=258 slices at 2µm = 516µm total
# Ex-vivo at 1µm iso has ex_nz_full slices, DS4 => ex_nz slices at 4µm
# Map: warped z (in 2µm) -> ex z (in 4µm): z_ex = z_warped * 2 / 4 = z_warped / 2
# After DS_EX: z_ex_ds = z_warped / 2 / DS_EX

for z_ex in range(ex_nz):
    z_um = z_ex * DS_EX  # physical z in µm (1µm iso)
    z_native = z_um / ND2_Z_UM  # z in native 2µm slices
    z_int = int(round(z_native))
    if 0 <= z_int < total_z_native:
        # XY: resize warped slice to ex dims
        sl = warped_vol[z_int]
        if sl.max() > 0:
            iv_resized[z_ex] = cv2.resize(sl, (ex_nx, ex_ny), interpolation=cv2.INTER_LINEAR)

# Normalize to uint8
iv_vals = iv_resized[iv_resized > 0]
if len(iv_vals) > 100:
    lo, hi = np.percentile(iv_vals, [1, 99.5])
    iv_u8 = np.clip((iv_resized - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
else:
    iv_u8 = np.zeros_like(iv_resized, dtype=np.uint8)
del warped_vol, iv_resized

print(f"  Ex-vivo volume: {ex_u8.shape}, non-zero: {(ex_u8 > 0).sum():,}")
print(f"  Warped in-vivo volume: {iv_u8.shape}, non-zero: {(iv_u8 > 0).sum():,}")

# ============================================================
# 7. Extract sparse voxels for both volumes
# ============================================================
print(f"\nExtracting sparse voxels...")

# Ex-vivo
ez, ey, exx = np.where(ex_u8 > VOXEL_THRESH_EX)
ex_vals = ex_u8[ez, ey, exx]
n_ex = len(ez)
print(f"  Ex-vivo: {n_ex:,} voxels (threshold={VOXEL_THRESH_EX})")

# In-vivo warped
iz, iy, ix = np.where(iv_u8 > VOXEL_THRESH_IV)
iv_vals_arr = iv_u8[iz, iy, ix]
n_iv = len(iz)
print(f"  In-vivo warped: {n_iv:,} voxels (threshold={VOXEL_THRESH_IV})")

# Normalize coordinates to [0, 1] centered on centroid
# Use same span for both so they're in the same coordinate system
span = float(max(ex_nx, ex_ny, ex_nz))

ex_vx = exx.astype(np.float32) / span
ex_vy = ey.astype(np.float32) / span
ex_vz = ez.astype(np.float32) / span
ex_cx, ex_cy, ex_cz = ex_vx.mean(), ex_vy.mean(), ex_vz.mean()

iv_vx = ix.astype(np.float32) / span
iv_vy = iy.astype(np.float32) / span
iv_vz = iz.astype(np.float32) / span

# Center both on ex-vivo centroid (so they overlap)
ex_vx += (0.5 - ex_cx); ex_vy += (0.5 - ex_cy); ex_vz += (0.5 - ex_cz)
iv_vx += (0.5 - ex_cx); iv_vy += (0.5 - ex_cy); iv_vz += (0.5 - ex_cz)

ex_vv = ex_vals.astype(np.float32) / 255.0
iv_vv = iv_vals_arr.astype(np.float32) / 255.0

del ex_u8, iv_u8

# ============================================================
# 8. Build HTML viewer
# ============================================================
print("\nBuilding HTML...")

html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Warped In-vivo → Ex-vivo Stitched Space</title>
<style>
  body {{ margin:0; overflow:hidden; background:#000; color:#ddd; font-family:monospace; font-size:11px; }}
  #info {{ position:absolute; top:8px; left:8px; z-index:10; background:rgba(0,0,0,0.8); padding:10px; border-radius:6px; max-width:360px; }}
  #controls {{ position:absolute; top:8px; right:8px; z-index:10; background:rgba(0,0,0,0.8); padding:10px 14px; border-radius:6px; min-width:230px; }}
  #controls label {{ display:block; margin:4px 0; }}
  #controls hr {{ border-color:#444; margin:8px 0; }}
  .section-ex {{ color:#0f0; font-weight:bold; }}
  .section-iv {{ color:#f0f; font-weight:bold; }}
</style>
</head><body>
<div id="info">
  <b>In-vivo warped into Ex-vivo stitched space</b><br>
  Per-tile 3D affine registration ({len(tile_lm_files)} tiles)<br>
  <span style="color:#0f0">Green</span> = ex-vivo stitched &nbsp;
  <span style="color:#f0f">Magenta</span> = in-vivo warped<br>
  Drag: rotate | Scroll: zoom | Shift+drag: pan<br>
  <hr style="border-color:#444;margin:6px 0">
  <table style="font-size:10px;border-collapse:collapse;width:100%">
    <tr><td></td><td style="color:#0f0"><b>Ex-vivo</b></td><td style="color:#f0f"><b>In-vivo warped</b></td></tr>
    <tr><td>Voxels shown</td><td style="color:#0f0">{n_ex:,}</td><td style="color:#f0f">{n_iv:,}</td></tr>
    <tr><td>Volume dims</td><td colspan="2">({ex_nz}, {ex_ny}, {ex_nx}) @ DS{DS_EX}</td></tr>
  </table>
</div>
<div id="controls">
  <span class="section-ex">Ex-vivo (stitched)</span>
  <label>Opacity: <input type="range" id="exOpac" min="0" max="100" value="50" style="width:90px"><span id="exOpVal">50</span></label>
  <label>Pt size: <input type="range" id="exPsize" min="1" max="30" value="2" style="width:90px"><span id="exPsVal">2</span></label>
  <hr>
  <span class="section-iv">In-vivo (warped)</span>
  <label>Opacity: <input type="range" id="ivOpac" min="0" max="100" value="70" style="width:90px"><span id="ivOpVal">70</span></label>
  <label>Pt size: <input type="range" id="ivPsize" min="1" max="30" value="2" style="width:90px"><span id="ivPsVal">2</span></label>
  <hr>
  <label><input type="checkbox" id="autorot"> Auto-rotate</label>
  <label>Ex cmap: <select id="exCmap"><option value="green">Green</option><option value="hot">Hot</option><option value="cyan">Cyan</option><option value="gray">Gray</option></select></label>
  <label>IV cmap: <select id="ivCmap"><option value="magenta" selected>Magenta</option><option value="green">Green</option><option value="hot">Hot</option><option value="cyan">Cyan</option><option value="gray">Gray</option></select></label>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const SCALE=4.0;
const exVox={{x:"{encode_f32(ex_vx)}",y:"{encode_f32(ex_vy)}",z:"{encode_f32(ex_vz)}",v:"{encode_f32(ex_vv)}",n:{n_ex}}};
const ivVox={{x:"{encode_f32(iv_vx)}",y:"{encode_f32(iv_vy)}",z:"{encode_f32(iv_vz)}",v:"{encode_f32(iv_vv)}",n:{n_iv}}};

let scene, camera, renderer, pivotGroup;
let exPoints, ivPoints;
let rotY=0, rotX=-0.3, zoom=6.0, panX=0, panY=0;
let dragging=false, lastX=0, lastY=0;

function b64toF32(b64, n) {{
  const bin = atob(b64);
  const buf = new ArrayBuffer(n * 4);
  const u8 = new Uint8Array(buf);
  for(let i=0; i<bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}}

function colormap(v, name) {{
  if(name==='green') return [0, v, 0];
  if(name==='magenta') return [v, 0, v];
  if(name==='hot') return [Math.min(v*2,1), Math.max(v*2-1,0)*0.8, Math.max(v*3-2,0)];
  if(name==='cyan') return [0, v*0.8, v];
  return [v, v, v];
}}

function buildPoints(data, sx, sy, sz, cmapName) {{
  const n=data.n;
  const xs=b64toF32(data.x,n), ys=b64toF32(data.y,n);
  const zs=b64toF32(data.z,n), vs=b64toF32(data.v,n);
  const pos=new Float32Array(n*3), col=new Float32Array(n*3);
  for(let i=0;i<n;i++) {{
    pos[i*3]   = (xs[i]-0.5)*sx*2;
    pos[i*3+1] = -(ys[i]-0.5)*sy*2;
    pos[i*3+2] = (zs[i]-0.5)*sz*2;
    const [r,g,b]=colormap(vs[i], cmapName);
    col[i*3]=r; col[i*3+1]=g; col[i*3+2]=b;
  }}
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color', new THREE.BufferAttribute(col,3));
  return geo;
}}

function rebuildPoints() {{
  const exCmap=document.getElementById('exCmap').value;
  const ivCmap=document.getElementById('ivCmap').value;
  if(exPoints) pivotGroup.remove(exPoints);
  if(ivPoints) pivotGroup.remove(ivPoints);
  const exGeo=buildPoints(exVox,SCALE,SCALE,SCALE,exCmap);
  const ivGeo=buildPoints(ivVox,SCALE,SCALE,SCALE,ivCmap);
  const exOpac=+document.getElementById('exOpac').value/100;
  const ivOpac=+document.getElementById('ivOpac').value/100;
  const exPsz=+document.getElementById('exPsize').value;
  const ivPsz=+document.getElementById('ivPsize').value;
  exPoints=new THREE.Points(exGeo,new THREE.PointsMaterial({{
    size:exPsz*0.02,vertexColors:true,transparent:true,opacity:exOpac,
    blending:THREE.AdditiveBlending,depthWrite:false}}));
  ivPoints=new THREE.Points(ivGeo,new THREE.PointsMaterial({{
    size:ivPsz*0.02,vertexColors:true,transparent:true,opacity:ivOpac,
    blending:THREE.AdditiveBlending,depthWrite:false}}));
  pivotGroup.add(exPoints);
  pivotGroup.add(ivPoints);
}}

function init() {{
  scene=new THREE.Scene();
  camera=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.1,100);
  camera.position.z=zoom;
  renderer=new THREE.WebGLRenderer({{antialias:true}});
  renderer.setSize(innerWidth,innerHeight);
  renderer.setPixelRatio(devicePixelRatio);
  document.body.appendChild(renderer.domElement);
  pivotGroup=new THREE.Group();
  scene.add(pivotGroup);

  rebuildPoints();

  // Controls
  ['exOpac','exPsize','ivOpac','ivPsize'].forEach(id=>{{
    const el=document.getElementById(id);
    el.addEventListener('input',()=>{{
      document.getElementById(id.replace('Opac','OpVal').replace('Psize','PsVal')).textContent=el.value;
      if(exPoints) exPoints.material.opacity=+document.getElementById('exOpac').value/100;
      if(ivPoints) ivPoints.material.opacity=+document.getElementById('ivOpac').value/100;
      if(exPoints) exPoints.material.size=+document.getElementById('exPsize').value*0.02;
      if(ivPoints) ivPoints.material.size=+document.getElementById('ivPsize').value*0.02;
    }});
  }});
  ['exCmap','ivCmap'].forEach(id=>{{
    document.getElementById(id).addEventListener('change',rebuildPoints);
  }});

  // Mouse
  renderer.domElement.addEventListener('mousedown',e=>{{dragging=true;lastX=e.clientX;lastY=e.clientY;}});
  window.addEventListener('mouseup',()=>{{dragging=false;}});
  window.addEventListener('mousemove',e=>{{
    if(!dragging)return;
    const dx=e.clientX-lastX, dy=e.clientY-lastY;
    lastX=e.clientX; lastY=e.clientY;
    if(e.shiftKey){{ panX+=dx*0.003; panY-=dy*0.003; }}
    else{{ rotY+=dx*0.005; rotX+=dy*0.005; }}
  }});
  renderer.domElement.addEventListener('wheel',e=>{{
    e.preventDefault();
    zoom*=1+e.deltaY*0.001;
    zoom=Math.max(1,Math.min(30,zoom));
  }},{{passive:false}});
  window.addEventListener('resize',()=>{{
    camera.aspect=innerWidth/innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth,innerHeight);
  }});
}}

function animate() {{
  requestAnimationFrame(animate);
  if(document.getElementById('autorot').checked) rotY+=0.002;
  pivotGroup.rotation.y=rotY;
  pivotGroup.rotation.x=rotX;
  pivotGroup.position.x=panX;
  pivotGroup.position.y=panY;
  camera.position.z=zoom;
  renderer.render(scene,camera);
}}

init();
animate();
</script>
</body></html>
'''

with open(OUT, 'w') as f:
    f.write(html)
print(f"\nSaved: {OUT}")
print(f"Ex-vivo: {n_ex:,} voxels | In-vivo warped: {n_iv:,} voxels")
