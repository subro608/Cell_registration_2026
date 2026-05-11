#!/usr/bin/env python3
"""
Dual 3D HTML viewer: stitched ex-vivo v5 + JY306 in-vivo side by side.
Green lines connect 878 matched cell pairs in 3D.
Click a line to show local MIP patches below.
Pre-extracts sparse voxel lists for fast WebGL rendering.
Separate controls for ex-vivo and in-vivo.
"""
import numpy as np
import tifffile
from PIL import Image
import io, base64, json, glob

BASE = "/Users/neurolab/neuroinformatics/margaret"
OUT = f"{BASE}/3d_viewer/viewer_dual_3d.html"

DS_EX = 4
DS_IV = 1   # s80 is already small: (16, 658, 629)
NORM = 4000
PATCH_R = 40
PATCH_SZ = 80
VOXEL_THRESH_EX = 8    # ex-vivo: sparse tissue on black canvas
VOXEL_THRESH_IV = 25   # in-vivo: after background subtraction

# In-vivo pixel sizes (JY306_in_Vivo_stack_flipped_s80.tif)
IV_XY_UM = 0.6835   # µm/px XY
IV_Z_UM  = 3.0      # µm/px Z

# ============================================================
# 1. Load + process stitched ex-vivo v5
# ============================================================
print("Loading stitched ex-vivo v5...")
EX_TIFF = f"{BASE}/registration_video/stitched/stitched_gfp_fullres_v5_1um_isotropic.tif"
with tifffile.TiffFile(EX_TIFF) as tif:
    ex_nz_full = len(tif.pages)
    ex_h, ex_w = tif.pages[0].shape
    ex_nz = ex_nz_full // DS_EX
    ex_ny = ex_h // DS_EX
    ex_nx = ex_w // DS_EX
    print(f"  Full: ({ex_nz_full}, {ex_h}, {ex_w}) -> DS{DS_EX}: ({ex_nz}, {ex_ny}, {ex_nx})")

    ex_vol = np.zeros((ex_nz, ex_ny, ex_nx), dtype=np.float32)
    for zi in range(ex_nz):
        sl = tif.pages[zi * DS_EX].asarray().astype(np.float32)
        ex_vol[zi] = sl[::DS_EX, ::DS_EX][:ex_ny, :ex_nx]
        if zi % 10 == 0:
            print(f"  z={zi}/{ex_nz}")

# Per-slice equalization
print("  Per-slice equalization...")
nz_vals = ex_vol[ex_vol > 0]
gmean = nz_vals.mean()
for z in range(ex_nz):
    sl = ex_vol[z]
    mask = sl > 0
    if mask.sum() > 100:
        smean = sl[mask].mean()
        if smean > 1:
            ex_vol[z][mask] *= (gmean / smean)

ex_u8 = np.clip(ex_vol / NORM * 255, 0, 255).astype(np.uint8)

# Encode as slice PNGs (same approach as viewer_equalized.html)
print("  Encoding slice PNGs...")
ex_b64 = []
for z in range(ex_nz):
    img = Image.fromarray(ex_u8[z])
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    ex_b64.append(base64.b64encode(buf.getvalue()).decode('ascii'))
print(f"  {ex_nz} slices, ~{sum(len(s) for s in ex_b64)//1024}KB total")
del ex_vol, ex_u8

# ============================================================
# 2. Load JY306 in-vivo (s80 native) -> resample to 1µm isotropic
# ============================================================
print("Loading JY306 in-vivo (s80)...")
IV_TIFF = f"{BASE}/JY306_in_Vivo_stack_flipped_s80.tif"
iv_vol_native = tifffile.imread(IV_TIFF).astype(np.float32)
iv_nz_nat, iv_h_nat, iv_w_nat = iv_vol_native.shape
print(f"  Native: {iv_vol_native.shape} @ {IV_XY_UM}×{IV_XY_UM}×{IV_Z_UM} µm/px")

# Resample to 1µm isotropic using scipy zoom
from scipy.ndimage import zoom as ndizoom
iv_zoom_z = IV_Z_UM / 1.0    # 3.0
iv_zoom_y = IV_XY_UM / 1.0   # 0.6835
iv_zoom_x = IV_XY_UM / 1.0   # 0.6835
print(f"  Resampling to 1µm iso (zoom={iv_zoom_z:.2f}, {iv_zoom_y:.4f}, {iv_zoom_x:.4f})...")
iv_vol_iso = ndizoom(iv_vol_native, (iv_zoom_z, iv_zoom_y, iv_zoom_x), order=1)
iv_nz_full, iv_h, iv_w = iv_vol_iso.shape
print(f"  1µm iso: {iv_vol_iso.shape}")
# Physical size in µm (now = voxel dims since 1µm iso)
iv_z_um = iv_nz_full
iv_y_um = iv_h
iv_x_um = iv_w

iv_vol = iv_vol_iso[::DS_IV, ::DS_IV, ::DS_IV].copy()
iv_nz, iv_ny, iv_nx = iv_vol.shape
print(f"  DS{DS_IV}: ({iv_nz}, {iv_ny}, {iv_nx})")
del iv_vol_iso

# Local background subtraction: median filter removes neuropil, keeps cells
from scipy.ndimage import median_filter
print("  Local background subtraction (median filter)...")
iv_p99 = np.percentile(iv_vol[iv_vol > 0], 99) if (iv_vol > 0).any() else 1
iv_norm = np.clip(iv_vol / iv_p99 * 255, 0, 255)
iv_sub = np.zeros_like(iv_norm)
for z in range(iv_nz):
    bg = median_filter(iv_norm[z], size=15)
    iv_sub[z] = np.clip(iv_norm[z] - bg, 0, 255)
iv_u8 = iv_sub.astype(np.uint8)

print("  Extracting sparse voxels...")
VOXEL_THRESH_IV = 25
izz, iyy, ixx = np.where(iv_u8 > VOXEL_THRESH_IV)
iv_vals = iv_u8[izz, iyy, ixx]
print(f"  {len(izz)} voxels above threshold {VOXEL_THRESH_IV}")

iv_vx = (ixx.astype(np.float32) / iv_nx)
iv_vy = (iyy.astype(np.float32) / iv_ny)
iv_vz = (izz.astype(np.float32) / iv_nz)
iv_vv = iv_vals.astype(np.float32) / 255.0
del iv_vol, iv_u8

# ============================================================
# 3. Load landmarks + generate MIP patches
# ============================================================
print("Loading landmarks...")
files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_stitched_v5_*.npz'))
all_st, all_iv_pts, all_tiles = [], [], []
unique_tiles = []
tile_ranges = {}  # tile -> (start_idx, end_idx)
idx = 0
for f in files:
    d = np.load(f)
    n = d['stitched_coords'].shape[0]
    tile = f.split('landmarks_stitched_v5_')[1].replace('.npz', '')
    unique_tiles.append(tile)
    tile_ranges[tile] = (idx, idx + n)
    idx += n
    all_st.append(d['stitched_coords'])
    all_iv_pts.append(d['pcd_invivo_jy306'])
    all_tiles.extend([tile] * n)

st_pts = np.vstack(all_st)
iv_pts = np.vstack(all_iv_pts)
N_CELLS = st_pts.shape[0]
print(f"  {N_CELLS} matched cells")

# Generate MIP patches — write to separate JS file to avoid huge inline strings
print("Generating MIP patches...")
ex_needed_z = set()
for i in range(N_CELLS):
    z_center = int(round(st_pts[i, 0]))
    for dz in range(-2, 3):
        zz = z_center + dz
        if 0 <= zz < ex_nz_full:
            ex_needed_z.add(zz)

print(f"  Loading {len(ex_needed_z)} ex-vivo pages...")
ex_pages = {}
with tifffile.TiffFile(EX_TIFF) as tif:
    for zz in sorted(ex_needed_z):
        ex_pages[zz] = tif.pages[zz].asarray()

# Build a single concatenated image strip for patches (more compact)
# Each pair: ex-vivo patch | in-vivo patch side by side
patch_strip_w = PATCH_SZ * 2
patch_strip_h = PATCH_SZ * N_CELLS
patch_strip = np.zeros((patch_strip_h, patch_strip_w), dtype=np.uint8)

for i in range(N_CELLS):
    # Ex-vivo MIP patch
    z_c = int(round(st_pts[i, 0]))
    y_c = int(round(st_pts[i, 1]))
    x_c = int(round(st_pts[i, 2]))
    slices = []
    for dz in range(-2, 3):
        zz = z_c + dz
        if zz in ex_pages:
            page = ex_pages[zz]
            y0, y1 = max(0, y_c-PATCH_R), min(page.shape[0], y_c+PATCH_R)
            x0, x1 = max(0, x_c-PATCH_R), min(page.shape[1], x_c+PATCH_R)
            slices.append(page[y0:y1, x0:x1].astype(np.float32))
    if slices:
        mip = np.max(np.array(slices), axis=0)
        p99 = np.percentile(mip[mip>0], 99) if (mip>0).any() else 1
        mip_u8 = np.clip(mip / max(p99, 1) * 255, 0, 255).astype(np.uint8)
        mip_img = np.array(Image.fromarray(mip_u8).resize((PATCH_SZ, PATCH_SZ), Image.LANCZOS))
    else:
        mip_img = np.zeros((PATCH_SZ, PATCH_SZ), dtype=np.uint8)
    row = i * PATCH_SZ
    patch_strip[row:row+PATCH_SZ, 0:PATCH_SZ] = mip_img

    # In-vivo MIP patch
    z_c_iv = int(round(iv_pts[i, 0]))
    y_c_iv = int(round(iv_pts[i, 1]))
    x_c_iv = int(round(iv_pts[i, 2]))
    slices_iv = []
    for dz in range(-2, 3):
        zz = z_c_iv + dz
        if 0 <= zz < iv_nz_nat:
            page = iv_vol_native[zz]
            y0, y1 = max(0, y_c_iv-PATCH_R), min(page.shape[0], y_c_iv+PATCH_R)
            x0, x1 = max(0, x_c_iv-PATCH_R), min(page.shape[1], x_c_iv+PATCH_R)
            slices_iv.append(page[y0:y1, x0:x1].astype(np.float32))
    if slices_iv:
        mip_iv = np.max(np.array(slices_iv), axis=0)
        p99_iv = np.percentile(mip_iv[mip_iv>0], 99) if (mip_iv>0).any() else 1
        mip_iv_u8 = np.clip(mip_iv / max(p99_iv, 1) * 255, 0, 255).astype(np.uint8)
        mip_iv_img = np.array(Image.fromarray(mip_iv_u8).resize((PATCH_SZ, PATCH_SZ), Image.LANCZOS))
    else:
        mip_iv_img = np.zeros((PATCH_SZ, PATCH_SZ), dtype=np.uint8)
    patch_strip[row:row+PATCH_SZ, PATCH_SZ:PATCH_SZ*2] = mip_iv_img

    if i % 200 == 0:
        print(f"  patch {i}/{N_CELLS}")

del ex_pages, iv_vol_native

# Encode patch strip as single PNG
print("  Encoding patch strip...")
patch_img = Image.fromarray(patch_strip)
buf = io.BytesIO()
patch_img.save(buf, format='PNG', optimize=True)
patch_strip_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
print(f"  Patch strip: {patch_strip_w}x{patch_strip_h}, {len(patch_strip_b64)//1024}KB")

# ============================================================
# 4. Prepare landmark data
# ============================================================
# Ex-vivo landmarks: (z, y, x) in 1µm iso -> normalize by volume dims
# In-vivo landmarks: (z, y, x) in native s80 pixels -> convert to 1µm iso, then normalize

landmarks_js = []
for i in range(N_CELLS):
    # Ex-vivo: normalize by full 1µm iso volume dims
    ex_z_n = st_pts[i, 0] / ex_nz_full
    ex_y_n = st_pts[i, 1] / ex_h
    ex_x_n = st_pts[i, 2] / ex_w
    # In-vivo: native pixels -> 1µm iso coords -> normalize by 1µm iso volume dims
    iv_z_n = (iv_pts[i, 0] * IV_Z_UM) / iv_nz_full   # z_native * 3.0 / nz_iso
    iv_y_n = (iv_pts[i, 1] * IV_XY_UM) / iv_h         # y_native * 0.6835 / ny_iso
    iv_x_n = (iv_pts[i, 2] * IV_XY_UM) / iv_w         # x_native * 0.6835 / nx_iso
    landmarks_js.append(f'[{ex_x_n:.4f},{ex_y_n:.4f},{ex_z_n:.4f},{iv_x_n:.4f},{iv_y_n:.4f},{iv_z_n:.4f}]')

def encode_f32(arr):
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode('ascii')

n_iv = len(iv_vx)
print(f"Ex-vivo: {ex_nz} slices ({ex_ny}x{ex_nx}), In-vivo: {n_iv} sparse voxels")

# Physical scale factors in µm, scale relative to largest dimension
# Ex-vivo: already 1µm iso -> dims in µm = dims in voxels
ex_x_um = ex_w       # 3545 µm
ex_y_um = ex_h       # 3554 µm
ex_z_um = ex_nz_full # 516 µm
# In-vivo: native pixels -> µm
# iv_x_um, iv_y_um, iv_z_um already computed above
# Shared global scale so physical proportions are correct
global_max = max(ex_x_um, ex_y_um, ex_z_um, iv_x_um, iv_y_um, iv_z_um)
ex_sx = ex_x_um / global_max
ex_sy = ex_y_um / global_max
ex_sz = ex_z_um / global_max
iv_sx = iv_x_um / global_max
iv_sy = iv_y_um / global_max
iv_sz = iv_z_um / global_max
print(f"Physical scales: ex=({ex_sx:.3f},{ex_sy:.3f},{ex_sz:.3f}), iv=({iv_sx:.3f},{iv_sy:.3f},{iv_sz:.3f})")
print(f"  Ex-vivo: {ex_z_um:.0f}×{ex_y_um:.0f}×{ex_x_um:.0f} µm")
print(f"  In-vivo: {iv_z_um:.0f}×{iv_y_um:.0f}×{iv_x_um:.0f} µm")

# aspect ratios no longer needed separately — use physical scales directly

# ============================================================
# 5. Build HTML
# ============================================================
print("Building HTML...")

html_parts = []
html_parts.append(f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Ex-vivo + In-vivo 3D — Matched Cells</title>
<style>
  body {{ margin:0; overflow:hidden; background:#000; color:#ddd; font-family:sans-serif; font-size:11px; }}
  #info {{ position:absolute; top:8px; left:8px; z-index:10; background:rgba(0,0,0,0.75); padding:8px; border-radius:4px; max-width:300px; }}
  #controls {{ position:absolute; top:8px; right:8px; z-index:10; background:rgba(0,0,0,0.75);
               padding:8px 12px; border-radius:4px; min-width:220px; }}
  #controls label {{ display:block; margin:3px 0; }}
  #controls hr {{ border-color:#444; margin:6px 0; }}
  .section-title {{ color:#0f0; font-weight:bold; font-size:11px; }}
  .section-title-iv {{ color:#f0f; font-weight:bold; font-size:11px; }}
  #patchPanel {{ position:absolute; bottom:0; left:0; right:0; height:0; background:rgba(0,0,0,0.92);
                 z-index:20; transition:height 0.3s; overflow:hidden; }}
  #patchPanel.show {{ height:170px; }}
  #patchInner {{ display:flex; align-items:center; justify-content:center; gap:30px; height:100%; }}
  #patchPanel canvas {{ width:120px; height:120px; image-rendering:pixelated; border:2px solid #0f0; }}
  .plabel {{ color:#0f0; font-size:12px; text-align:center; margin-bottom:4px; }}
  .plabel-iv {{ color:#f0f; font-size:12px; text-align:center; margin-bottom:4px; }}
  .ppair {{ text-align:center; }}
  #closeBtn {{ position:absolute; top:5px; right:15px; cursor:pointer; color:#f00; font-size:18px; font-weight:bold; z-index:21; }}
</style>
</head><body>
<div id="info">
  <b>Ex-vivo (stitched v5) + In-vivo (JY306 s80)</b><br>
  {N_CELLS} matched cell pairs<br>
  <span style="color:#0f0">Green</span> = ex-vivo &nbsp; <span style="color:#f0f">Magenta</span> = in-vivo<br>
  <b>Click a line</b> to see MIP patches below<br>
  Drag: rotate | Scroll: zoom | Shift+drag: pan<br>
  <span id="ptCount" style="color:#0f0"></span>
  <hr style="border-color:#444;margin:6px 0">
  <table style="font-size:10px;border-collapse:collapse;width:100%">
    <tr><td></td><td style="color:#0f0"><b>Ex-vivo</b></td><td style="color:#f0f"><b>In-vivo</b></td></tr>
    <tr><td>Native voxels</td><td>({ex_nz_full}, {ex_h}, {ex_w})</td><td>({iv_nz_nat}, {iv_h_nat}, {iv_w_nat})</td></tr>
    <tr><td>Native px size</td><td>1.0 &times; 1.0 &times; 1.0 &micro;m</td><td>{IV_XY_UM} &times; {IV_XY_UM} &times; {IV_Z_UM} &micro;m</td></tr>
    <tr><td>1&micro;m iso voxels</td><td>({ex_nz_full}, {ex_h}, {ex_w})</td><td>({iv_nz_full}, {iv_h}, {iv_w})</td></tr>
    <tr><td>Pixel size (both)</td><td colspan="2" style="text-align:center">1.0 &times; 1.0 &times; 1.0 &micro;m</td></tr>
    <tr><td>Physical size</td><td>{ex_z_um:.0f} &times; {ex_y_um:.0f} &times; {ex_x_um:.0f} &micro;m</td><td>{iv_z_um:.0f} &times; {iv_y_um:.0f} &times; {iv_x_um:.0f} &micro;m</td></tr>
    <tr><td>Displayed (DS{DS_EX}/{DS_IV})</td><td>({ex_nz}, {ex_ny}, {ex_nx})</td><td>({iv_nz}, {iv_ny}, {iv_nx})</td></tr>
    <tr><td>Voxels shown</td><td style="color:#0f0" id="exVoxCount">loading...</td><td style="color:#f0f">{n_iv:,}</td></tr>
  </table>
</div>
<div id="controls">
  <span class="section-title">Ex-vivo (stitched)</span>
  <label>Threshold: <input type="range" id="exThresh" min="0" max="60" value="8" style="width:90px"><span id="exThVal">8</span></label>
  <label>Opacity: <input type="range" id="exOpac" min="1" max="100" value="26" style="width:90px"><span id="exOpVal">26</span></label>
  <label>Pt size: <input type="range" id="exPsize" min="1" max="20" value="1" style="width:90px"><span id="exPsVal">1</span></label>
  <label>GP length: <input type="range" id="gp_l" min="1" max="100" value="10" style="width:90px"><span id="gpLVal">10</span></label>
  <label>GP interp: <input type="range" id="gp_interp" min="1" max="4" value="2" style="width:90px"><span id="gpIVal">2</span></label>
  <hr>
  <span class="section-title-iv">In-vivo (JY306)</span>
  <label>Opacity: <input type="range" id="ivOpac" min="1" max="100" value="5" style="width:90px"><span id="ivOpVal">5</span></label>
  <label>Pt size: <input type="range" id="ivPsize" min="1" max="20" value="1" style="width:90px"><span id="ivPsVal">1</span></label>
  <hr>
  <label>Line opacity: <input type="range" id="lineOpac" min="1" max="100" value="30" style="width:90px"><span id="loVal">30</span></label>
  <label>Tile: <select id="tileSelect"><option value="all">All ({N_CELLS})</option>{"".join(f'<option value="{t}"{"selected" if t=="row1_3" else ""}>{t} ({tile_ranges[t][1]-tile_ranges[t][0]})</option>' for t in unique_tiles)}</select></label>
  <label><input type="checkbox" id="autorot"> Auto-rotate</label>
  <label><input type="checkbox" id="showLines" checked> Show lines</label>
  <label>Ex colormap: <select id="exCmap"><option value="green">Green</option><option value="hot">Hot</option><option value="cyan">Cyan</option><option value="gray">Gray</option></select></label>
  <label>IV colormap: <select id="ivCmap"><option value="magenta" selected>Magenta</option><option value="green">Green</option><option value="hot">Hot</option><option value="cyan">Cyan</option><option value="gray">Gray</option></select></label>
</div>
<div id="patchPanel">
  <span id="closeBtn" onclick="document.getElementById('patchPanel').classList.remove('show')">&times;</span>
  <div id="patchInner">
    <div class="ppair">
      <div class="plabel">Ex-vivo (stitched)</div>
      <canvas id="patchExCv" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas>
    </div>
    <div style="color:#0f0;font-size:16px;text-align:center" id="pairInfo">&#8596;</div>
    <div class="ppair">
      <div class="plabel-iv">In-vivo (JY306)</div>
      <canvas id="patchIvCv" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas>
    </div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// Physical scale factors (µm-proportional)
const EX_SX={ex_sx:.6f}, EX_SY={ex_sy:.6f}, EX_SZ={ex_sz:.6f};
const IV_SX={iv_sx:.6f}, IV_SY={iv_sy:.6f}, IV_SZ={iv_sz:.6f};
const N_CELLS={N_CELLS}, PATCH_SZ={PATCH_SZ};
const SPACING=0.4;

const EX_NZ={ex_nz}, EX_NY={ex_ny}, EX_NX={ex_nx};
const landmarks=[{",".join(landmarks_js)}];
const tileNames={json.dumps(all_tiles)};
const tileRanges={json.dumps(tile_ranges)};
''')

# Write ex-vivo slice data as array of b64 PNGs
ex_slice_js = ",".join(f'"{s}"' for s in ex_b64)
html_parts.append(f'const exSliceData=[{ex_slice_js}];\n')

# Write in-vivo sparse voxels
html_parts.append(f'const ivVoxB64={{x:"{encode_f32(iv_vx)}",y:"{encode_f32(iv_vy)}",z:"{encode_f32(iv_vz)}",v:"{encode_f32(iv_vv)}",n:{n_iv}}};\n')
html_parts.append(f'const patchStripB64="{patch_strip_b64}";\n')

html_parts.append('''
let scene, camera, renderer, raycaster, mouse;
let exPoints, ivPoints, linesMesh, glowMesh, markerGroup;
let rotY=0, rotX=-0.3, zoom=3.0, panX=0, panY=0;
let dragging=false, lastX=0, lastY=0;
let autoRotate=false;
let pivotGroup;
let patchStripImg = null;
let hoveredIdx = -1;
let selectedSet = new Set();
let visibleIndices = [];
let exRawData = null; // loaded from slice PNGs
const MARKER_RADIUS = 0.012; // endpoint sphere radius

function b64toF32(b64, n) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(n * 4);
  const u8 = new Uint8Array(buf);
  for(let i=0; i<bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

function colormap(v, name) {
  if(name==='green') return [0, v, 0];
  if(name==='magenta') return [v, 0, v];
  if(name==='hot') return [Math.min(v*2,1), Math.max(v*2-1,0)*0.8, Math.max(v*3-2,0)];
  if(name==='cyan') return [0, v*0.8, v];
  return [v, v, v];
}

// ====== GP MATH (from viewer_equalized) ======
function rbfKernel(a, b, l) { const d=a-b; return Math.exp(-(d*d)/(2*l*l)); }
function buildK(N, l, noise) {
  const K = new Float64Array(N*N);
  for(let i=0;i<N;i++) for(let j=0;j<N;j++) { K[i*N+j]=rbfKernel(i,j,l); if(i===j) K[i*N+j]+=noise*noise; }
  return K;
}
function cholesky(K,N) {
  const L=new Float64Array(N*N);
  for(let i=0;i<N;i++) for(let j=0;j<=i;j++) {
    let s=0; for(let k=0;k<j;k++) s+=L[i*N+k]*L[j*N+k];
    L[i*N+j]= i===j ? (K[i*N+i]-s>0?Math.sqrt(K[i*N+i]-s):1e-10) : (K[i*N+j]-s)/L[j*N+j];
  }
  return L;
}
function solveL(L,b,N) { const x=new Float64Array(N); for(let i=0;i<N;i++){let s=0;for(let k=0;k<i;k++)s+=L[i*N+k]*x[k];x[i]=(b[i]-s)/L[i*N+i];}return x;}
function solveLT(L,b,N) { const x=new Float64Array(N); for(let i=N-1;i>=0;i--){let s=0;for(let k=i+1;k<N;k++)s+=L[k*N+i]*x[k];x[i]=(b[i]-s)/L[i*N+i];}return x;}
function solveChol(L,b,N) { return solveLT(L,solveL(L,b,N),N); }
function gpWeights(NZ, targetZs, l, noise) {
  const K=buildK(NZ,l,noise), L=cholesky(K,NZ), W=[];
  for(let t=0;t<targetZs.length;t++) {
    const zstar=targetZs[t], kstar=new Float64Array(NZ);
    for(let i=0;i<NZ;i++) kstar[i]=rbfKernel(zstar,i,l);
    W.push(solveChol(L,kstar,NZ));
  }
  return W;
}

// Load ex-vivo slice PNGs into rawData array
async function loadExSlices() {
  exRawData = new Uint8Array(EX_NZ * EX_NY * EX_NX);
  for(let z=0; z<EX_NZ; z++) {
    const img = new Image();
    await new Promise(r => { img.onload=r; img.src='data:image/png;base64,'+exSliceData[z]; });
    const c = document.createElement('canvas');
    c.width=img.width; c.height=img.height;
    const ctx = c.getContext('2d');
    ctx.drawImage(img,0,0);
    const d = ctx.getImageData(0,0,img.width,img.height).data;
    const off = z * EX_NY * EX_NX;
    for(let y=0; y<EX_NY; y++)
      for(let x=0; x<EX_NX; x++)
        exRawData[off + y*EX_NX + x] = d[(y*EX_NX+x)*4];
    if(z%20===0) document.getElementById('ptCount').textContent = 'Loading ex-vivo slice '+z+'/'+EX_NZ;
  }
}

// Build ex-vivo points from raw data with GP interpolation
function buildExPoints() {
  const thresh = +document.getElementById('exThresh').value;
  const cmapName = document.getElementById('exCmap').value;
  const l = +document.getElementById('gp_l').value / 10;
  const interp = +document.getElementById('gp_interp').value;
  const noise = 0.01;

  const targetZs = [];
  for(let z=0; z<EX_NZ; z++) {
    targetZs.push(z);
    if(interp > 1 && z < EX_NZ-1)
      for(let k=1; k<interp; k++) targetZs.push(z + k/interp);
  }
  const nZout = targetZs.length;
  const W = gpWeights(EX_NZ, targetZs, l, noise);

  const ptsX=[], ptsY=[], ptsT=[], ptsV=[];
  const colBuf = new Float64Array(EX_NZ);
  for(let y=0; y<EX_NY; y++) {
    for(let x=0; x<EX_NX; x++) {
      let hasData=false;
      for(let z=0;z<EX_NZ;z++) { colBuf[z]=exRawData[z*EX_NY*EX_NX+y*EX_NX+x]; if(colBuf[z]>0)hasData=true; }
      if(!hasData) continue;
      for(let t=0;t<nZout;t++) {
        let val=0; const w=W[t];
        for(let i=0;i<EX_NZ;i++) val+=w[i]*colBuf[i];
        val=Math.max(0,Math.min(255,val));
        if(val>thresh) { ptsX.push(x); ptsY.push(y); ptsT.push(t); ptsV.push(val); }
      }
    }
  }

  const n=ptsX.length;
  document.getElementById('exVoxCount').textContent = n.toLocaleString();
  const pos=new Float32Array(n*3), col=new Float32Array(n*3);
  for(let i=0;i<n;i++) {
    const zReal=targetZs[ptsT[i]];
    pos[i*3]   = (ptsX[i]/EX_NX - 0.5)*EX_SX*2 - SPACING/2;
    pos[i*3+1] = -(ptsY[i]/EX_NY - 0.5)*EX_SY*2;
    pos[i*3+2] = (zReal/EX_NZ - 0.5)*EX_SZ*2;
    const nv=ptsV[i]/255;
    const [r,g,b]=colormap(nv, cmapName);
    col[i*3]=r; col[i*3+1]=g; col[i*3+2]=b;
  }
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color', new THREE.BufferAttribute(col,3));
  return {geo, count:n};
}

// Build in-vivo points from pre-extracted sparse voxels
function buildIvPoints(cmapName) {
  const n=ivVoxB64.n;
  const xs=b64toF32(ivVoxB64.x,n), ys=b64toF32(ivVoxB64.y,n);
  const zs=b64toF32(ivVoxB64.z,n), vs=b64toF32(ivVoxB64.v,n);
  const pos=new Float32Array(n*3), col=new Float32Array(n*3);
  for(let i=0;i<n;i++) {
    pos[i*3]   = (xs[i]-0.5)*IV_SX*2 + SPACING/2;
    pos[i*3+1] = -(ys[i]-0.5)*IV_SY*2;
    pos[i*3+2] = (zs[i]-0.5)*IV_SZ*2;
    const [r,g,b]=colormap(vs[i], cmapName);
    col[i*3]=r; col[i*3+1]=g; col[i*3+2]=b;
  }
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color', new THREE.BufferAttribute(col,3));
  return {geo, count:n};
}

function getVisibleIndices() {
  const sel = document.getElementById('tileSelect').value;
  if(sel === 'all') { const arr=[]; for(let i=0;i<landmarks.length;i++) arr.push(i); return arr; }
  const range=tileRanges[sel]; const arr=[];
  for(let i=range[0];i<range[1];i++) arr.push(i); return arr;
}

function linePos(lm) {
  return [
    (lm[0]-0.5)*EX_SX*2 - SPACING/2, -(lm[1]-0.5)*EX_SY*2, (lm[2]-0.5)*EX_SZ*2,
    (lm[3]-0.5)*IV_SX*2 + SPACING/2, -(lm[4]-0.5)*IV_SY*2, (lm[5]-0.5)*IV_SZ*2
  ];
}

function buildLines() {
  if(linesMesh) pivotGroup.remove(linesMesh);
  if(glowMesh) pivotGroup.remove(glowMesh);
  linesMesh=null; glowMesh=null;
  if(!document.getElementById('showLines').checked) return;
  const lineOpac = +document.getElementById('lineOpac').value/100;
  visibleIndices = getVisibleIndices();
  const n=visibleIndices.length;
  const pos=new Float32Array(n*6), col=new Float32Array(n*6);
  for(let j=0;j<n;j++) {
    const i=visibleIndices[j], lm=landmarks[i], p=linePos(lm);
    for(let k=0;k<6;k++) pos[j*6+k]=p[k];
    const sel=selectedSet.has(i);
    const r=sel?0.3:0, g=sel?1:0.6, b=sel?0.3:0;
    col[j*6]=r;col[j*6+1]=g;col[j*6+2]=b;col[j*6+3]=r;col[j*6+4]=g;col[j*6+5]=b;
  }
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color',new THREE.BufferAttribute(col,3));
  linesMesh=new THREE.LineSegments(geo,new THREE.LineBasicMaterial({
    vertexColors:true,transparent:true,opacity:lineOpac,blending:THREE.AdditiveBlending,depthWrite:false
  }));
  pivotGroup.add(linesMesh);
}

function updateLineColors() {
  if(!linesMesh) return;
  const col=linesMesh.geometry.attributes.color.array;
  for(let j=0;j<visibleIndices.length;j++) {
    const i=visibleIndices[j];
    const isHover=(i===hoveredIdx), isSel=selectedSet.has(i);
    let r,g,b;
    if(isHover){r=1;g=1;b=0;} else if(isSel){r=0.5;g=1;b=0.5;} else{r=0;g=0.35;b=0;}
    col[j*6]=r;col[j*6+1]=g;col[j*6+2]=b;col[j*6+3]=r;col[j*6+4]=g;col[j*6+5]=b;
  }
  linesMesh.geometry.attributes.color.needsUpdate=true;
  updateMarkers();
}

function updateMarkers() {
  // Remove old markers
  if(markerGroup) { pivotGroup.remove(markerGroup); markerGroup=null; }
  // Collect all active indices (hovered + selected)
  const active = new Set(selectedSet);
  if(hoveredIdx >= 0) active.add(hoveredIdx);
  if(active.size === 0) return;

  markerGroup = new THREE.Group();
  const sphereGeo = new THREE.SphereGeometry(MARKER_RADIUS, 12, 12);

  active.forEach(idx => {
    const lm = landmarks[idx];
    const p = linePos(lm);
    const isHover = (idx === hoveredIdx);
    const color = isHover ? 0xffff00 : 0x00ff88;

    // Ex-vivo endpoint sphere
    const matEx = new THREE.MeshBasicMaterial({color: color, transparent:true, opacity:0.9});
    const sEx = new THREE.Mesh(sphereGeo, matEx);
    sEx.position.set(p[0], p[1], p[2]);
    markerGroup.add(sEx);

    // In-vivo endpoint sphere
    const matIv = new THREE.MeshBasicMaterial({color: color, transparent:true, opacity:0.9});
    const sIv = new THREE.Mesh(sphereGeo, matIv);
    sIv.position.set(p[3], p[4], p[5]);
    markerGroup.add(sIv);

    // Bright glow line overlay (additive, full opacity)
    const glowPos = new Float32Array([p[0],p[1],p[2], p[3],p[4],p[5]]);
    const glowGeo = new THREE.BufferGeometry();
    glowGeo.setAttribute('position', new THREE.BufferAttribute(glowPos, 3));
    const glowMat = new THREE.LineBasicMaterial({
      color: color, transparent:true, opacity:1.0,
      blending: THREE.AdditiveBlending, depthWrite:false
    });
    markerGroup.add(new THREE.LineSegments(glowGeo, glowMat));
  });

  pivotGroup.add(markerGroup);
}

function findNearestLine(e) {
  mouse.x=(e.clientX/window.innerWidth)*2-1; mouse.y=-(e.clientY/window.innerHeight)*2+1;
  raycaster.setFromCamera(mouse,camera);
  if(!linesMesh) return -1;
  const positions=linesMesh.geometry.attributes.position.array;
  const mat4=pivotGroup.matrixWorld;
  let bestDist=0.12, bestIdx=-1;
  for(let j=0;j<visibleIndices.length;j++) {
    const p1=new THREE.Vector3(positions[j*6],positions[j*6+1],positions[j*6+2]).applyMatrix4(mat4);
    const p2=new THREE.Vector3(positions[j*6+3],positions[j*6+4],positions[j*6+5]).applyMatrix4(mat4);
    const dir=new THREE.Vector3().subVectors(p2,p1); const len=dir.length(); if(len<1e-6) continue; dir.normalize();
    const toP1=new THREE.Vector3().subVectors(p1,raycaster.ray.origin);
    const rayDir=raycaster.ray.direction.clone().normalize();
    const cross=new THREE.Vector3().crossVectors(rayDir,dir); const crossLen=cross.length(); if(crossLen<1e-6) continue;
    const dist=Math.abs(toP1.dot(cross))/crossLen;
    const mid=new THREE.Vector3().addVectors(p1,p2).multiplyScalar(0.5);
    const finalDist=Math.min(dist,raycaster.ray.distanceToPoint(mid));
    if(finalDist<bestDist){bestDist=finalDist;bestIdx=visibleIndices[j];}
  }
  return bestIdx;
}

function rebuild() {
  const exOpac=+document.getElementById('exOpac').value/100;
  const exPs=+document.getElementById('exPsize').value;
  const ivOpac=+document.getElementById('ivOpac').value/100;
  const ivPs=+document.getElementById('ivPsize').value;
  const exCmap=document.getElementById('exCmap').value;
  const ivCmap=document.getElementById('ivCmap').value;

  document.getElementById('exThVal').textContent=document.getElementById('exThresh').value;
  document.getElementById('exOpVal').textContent=document.getElementById('exOpac').value;
  document.getElementById('exPsVal').textContent=exPs;
  document.getElementById('ivOpVal').textContent=document.getElementById('ivOpac').value;
  document.getElementById('ivPsVal').textContent=ivPs;
  document.getElementById('loVal').textContent=document.getElementById('lineOpac').value;
  document.getElementById('gpLVal').textContent=document.getElementById('gp_l').value;
  document.getElementById('gpIVal').textContent=document.getElementById('gp_interp').value;

  if(exPoints) pivotGroup.remove(exPoints);
  if(ivPoints) pivotGroup.remove(ivPoints);

  document.getElementById('ptCount').textContent='Computing GP...';
  setTimeout(() => {
    const ex=buildExPoints();
    exPoints=new THREE.Points(ex.geo, new THREE.PointsMaterial({
      size:exPs*0.006,vertexColors:true,transparent:true,opacity:exOpac,
      blending:THREE.AdditiveBlending,depthWrite:false
    }));
    pivotGroup.add(exPoints);

    const iv=buildIvPoints(ivCmap);
    ivPoints=new THREE.Points(iv.geo, new THREE.PointsMaterial({
      size:ivPs*0.006,vertexColors:true,transparent:true,opacity:ivOpac,
      blending:THREE.AdditiveBlending,depthWrite:false
    }));
    pivotGroup.add(ivPoints);

    buildLines();
    document.getElementById('ptCount').textContent=(ex.count+iv.count).toLocaleString()+' pts';
  }, 50);
}

function animate() {
  requestAnimationFrame(animate);
  if(autoRotate && !dragging) rotY+=0.002;
  pivotGroup.rotation.y=rotY; pivotGroup.rotation.x=rotX;
  pivotGroup.position.x=panX; pivotGroup.position.y=panY;
  camera.position.z=zoom;
  renderer.render(scene,camera);
}

function showPatch(idx) {
  if(!patchStripImg) return;
  const sy=idx*PATCH_SZ;
  const exCv=document.getElementById('patchExCv'), exCtx=exCv.getContext('2d');
  exCtx.clearRect(0,0,PATCH_SZ,PATCH_SZ);
  exCtx.drawImage(patchStripImg,0,sy,PATCH_SZ,PATCH_SZ,0,0,PATCH_SZ,PATCH_SZ);
  const ivCv=document.getElementById('patchIvCv'), ivCtx=ivCv.getContext('2d');
  ivCtx.clearRect(0,0,PATCH_SZ,PATCH_SZ);
  ivCtx.drawImage(patchStripImg,PATCH_SZ,sy,PATCH_SZ,PATCH_SZ,0,0,PATCH_SZ,PATCH_SZ);
  document.getElementById('pairInfo').innerHTML='#'+idx+' ('+tileNames[idx]+')<br>&#8596;';
  document.getElementById('patchPanel').classList.add('show');
}

function onMouseClick(e) {
  if(e.shiftKey) return;
  const idx = linesMesh ? findNearestLine(e) : -1;
  if(idx>=0) {
    if(selectedSet.has(idx)){selectedSet.delete(idx);if(selectedSet.size===0)document.getElementById('patchPanel').classList.remove('show');}
    else{selectedSet.add(idx);showPatch(idx);}
  } else {
    // Clicked empty space — clear all selections
    selectedSet.clear();
    document.getElementById('patchPanel').classList.remove('show');
  }
  updateLineColors();
}

function onMouseMove(e) {
  if(dragging) {
    const dx=e.clientX-lastX, dy=e.clientY-lastY;
    if(e.shiftKey){panX+=dx*0.002;panY-=dy*0.002;} else{rotY+=dx*0.005;rotX+=dy*0.005;}
    lastX=e.clientX;lastY=e.clientY; return;
  }
  const idx=findNearestLine(e);
  if(idx!==hoveredIdx){hoveredIdx=idx;updateLineColors();renderer.domElement.style.cursor=idx>=0?'pointer':'default';}
}

async function init() {
  scene=new THREE.Scene();
  camera=new THREE.PerspectiveCamera(50,window.innerWidth/window.innerHeight,0.1,100);
  camera.position.z=zoom;
  renderer=new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(window.innerWidth,window.innerHeight);
  renderer.setClearColor(0x000000);
  document.body.appendChild(renderer.domElement);
  raycaster=new THREE.Raycaster(); mouse=new THREE.Vector2();
  pivotGroup=new THREE.Group(); scene.add(pivotGroup);

  patchStripImg=new Image();
  patchStripImg.src='data:image/png;base64,'+patchStripB64;

  await loadExSlices();
  rebuild();
  animate();
}

document.addEventListener('mousedown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;});
document.addEventListener('mouseup',e=>{if(Math.abs(e.clientX-lastX)<3&&Math.abs(e.clientY-lastY)<3)onMouseClick(e);dragging=false;});
document.addEventListener('mousemove',onMouseMove);
document.addEventListener('wheel',e=>{zoom=Math.max(0.5,Math.min(15,zoom+e.deltaY*0.003));});
window.addEventListener('resize',()=>{camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();renderer.setSize(window.innerWidth,window.innerHeight);});

let rebuildTimer=null;
['exThresh','exOpac','exPsize','ivOpac','ivPsize','lineOpac','exCmap','ivCmap','gp_l','gp_interp'].forEach(id=>{
  document.getElementById(id).addEventListener('input',()=>{clearTimeout(rebuildTimer);rebuildTimer=setTimeout(rebuild,500);});
});
document.getElementById('showLines').addEventListener('change',buildLines);
document.getElementById('tileSelect').addEventListener('change',buildLines);
document.getElementById('autorot').addEventListener('change',e=>autoRotate=e.target.checked);
init();
</script></body></html>
''')

html = "".join(html_parts)
with open(OUT, 'w') as f:
    f.write(html)
print(f"Done! {OUT} ({len(html)/1e6:.1f} MB)")
