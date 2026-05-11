#!/usr/bin/env python3
"""
Dual 3D HTML viewer v2: Python-side GP interpolation at DS3 for higher quality.
GP is computed as numpy matrix multiply — no browser-side computation needed.
Instant rendering on load.
"""
import numpy as np
import cv2
import tifffile
from PIL import Image
import io, base64, json, glob
from scipy.ndimage import zoom as ndizoom, median_filter

BASE = "/Users/neurolab/neuroinformatics/margaret"
OUT = f"{BASE}/3d_viewer/viewer_dual_3d_v2.html"

DS_EX = 3   # Higher res than v1 (DS4)
DS_IV = 1   # s80 already small
NORM = 4000
PATCH_SZ = 80
# Matched physical patch volume: 100×100 µm XY, 12 µm Z
PHYS_RADIUS_UM = 50   # µm half-width
PHYS_Z_HALF_UM = 6    # µm half-depth
CROP_ND2 = int(round(PHYS_RADIUS_UM / 0.645))   # 78 px in nd2
CROP_JY  = int(round(PHYS_RADIUS_UM / 0.6835))  # 73 px in s80
DZ_ND2   = int(np.ceil(PHYS_Z_HALF_UM / 2.0))   # ±3 nd2 slices (±6µm)
DZ_JY    = int(np.ceil(PHYS_Z_HALF_UM / 3.0))   # ±2 s80 slices (±6µm)
VOXEL_THRESH_EX = 8
VOXEL_THRESH_IV = 25

# GP parameters (baked in, no longer interactive)
GP_LENGTHSCALE = 1.0   # RBF lengthscale (in z-index units)
GP_INTERP = 2           # interpolation factor (2 = double z resolution)
GP_NOISE = 0.01

IV_XY_UM = 0.6835
IV_Z_UM  = 3.0

# ============================================================
# GP math in numpy (vectorized)
# ============================================================
def rbf_kernel_matrix(n, lengthscale, noise):
    """Build (n, n) RBF kernel + noise."""
    z = np.arange(n, dtype=np.float64)
    K = np.exp(-0.5 * (z[:, None] - z[None, :]) ** 2 / lengthscale ** 2)
    K += noise ** 2 * np.eye(n)
    return K

def gp_weight_matrix(nz, interp, lengthscale, noise):
    """Compute (n_target, nz) GP weight matrix W such that out = W @ data."""
    target_zs = []
    for z in range(nz):
        target_zs.append(z)
        if interp > 1 and z < nz - 1:
            for k in range(1, interp):
                target_zs.append(z + k / interp)
    target_zs = np.array(target_zs, dtype=np.float64)
    z_orig = np.arange(nz, dtype=np.float64)

    K = rbf_kernel_matrix(nz, lengthscale, noise)
    K_inv = np.linalg.solve(K, np.eye(nz))  # K^-1

    # k* for each target: (n_target, nz)
    kstar = np.exp(-0.5 * (target_zs[:, None] - z_orig[None, :]) ** 2 / lengthscale ** 2)
    W = kstar @ K_inv  # (n_target, nz)
    return W, target_zs

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
        if zi % 20 == 0:
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
del ex_vol

# GP interpolation in numpy
print(f"  Computing GP weights (l={GP_LENGTHSCALE}, interp={GP_INTERP})...")
W, target_zs = gp_weight_matrix(ex_nz, GP_INTERP, GP_LENGTHSCALE, GP_NOISE)
nz_out = len(target_zs)
print(f"  GP: {ex_nz} -> {nz_out} z-levels, weight matrix {W.shape}")

print(f"  Applying GP to ({ex_nz}, {ex_ny}, {ex_nx}) volume...")
# Reshape: (nz, ny*nx) for matrix multiply
flat = ex_u8.reshape(ex_nz, -1).astype(np.float64)
gp_flat = W @ flat  # (nz_out, ny*nx)
gp_flat = np.clip(gp_flat, 0, 255)
gp_u8 = gp_flat.astype(np.uint8)
del ex_u8, flat, gp_flat
print(f"  GP output: ({nz_out}, {ex_ny}, {ex_nx})")

# Extract sparse voxels from GP-interpolated volume
print(f"  Extracting sparse voxels (threshold={VOXEL_THRESH_EX})...")
gp_vol = gp_u8.reshape(nz_out, ex_ny, ex_nx)
ez, ey, exx = np.where(gp_vol > VOXEL_THRESH_EX)
ex_vals = gp_vol[ez, ey, exx]
n_ex = len(ez)
print(f"  {n_ex:,} ex-vivo voxels")

# Normalize coordinates to [0, 1]
ex_vx = (exx.astype(np.float32) / ex_nx)
ex_vy = (ey.astype(np.float32) / ex_ny)
# Map GP z-indices back to original z-fraction
ex_vz = np.array([target_zs[z] / ex_nz for z in ez], dtype=np.float32)
ex_vv = ex_vals.astype(np.float32) / 255.0
del gp_vol, gp_u8

# ============================================================
# 2. Load JY306 in-vivo (s80) -> 1µm isotropic
# ============================================================
print("Loading JY306 in-vivo (s80)...")
IV_TIFF = f"{BASE}/JY306_in_Vivo_stack_flipped_s80.tif"
iv_vol_native = tifffile.imread(IV_TIFF).astype(np.float32)
iv_nz_nat, iv_h_nat, iv_w_nat = iv_vol_native.shape
print(f"  Native: {iv_vol_native.shape} @ {IV_XY_UM}x{IV_XY_UM}x{IV_Z_UM} µm/px")

print(f"  Resampling to 1µm iso...")
iv_vol_iso = ndizoom(iv_vol_native, (IV_Z_UM, IV_XY_UM, IV_XY_UM), order=1)
iv_nz_full, iv_h, iv_w = iv_vol_iso.shape
print(f"  1µm iso: {iv_vol_iso.shape}")
iv_z_um = iv_nz_full
iv_y_um = iv_h
iv_x_um = iv_w

iv_vol = iv_vol_iso[::DS_IV, ::DS_IV, ::DS_IV].copy()
iv_nz, iv_ny, iv_nx = iv_vol.shape
del iv_vol_iso

print("  Background subtraction (median filter)...")
iv_p99 = np.percentile(iv_vol[iv_vol > 0], 99) if (iv_vol > 0).any() else 1
iv_norm = np.clip(iv_vol / iv_p99 * 255, 0, 255)
iv_sub = np.zeros_like(iv_norm)
for z in range(iv_nz):
    bg = median_filter(iv_norm[z], size=15)
    iv_sub[z] = np.clip(iv_norm[z] - bg, 0, 255)
iv_u8 = iv_sub.astype(np.uint8)

print(f"  Extracting sparse voxels (threshold={VOXEL_THRESH_IV})...")
izz, iyy, ixx = np.where(iv_u8 > VOXEL_THRESH_IV)
iv_vals = iv_u8[izz, iyy, ixx]
n_iv = len(izz)
print(f"  {n_iv:,} in-vivo voxels")

iv_vx = (ixx.astype(np.float32) / iv_nx)
iv_vy = (iyy.astype(np.float32) / iv_ny)
iv_vz = (izz.astype(np.float32) / iv_nz)
iv_vv = iv_vals.astype(np.float32) / 255.0
del iv_vol, iv_u8

# ============================================================
# 3. Load landmarks + generate MIP patches
# ============================================================
print("Loading landmarks...")
CROP_ND2 = 150    # crop radius in nd2 pixels (4200 space, ~97µm)
PNG_DIR = f'{BASE}/png_exports/registration_video'
SLICES_PER_TILE = 12

files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_stitched_v5_*.npz'))
all_st, all_iv_pts, all_ev_nd2, all_cell_nd2_z, all_tiles = [], [], [], [], []
unique_tiles = []
tile_ranges = {}
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
    all_ev_nd2.append(d['ev_nd2'])          # (N, 3) as (col, row, z_merc)
    all_cell_nd2_z.append(d['cell_nd2_z'])  # (N,) best nd2 z-slice in tile
    all_tiles.extend([tile] * n)

st_pts = np.vstack(all_st)
iv_pts = np.vstack(all_iv_pts)
ev_nd2 = np.vstack(all_ev_nd2)
cell_nd2_z = np.concatenate(all_cell_nd2_z)
N_CELLS = st_pts.shape[0]
print(f"  {N_CELLS} matched cells")

# MIP patches — ex-vivo from RAW nd2 PNGs (not stitched), in-vivo from s80
print("Generating MIP patches (ex-vivo from raw nd2, in-vivo from s80)...")

# Pre-load nd2 slices needed for patches
patch_tiles = set(all_tiles)
nd2_pages = {}
for tile in patch_tiles:
    tile_dir = f'{PNG_DIR}/{tile}'
    for zi in range(SLICES_PER_TILE):
        import os
        png_path = os.path.join(tile_dir, f'GFP_z{zi:03d}.png')
        nd2_pages[(tile, zi)] = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
print(f"  Loaded nd2 PNGs for {len(patch_tiles)} tiles")

patch_strip_w = PATCH_SZ * 2
patch_strip_h = PATCH_SZ * N_CELLS
patch_strip = np.zeros((patch_strip_h, patch_strip_w, 3), dtype=np.uint8)

for i in range(N_CELLS):
    tile = all_tiles[i]
    col_c = int(round(ev_nd2[i, 0]))  # x in nd2 4200 space
    row_c = int(round(ev_nd2[i, 1]))  # y in nd2 4200 space
    z_in_tile = int(cell_nd2_z[i])

    # Ex-vivo: MIP ±DZ_ND2 from raw nd2 PNGs (matched physical depth)
    slices = []
    for dz in range(-DZ_ND2, DZ_ND2 + 1):
        zz = z_in_tile + dz
        if 0 <= zz < SLICES_PER_TILE and (tile, zz) in nd2_pages:
            page = nd2_pages[(tile, zz)]
            y0, y1 = max(0, row_c - CROP_ND2), min(page.shape[0], row_c + CROP_ND2)
            x0, x1 = max(0, col_c - CROP_ND2), min(page.shape[1], col_c + CROP_ND2)
            slices.append(page[y0:y1, x0:x1].astype(np.float32))
    if slices:
        mip = np.max(np.array(slices), axis=0)
        p99 = np.percentile(mip[mip>0], 99) if (mip>0).any() else 1
        mip_u8 = np.clip(mip / max(p99, 1) * 255, 0, 255).astype(np.uint8)
        mip_resized = np.array(Image.fromarray(mip_u8).resize((PATCH_SZ, PATCH_SZ), Image.LANCZOS))
        mip_rgb = cv2.cvtColor(mip_resized, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(mip_rgb, (PATCH_SZ//2, PATCH_SZ//2), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
    else:
        mip_rgb = np.zeros((PATCH_SZ, PATCH_SZ, 3), dtype=np.uint8)
    row = i * PATCH_SZ
    patch_strip[row:row+PATCH_SZ, 0:PATCH_SZ] = mip_rgb

    # In-vivo: MIP ±DZ_JY from s80 native (matched physical depth)
    z_c_iv = int(round(iv_pts[i, 0]))
    y_c_iv = int(round(iv_pts[i, 1]))
    x_c_iv = int(round(iv_pts[i, 2]))
    slices_iv = []
    for dz in range(-DZ_JY, DZ_JY + 1):
        zz = z_c_iv + dz
        if 0 <= zz < iv_nz_nat:
            page = iv_vol_native[zz]
            y0, y1 = max(0, y_c_iv-CROP_JY), min(page.shape[0], y_c_iv+CROP_JY)
            x0, x1 = max(0, x_c_iv-CROP_JY), min(page.shape[1], x_c_iv+CROP_JY)
            slices_iv.append(page[y0:y1, x0:x1].astype(np.float32))
    if slices_iv:
        mip_iv = np.max(np.array(slices_iv), axis=0)
        p99_iv = np.percentile(mip_iv[mip_iv>0], 99) if (mip_iv>0).any() else 1
        mip_iv_u8 = np.clip(mip_iv / max(p99_iv, 1) * 255, 0, 255).astype(np.uint8)
        mip_iv_resized = np.array(Image.fromarray(mip_iv_u8).resize((PATCH_SZ, PATCH_SZ), Image.LANCZOS))
        mip_iv_rgb = cv2.cvtColor(mip_iv_resized, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(mip_iv_rgb, (PATCH_SZ//2, PATCH_SZ//2), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
    else:
        mip_iv_rgb = np.zeros((PATCH_SZ, PATCH_SZ, 3), dtype=np.uint8)
    patch_strip[row:row+PATCH_SZ, PATCH_SZ:PATCH_SZ*2] = mip_iv_rgb

    if i % 200 == 0:
        print(f"  patch {i}/{N_CELLS}")

del nd2_pages, iv_vol_native

print("  Encoding patch strip (RGB with crosshairs)...")
patch_img = Image.fromarray(patch_strip, 'RGB')
buf = io.BytesIO()
patch_img.save(buf, format='PNG', optimize=True)
patch_strip_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
print(f"  Patch strip: {patch_strip_w}x{patch_strip_h}, {len(patch_strip_b64)//1024}KB")

# ============================================================
# 4. Prepare landmark + scale data
# ============================================================
landmarks_js = []
cell_info_js = []
for i in range(N_CELLS):
    ex_z_n = st_pts[i, 0] / ex_nz_full
    ex_y_n = st_pts[i, 1] / ex_h
    ex_x_n = st_pts[i, 2] / ex_w
    iv_z_n = (iv_pts[i, 0] * IV_Z_UM) / iv_nz_full
    iv_y_n = (iv_pts[i, 1] * IV_XY_UM) / iv_h
    iv_x_n = (iv_pts[i, 2] * IV_XY_UM) / iv_w
    landmarks_js.append(f'[{ex_x_n:.4f},{ex_y_n:.4f},{ex_z_n:.4f},{iv_x_n:.4f},{iv_y_n:.4f},{iv_z_n:.4f}]')
    # Per-cell z-slice info for patch display
    ez = int(cell_nd2_z[i])
    ez_lo = max(0, ez - DZ_ND2)
    ez_hi = min(SLICES_PER_TILE - 1, ez + DZ_ND2)
    ivz = int(round(iv_pts[i, 0]))
    ivz_lo = max(0, ivz - DZ_JY)
    ivz_hi = min(15, ivz + DZ_JY)  # s80 has 16 z-slices (0-15)
    cell_info_js.append(f'[{ez},{ez_lo},{ez_hi},{ivz},{ivz_lo},{ivz_hi}]')

def encode_f32(arr):
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode('ascii')

print(f"Ex-vivo: {n_ex:,} voxels (DS{DS_EX} + GP interp {GP_INTERP}x), In-vivo: {n_iv:,} voxels")

# Physical scales
ex_x_um = ex_w
ex_y_um = ex_h
ex_z_um = ex_nz_full
global_max = max(ex_x_um, ex_y_um, ex_z_um, iv_x_um, iv_y_um, iv_z_um)
ex_sx = ex_x_um / global_max
ex_sy = ex_y_um / global_max
ex_sz = ex_z_um / global_max
iv_sx = iv_x_um / global_max
iv_sy = iv_y_um / global_max
iv_sz = iv_z_um / global_max
print(f"Physical scales: ex=({ex_sx:.3f},{ex_sy:.3f},{ex_sz:.3f}), iv=({iv_sx:.3f},{iv_sy:.3f},{iv_sz:.3f})")

# ============================================================
# 5. Build HTML — simplified JS (no GP in browser)
# ============================================================
print("Building HTML...")

html_parts = []
html_parts.append(f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Ex-vivo + In-vivo 3D v2 — Matched Cells</title>
<style>
  body {{ margin:0; overflow:hidden; background:#000; color:#ddd; font-family:sans-serif; font-size:11px; }}
  #info {{ position:absolute; top:8px; left:8px; z-index:10; background:rgba(0,0,0,0.75); padding:8px; border-radius:4px; max-width:340px; }}
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
  <b>Ex-vivo (stitched v5) + In-vivo (JY306 s80)</b> — v2<br>
  {N_CELLS} matched cell pairs | GP pre-computed (l={GP_LENGTHSCALE}, interp={GP_INTERP}x)<br>
  <span style="color:#0f0">Green</span> = ex-vivo &nbsp; <span style="color:#f0f">Magenta</span> = in-vivo<br>
  <b>Click a line</b> to see MIP patches below<br>
  Drag: rotate | Scroll: zoom | Shift+drag: pan<br>
  <span id="ptCount" style="color:#0f0">{n_ex + n_iv:,} pts</span>
  <hr style="border-color:#444;margin:6px 0">
  <table style="font-size:10px;border-collapse:collapse;width:100%">
    <tr><td></td><td style="color:#0f0"><b>Ex-vivo</b></td><td style="color:#f0f"><b>In-vivo</b></td></tr>
    <tr><td>Native voxels</td><td>({ex_nz_full}, {ex_h}, {ex_w})</td><td>({iv_nz_nat}, {iv_h_nat}, {iv_w_nat})</td></tr>
    <tr><td>Native px size</td><td>1.0 &times; 1.0 &times; 1.0 &micro;m</td><td>{IV_XY_UM} &times; {IV_XY_UM} &times; {IV_Z_UM} &micro;m</td></tr>
    <tr><td>1&micro;m iso voxels</td><td>({ex_nz_full}, {ex_h}, {ex_w})</td><td>({iv_nz_full}, {iv_h}, {iv_w})</td></tr>
    <tr><td>Pixel size (both)</td><td colspan="2" style="text-align:center">1.0 &times; 1.0 &times; 1.0 &micro;m</td></tr>
    <tr><td>Physical size</td><td>{ex_z_um:.0f} &times; {ex_y_um:.0f} &times; {ex_x_um:.0f} &micro;m</td><td>{iv_z_um:.0f} &times; {iv_y_um:.0f} &times; {iv_x_um:.0f} &micro;m</td></tr>
    <tr><td>Displayed</td><td>DS{DS_EX} + GP{GP_INTERP}x ({nz_out}&times;{ex_ny}&times;{ex_nx})</td><td>DS{DS_IV} ({iv_nz}&times;{iv_ny}&times;{iv_nx})</td></tr>
    <tr><td>Voxels shown</td><td style="color:#0f0">{n_ex:,}</td><td style="color:#f0f">{n_iv:,}</td></tr>
  </table>
</div>
<div id="controls">
  <span class="section-title">Ex-vivo (stitched)</span>
  <label>Opacity: <input type="range" id="exOpac" min="1" max="100" value="26" style="width:90px"><span id="exOpVal">26</span></label>
  <label>Pt size: <input type="range" id="exPsize" min="1" max="20" value="1" style="width:90px"><span id="exPsVal">1</span></label>
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
const EX_SX={ex_sx:.6f}, EX_SY={ex_sy:.6f}, EX_SZ={ex_sz:.6f};
const IV_SX={iv_sx:.6f}, IV_SY={iv_sy:.6f}, IV_SZ={iv_sz:.6f};
const N_CELLS={N_CELLS}, PATCH_SZ={PATCH_SZ};
const SPACING=0.4;
const landmarks=[{",".join(landmarks_js)}];
const tileNames={json.dumps(all_tiles)};
const tileRanges={json.dumps(tile_ranges)};
const cellInfo=[{",".join(cell_info_js)}]; // [ez_center,ez_lo,ez_hi,ivz_center,ivz_lo,ivz_hi]
const DZ_ND2={DZ_ND2}, DZ_JY={DZ_JY};
const ND2_Z_UM=2.0, IV_Z_UM_STEP=3.0;
''')

# Write pre-computed sparse voxels for both volumes
html_parts.append(f'const exVox={{x:"{encode_f32(ex_vx)}",y:"{encode_f32(ex_vy)}",z:"{encode_f32(ex_vz)}",v:"{encode_f32(ex_vv)}",n:{n_ex}}};\n')
html_parts.append(f'const ivVox={{x:"{encode_f32(iv_vx)}",y:"{encode_f32(iv_vy)}",z:"{encode_f32(iv_vz)}",v:"{encode_f32(iv_vv)}",n:{n_iv}}};\n')
html_parts.append(f'const patchStripB64="{patch_strip_b64}";\n')

# Simplified JS — no GP, no slice loading, just render pre-computed voxels
html_parts.append('''
let scene, camera, renderer, raycaster, mouse;
let exPoints, ivPoints, linesMesh, markerGroup;
let rotY=0, rotX=-0.3, zoom=3.0, panX=0, panY=0;
let dragging=false, lastX=0, lastY=0;
let autoRotate=false;
let pivotGroup;
let patchStripImg = null;
let hoveredIdx = -1;
let selectedSet = new Set();
let visibleIndices = [];
const MARKER_RADIUS = 0.012;

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

function buildVolumePoints(data, sx, sy, sz, offsetX, cmapName) {
  const n = data.n;
  const xs=b64toF32(data.x,n), ys=b64toF32(data.y,n);
  const zs=b64toF32(data.z,n), vs=b64toF32(data.v,n);
  const pos=new Float32Array(n*3), col=new Float32Array(n*3);
  for(let i=0;i<n;i++) {
    pos[i*3]   = (xs[i]-0.5)*sx*2 + offsetX;
    pos[i*3+1] = -(ys[i]-0.5)*sy*2;
    pos[i*3+2] = (zs[i]-0.5)*sz*2;
    const [r,g,b]=colormap(vs[i], cmapName);
    col[i*3]=r; col[i*3+1]=g; col[i*3+2]=b;
  }
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color', new THREE.BufferAttribute(col,3));
  return {geo, count:n};
}

function getVisibleIndices() {
  const sel=document.getElementById('tileSelect').value;
  if(sel==='all'){const a=[];for(let i=0;i<landmarks.length;i++)a.push(i);return a;}
  const r=tileRanges[sel];const a=[];for(let i=r[0];i<r[1];i++)a.push(i);return a;
}

function linePos(lm) {
  return [
    (lm[0]-0.5)*EX_SX*2-SPACING/2, -(lm[1]-0.5)*EX_SY*2, (lm[2]-0.5)*EX_SZ*2,
    (lm[3]-0.5)*IV_SX*2+SPACING/2, -(lm[4]-0.5)*IV_SY*2, (lm[5]-0.5)*IV_SZ*2
  ];
}

function buildLines() {
  if(linesMesh) pivotGroup.remove(linesMesh);
  linesMesh=null;
  if(!document.getElementById('showLines').checked) return;
  const lineOpac=+document.getElementById('lineOpac').value/100;
  visibleIndices=getVisibleIndices();
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
  if(markerGroup){pivotGroup.remove(markerGroup);markerGroup=null;}
  const active=new Set(selectedSet);
  if(hoveredIdx>=0) active.add(hoveredIdx);
  if(active.size===0) return;
  markerGroup=new THREE.Group();
  const sphereGeo=new THREE.SphereGeometry(MARKER_RADIUS,12,12);
  active.forEach(idx=>{
    const lm=landmarks[idx], p=linePos(lm);
    const isHover=(idx===hoveredIdx);
    const color=isHover?0xffff00:0x00ff88;
    const matEx=new THREE.MeshBasicMaterial({color,transparent:true,opacity:0.9});
    const sEx=new THREE.Mesh(sphereGeo,matEx); sEx.position.set(p[0],p[1],p[2]); markerGroup.add(sEx);
    const matIv=new THREE.MeshBasicMaterial({color,transparent:true,opacity:0.9});
    const sIv=new THREE.Mesh(sphereGeo,matIv); sIv.position.set(p[3],p[4],p[5]); markerGroup.add(sIv);
    const glowPos=new Float32Array([p[0],p[1],p[2],p[3],p[4],p[5]]);
    const glowGeo=new THREE.BufferGeometry();
    glowGeo.setAttribute('position',new THREE.BufferAttribute(glowPos,3));
    markerGroup.add(new THREE.LineSegments(glowGeo,new THREE.LineBasicMaterial({
      color,transparent:true,opacity:1.0,blending:THREE.AdditiveBlending,depthWrite:false
    })));
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
    const dir=new THREE.Vector3().subVectors(p2,p1);const len=dir.length();if(len<1e-6)continue;dir.normalize();
    const toP1=new THREE.Vector3().subVectors(p1,raycaster.ray.origin);
    const rayDir=raycaster.ray.direction.clone().normalize();
    const cross=new THREE.Vector3().crossVectors(rayDir,dir);const crossLen=cross.length();if(crossLen<1e-6)continue;
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

  document.getElementById('exOpVal').textContent=document.getElementById('exOpac').value;
  document.getElementById('exPsVal').textContent=exPs;
  document.getElementById('ivOpVal').textContent=document.getElementById('ivOpac').value;
  document.getElementById('ivPsVal').textContent=ivPs;
  document.getElementById('loVal').textContent=document.getElementById('lineOpac').value;

  if(exPoints) pivotGroup.remove(exPoints);
  if(ivPoints) pivotGroup.remove(ivPoints);

  const ex=buildVolumePoints(exVox,EX_SX,EX_SY,EX_SZ,-SPACING/2,exCmap);
  exPoints=new THREE.Points(ex.geo, new THREE.PointsMaterial({
    size:exPs*0.006,vertexColors:true,transparent:true,opacity:exOpac,
    blending:THREE.AdditiveBlending,depthWrite:false
  }));
  pivotGroup.add(exPoints);

  const iv=buildVolumePoints(ivVox,IV_SX,IV_SY,IV_SZ,SPACING/2,ivCmap);
  ivPoints=new THREE.Points(iv.geo, new THREE.PointsMaterial({
    size:ivPs*0.006,vertexColors:true,transparent:true,opacity:ivOpac,
    blending:THREE.AdditiveBlending,depthWrite:false
  }));
  pivotGroup.add(ivPoints);

  buildLines();
}

function animate() {
  requestAnimationFrame(animate);
  if(autoRotate&&!dragging) rotY+=0.002;
  pivotGroup.rotation.y=rotY; pivotGroup.rotation.x=rotX;
  pivotGroup.position.x=panX; pivotGroup.position.y=panY;
  camera.position.z=zoom;
  renderer.render(scene,camera);
}

function showPatch(idx) {
  if(!patchStripImg) return;
  const sy=idx*PATCH_SZ;
  const exCv=document.getElementById('patchExCv'),exCtx=exCv.getContext('2d');
  exCtx.clearRect(0,0,PATCH_SZ,PATCH_SZ);
  exCtx.drawImage(patchStripImg,0,sy,PATCH_SZ,PATCH_SZ,0,0,PATCH_SZ,PATCH_SZ);
  const ivCv=document.getElementById('patchIvCv'),ivCtx=ivCv.getContext('2d');
  ivCtx.clearRect(0,0,PATCH_SZ,PATCH_SZ);
  ivCtx.drawImage(patchStripImg,PATCH_SZ,sy,PATCH_SZ,PATCH_SZ,0,0,PATCH_SZ,PATCH_SZ);
  const ci=cellInfo[idx];
  const exZstr='z'+ci[1]+'-'+ci[2]+' (center '+ci[0]+')';
  const ivZstr='z'+ci[3]+'±'+DZ_JY+' ('+ci[4]+'-'+ci[5]+')';
  document.getElementById('pairInfo').innerHTML='#'+idx+' ('+tileNames[idx]+')<br>'+
    '<span style="color:#0f0;font-size:10px">MIP z'+ci[1]+'-'+ci[2]+' (±'+DZ_ND2+'×2µm)</span><br>&#8596;<br>'+
    '<span style="color:#f0f;font-size:10px">MIP z'+ci[4]+'-'+ci[5]+' (±'+DZ_JY+'×3µm)</span>';
  document.getElementById('patchPanel').classList.add('show');
}

function onMouseClick(e) {
  if(e.shiftKey) return;
  const idx=linesMesh?findNearestLine(e):-1;
  if(idx>=0){
    if(selectedSet.has(idx)){selectedSet.delete(idx);if(selectedSet.size===0)document.getElementById('patchPanel').classList.remove('show');}
    else{selectedSet.add(idx);showPatch(idx);}
  } else { selectedSet.clear(); document.getElementById('patchPanel').classList.remove('show'); }
  updateLineColors();
}

function onMouseMove(e) {
  if(dragging){
    const dx=e.clientX-lastX,dy=e.clientY-lastY;
    if(e.shiftKey){panX+=dx*0.002;panY-=dy*0.002;} else{rotY+=dx*0.005;rotX+=dy*0.005;}
    lastX=e.clientX;lastY=e.clientY;return;
  }
  const idx=findNearestLine(e);
  if(idx!==hoveredIdx){hoveredIdx=idx;updateLineColors();renderer.domElement.style.cursor=idx>=0?'pointer':'default';}
}

function init() {
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

  rebuild();
  animate();
}

document.addEventListener('mousedown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;});
document.addEventListener('mouseup',e=>{if(Math.abs(e.clientX-lastX)<3&&Math.abs(e.clientY-lastY)<3)onMouseClick(e);dragging=false;});
document.addEventListener('mousemove',onMouseMove);
document.addEventListener('wheel',e=>{zoom=Math.max(0.5,Math.min(15,zoom+e.deltaY*0.003));});
window.addEventListener('resize',()=>{camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();renderer.setSize(window.innerWidth,window.innerHeight);});

let rebuildTimer=null;
['exOpac','exPsize','ivOpac','ivPsize','lineOpac','exCmap','ivCmap'].forEach(id=>{
  document.getElementById(id).addEventListener('input',()=>{clearTimeout(rebuildTimer);rebuildTimer=setTimeout(rebuild,200);});
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
