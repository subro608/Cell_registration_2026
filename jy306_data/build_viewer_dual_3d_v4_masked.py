#!/usr/bin/env python3
"""
Dual 3D HTML viewer v4: Raw nd2 tiles stacked in z (no IOU/elastix transforms).
Landmarks stay in nd2 native space — no transform chain needed.
Python-side GP interpolation at DS3.
"""
import numpy as np
import cv2
from PIL import Image
import io, base64, json, glob, os
from scipy.ndimage import zoom as ndizoom, median_filter

BASE = "/Users/neurolab/neuroinformatics/margaret"
PNG_DIR = os.path.join(BASE, "png_exports/registration_video")
OUT = f"{BASE}/3d_viewer/viewer_dual_3d_v4_masked.html"

DS_EX = 3
NORM = 4000
PATCH_SZ = 80
PHYS_RADIUS_UM = 50
PHYS_Z_HALF_UM = 6
CROP_ND2 = int(round(PHYS_RADIUS_UM / 0.645))
CROP_JY  = int(round(PHYS_RADIUS_UM / 0.6835))
DZ_ND2   = int(np.ceil(PHYS_Z_HALF_UM / 2.0))
DZ_JY    = int(np.ceil(PHYS_Z_HALF_UM / 3.0))
VOXEL_THRESH_EX = 14
VOXEL_THRESH_IV = 25

# GP parameters
GP_LENGTHSCALE = 1.0
GP_INTERP = 2
GP_NOISE = 0.01

# nd2 native pixel sizes
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0

# In-vivo pixel sizes
IV_XY_UM = 0.6835
IV_Z_UM = 3.0

# Tile ordering (sequential z-sections)
TILE_ORDER = [
    'row1_1', 'row1_2', 'row1_3', 'row1_4',
    'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
    'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
    'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
    'row5_1',
]
SLICES_PER_TILE = 12

# Compute tile z-offsets
tile_z_offset = {}
for i, t in enumerate(TILE_ORDER):
    tile_z_offset[t] = i * SLICES_PER_TILE

# ============================================================
# GP math
# ============================================================
def gp_weight_matrix(nz, interp, lengthscale, noise):
    target_zs = []
    for z in range(nz):
        target_zs.append(z)
        if interp > 1 and z < nz - 1:
            for k in range(1, interp):
                target_zs.append(z + k / interp)
    target_zs = np.array(target_zs, dtype=np.float64)
    z_orig = np.arange(nz, dtype=np.float64)
    K = np.exp(-0.5 * (z_orig[:, None] - z_orig[None, :]) ** 2 / lengthscale ** 2)
    K += noise ** 2 * np.eye(nz)
    K_inv = np.linalg.solve(K, np.eye(nz))
    kstar = np.exp(-0.5 * (target_zs[:, None] - z_orig[None, :]) ** 2 / lengthscale ** 2)
    W = kstar @ K_inv
    return W, target_zs

# ============================================================
# 1. Load nd2 GFP PNGs — stack all tiles in z, MASKED
# ============================================================
print("Loading nd2 GFP tiles (masked, no alignment)...")
n_tiles = len(TILE_ORDER)
total_z = n_tiles * SLICES_PER_TILE  # 264

# Load v4 masks
MASK_FILE = os.path.join(BASE, "registration_video/via_masks_v4.npz")
masks = np.load(MASK_FILE)
print(f"  Loaded v4 masks: {len(masks.keys())} tiles")

# Read first PNG to get dimensions
first_png = os.path.join(PNG_DIR, TILE_ORDER[0], 'GFP_z000.png')
first_img = cv2.imread(first_png, cv2.IMREAD_UNCHANGED)
full_h, full_w = first_img.shape[:2]
print(f"  Native per tile: ({SLICES_PER_TILE}, {full_h}, {full_w}) @ {ND2_XY_UM}x{ND2_XY_UM}x{ND2_Z_UM} µm")
print(f"  Total: ({total_z}, {full_h}, {full_w}) = {n_tiles} tiles × {SLICES_PER_TILE} z-slices")

ex_ny = full_h // DS_EX
ex_nx = full_w // DS_EX
print(f"  DS{DS_EX}: ({total_z}, {ex_ny}, {ex_nx})")

# Load all slices, apply mask, downsample, store
ex_vol = np.zeros((total_z, ex_ny, ex_nx), dtype=np.float32)
for ti, tile in enumerate(TILE_ORDER):
    tile_dir = os.path.join(PNG_DIR, tile)
    if tile in masks:
        mask_full = masks[tile] > 0
    else:
        print(f"  WARNING: no mask for {tile}, using full image")
        mask_full = np.ones((full_h, full_w), dtype=bool)
    mask_ds = mask_full[::DS_EX, ::DS_EX][:ex_ny, :ex_nx]
    for zi in range(SLICES_PER_TILE):
        png_path = os.path.join(tile_dir, f'GFP_z{zi:03d}.png')
        img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        img_ds = img[::DS_EX, ::DS_EX][:ex_ny, :ex_nx]
        img_ds[~mask_ds] = 0
        global_z = ti * SLICES_PER_TILE + zi
        ex_vol[global_z] = img_ds
    if ti % 5 == 0:
        print(f"  tile {ti+1}/{n_tiles} ({tile})")
print(f"  Loaded all {n_tiles} tiles (masked)")

# Per-slice equalization
print("  Per-slice equalization...")
nz_vals = ex_vol[ex_vol > 0]
gmean = nz_vals.mean() if len(nz_vals) > 0 else 1.0
for z in range(total_z):
    sl = ex_vol[z]
    mask = sl > 0
    if mask.sum() > 100:
        smean = sl[mask].mean()
        if smean > 1:
            ex_vol[z][mask] *= (gmean / smean)

ex_u8 = np.clip(ex_vol / NORM * 255, 0, 255).astype(np.uint8)
del ex_vol

# GP interpolation
print(f"  Computing GP weights (l={GP_LENGTHSCALE}, interp={GP_INTERP})...")
W, target_zs = gp_weight_matrix(total_z, GP_INTERP, GP_LENGTHSCALE, GP_NOISE)
nz_out = len(target_zs)
print(f"  GP: {total_z} -> {nz_out} z-levels, W shape {W.shape}")

# Apply GP in chunks to avoid OOM
print(f"  Applying GP in chunks...")
CHUNK = 200000  # columns per chunk
n_cols = ex_ny * ex_nx
gp_u8 = np.zeros((nz_out, ex_ny, ex_nx), dtype=np.uint8)
flat = ex_u8.reshape(total_z, -1)

for c0 in range(0, n_cols, CHUNK):
    c1 = min(c0 + CHUNK, n_cols)
    chunk = flat[:, c0:c1].astype(np.float32)
    gp_chunk = W.astype(np.float32) @ chunk
    gp_chunk = np.clip(gp_chunk, 0, 255)
    gp_u8_flat = gp_u8.reshape(nz_out, -1)
    gp_u8_flat[:, c0:c1] = gp_chunk.astype(np.uint8)
    if c0 % (CHUNK * 5) == 0:
        print(f"    chunk {c0//CHUNK}/{(n_cols+CHUNK-1)//CHUNK}")

del ex_u8, flat
print(f"  GP output: ({nz_out}, {ex_ny}, {ex_nx})")

# Extract sparse voxels
print(f"  Extracting sparse voxels (threshold={VOXEL_THRESH_EX})...")
ez, ey, exx = np.where(gp_u8 > VOXEL_THRESH_EX)
ex_vals = gp_u8[ez, ey, exx]
n_ex = len(ez)
print(f"  {n_ex:,} ex-vivo voxels")

ex_vx = exx.astype(np.float32) / ex_nx
ex_vy = ey.astype(np.float32) / ex_ny
ex_vz = np.array([target_zs[z] / total_z for z in ez], dtype=np.float32)
ex_vv = ex_vals.astype(np.float32) / 255.0
del gp_u8

# Physical size of nd2 stack
nd2_x_um = full_w * ND2_XY_UM  # 4200 * 0.645 = 2709 µm
nd2_y_um = full_h * ND2_XY_UM
nd2_z_um = total_z * ND2_Z_UM  # 264 * 2 = 528 µm

# ============================================================
# 2. Load JY306 in-vivo (s80) -> 1µm isotropic
# ============================================================
print("Loading JY306 in-vivo (s80)...")
import tifffile
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

iv_nz, iv_ny, iv_nx = iv_vol_iso.shape

print("  Background subtraction (median filter)...")
iv_p99 = np.percentile(iv_vol_iso[iv_vol_iso > 0], 99) if (iv_vol_iso > 0).any() else 1
iv_norm = np.clip(iv_vol_iso / iv_p99 * 255, 0, 255)
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

iv_vx = ixx.astype(np.float32) / iv_nx
iv_vy = iyy.astype(np.float32) / iv_ny
iv_vz = izz.astype(np.float32) / iv_nz
iv_vv = iv_vals.astype(np.float32) / 255.0
del iv_vol_iso, iv_u8, iv_sub, iv_norm

# ============================================================
# 3. Load landmarks in nd2 native space
# ============================================================
print("Loading landmarks (nd2 native)...")
lm_nd2_files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz'))
lm_st_files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_stitched_v5_*.npz'))

# Build lookup for cell_nd2_z from stitched files
cell_nd2_z_lookup = {}
for f in lm_st_files:
    tile = f.split('landmarks_stitched_v5_')[1].replace('.npz', '')
    d = np.load(f)
    cell_nd2_z_lookup[tile] = d['cell_nd2_z']

all_ev_nd2 = []
all_iv_pts = []
all_cell_z = []  # global z in combined nd2 stack
all_tiles = []
unique_tiles = []
tile_ranges = {}
idx = 0

for f in lm_nd2_files:
    d = np.load(f)
    tile = f.split('landmarks_nd2_native_')[1].replace('.npz', '')
    n = d['ev_nd2'].shape[0]
    unique_tiles.append(tile)
    tile_ranges[tile] = (idx, idx + n)
    idx += n

    all_ev_nd2.append(d['ev_nd2'])  # (N, 3) as (col, row, z_merc)
    all_iv_pts.append(d['pcd_invivo_jy306'])  # (N, 3) as (z, y, x) in s80 pixels

    # Get nd2 z-slice from stitched landmarks
    if tile in cell_nd2_z_lookup:
        nd2_z = cell_nd2_z_lookup[tile]
    else:
        # Fallback: use middle slice
        nd2_z = np.full(n, SLICES_PER_TILE // 2, dtype=np.int64)

    # Global z = tile offset + cell's z-slice within tile
    global_z = tile_z_offset[tile] + nd2_z
    all_cell_z.append(global_z)
    all_tiles.extend([tile] * n)

ev_nd2 = np.vstack(all_ev_nd2)   # (N, 3) as (col, row, z_merc)
iv_pts = np.vstack(all_iv_pts)    # (N, 3) as (z, y, x)
cell_z = np.concatenate(all_cell_z)  # global z-index in nd2 stack
N_CELLS = ev_nd2.shape[0]
print(f"  {N_CELLS} matched cells from {len(unique_tiles)} tiles")

# ============================================================
# 4. MIP patches (nd2 native for ex-vivo, s80 for in-vivo)
# ============================================================
print("Generating MIP patches...")

# Load needed nd2 pages for ex-vivo patches (full resolution)
patch_tiles_needed = set()
for i in range(N_CELLS):
    patch_tiles_needed.add(all_tiles[i])

# Pre-load nd2 slices for patch generation
nd2_pages = {}  # (tile, z_slice) -> image
for tile in patch_tiles_needed:
    tile_dir = os.path.join(PNG_DIR, tile)
    for zi in range(SLICES_PER_TILE):
        png_path = os.path.join(tile_dir, f'GFP_z{zi:03d}.png')
        nd2_pages[(tile, zi)] = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)

patch_strip_w = PATCH_SZ * 2
patch_strip_h = PATCH_SZ * N_CELLS
patch_strip = np.zeros((patch_strip_h, patch_strip_w, 3), dtype=np.uint8)

for i in range(N_CELLS):
    tile = all_tiles[i]
    col_c = int(round(ev_nd2[i, 0]))  # x in nd2 4200 space
    row_c = int(round(ev_nd2[i, 1]))  # y in nd2 4200 space
    z_in_tile = int(cell_z[i] - tile_z_offset[tile])

    # Ex-vivo MIP ±DZ_ND2 z-slices (matched physical depth)
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
        p99 = np.percentile(mip[mip > 0], 99) if (mip > 0).any() else 1
        mip_u8 = np.clip(mip / max(p99, 1) * 255, 0, 255).astype(np.uint8)
        mip_resized = np.array(Image.fromarray(mip_u8).resize((PATCH_SZ, PATCH_SZ), Image.LANCZOS))
        mip_rgb = cv2.cvtColor(mip_resized, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(mip_rgb, (PATCH_SZ//2, PATCH_SZ//2), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
    else:
        mip_rgb = np.zeros((PATCH_SZ, PATCH_SZ, 3), dtype=np.uint8)
    row = i * PATCH_SZ
    patch_strip[row:row + PATCH_SZ, 0:PATCH_SZ] = mip_rgb

    # In-vivo patch — matched physical depth
    z_c_iv = int(round(iv_pts[i, 0]))
    y_c_iv = int(round(iv_pts[i, 1]))
    x_c_iv = int(round(iv_pts[i, 2]))
    slices_iv = []
    for dz in range(-DZ_JY, DZ_JY + 1):
        zz = z_c_iv + dz
        if 0 <= zz < iv_nz_nat:
            page = iv_vol_native[zz]
            y0, y1 = max(0, y_c_iv - CROP_JY), min(page.shape[0], y_c_iv + CROP_JY)
            x0, x1 = max(0, x_c_iv - CROP_JY), min(page.shape[1], x_c_iv + CROP_JY)
            slices_iv.append(page[y0:y1, x0:x1].astype(np.float32))
    if slices_iv:
        mip_iv = np.max(np.array(slices_iv), axis=0)
        p99_iv = np.percentile(mip_iv[mip_iv > 0], 99) if (mip_iv > 0).any() else 1
        mip_iv_u8 = np.clip(mip_iv / max(p99_iv, 1) * 255, 0, 255).astype(np.uint8)
        mip_iv_resized = np.array(Image.fromarray(mip_iv_u8).resize((PATCH_SZ, PATCH_SZ), Image.LANCZOS))
        mip_iv_rgb = cv2.cvtColor(mip_iv_resized, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(mip_iv_rgb, (PATCH_SZ//2, PATCH_SZ//2), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
    else:
        mip_iv_rgb = np.zeros((PATCH_SZ, PATCH_SZ, 3), dtype=np.uint8)
    patch_strip[row:row + PATCH_SZ, PATCH_SZ:PATCH_SZ * 2] = mip_iv_rgb

    if i % 200 == 0:
        print(f"  patch {i}/{N_CELLS}")

del nd2_pages, iv_vol_native

print("  Encoding patch strip (RGB with crosshairs)...")
patch_img = Image.fromarray(patch_strip, 'RGB')
buf = io.BytesIO()
patch_img.save(buf, format='PNG', optimize=True)
patch_strip_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
print(f"  Patch strip: {patch_strip_w}x{patch_strip_h}, {len(patch_strip_b64) // 1024}KB")

# ============================================================
# 5. Prepare landmark + scale data
# ============================================================
landmarks_js = []
cell_info_js = []
for i in range(N_CELLS):
    # Ex-vivo: nd2 native coords -> normalized [0,1]
    ex_x_n = ev_nd2[i, 0] / full_w  # col / 4200
    ex_y_n = ev_nd2[i, 1] / full_h  # row / 4200
    ex_z_n = cell_z[i] / total_z     # global z / 264

    # In-vivo: s80 native pixels -> physical µm -> normalized
    iv_z_n = (iv_pts[i, 0] * IV_Z_UM) / iv_nz_full
    iv_y_n = (iv_pts[i, 1] * IV_XY_UM) / iv_h
    iv_x_n = (iv_pts[i, 2] * IV_XY_UM) / iv_w

    landmarks_js.append(f'[{ex_x_n:.4f},{ex_y_n:.4f},{ex_z_n:.4f},{iv_x_n:.4f},{iv_y_n:.4f},{iv_z_n:.4f}]')
    tile = all_tiles[i]
    ez = int(cell_z[i] - tile_z_offset[tile])
    ez_lo = max(0, ez - DZ_ND2)
    ez_hi = min(SLICES_PER_TILE - 1, ez + DZ_ND2)
    ivz = int(round(iv_pts[i, 0]))
    ivz_lo = max(0, ivz - DZ_JY)
    ivz_hi = min(15, ivz + DZ_JY)
    cell_info_js.append(f'[{ez},{ez_lo},{ez_hi},{ivz},{ivz_lo},{ivz_hi}]')

def encode_f32(arr):
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode('ascii')

print(f"Ex-vivo: {n_ex:,} voxels (nd2 raw DS{DS_EX} + GP{GP_INTERP}x), In-vivo: {n_iv:,} voxels")

# Physical scales — shared global_max for true proportions
global_max = max(nd2_x_um, nd2_y_um, nd2_z_um, iv_x_um, iv_y_um, iv_z_um)
ex_sx = nd2_x_um / global_max
ex_sy = nd2_y_um / global_max
ex_sz = nd2_z_um / global_max
iv_sx = iv_x_um / global_max
iv_sy = iv_y_um / global_max
iv_sz = iv_z_um / global_max
print(f"Physical: nd2=({nd2_z_um:.0f}×{nd2_y_um:.0f}×{nd2_x_um:.0f}µm), iv=({iv_z_um:.0f}×{iv_y_um:.0f}×{iv_x_um:.0f}µm)")
print(f"Scales: ex=({ex_sx:.3f},{ex_sy:.3f},{ex_sz:.3f}), iv=({iv_sx:.3f},{iv_sy:.3f},{iv_sz:.3f})")

# ============================================================
# 6. Build HTML
# ============================================================
print("Building HTML...")

html_parts = []
html_parts.append(f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Ex-vivo (nd2 masked) + In-vivo 3D v4 — Matched Cells</title>
<style>
  body {{ margin:0; overflow:hidden; background:#000; color:#ddd; font-family:sans-serif; font-size:11px; }}
  #info {{ position:absolute; top:8px; left:8px; z-index:10; background:rgba(0,0,0,0.75); padding:8px; border-radius:4px; max-width:380px; }}
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
  <b>Ex-vivo (nd2 masked stack) + In-vivo (JY306 s80)</b> — v4<br>
  {N_CELLS} matched cells | nd2 native space (v4 masks, no IOU/elastix)<br>
  GP pre-computed (l={GP_LENGTHSCALE}, interp={GP_INTERP}x)<br>
  <span style="color:#0f0">Green</span> = ex-vivo &nbsp; <span style="color:#f0f">Magenta</span> = in-vivo<br>
  <b>Click a line</b> to see MIP patches below<br>
  Drag: rotate | Scroll: zoom | Shift+drag: pan<br>
  <span id="ptCount" style="color:#0f0">{n_ex + n_iv:,} pts</span>
  <hr style="border-color:#444;margin:6px 0">
  <table style="font-size:10px;border-collapse:collapse;width:100%">
    <tr><td></td><td style="color:#0f0"><b>Ex-vivo (nd2)</b></td><td style="color:#f0f"><b>In-vivo</b></td></tr>
    <tr><td>Source</td><td>{n_tiles} nd2 tiles &times; 12z</td><td>JY306 s80</td></tr>
    <tr><td>Native voxels</td><td>({total_z}, {full_h}, {full_w})</td><td>({iv_nz_nat}, {iv_h_nat}, {iv_w_nat})</td></tr>
    <tr><td>Native px size</td><td>{ND2_XY_UM} &times; {ND2_XY_UM} &times; {ND2_Z_UM} &micro;m</td><td>{IV_XY_UM} &times; {IV_XY_UM} &times; {IV_Z_UM} &micro;m</td></tr>
    <tr><td>Physical size</td><td>{nd2_z_um:.0f} &times; {nd2_y_um:.0f} &times; {nd2_x_um:.0f} &micro;m</td><td>{iv_z_um:.0f} &times; {iv_y_um:.0f} &times; {iv_x_um:.0f} &micro;m</td></tr>
    <tr><td>Displayed</td><td>DS{DS_EX}+GP{GP_INTERP}x ({nz_out}&times;{ex_ny}&times;{ex_nx})</td><td>1&micro;m iso ({iv_nz}&times;{iv_ny}&times;{iv_nx})</td></tr>
    <tr><td>Voxels shown</td><td style="color:#0f0">{n_ex:,}</td><td style="color:#f0f">{n_iv:,}</td></tr>
    <tr><td>Alignment</td><td colspan="2">None (masked nd2 stack, landmarks in native space)</td></tr>
  </table>
</div>
<div id="controls">
  <span class="section-title">Ex-vivo (nd2 masked)</span>
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
      <div class="plabel">Ex-vivo (nd2)</div>
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
const cellInfo=[{",".join(cell_info_js)}];
const DZ_ND2={DZ_ND2}, DZ_JY={DZ_JY};
''')

html_parts.append(f'const exVox={{x:"{encode_f32(ex_vx)}",y:"{encode_f32(ex_vy)}",z:"{encode_f32(ex_vz)}",v:"{encode_f32(ex_vv)}",n:{n_ex}}};\n')
html_parts.append(f'const ivVox={{x:"{encode_f32(iv_vx)}",y:"{encode_f32(iv_vy)}",z:"{encode_f32(iv_vz)}",v:"{encode_f32(iv_vv)}",n:{n_iv}}};\n')
html_parts.append(f'const patchStripB64="{patch_strip_b64}";\n')

# JS rendering — same as v2
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
