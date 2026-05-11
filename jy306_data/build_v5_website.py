#!/usr/bin/env python3
"""
Build v5 website data: 4 modalities as binary .bin files + landmarks JSON + per-cell panels.

Modalities:
  1. Ex-vivo structural (nd2 GFP tiles, magenta)
  2. In-vivo structural (JY306 z-stack warped to nd2 space, green)
  3. In-vivo calcium (time-averaged movie warped to nd2 space, green functional)
  4. MERSCOPE gene dots (transcript positions in nd2 space, rainbow)

Output: invivo-exvivo-cell-registration/data/
"""
import numpy as np, cv2, os, glob, json, re, pickle, sys, time
import tifffile
import pandas as pd
from PIL import Image
from scipy.ndimage import median_filter
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
PNG_DIR = os.path.join(BASE, 'png_exports/registration_video')
PKL_BASE = os.path.join(BASE, 'png_exports/registration_per_tile_pkl')
VAROL = os.path.join(BASE, 'jy306_varol')
PKL_MERC_DIR = os.path.join(BASE, 'merscope_exvivo ')
OUT_DIR = os.path.join(BASE, 'invivo-exvivo-cell-registration/data')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'panels'), exist_ok=True)

# ── Constants ──
TILE_ORDER = [
    'row1_1', 'row1_2', 'row1_3', 'row1_4',
    'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
    'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
    'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
    'row5_1',
]
PKL_TILES = [
    'row1_3',
    'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
    'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
    'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
    'row5_1',
]
TILE_TO_REGION = {
    'row1_3': 23, 'row2_1': 17, 'row2_2': 18, 'row2_3': 19,
    'row2_4': 20, 'row2_5': 21, 'row3_1': 16, 'row3_2': 15,
    'row3_3': 14, 'row3_4': 13, 'row3_5': 12, 'row3_6': 11,
    'row4_1': 5, 'row4_2': 6, 'row4_3': 7, 'row4_4': 8,
    'row4_5': 9, 'row4_6': 10, 'row5_1': 4,
}
SLICES_PER_TILE = 12
N_TILES = len(TILE_ORDER)
TOTAL_Z = N_TILES * SLICES_PER_TILE  # 264

DS = 5  # downsample factor for ex-vivo
NORM = 4000
VOXEL_THRESH_EX = 12
VOXEL_THRESH_IV = 25
VOXEL_THRESH_CAL = 15

ND2_XY_UM = 0.645
ND2_Z_UM = 2.0
IV_XY_UM = 0.6835
IV_Z_UM = 3.0

PATCH_SZ = 120  # output panel size
CROP_ND2 = 78   # ~50µm radius in nd2 pixels
CROP_JY = 73    # ~50µm radius in JY306 pixels
DZ_ND2 = 3      # ±3 slices for MIP
DZ_JY = 2       # ±2 slices

N_GENE_COLORS = 550

tile_z_offset = {t: i * SLICES_PER_TILE for i, t in enumerate(TILE_ORDER)}


def make_rainbow_palette(n):
    colors = []
    for i in range(n):
        h = int(180 * i / n)
        s = 200 + int(55 * ((i % 3) / 2))
        v = 200 + int(55 * ((i % 5) / 4))
        bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append([int(bgr[2]), int(bgr[1]), int(bgr[0])])  # RGB
    return colors

GENE_PALETTE = make_rainbow_palette(N_GENE_COLORS)


def build_pkl_affine(T_dict):
    B = np.eye(4)
    for step in T_dict:
        for k, v in step.items():
            if k == 'bhat':
                B = B @ np.c_[v, np.array((0, 0, 0, 1))]
            if k == 'scale':
                B[:, :3] *= v
    R_3 = np.linalg.inv(B[:3, :3]).T
    offset_3 = -B[-1, :-1] @ np.linalg.inv(B[:3, :3])
    R_3_inv = np.linalg.inv(R_3)
    return R_3_inv, offset_3


def norm_u8(img, lo=1, hi=99.5):
    v = img.ravel()
    v = v[v > 0]
    if len(v) < 50:
        return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img.astype(np.float32) - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)


def crop_centered(img, cx, cy, radius):
    h, w = img.shape[:2]
    out_shape = (radius * 2, radius * 2, img.shape[2]) if img.ndim == 3 else (radius * 2, radius * 2)
    out = np.zeros(out_shape, dtype=img.dtype)
    sy0 = max(0, cy - radius); sy1 = min(h, cy + radius)
    sx0 = max(0, cx - radius); sx1 = min(w, cx + radius)
    dy0 = sy0 - (cy - radius); dy1 = dy0 + (sy1 - sy0)
    dx0 = sx0 - (cx - radius); dx1 = dx0 + (sx1 - sx0)
    out[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return out


t0 = time.time()

# ================================================================
# 1. EX-VIVO: Load nd2 GFP tiles, downsample, extract voxels
# ================================================================
print("=" * 60)
print("1. Ex-vivo structural (nd2 GFP tiles)")
print("=" * 60)

first_img = cv2.imread(os.path.join(PNG_DIR, TILE_ORDER[0], 'GFP_z000.png'), cv2.IMREAD_UNCHANGED)
FULL_H, FULL_W = first_img.shape[:2]
ex_ny = FULL_H // DS
ex_nx = FULL_W // DS
print(f"  Native: ({TOTAL_Z}, {FULL_H}, {FULL_W}), DS{DS}: ({TOTAL_Z}, {ex_ny}, {ex_nx})")

ex_vol = np.zeros((TOTAL_Z, ex_ny, ex_nx), dtype=np.float32)
for ti, tile in enumerate(TILE_ORDER):
    tile_dir = os.path.join(PNG_DIR, tile)
    for zi in range(SLICES_PER_TILE):
        png_path = os.path.join(tile_dir, f'GFP_z{zi:03d}.png')
        img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        gz = ti * SLICES_PER_TILE + zi
        ex_vol[gz] = img[::DS, ::DS][:ex_ny, :ex_nx]
    if ti % 5 == 0:
        print(f"  tile {ti+1}/{N_TILES} ({tile})")

# Per-slice equalization
nz_vals = ex_vol[ex_vol > 0]
gmean = nz_vals.mean() if len(nz_vals) > 0 else 1.0
for z in range(TOTAL_Z):
    sl = ex_vol[z]
    mask = sl > 0
    if mask.sum() > 100:
        smean = sl[mask].mean()
        if smean > 1:
            ex_vol[z][mask] *= (gmean / smean)

ex_u8 = np.clip(ex_vol / NORM * 255, 0, 255).astype(np.uint8)
del ex_vol

# Extract sparse voxels
ez, ey, exx = np.where(ex_u8 > VOXEL_THRESH_EX)
ex_vals = ex_u8[ez, ey, exx]
n_ex = len(ez)
print(f"  {n_ex:,} ex-vivo voxels")

# Normalize to [0,1]
ex_vx = exx.astype(np.float32) / ex_nx
ex_vy = ey.astype(np.float32) / ex_ny
ex_vz = ez.astype(np.float32) / TOTAL_Z
ex_vv = ex_vals.astype(np.float32) / 255.0
del ex_u8

# Write binary: interleaved [x,y,z,v] float32
ex_bin = np.column_stack([ex_vx, ex_vy, ex_vz, ex_vv]).astype(np.float32)
ex_bin.tofile(os.path.join(OUT_DIR, 'exvivo_voxels.bin'))
print(f"  Wrote exvivo_voxels.bin ({ex_bin.nbytes / 1e6:.1f} MB, {n_ex:,} pts)")
del ex_bin, ex_vx, ex_vy, ex_vz, ex_vv, ez, ey, exx, ex_vals

# ================================================================
# 2. IN-VIVO STRUCTURAL: JY306 z-stack → 1µm iso → voxels
# ================================================================
print("\n" + "=" * 60)
print("2. In-vivo structural (JY306 z-stack)")
print("=" * 60)

from scipy.ndimage import zoom as ndizoom

iv_tiff = os.path.join(BASE, 'JY306_in_Vivo_stack_flipped_s80.tif')
iv_vol_native = tifffile.imread(iv_tiff).astype(np.float32)
iv_nz_nat, iv_h_nat, iv_w_nat = iv_vol_native.shape
print(f"  Native: {iv_vol_native.shape} @ {IV_XY_UM}x{IV_XY_UM}x{IV_Z_UM} µm/px")

print("  Resampling to 1µm iso...")
iv_vol_iso = ndizoom(iv_vol_native, (IV_Z_UM, IV_XY_UM, IV_XY_UM), order=1)
iv_nz, iv_ny, iv_nx = iv_vol_iso.shape
print(f"  1µm iso: {iv_vol_iso.shape}")

print("  Background subtraction (median filter)...")
iv_p99 = np.percentile(iv_vol_iso[iv_vol_iso > 0], 99) if (iv_vol_iso > 0).any() else 1
iv_norm = np.clip(iv_vol_iso / iv_p99 * 255, 0, 255)
iv_sub = np.zeros_like(iv_norm)
for z in range(iv_nz):
    bg = median_filter(iv_norm[z], size=15)
    iv_sub[z] = np.clip(iv_norm[z] - bg, 0, 255)
iv_u8 = iv_sub.astype(np.uint8)
del iv_vol_iso, iv_sub, iv_norm

izz, iyy, ixx = np.where(iv_u8 > VOXEL_THRESH_IV)
iv_vals = iv_u8[izz, iyy, ixx]
n_iv = len(izz)
print(f"  {n_iv:,} in-vivo structural voxels")

iv_vx = ixx.astype(np.float32) / iv_nx
iv_vy = iyy.astype(np.float32) / iv_ny
iv_vz = izz.astype(np.float32) / iv_nz
iv_vv = iv_vals.astype(np.float32) / 255.0
del iv_u8

iv_bin = np.column_stack([iv_vx, iv_vy, iv_vz, iv_vv]).astype(np.float32)
iv_bin.tofile(os.path.join(OUT_DIR, 'invivo_struct_voxels.bin'))
print(f"  Wrote invivo_struct_voxels.bin ({iv_bin.nbytes / 1e6:.1f} MB, {n_iv:,} pts)")
del iv_bin, iv_vx, iv_vy, iv_vz, iv_vv, izz, iyy, ixx, iv_vals

# ================================================================
# 3. IN-VIVO CALCIUM: Time-averaged movie warped per-tile to nd2
# ================================================================
print("\n" + "=" * 60)
print("3. In-vivo calcium (time-averaged movie)")
print("=" * 60)

avi_path = os.path.join(BASE, 'movie_rolling_avg_win12_step3_short.avi')
cap = cv2.VideoCapture(avi_path)
cal_frames = []
while True:
    ret, fr = cap.read()
    if not ret:
        break
    cal_frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr)
cap.release()
cal_movie = np.array(cal_frames, dtype=np.float32)
n_cal = len(cal_movie)
print(f"  {n_cal} calcium frames loaded")

# Time-average: mean projection
cal_mean = cal_movie.mean(axis=0)
# Also compute temporal std (activity map)
cal_std = cal_movie.std(axis=0)
del cal_movie

M_movie_to_jy306 = np.load(os.path.join(BASE, 'animation/movie_avi_to_jy306_affine.npz'))['M_affine']
M_m2j_3x3 = np.vstack([M_movie_to_jy306, [0, 0, 1]])

# For calcium, warp mean+std for each tile with landmarks, accumulate into DS volume
# We replicate across the tile's z-slices (calcium is 2D, but we give it depth via tile z)
cal_vol = np.zeros((TOTAL_Z, ex_ny, ex_nx), dtype=np.float32)
cal_count = np.zeros((TOTAL_Z, ex_ny, ex_nx), dtype=np.float32)

for tile in PKL_TILES:
    npz_path = os.path.join(PKL_BASE, tile, f'pkl_transform_{tile}.npz')
    npz = np.load(npz_path, allow_pickle=True)
    M2d = npz['M2d_jy306_to_nd2']

    M_j2n_3x3 = np.vstack([M2d, [0, 0, 1]])
    M_movie_to_nd2 = (M_j2n_3x3 @ M_m2j_3x3)[:2, :]

    # Warp mean calcium to nd2 space
    warped_mean = cv2.warpAffine(cal_mean, M_movie_to_nd2, (FULL_W, FULL_H), borderValue=0)
    warped_std = cv2.warpAffine(cal_std, M_movie_to_nd2, (FULL_W, FULL_H), borderValue=0)

    # Combine mean + std for activity-weighted brightness
    combined = np.clip(warped_mean * 0.5 + warped_std * 2.0, 0, 255)
    ds_combined = combined[::DS, ::DS][:ex_ny, :ex_nx]

    # Place into the tile's z-range (single z-slice in the middle)
    z_off = tile_z_offset[tile]
    mid_z = z_off + SLICES_PER_TILE // 2
    if 0 <= mid_z < TOTAL_Z:
        mask = ds_combined > 5
        cal_vol[mid_z][mask] = np.maximum(cal_vol[mid_z][mask], ds_combined[mask])
        cal_count[mid_z][mask] = 1

del cal_mean, cal_std

cal_u8 = np.clip(cal_vol / max(cal_vol.max(), 1) * 255, 0, 255).astype(np.uint8)
del cal_vol, cal_count

cz, cy, cx = np.where(cal_u8 > VOXEL_THRESH_CAL)
cal_vals = cal_u8[cz, cy, cx]
n_cal_vox = len(cz)
print(f"  {n_cal_vox:,} calcium voxels")

cal_vx = cx.astype(np.float32) / ex_nx
cal_vy = cy.astype(np.float32) / ex_ny
cal_vz = cz.astype(np.float32) / TOTAL_Z
cal_vv = cal_vals.astype(np.float32) / 255.0

cal_bin = np.column_stack([cal_vx, cal_vy, cal_vz, cal_vv]).astype(np.float32)
cal_bin.tofile(os.path.join(OUT_DIR, 'invivo_calcium_voxels.bin'))
print(f"  Wrote invivo_calcium_voxels.bin ({cal_bin.nbytes / 1e6:.1f} MB, {n_cal_vox:,} pts)")
del cal_bin, cal_u8, cal_vx, cal_vy, cal_vz, cal_vv

# ================================================================
# 4. MERSCOPE: Transform transcript positions to nd2 space
# ================================================================
print("\n" + "=" * 60)
print("4. MERSCOPE gene dots")
print("=" * 60)

all_dots_x = []
all_dots_y = []
all_dots_z = []
all_dots_gene = []
gene_set = set()

for tile in PKL_TILES:
    region_id = TILE_TO_REGION.get(tile)
    if region_id is None:
        continue

    csv_path = os.path.join(VAROL, f'region_{region_id}_resegmentation/detected_transcripts.csv')
    mnf_path = os.path.join(VAROL, f'region_{region_id}_resegmentation/manifest.json')
    m2m_path = os.path.join(VAROL, f'region_{region_id}_resegmentation/micron_to_mosaic_pixel_transform.csv')

    # Find pkl
    pkl_merc_file = None
    for fname in os.listdir(PKL_MERC_DIR):
        if fname.endswith('.pkl'):
            mm = re.match(r'[\d_]+_reg(\d+)_transformed.*\.pkl', fname)
            if mm and int(mm.group(1)) == region_id:
                if pkl_merc_file is None or fname > pkl_merc_file:
                    pkl_merc_file = fname

    if not all(os.path.exists(p) for p in [csv_path, mnf_path, m2m_path]) or not pkl_merc_file:
        print(f"  {tile} (region {region_id}): missing data, skipping")
        continue

    m2m = np.loadtxt(m2m_path, delimiter=' ')
    scale_m, tx_m, ty_m = m2m[0, 0], m2m[0, 2], m2m[1, 2]

    with open(mnf_path) as f:
        mnf = json.load(f)
    W_mosaic = mnf['mosaic_width_pixels']

    with open(os.path.join(PKL_MERC_DIR, pkl_merc_file), 'rb') as f:
        pdat = pickle.load(f)
    R_3_inv, offset_3 = build_pkl_affine(pdat['transformations'])
    tif_size = pdat['transformed'].shape[-1]
    nd2_scale = 4200 / tif_size

    df = pd.read_csv(csv_path, usecols=['global_x', 'global_y', 'global_z', 'gene'])
    gx, gy = df.global_x.values, df.global_y.values
    gz = df.global_z.values

    # Transform: MERSCOPE global → mosaic → fliplr → microns → PKL affine → nd2
    x_mos = scale_m * gx + tx_m
    y_mos = scale_m * gy + ty_m
    merc_x = (W_mosaic - 1 - x_mos) * 0.108
    merc_y = y_mos * 0.108
    adj_y = merc_y - offset_3[1]
    adj_x = merc_x - offset_3[2]
    nd2_x = (R_3_inv[2, 1] * adj_y + R_3_inv[2, 2] * adj_x) * nd2_scale
    nd2_y = (R_3_inv[1, 1] * adj_y + R_3_inv[1, 2] * adj_x) * nd2_scale

    # Filter to within tile bounds (0-4200) with margin
    in_bounds = (nd2_x >= -200) & (nd2_x < 4400) & (nd2_y >= -200) & (nd2_y < 4400)
    nd2_x = nd2_x[in_bounds]
    nd2_y = nd2_y[in_bounds]
    gz_filt = gz[in_bounds]
    genes_filt = df.gene.values[in_bounds]

    # Normalize to [0,1] in the full nd2 volume
    z_off = tile_z_offset[tile]
    # Map MERSCOPE z (0-7) across the tile's 12 slices
    nd2_z_norm = (z_off + gz_filt * (SLICES_PER_TILE - 1) / 7.0) / TOTAL_Z

    all_dots_x.append(nd2_x / FULL_W)
    all_dots_y.append(nd2_y / FULL_H)
    all_dots_z.append(nd2_z_norm.astype(np.float32))
    all_dots_gene.append(genes_filt)
    gene_set.update(genes_filt)

    print(f"  {tile} (region {region_id}): {in_bounds.sum():,} dots")

# Concatenate and assign gene indices
all_x = np.concatenate(all_dots_x).astype(np.float32)
all_y = np.concatenate(all_dots_y).astype(np.float32)
all_z = np.concatenate(all_dots_z).astype(np.float32)
all_genes = np.concatenate(all_dots_gene)
n_dots = len(all_x)
print(f"  Total: {n_dots:,} MERSCOPE dots, {len(gene_set)} genes")

# Sort genes by frequency for consistent coloring
gene_counts = Counter(all_genes)
genes_sorted = [g for g, _ in gene_counts.most_common()]
gene_to_idx = {g: i for i, g in enumerate(genes_sorted)}
gene_idx = np.array([gene_to_idx[g] for g in all_genes], dtype=np.float32)

# Subsample if too many (>2M dots would be slow in browser)
MAX_DOTS = 2_000_000
if n_dots > MAX_DOTS:
    print(f"  Subsampling from {n_dots:,} to {MAX_DOTS:,}")
    rng = np.random.default_rng(42)
    sel = rng.choice(n_dots, MAX_DOTS, replace=False)
    sel.sort()
    all_x, all_y, all_z, gene_idx = all_x[sel], all_y[sel], all_z[sel], gene_idx[sel]
    n_dots = MAX_DOTS

merc_bin = np.column_stack([all_x, all_y, all_z, gene_idx]).astype(np.float32)
merc_bin.tofile(os.path.join(OUT_DIR, 'merscope_dots.bin'))
print(f"  Wrote merscope_dots.bin ({merc_bin.nbytes / 1e6:.1f} MB, {n_dots:,} pts)")

# Gene color mapping
gene_colors = {}
for g in genes_sorted[:N_GENE_COLORS]:
    idx = gene_to_idx[g]
    gene_colors[str(idx)] = GENE_PALETTE[idx % N_GENE_COLORS]
# For remaining genes, wrap around
for g in genes_sorted[N_GENE_COLORS:]:
    idx = gene_to_idx[g]
    gene_colors[str(idx)] = GENE_PALETTE[idx % N_GENE_COLORS]

with open(os.path.join(OUT_DIR, 'gene_colors.json'), 'w') as f:
    json.dump(gene_colors, f)
print(f"  Wrote gene_colors.json ({len(gene_colors)} entries)")

del all_x, all_y, all_z, gene_idx, merc_bin, all_dots_x, all_dots_y, all_dots_z, all_dots_gene

# ================================================================
# 5. LANDMARKS: Load all landmark coords in normalized space
# ================================================================
print("\n" + "=" * 60)
print("5. Landmarks")
print("=" * 60)

# Reload JY306 for panels
jy306 = tifffile.imread(iv_tiff).astype(np.float32)

# Reload calcium movie for panels
cap = cv2.VideoCapture(avi_path)
cal_frames_raw = []
while True:
    ret, fr = cap.read()
    if not ret:
        break
    cal_frames_raw.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr)
cap.release()
cal_movie_raw = np.array(cal_frames_raw, dtype=np.uint8)
n_cal_raw = len(cal_movie_raw)
del cal_frames_raw

landmarks = []
cell_idx = 0

for tile in PKL_TILES:
    npz_path = os.path.join(PKL_BASE, tile, f'pkl_transform_{tile}.npz')
    npz = np.load(npz_path, allow_pickle=True)
    ev_nd2 = npz['ev_nd2']       # (N, 3): col, row, z
    iv_pts = npz['pcd_invivo_jy306']  # (N, 3): z, y, x in JY306 s80
    M2d = npz['M2d_jy306_to_nd2']
    nd2_z_gauss = npz['nd2_z_gauss']
    pkl_dist = npz['pkl_dist_um']
    n_cells = ev_nd2.shape[0]

    z_off = tile_z_offset[tile]

    # Pre-load nd2 slices for this tile
    nd2_slices = {}
    for zi in range(SLICES_PER_TILE):
        png_path = os.path.join(PNG_DIR, tile, f'GFP_z{zi:03d}.png')
        nd2_slices[zi] = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)

    # Calcium warp transform for this tile
    M_j2n_3x3 = np.vstack([M2d, [0, 0, 1]])
    M_movie_to_nd2 = (M_j2n_3x3 @ M_m2j_3x3)[:2, :]

    # MERSCOPE data for this tile
    region_id = TILE_TO_REGION.get(tile)
    merc_nd2_x, merc_nd2_y, merc_genes = None, None, None
    if region_id is not None:
        csv_path = os.path.join(VAROL, f'region_{region_id}_resegmentation/detected_transcripts.csv')
        mnf_path = os.path.join(VAROL, f'region_{region_id}_resegmentation/manifest.json')
        m2m_path = os.path.join(VAROL, f'region_{region_id}_resegmentation/micron_to_mosaic_pixel_transform.csv')
        pkl_f = None
        for fname in os.listdir(PKL_MERC_DIR):
            if fname.endswith('.pkl'):
                mm2 = re.match(r'[\d_]+_reg(\d+)_transformed.*\.pkl', fname)
                if mm2 and int(mm2.group(1)) == region_id:
                    if pkl_f is None or fname > pkl_f:
                        pkl_f = fname
        if pkl_f and all(os.path.exists(p) for p in [csv_path, mnf_path, m2m_path]):
            m2m = np.loadtxt(m2m_path, delimiter=' ')
            sc_m, tx_m2, ty_m2 = m2m[0, 0], m2m[0, 2], m2m[1, 2]
            with open(mnf_path) as f:
                mnf = json.load(f)
            W_mos = mnf['mosaic_width_pixels']
            with open(os.path.join(PKL_MERC_DIR, pkl_f), 'rb') as f:
                pd2 = pickle.load(f)
            R3i, off3 = build_pkl_affine(pd2['transformations'])
            tif_sz = pd2['transformed'].shape[-1]
            nd2_sc = 4200 / tif_sz
            df_m = pd.read_csv(csv_path, usecols=['global_x', 'global_y', 'gene'])
            gx2, gy2 = df_m.global_x.values, df_m.global_y.values
            xm = sc_m * gx2 + tx_m2
            ym = sc_m * gy2 + ty_m2
            mx = (W_mos - 1 - xm) * 0.108
            my = ym * 0.108
            ay = my - off3[1]
            ax = mx - off3[2]
            merc_nd2_x = (R3i[2, 1] * ay + R3i[2, 2] * ax) * nd2_sc
            merc_nd2_y = (R3i[1, 1] * ay + R3i[1, 2] * ax) * nd2_sc
            merc_genes = df_m.gene.values

    for ci in range(n_cells):
        x_nd2 = int(round(ev_nd2[ci, 0]))
        y_nd2 = int(round(ev_nd2[ci, 1]))
        z_nd2 = nd2_z_gauss[ci]
        z_in_tile = int(round(z_nd2))
        z_global = z_off + z_nd2

        # Ex-vivo normalized coords
        ex_xn = ev_nd2[ci, 0] / FULL_W
        ex_yn = ev_nd2[ci, 1] / FULL_H
        ex_zn = z_global / TOTAL_Z

        # In-vivo normalized coords (in iso space for consistency with volume)
        iv_z_s80 = iv_pts[ci, 0]
        iv_y_s80 = iv_pts[ci, 1]
        iv_x_s80 = iv_pts[ci, 2]
        iv_xn = (iv_x_s80 * IV_XY_UM) / (iv_w_nat * IV_XY_UM)
        iv_yn = (iv_y_s80 * IV_XY_UM) / (iv_h_nat * IV_XY_UM)
        iv_zn = (iv_z_s80 * IV_Z_UM) / (iv_nz_nat * IV_Z_UM)

        # Calcium normalized (same as ex-vivo since warped to nd2)
        cal_xn = ex_xn
        cal_yn = ex_yn
        cal_zn = (z_off + SLICES_PER_TILE // 2) / TOTAL_Z  # mid tile z

        # MERSCOPE normalized (same x,y as ex-vivo in nd2 space)
        merc_xn = ex_xn
        merc_yn = ex_yn
        merc_zn = ex_zn

        landmarks.append({
            'idx': cell_idx,
            'tile': tile,
            'local': ci,
            'dist_um': float(pkl_dist[ci]),
            'ex': [float(ex_xn), float(ex_yn), float(ex_zn)],
            'iv': [float(iv_xn), float(iv_yn), float(iv_zn)],
            'cal': [float(cal_xn), float(cal_yn), float(cal_zn)],
            'merc': [float(merc_xn), float(merc_yn), float(merc_zn)],
        })

        # ── Generate 4 panels per cell ──
        # 1. Ex-vivo MIP
        slices_ev = []
        for dz in range(-DZ_ND2, DZ_ND2 + 1):
            zz = z_in_tile + dz
            if 0 <= zz < SLICES_PER_TILE and zz in nd2_slices:
                page = nd2_slices[zz]
                slices_ev.append(crop_centered(page, x_nd2, y_nd2, CROP_ND2).astype(np.float32))
        if slices_ev:
            mip_ev = np.max(np.array(slices_ev), axis=0)
            mip_ev = norm_u8(mip_ev)
            mip_ev_r = np.array(Image.fromarray(mip_ev).resize((PATCH_SZ, PATCH_SZ), Image.LANCZOS))
            ev_panel = np.zeros((PATCH_SZ, PATCH_SZ, 3), np.uint8)
            ev_panel[:, :, 2] = mip_ev_r  # magenta: R+B
            ev_panel[:, :, 0] = mip_ev_r
        else:
            ev_panel = np.zeros((PATCH_SZ, PATCH_SZ, 3), np.uint8)

        # 2. In-vivo structural
        z_c_iv = int(round(iv_z_s80))
        y_c_iv = int(round(iv_y_s80))
        x_c_iv = int(round(iv_x_s80))
        slices_iv = []
        for dz in range(-DZ_JY, DZ_JY + 1):
            zz = z_c_iv + dz
            if 0 <= zz < iv_nz_nat:
                page = jy306[zz]
                slices_iv.append(crop_centered(page, x_c_iv, y_c_iv, CROP_JY).astype(np.float32))
        if slices_iv:
            mip_iv = np.max(np.array(slices_iv), axis=0)
            mip_iv = norm_u8(mip_iv)
            mip_iv_r = np.array(Image.fromarray(mip_iv).resize((PATCH_SZ, PATCH_SZ), Image.LANCZOS))
            iv_panel = np.zeros((PATCH_SZ, PATCH_SZ, 3), np.uint8)
            iv_panel[:, :, 1] = mip_iv_r  # green
        else:
            iv_panel = np.zeros((PATCH_SZ, PATCH_SZ, 3), np.uint8)

        # 3. In-vivo calcium (single frame temporal std map, warped)
        cal_std_frame = np.zeros((cal_movie_raw.shape[1], cal_movie_raw.shape[2]), dtype=np.float32)
        # Use std across time as activity indicator
        cal_std_frame = cal_movie_raw.astype(np.float32).std(axis=0)
        warped_cal = cv2.warpAffine(cal_std_frame, M_movie_to_nd2, (FULL_W, FULL_H), borderValue=0)
        cal_crop = crop_centered(warped_cal, x_nd2, y_nd2, CROP_ND2)
        cal_crop_u8 = norm_u8(cal_crop)
        cal_r = np.array(Image.fromarray(cal_crop_u8).resize((PATCH_SZ, PATCH_SZ), Image.LANCZOS))
        cal_panel = np.zeros((PATCH_SZ, PATCH_SZ, 3), np.uint8)
        cal_panel[:, :, 1] = cal_r  # green (functional)

        # 4. MERSCOPE dots panel
        dot_canvas = np.zeros((CROP_ND2 * 2, CROP_ND2 * 2, 3), np.uint8)
        if merc_nd2_x is not None:
            dx = merc_nd2_x - x_nd2
            dy = merc_nd2_y - y_nd2
            in_crop = (np.abs(dx) < CROP_ND2) & (np.abs(dy) < CROP_ND2)
            px = (dx[in_crop] + CROP_ND2).astype(int)
            py = (dy[in_crop] + CROP_ND2).astype(int)
            genes_crop = merc_genes[in_crop]
            gc = Counter(genes_crop)
            gs = [g for g, _ in gc.most_common()]
            g2c = {g: GENE_PALETTE[i % N_GENE_COLORS] for i, g in enumerate(gs)}
            for j in range(len(px)):
                if 0 <= px[j] < CROP_ND2*2 and 0 <= py[j] < CROP_ND2*2:
                    c = g2c[genes_crop[j]]
                    dot_canvas[py[j], px[j]] = c
        merc_panel = np.array(Image.fromarray(dot_canvas).resize((PATCH_SZ, PATCH_SZ), Image.LANCZOS))

        # Save panels as PNG
        for name, panel in [('ev', ev_panel), ('iv_struct', iv_panel),
                            ('iv_cal', cal_panel), ('merscope', merc_panel)]:
            out_path = os.path.join(OUT_DIR, 'panels', f'cell_{cell_idx:04d}_{name}.png')
            cv2.imwrite(out_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR) if name == 'merscope' else panel)

        cell_idx += 1
        if cell_idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {cell_idx} cells processed ({elapsed:.0f}s)")

    del nd2_slices

del jy306, cal_movie_raw

N_CELLS = cell_idx
print(f"  {N_CELLS} landmarks total")

# Write landmarks JSON
with open(os.path.join(OUT_DIR, 'landmarks.json'), 'w') as f:
    json.dump({'n_cells': N_CELLS, 'landmarks': landmarks}, f)
print(f"  Wrote landmarks.json")

# ================================================================
# 6. Metadata
# ================================================================
meta = {
    'n_exvivo': n_ex,
    'n_invivo_struct': n_iv,
    'n_calcium': n_cal_vox,
    'n_merscope': n_dots,
    'n_landmarks': N_CELLS,
    'n_genes': len(gene_set),
    'total_z': TOTAL_Z,
    'full_w': FULL_W,
    'full_h': FULL_H,
    'ds': DS,
    'nd2_xy_um': ND2_XY_UM,
    'nd2_z_um': ND2_Z_UM,
    'iv_xy_um': IV_XY_UM,
    'iv_z_um': IV_Z_UM,
    'iv_shape': [int(iv_nz_nat), int(iv_h_nat), int(iv_w_nat)],
    'iv_iso_shape': [int(iv_nz), int(iv_ny), int(iv_nx)],
    'ex_physical_um': [TOTAL_Z * ND2_Z_UM, FULL_H * ND2_XY_UM, FULL_W * ND2_XY_UM],
    'iv_physical_um': [iv_nz_nat * IV_Z_UM, iv_h_nat * IV_XY_UM, iv_w_nat * IV_XY_UM],
}
with open(os.path.join(OUT_DIR, 'metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2)

elapsed = time.time() - t0
print(f"\nDone! {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"Output: {OUT_DIR}/")
