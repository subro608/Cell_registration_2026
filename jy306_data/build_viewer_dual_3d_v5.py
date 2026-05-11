#!/usr/bin/env python3
"""
4-modality 2×2 grid viewer (v5).
Reuses ex-vivo + in-vivo volumes from the existing dual 3D v5 viewer.
Adds calcium (temporal std) and MERSCOPE (gene dots) as new quadrants.
"""
import numpy as np
import cv2
from PIL import Image
import io, base64, json, glob, os, re, pickle
import pandas as pd
from scipy.ndimage import median_filter
from collections import Counter

BASE = "/Users/neurolab/neuroinformatics/margaret"
PNG_DIR = os.path.join(BASE, "png_exports/registration_video")
PKL_BASE = os.path.join(BASE, "png_exports/registration_per_tile_pkl")
VAROL = os.path.join(BASE, "jy306_varol")
PKL_MERC_DIR = os.path.join(BASE, "merscope_exvivo ")
SRC_HTML = f"{BASE}/3d_viewer/viewer_dual_3d_v5.html"
OUT = f"{BASE}/invivo-exvivo-cell-registration/dual_v5.html"

DS_EX = 5
PATCH_SZ = 80
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0
IV_XY_UM = 0.6835
IV_Z_UM = 3.0

TILE_ORDER = [
    'row1_1', 'row1_2', 'row1_3', 'row1_4',
    'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
    'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
    'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
    'row5_1',
]
SLICES_PER_TILE = 12
tile_z_offset = {t: i * SLICES_PER_TILE for i, t in enumerate(TILE_ORDER)}
n_tiles = len(TILE_ORDER)
total_z = n_tiles * SLICES_PER_TILE
full_h, full_w = 4200, 4200
ex_ny = full_h // DS_EX
ex_nx = full_w // DS_EX

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

def encode_f32(arr):
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode('ascii')

# ============================================================
# 1. Extract data from existing dual 3D v5 viewer
# ============================================================
print("Extracting data from existing dual 3D v5 viewer...")
with open(SRC_HTML) as f:
    src = f.read()

def extract_vox(name):
    m = re.search(rf'const {name}=\{{(.*?)\}};', src, re.DOTALL)
    body = m.group(1)
    fields = {}
    for key in ['x', 'y', 'z', 'v']:
        km = re.search(rf'{key}:"([^"]+)"', body)
        fields[key] = km.group(1)
    nm = re.search(r'n:(\d+)', body)
    fields['n'] = int(nm.group(1))
    return fields

exVox = extract_vox('exVox')
ivVox = extract_vox('ivVox')
print(f"  exVox: {exVox['n']:,} voxels")
print(f"  ivVox: {ivVox['n']:,} voxels")

# Subsample ex-vivo to reduce file size (GitHub 100MB limit)
MAX_EX_DUAL = 600_000
if exVox['n'] > MAX_EX_DUAL:
    def decode_f32(b64, n):
        return np.frombuffer(base64.b64decode(b64), dtype=np.float32)[:n]
    def encode_f32(arr):
        return base64.b64encode(arr.tobytes()).decode('ascii')
    rng = np.random.default_rng(42)
    n_ex = exVox['n']
    vals = decode_f32(exVox['v'], n_ex)
    weights = vals / vals.sum()
    sel = np.sort(rng.choice(n_ex, MAX_EX_DUAL, replace=False, p=weights))
    for k in ['x', 'y', 'z', 'v']:
        arr = decode_f32(exVox[k], n_ex)[sel]
        exVox[k] = encode_f32(arr)
    exVox['n'] = MAX_EX_DUAL
    print(f"  exVox subsampled to {MAX_EX_DUAL:,} (brightness-weighted)")

# Extract constants
def extract_const(name):
    m = re.search(rf'const {name}=([\d.e+-]+)', src)
    return float(m.group(1)) if m else None

EX_SX = extract_const('EX_SX')
EX_SY = extract_const('EX_SY') or EX_SX  # v5 might use single scale
EX_SZ = extract_const('EX_SZ') or extract_const('EX_SZ_RATIO') or 0.195
IV_SX = extract_const('IV_SX')
IV_SY = extract_const('IV_SY') or IV_SX
IV_SZ = extract_const('IV_SZ') or extract_const('IV_SZ_RATIO') or 0.111

# Extract landmarks, tileNames, cellInfo, patchStrip, DZ values
m = re.search(r'const landmarks=(\[.*?\]);', src, re.DOTALL)
landmarks_all = json.loads(m.group(1))

m = re.search(r'const tileNames=(\[.*?\]);', src, re.DOTALL)
tileNames_all = json.loads(m.group(1).replace("'", '"'))

m = re.search(r'const cellInfo=(\[.*?\]);', src, re.DOTALL)
cellInfo_all = json.loads(m.group(1))

# Read patch strip from REGISTRATION viewer (colored: magenta EV, green IV, colored MERSCOPE)
REG_PATCH_CACHE = f'{BASE}/3d_viewer/patch_strip_v5.png'
with open(REG_PATCH_CACHE, 'rb') as f:
    patchStripB64 = base64.b64encode(f.read()).decode('ascii')
print(f"  Registration patch strip: {len(patchStripB64)//1024}KB")

DZ_ND2 = int(extract_const('DZ_ND2') or 2)
DZ_JY = int(extract_const('DZ_JY') or 2)

landmarks_raw = json.dumps(landmarks_all)
tileNames_raw = json.dumps(tileNames_all)
cellInfo_raw = json.dumps(cellInfo_all)
N_CELLS = len(landmarks_all)

print(f"  {N_CELLS} landmarks, DZ_ND2={DZ_ND2}, DZ_JY={DZ_JY}")
print(f"  EX_SX={EX_SX}, IV_SX={IV_SX}")

# Build tileRanges from data
tile_ranges = {}
for i, t in enumerate(tileNames_all):
    if t not in tile_ranges:
        tile_ranges[t] = [i, i + 1]
    else:
        tile_ranges[t][1] = i + 1
tileRanges_raw = json.dumps(tile_ranges)
tile_options_html = f'<option value="all">All ({N_CELLS})</option>'
for tname, (start, end) in tile_ranges.items():
    tile_options_html += f'<option value="{tname}">{tname} ({end-start})</option>'

del src  # free memory

# ============================================================
# 2. Calcium: native movie space (temporal std, no warping)
# ============================================================
print("\nLoading calcium movie (native space)...")
avi_path = os.path.join(BASE, 'movie_rolling_avg_win12_step3_short.avi')
cap = cv2.VideoCapture(avi_path)
cal_frames = []
while True:
    ret, fr = cap.read()
    if not ret: break
    cal_frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr)
cap.release()
cal_movie = np.array(cal_frames, dtype=np.float32)
n_cal = len(cal_movie)
cal_h, cal_w = cal_movie.shape[1], cal_movie.shape[2]
print(f"  {n_cal} frames, {cal_w}x{cal_h}")

cal_std = cal_movie.std(axis=0)
del cal_movie

# Normalize to uint8
cal_max = cal_std.max()
cal_u8 = np.clip(cal_std / max(cal_max, 1) * 255, 0, 255).astype(np.uint8)
del cal_std

VOXEL_THRESH_CAL = 15
cy, cx = np.where(cal_u8 > VOXEL_THRESH_CAL)
cal_vals = cal_u8[cy, cx]
n_cal_vox = len(cy)
print(f"  {n_cal_vox:,} calcium voxels above threshold")

MAX_CAL = 200_000
if n_cal_vox > MAX_CAL:
    rng = np.random.default_rng(42)
    weights = cal_vals.astype(np.float64) / cal_vals.sum()
    sel = rng.choice(n_cal_vox, MAX_CAL, replace=False, p=weights)
    sel.sort()
    cy, cx, cal_vals = cy[sel], cx[sel], cal_vals[sel]
    n_cal_vox = MAX_CAL
    print(f"  Subsampled to {n_cal_vox:,}")

# Native movie pixel coords — keep as raw pixels, buildCloud auto-scales
cal_vx = cx.astype(np.float32)
cal_vy = cy.astype(np.float32)
# Calcium is 2D — add slight random z jitter for mild 3D texture
rng_cal = np.random.default_rng(99)
cal_vz = rng_cal.uniform(-0.02, 0.02, n_cal_vox).astype(np.float32) * max(cal_w, cal_h)
cal_vv = cal_vals.astype(np.float32) / 255.0
# cal_u8 kept for landmark mask check later

# ============================================================
# 3. MERSCOPE gene dots in nd2 space
# ============================================================
print("\nLoading MERSCOPE transcripts...")

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

N_GENE_COLORS = 500
def make_rainbow_palette(n):
    colors = []
    for i in range(n):
        h = int(180 * i / n)
        s = 200 + int(55 * ((i % 3) / 2))
        v = 200 + int(55 * ((i % 5) / 4))
        bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(c) for c in bgr))  # BGR, same as scene7
    return colors
GENE_PALETTE = make_rainbow_palette(N_GENE_COLORS)

all_merc_x, all_merc_y, all_merc_z, all_merc_genes = [], [], [], []
gene_counter = Counter()

# Load stitch params for cumulative IOU transforms (same alignment as ex-vivo tiles)
with open(f'{BASE}/registration_video/stitch_v5_params.json') as f:
    stitch_params = json.load(f)
stitch_z_offsets = stitch_params['tile_z_offsets']

# Load tissue masks for filtering
via_masks_merc = np.load(f'{BASE}/registration_video/via_masks_v4.npz')

for tile in PKL_TILES:
    region_id = TILE_TO_REGION.get(tile)
    if region_id is None: continue
    csv_path = f'{VAROL}/region_{region_id}_resegmentation/detected_transcripts.csv'
    if not os.path.exists(csv_path): continue
    if tile not in stitch_params['cumulative_iou']: continue
    df = pd.read_csv(csv_path, usecols=['global_x', 'global_y', 'global_z', 'gene'])
    df = df[~df['gene'].str.startswith('Blank')]
    gx, gy, gz = df.global_x.values, df.global_y.values, df.global_z.values

    # Normalize region's x,y to tile-local [0, 4200] (like nd2 tile footprint)
    gx_min, gx_max = gx.min(), gx.max()
    gy_min, gy_max = gy.min(), gy.max()
    gx_range = max(gx_max - gx_min, 1)
    gy_range = max(gy_max - gy_min, 1)
    local_x = (4200 - (gx - gx_min) / gx_range * 4200)  # flip x (MERSCOPE mirror)
    local_y = (gy - gy_min) / gy_range * 4200

    # Apply tissue mask: filter dots outside tissue boundary
    if tile in via_masks_merc:
        mask = via_masks_merc[tile]  # 4200×4200 uint8
        ix = np.clip(local_x.astype(int), 0, 4199)
        iy = np.clip(local_y.astype(int), 0, 4199)
        in_tissue = mask[iy, ix] > 0
        local_x, local_y, gz = local_x[in_tissue], local_y[in_tissue], gz[in_tissue]
        genes_ok = df.gene.values[in_tissue]
        print(f"    mask: {in_tissue.sum():,}/{len(in_tissue):,} kept")
    else:
        genes_ok = df.gene.values

    # Apply cumulative IOU transform to place on stitched canvas
    M = np.array(stitch_params['cumulative_iou'][tile])  # 3×3
    canvas_x = M[0, 0] * local_x + M[0, 1] * local_y + M[0, 2]
    canvas_y = M[1, 0] * local_x + M[1, 1] * local_y + M[1, 2]

    # z: use stitch z-offset + MERSCOPE z-plane within tile
    z_off = stitch_z_offsets[tile]
    max_gz = max(df.global_z.max(), 1)
    tile_z = z_off + gz * (SLICES_PER_TILE - 1) / max_gz

    all_merc_x.append(canvas_x.astype(np.float32))
    all_merc_y.append(canvas_y.astype(np.float32))
    all_merc_z.append(tile_z.astype(np.float32))
    all_merc_genes.append(genes_ok)
    gene_counter.update(genes_ok)
    print(f"  {tile} (region {region_id}): {len(canvas_x):,} dots")

merc_x = np.concatenate(all_merc_x)
merc_y = np.concatenate(all_merc_y)
merc_z = np.concatenate(all_merc_z)
merc_genes = np.concatenate(all_merc_genes)
n_merc = len(merc_x)
print(f"  Total: {n_merc:,} dots, {len(gene_counter)} unique genes")
print(f"  x: [{merc_x.min():.0f}, {merc_x.max():.0f}] µm")
print(f"  y: [{merc_y.min():.0f}, {merc_y.max():.0f}] µm")
print(f"  z: [{merc_z.min():.1f}, {merc_z.max():.1f}] (tile-stacked)")

MAX_MERC = 150_000
if n_merc > MAX_MERC:
    rng2 = np.random.default_rng(42)
    sel2 = rng2.choice(n_merc, MAX_MERC, replace=False)
    sel2.sort()
    merc_x, merc_y, merc_z = merc_x[sel2], merc_y[sel2], merc_z[sel2]
    merc_genes = merc_genes[sel2]
    n_merc = MAX_MERC
    print(f"  Subsampled to {n_merc:,}")

# Per-gene colors by frequency (scene7 approach: most common genes get warm HSV = orange/red)
all_genes_sorted = [g for g, _ in gene_counter.most_common()]
gene_to_color_idx = {g: i % N_GENE_COLORS for i, g in enumerate(all_genes_sorted)}
merc_color_idx = np.array([gene_to_color_idx[g] for g in merc_genes], dtype=np.float32)
merc_vv = np.ones(n_merc, dtype=np.float32) * 0.8
print(f"  {len(all_genes_sorted)} genes mapped to {N_GENE_COLORS} rainbow colors")
del all_merc_x, all_merc_y, all_merc_z, all_merc_genes, merc_genes

gene_palette_js = json.dumps([[c[2], c[1], c[0]] for c in GENE_PALETTE[:N_GENE_COLORS]])  # BGR→RGB for JS

# ============================================================
# 4. Generate 4-column patch strip (ex-vivo, in-vivo, calcium, MERSCOPE)
# ============================================================
print("\nGenerating 4-column patch strip...")

# Load raw landmarks from npz files
LM_DIR = f'{BASE}/registration_video'
lm_nd2_files = sorted(glob.glob(f'{LM_DIR}/landmarks_nd2_native_*.npz'))
lm_st_files = sorted(glob.glob(f'{LM_DIR}/landmarks_stitched_v5_*.npz'))

cell_nd2_z_lookup = {}
for f in lm_st_files:
    tile = f.split('landmarks_stitched_v5_')[1].replace('.npz', '')
    d = np.load(f)
    cell_nd2_z_lookup[tile] = d['cell_nd2_z']

all_ev_nd2, all_iv_pts, all_cell_z, all_tiles_lm = [], [], [], []
# Build set of tiles from nd2_native files
nd2_native_tiles = set()
for f in lm_nd2_files:
    nd2_native_tiles.add(f.split('landmarks_nd2_native_')[1].replace('.npz', ''))
# Also include tiles that only have stitched_v5 files (e.g. row2_1)
all_lm_tiles = sorted(nd2_native_tiles | set(cell_nd2_z_lookup.keys()))
for tile in all_lm_tiles:
    nd2_f = f'{LM_DIR}/landmarks_nd2_native_{tile}.npz'
    st_f = f'{LM_DIR}/landmarks_stitched_v5_{tile}.npz'
    if os.path.exists(nd2_f):
        d = np.load(nd2_f)
    elif os.path.exists(st_f):
        d = np.load(st_f)
        print(f"  {tile}: using stitched_v5 fallback (no nd2_native)")
    else:
        continue
    n = d['ev_nd2'].shape[0]
    all_ev_nd2.append(d['ev_nd2'])       # (N,3) as (col, row, z_merc)
    all_iv_pts.append(d['pcd_invivo_jy306'])  # (N,3) as (z, y, x) in s80 pixels
    nd2_z = cell_nd2_z_lookup.get(tile, np.full(n, SLICES_PER_TILE // 2, dtype=np.int64))
    all_cell_z.append(tile_z_offset.get(tile, 0) + nd2_z)
    all_tiles_lm.extend([tile] * n)

ev_nd2 = np.vstack(all_ev_nd2)    # (N,3) col, row, z_merc
iv_pts = np.vstack(all_iv_pts)     # (N,3) z, y, x in JY306 s80
cell_z_arr = np.concatenate(all_cell_z)
N_LM = ev_nd2.shape[0]
print(f"  {N_LM} landmarks loaded")

# ── Compute landmark coords in calcium (movie pixel) and MERSCOPE (canvas) spaces ──
M_m2j_raw = np.load(f'{BASE}/animation/movie_avi_to_jy306_affine.npz')
M_m2j = M_m2j_raw['M_affine']  # 2×3: movie → JY306
CAL_BEST_Z = int(M_m2j_raw['best_z'])
# JY306 → movie pixel: invert M_m2j
M_j2m = cv2.invertAffineTransform(M_m2j)  # 2×3
lm_cal_x = np.zeros(N_LM, dtype=np.float32)
lm_cal_y = np.zeros(N_LM, dtype=np.float32)
lm_cal_z = np.zeros(N_LM, dtype=np.float32)
for i in range(N_LM):
    jy_x, jy_y = iv_pts[i, 2], iv_pts[i, 1]  # x, y in JY306 s80
    movie_pt = M_j2m @ np.array([jy_x, jy_y, 1.0])
    lm_cal_x[i] = movie_pt[0]
    lm_cal_y[i] = movie_pt[1]

# MERSCOPE: ev_nd2 → stitched canvas via cum_iou
lm_merc_x = np.zeros(N_LM, dtype=np.float32)
lm_merc_y = np.zeros(N_LM, dtype=np.float32)
lm_merc_z = np.zeros(N_LM, dtype=np.float32)
idx = 0
for tile in all_lm_tiles:
    nd2_f = f'{LM_DIR}/landmarks_nd2_native_{tile}.npz'
    st_f = f'{LM_DIR}/landmarks_stitched_v5_{tile}.npz'
    if os.path.exists(nd2_f):
        d = np.load(nd2_f)
    elif os.path.exists(st_f):
        d = np.load(st_f)
    else:
        continue
    n = d['ev_nd2'].shape[0]
    if tile in stitch_params['cumulative_iou']:
        M_cum = np.array(stitch_params['cumulative_iou'][tile])
        for j in range(n):
            nd2_x, nd2_y = d['ev_nd2'][j, 0], d['ev_nd2'][j, 1]
            canvas_pt = M_cum @ np.array([nd2_x, nd2_y, 1.0])
            lm_merc_x[idx + j] = canvas_pt[0]
            lm_merc_y[idx + j] = canvas_pt[1]
            z_off = stitch_z_offsets.get(tile, 0)
            lm_merc_z[idx + j] = z_off + SLICES_PER_TILE / 2
    idx += n
print(f"  Landmark coords: cal range x=[{lm_cal_x.min():.0f},{lm_cal_x.max():.0f}], merc canvas x=[{lm_merc_x.min():.0f},{lm_merc_x.max():.0f}]")

# ── Extend landmarks with calcium and MERSCOPE coords ──
# landmarks_all currently has 6 values per landmark: [ev_x, ev_y, ev_z, iv_x, iv_y, iv_z]
# We append 6 more: [cal_x, cal_y, cal_z, merc_x, merc_y, merc_z]
# Also compute combined cal_ok flag (z-check AND within calcium tissue mask)
CAL_Z_TOL = 1
n_cal_ok = 0
for i in range(N_LM):
    cal_z_val = 0.0  # calcium is 2D, cloud z is near-zero jitter
    landmarks_all[i].extend([
        float(lm_cal_x[i]), float(lm_cal_y[i]), cal_z_val,
        float(lm_merc_x[i]), float(lm_merc_y[i]), float(lm_merc_z[i])
    ])
    # Check z: iv_pts is (z, y, x) in JY306 s80
    z_iv = iv_pts[i, 0]
    z_ok = abs(z_iv - CAL_BEST_Z) <= CAL_Z_TOL
    # Check mask: is landmark within calcium tissue boundary?
    cx_i, cy_i = int(round(lm_cal_x[i])), int(round(lm_cal_y[i]))
    in_mask = (0 <= cx_i < cal_w and 0 <= cy_i < cal_h and cal_u8[cy_i, cx_i] > VOXEL_THRESH_CAL)
    cal_ok = 1 if (z_ok and in_mask) else 0
    # Update cellInfo: overwrite index 6 with combined cal_ok
    if len(cellInfo_all[i]) > 6:
        cellInfo_all[i][6] = cal_ok
    else:
        cellInfo_all[i].append(cal_ok)
    n_cal_ok += cal_ok
landmarks_raw = json.dumps(landmarks_all)
cellInfo_raw = json.dumps(cellInfo_all)
del cal_u8
print(f"  Extended landmarks to 12 values each (added cal + merc coords)")
print(f"  Calcium OK (z + mask): {n_cal_ok}/{N_LM} landmarks")

# ── Load calcium video strip from registration viewer (nd2 space, pre-cropped) ──
CAL_STRIP_CACHE = f'{BASE}/3d_viewer/cal_vid_strip_v5.png'
with open(CAL_STRIP_CACHE, 'rb') as f:
    calStripB64 = base64.b64encode(f.read()).decode('ascii')
cap_tmp = cv2.VideoCapture(os.path.join(BASE, 'movie_rolling_avg_win12_step3_short.avi'))
n_cal_frames_raw = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
cap_tmp.release()
CAL_STEP = 3
n_cal = len(range(0, n_cal_frames_raw, CAL_STEP))
CAL_PSZ = 40
print(f"  Calcium strip (nd2 space): {len(calStripB64)//1024}KB, {n_cal} frames, {CAL_PSZ}px patches")

# ── MERSCOPE: pre-load transform chain per tile for dot rendering (scene7 approach) ──
print("  Loading MERSCOPE transform chains...")
merc_tile_data = {}
for tile in PKL_TILES:
    region_id = TILE_TO_REGION.get(tile)
    if region_id is None: continue
    csv_path = f'{VAROL}/region_{region_id}_resegmentation/detected_transcripts.csv'
    mnf_path = f'{VAROL}/region_{region_id}_resegmentation/manifest.json'
    m2m_path = f'{VAROL}/region_{region_id}_resegmentation/micron_to_mosaic_pixel_transform.csv'
    pkl_f = None
    for fname in os.listdir(PKL_MERC_DIR):
        if fname.endswith('.pkl'):
            mm = re.match(r'[\d_]+_reg(\d+)_transformed.*\.pkl', fname)
            if mm and int(mm.group(1)) == region_id:
                if pkl_f is None or fname > pkl_f: pkl_f = fname
    if not pkl_f or not all(os.path.exists(p) for p in [csv_path, mnf_path, m2m_path]):
        continue
    m2m = np.loadtxt(m2m_path, delimiter=' ')
    sc_m, tx_m, ty_m = m2m[0, 0], m2m[0, 2], m2m[1, 2]
    with open(mnf_path) as f: mnf = json.load(f)
    W_mos = mnf['mosaic_width_pixels']
    with open(f'{PKL_MERC_DIR}/{pkl_f}', 'rb') as f: pdat = pickle.load(f)
    R3i, off3 = build_pkl_affine(pdat['transformations'])
    tif_sz = pdat['transformed'].shape[-1]
    nd2_sc = 4200 / tif_sz
    merc_tile_data[tile] = {
        'csv': csv_path, 'sc_m': sc_m, 'tx_m': tx_m, 'ty_m': ty_m,
        'W_mos': W_mos, 'R3i': R3i, 'off3': off3, 'nd2_sc': nd2_sc,
    }
print(f"  {len(merc_tile_data)} tile transform chains loaded")

# Gene palette for dot rendering (same as 3D cloud)
all_genes_for_patches = sorted(gene_counter.keys())
gene_to_bgr = {}
for gi, g in enumerate(all_genes_for_patches):
    c = GENE_PALETTE[gi % N_GENE_COLORS]
    gene_to_bgr[g] = (c[2], c[1], c[0])  # RGB → BGR for cv2

CROP_RADIUS = 130  # nd2 pixels (~84µm), same as scene7 CROP_SM

# Load nd2 tiles for ex-vivo patches (same crop as scene7)
print("  Loading nd2 tiles for ex-vivo patches...")
nd2_pages = {}
for tile in set(all_tiles_lm):
    tile_dir = os.path.join(PNG_DIR, tile)
    for zi in range(SLICES_PER_TILE):
        png_path = os.path.join(tile_dir, f'GFP_z{zi:03d}.png')
        if os.path.exists(png_path):
            nd2_pages[(tile, zi)] = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)

# Get nd2 z-slice per landmark
cell_z_in_tile = []
for i in range(N_LM):
    tile = all_tiles_lm[i]
    gz = int(cell_z_arr[i]) - tile_z_offset.get(tile, 0)
    cell_z_in_tile.append(max(0, min(SLICES_PER_TILE - 1, gz)))

# Reuse registration viewer's 3-column patch strip (EV | IV warped | MERSCOPE, all in nd2 space)
# patchStripB64 was already extracted from the registration viewer at line 108-109
# Calcium is handled separately via calStripB64 (JS animation)
print("  Using registration viewer patch strip (registered space)")
N_CELLS = N_LM

# Crosshair positions: all at center since registration patches are landmark-centered
crosshairs = np.full((N_LM, 8), PATCH_SZ // 2, dtype=np.float32)
crosshairs_js = json.dumps(np.round(crosshairs, 1).tolist())


# patchStripB64 reused as-is from registration viewer (3-col: EV | IV warped | MERSCOPE, all nd2 space)
# Registration strip already has crosshairs baked in — disable JS crosshairs
print(f"  Reusing registration strip: {len(patchStripB64) // 1024}KB, {N_LM} landmarks")

# ============================================================
# 5. Build 2x2 grid HTML
# ============================================================
print("\nBuilding 2x2 grid HTML...")

# nd2 physical aspect (for ex-vivo, calcium, MERSCOPE quadrants)
nd2_x_um = full_w * ND2_XY_UM
nd2_y_um = full_h * ND2_XY_UM
nd2_z_um = total_z * ND2_Z_UM
nd2_max = max(nd2_x_um, nd2_y_um, nd2_z_um)

html_parts = []
html_parts.append(f'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>4-Modality 3D Viewer</title>
<style>
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#000;color:#ddd;font-family:'SF Mono','Fira Code',monospace;font-size:11px;overflow:hidden;height:100vh;display:flex;flex-direction:column}}
#grid{{flex:1;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:1px;background:#222}}
.cell{{position:relative;overflow:hidden;background:#000}}
.cell canvas{{display:block;width:100%;height:100%}}
.cell-label{{position:absolute;top:8px;left:10px;font-size:13px;font-weight:bold;letter-spacing:0.08em;text-shadow:0 0 8px rgba(0,0,0,0.9);z-index:5;pointer-events:none}}
.cell-info{{position:absolute;bottom:6px;left:10px;font-size:10px;color:#888;z-index:5;pointer-events:none}}
#controls{{position:fixed;top:8px;right:8px;z-index:30;background:rgba(0,0,0,0.88);padding:10px 14px;border-radius:6px;border:1px solid #333;min-width:180px;max-height:90vh;overflow-y:auto}}
#controls label{{display:block;margin:2px 0;font-size:10px}}
#controls hr{{border-color:#333;margin:5px 0}}
#controls input[type=range]{{width:80px;vertical-align:middle}}
.st{{font-weight:bold;font-size:10px;margin-bottom:1px}}
#patchPanel{{position:fixed;bottom:0;left:0;right:0;height:0;background:rgba(0,0,0,0.95);z-index:40;transition:height 0.3s;overflow:hidden;border-top:1px solid #444}}
#patchPanel.show{{height:160px}}
#patchInner{{display:flex;align-items:center;justify-content:center;gap:20px;height:100%;padding:8px}}
.ppair{{text-align:center}}
.ppair canvas{{width:100px;height:100px;image-rendering:pixelated;border:2px solid #555;border-radius:3px}}
.plabel{{font-size:10px;margin-bottom:3px}}
#closeBtn{{position:absolute;top:5px;right:12px;cursor:pointer;color:#f55;font-size:16px;font-weight:bold;z-index:41}}
#pairInfo{{font-size:10px;text-align:center}}
#lineCanvas{{position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:20}}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head><body>
<div id="viewToggle" style="position:fixed;top:8px;left:50%;transform:translateX(-50%);z-index:50;display:flex;gap:0;border-radius:6px;overflow:hidden;border:1px solid #555">
  <button style="background:#444;color:#fff;border:none;padding:6px 16px;cursor:pointer;font-size:11px;font-family:inherit;font-weight:bold">MODALITY</button>
  <button onclick="window.location.href='viewer_warped_invivo_3d_v5.html'" style="background:#222;color:#888;border:none;border-left:1px solid #555;padding:6px 16px;cursor:pointer;font-size:11px;font-family:inherit">REGISTRATION</button>
</div>
<div id="grid">
  <div class="cell" id="cell0">
    <div class="cell-label" style="color:#f0f">EX VIVO Z-STACK</div>
    <div class="cell-info">{exVox['n']:,} voxels | {n_tiles} tiles &times; 12z</div>
  </div>
  <div class="cell" id="cell1">
    <div class="cell-label" style="color:#0f0">IN VIVO Z-STACK</div>
    <div class="cell-info">{ivVox['n']:,} voxels | JY306 s80</div>
  </div>
  <div class="cell" id="cell2">
    <div class="cell-label" style="color:#0f0">IN VIVO CALCIUM</div>
    <div class="cell-info">{n_cal_vox:,} voxels | temporal std | {n_cal} frames</div>
  </div>
  <div class="cell" id="cell3">
    <div class="cell-label" style="color:#ff0">MERSCOPE mRNA EXPRESSION</div>
    <div class="cell-info">{n_merc:,} dots | {len(gene_counter)} genes | 19 tiles</div>
  </div>
</div>
<canvas id="lineCanvas"></canvas>
<div id="controls">
  <span class="st" style="color:#f0f">EX VIVO</span>
  <label>Opacity: <input type="range" id="op0" min="1" max="100" value="57"><span id="opV0">57</span></label>
  <label>Pt size: <input type="range" id="ps0" min="1" max="20" value="1"><span id="psV0">1</span></label>
  <hr>
  <span class="st" style="color:#0f0">IN VIVO</span>
  <label>Opacity: <input type="range" id="op1" min="1" max="100" value="74"><span id="opV1">74</span></label>
  <label>Pt size: <input type="range" id="ps1" min="1" max="20" value="1"><span id="psV1">1</span></label>
  <hr>
  <span class="st" style="color:#0f0">CALCIUM</span>
  <label>Opacity: <input type="range" id="op2" min="1" max="100" value="40"><span id="opV2">40</span></label>
  <label>Pt size: <input type="range" id="ps2" min="1" max="20" value="2"><span id="psV2">2</span></label>
  <hr>
  <span class="st" style="color:#ff0">MERSCOPE</span>
  <label>Opacity: <input type="range" id="op3" min="1" max="100" value="30"><span id="opV3">30</span></label>
  <label>Pt size: <input type="range" id="ps3" min="1" max="20" value="2"><span id="psV3">2</span></label>
  <hr>
  <label>Line opacity: <input type="range" id="lineOpac" min="1" max="100" value="12" style="width:80px"><span id="loVal">12</span></label>
  <label>Tile: <select id="tileSelect">{tile_options_html}</select></label>
  <label><input type="checkbox" id="showLines" checked> Show lines</label>
  <label><input type="checkbox" id="showCross" checked> Crosshairs</label>
  <label><input type="checkbox" id="autorot"> Auto-rotate</label>
</div>
<div id="patchPanel">
  <span id="closeBtn">&times;</span>
  <div id="patchInner">
    <div class="ppair"><div class="plabel" style="color:#0f0">CALCIUM</div><canvas id="pCalCv" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas></div>
    <div class="ppair"><div class="plabel" style="color:#0f0">IN VIVO</div><canvas id="pIvCv" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas></div>
    <div id="pairInfo"></div>
    <div class="ppair"><div class="plabel" style="color:#f0f">EX VIVO</div><canvas id="pExCv" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas></div>
    <div class="ppair"><div class="plabel" style="color:#ff0">MERSCOPE</div><canvas id="pMsCv" width="{PATCH_SZ}" height="{PATCH_SZ}"></canvas></div>
  </div>
</div>
<script>
const N_CELLS={N_CELLS}, PATCH_SZ={PATCH_SZ};
const landmarks={landmarks_raw};
const tileNames={tileNames_raw};
const tileRanges={tileRanges_raw};
const cellInfo={cellInfo_raw};
const DZ_ND2={DZ_ND2}, DZ_JY={DZ_JY};
const crosshairPos={crosshairs_js};
const CAL_N_FRAMES={n_cal}, CAL_PSZ={CAL_PSZ};
''')


# Embed the volume data directly from the source
html_parts.append(f'const exVox={{x:"{exVox["x"]}",y:"{exVox["y"]}",z:"{exVox["z"]}",v:"{exVox["v"]}",n:{exVox["n"]}}};\n')
html_parts.append(f'const ivVox={{x:"{ivVox["x"]}",y:"{ivVox["y"]}",z:"{ivVox["z"]}",v:"{ivVox["v"]}",n:{ivVox["n"]}}};\n')
html_parts.append(f'const calVox={{x:"{encode_f32(cal_vx)}",y:"{encode_f32(cal_vy)}",z:"{encode_f32(cal_vz)}",v:"{encode_f32(cal_vv)}",n:{n_cal_vox}}};\n')
html_parts.append(f'const mercVox={{x:"{encode_f32(merc_x)}",y:"{encode_f32(merc_y)}",z:"{encode_f32(merc_z)}",v:"{encode_f32(merc_vv)}",ci:"{encode_f32(merc_color_idx.astype(np.float32))}",n:{n_merc}}};\n')
html_parts.append(f'const genePalette={gene_palette_js};\n')
html_parts.append(f'const patchStripB64="{patchStripB64}";\n')
html_parts.append(f'const calStripB64="{calStripB64}";\n')

html_parts.append('''
const CELLS=['cell0','cell1','cell2','cell3'];
const CMAPS=['magenta','green','green','rainbow'];
const DEF_OPAC=[0.57, 0.74, 0.40, 0.30];
const DEF_PS=[1, 1, 2, 2];
const PS_MULT=[0.006, 0.006, 0.006, 0.006];
const LM_OFFSETS=[0, 3, 6, 9];  // landmark coord offsets per quadrant: EV, IV, CAL, MERC

let scenes=[], cameras=[], renderers=[], pivots=[];
let clouds=[], cloudInfos=[];
let rotX=-0.3, rotY=0.5, camZ=2.5, panX=0, panY=0;
let dragging=false, shiftDrag=false, lastMX=0, lastMY=0;
let autoRotate=false;
let patchStripImg=null;
let calStripImg=null;
let calAnimId=null, calFrameIdx=0, calCurrentLm=-1;
let lineCv=null, lineCtx=null;
let hoveredIdx=-1;
let selectedSet=new Set();
let visibleIndices=[];

function b64toF32(b64, n) {
  const bin=atob(b64);const buf=new ArrayBuffer(n*4);const u8=new Uint8Array(buf);
  for(let i=0;i<bin.length;i++) u8[i]=bin.charCodeAt(i);
  return new Float32Array(buf);
}

function colormap(v, name) {
  if(name==='green') return [0, v, 0];
  if(name==='green_bright') return [v*0.3, v, v*0.1];
  if(name==='magenta') return [v, 0, v];
  return [v, v, v];
}

function buildCloud(data, cmap) {
  const n=data.n;
  const xs=b64toF32(data.x,n),ys=b64toF32(data.y,n),zs=b64toF32(data.z,n),vs=b64toF32(data.v,n);
  let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity,zmin=Infinity,zmax=-Infinity;
  for(let i=0;i<n;i++){
    if(xs[i]<xmin)xmin=xs[i];if(xs[i]>xmax)xmax=xs[i];
    if(ys[i]<ymin)ymin=ys[i];if(ys[i]>ymax)ymax=ys[i];
    if(zs[i]<zmin)zmin=zs[i];if(zs[i]>zmax)zmax=zs[i];
  }
  const xr=xmax-xmin||1, yr=ymax-ymin||1, zr=zmax-zmin||1;
  const xc=(xmax+xmin)/2, yc=(ymax+ymin)/2, zc=(zmax+zmin)/2;
  const maxR=Math.max(xr,yr,zr);
  const pos=new Float32Array(n*3), col=new Float32Array(n*3);
  if(cmap==='rainbow' && data.ci) {
    const ci=b64toF32(data.ci,n);
    for(let i=0;i<n;i++){
      pos[i*3]=(xs[i]-xc)/maxR*2; pos[i*3+1]=-(ys[i]-yc)/maxR*2; pos[i*3+2]=(zs[i]-zc)/maxR*2;
      const c=genePalette[Math.floor(ci[i])%genePalette.length];
      col[i*3]=c[0]/255; col[i*3+1]=c[1]/255; col[i*3+2]=c[2]/255;
    }
  } else {
    for(let i=0;i<n;i++){
      pos[i*3]=(xs[i]-xc)/maxR*2; pos[i*3+1]=-(ys[i]-yc)/maxR*2; pos[i*3+2]=(zs[i]-zc)/maxR*2;
      const [r,g,b]=colormap(vs[i],cmap);
      col[i*3]=r; col[i*3+1]=g; col[i*3+2]=b;
    }
  }
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  geo.setAttribute('color',new THREE.BufferAttribute(col,3));
  return {geo, xc, yc, zc, maxR};
}

// Project a landmark's 3D position in quadrant qi to 2D screen coords
function projectLm(qi, lmIdx) {
  const lm=landmarks[lmIdx];
  const off=LM_OFFSETS[qi];
  const ci=cloudInfos[qi];
  const px=(lm[off]-ci.xc)/ci.maxR*2, py=-(lm[off+1]-ci.yc)/ci.maxR*2, pz=(lm[off+2]-ci.zc)/ci.maxR*2;
  const v=new THREE.Vector3(px,py,pz);
  v.applyMatrix4(pivots[qi].matrixWorld);
  v.project(cameras[qi]);
  const cell=document.getElementById(CELLS[qi]);
  const rect=cell.getBoundingClientRect();
  return {x: rect.left+(v.x+1)/2*rect.width, y: rect.top+(-v.y+1)/2*rect.height};
}

function getVisibleIndices() {
  const sel=document.getElementById('tileSelect').value;
  if(sel==='all'){const a=[];for(let i=0;i<landmarks.length;i++)a.push(i);return a;}
  const r=tileRanges[sel];const a=[];for(let i=r[0];i<r[1];i++)a.push(i);return a;
}

// Distance from point (px,py) to line segment (x1,y1)-(x2,y2)
function distToSeg(px,py,x1,y1,x2,y2){
  const dx=x2-x1,dy=y2-y1,len2=dx*dx+dy*dy;
  if(len2<1e-6)return Math.hypot(px-x1,py-y1);
  let t=((px-x1)*dx+(py-y1)*dy)/len2;
  t=Math.max(0,Math.min(1,t));
  return Math.hypot(px-(x1+t*dx),py-(y1+t*dy));
}

function findNearestLine(mx, my) {
  if(!document.getElementById('showLines').checked) return -1;
  let bestDist=15, bestIdx=-1;
  const pairs=[[0,1],[0,2],[0,3]];
  for(let j=0;j<visibleIndices.length;j++){
    const i=visibleIndices[j];
    for(const pa of pairs){
      const p0=projectLm(pa[0],i), p1=projectLm(pa[1],i);
      const d=distToSeg(mx,my,p0.x,p0.y,p1.x,p1.y);
      if(d<bestDist){bestDist=d;bestIdx=i;}
    }
  }
  return bestIdx;
}

function drawLines() {
  if(!lineCtx) return;
  lineCv.width=window.innerWidth; lineCv.height=window.innerHeight;
  lineCtx.clearRect(0,0,lineCv.width,lineCv.height);
  if(!document.getElementById('showLines').checked) return;
  const lineOpac=+document.getElementById('lineOpac').value/100;
  // Line pairs: EV↔IV, EV↔CAL, EV↔MERC
  const pairs=[[0,1],[0,2],[0,3]];
  const pairColors=['rgba(0,180,0,','rgba(0,180,0,','rgba(180,120,0,'];
  for(let j=0;j<visibleIndices.length;j++){
    const i=visibleIndices[j];
    const isHover=(i===hoveredIdx), isSel=selectedSet.has(i);
    const calZOk=cellInfo[i].length>6 ? cellInfo[i][6] : 1;
    const pts=[projectLm(0,i),projectLm(1,i),projectLm(2,i),projectLm(3,i)];
    for(let pi=0;pi<pairs.length;pi++){
      if(pi===1 && !calZOk) continue;  // skip calcium line if wrong z
      const pa=pairs[pi];
      lineCtx.beginPath();
      lineCtx.moveTo(pts[pa[0]].x, pts[pa[0]].y);
      lineCtx.lineTo(pts[pa[1]].x, pts[pa[1]].y);
      if(isHover){
        lineCtx.strokeStyle='rgba(180,180,0,0.5)';
        lineCtx.lineWidth=2;
      } else if(isSel){
        lineCtx.strokeStyle='rgba(0,160,80,0.4)';
        lineCtx.lineWidth=1.5;
      } else {
        lineCtx.strokeStyle=pairColors[pi]+lineOpac+')';
        lineCtx.lineWidth=0.6;
      }
      lineCtx.stroke();
    }
    if(isHover||isSel){
      const color=isHover?'rgba(180,180,0,0.5)':'rgba(0,160,80,0.4)';
      const r=isHover?4:3;
      pts.forEach((p,pIdx)=>{
        if(pIdx===2 && !calZOk) return;  // skip calcium dot if wrong z
        lineCtx.beginPath();
        lineCtx.arc(p.x,p.y,r,0,Math.PI*2);
        lineCtx.fillStyle=color;
        lineCtx.fill();
      });
    }
  }
}

function drawCalFrame(idx, fi) {
  if(!calStripImg||!calStripImg.complete) return;
  const cv=document.getElementById('pCalCv'), ctx=cv.getContext('2d');
  ctx.clearRect(0,0,PATCH_SZ,PATCH_SZ);
  const sy=(idx*CAL_N_FRAMES+fi)*CAL_PSZ;
  const off=document.createElement('canvas');off.width=CAL_PSZ;off.height=CAL_PSZ;
  const oCtx=off.getContext('2d');
  oCtx.drawImage(calStripImg,0,sy,CAL_PSZ,CAL_PSZ,0,0,CAL_PSZ,CAL_PSZ);
  const idata=oCtx.getImageData(0,0,CAL_PSZ,CAL_PSZ);
  const d=idata.data;
  for(let p=0;p<d.length;p+=4){const v=d[p]; d[p]=0; d[p+1]=v; d[p+2]=0; d[p+3]=255;}
  oCtx.putImageData(idata,0,0);
  ctx.imageSmoothingEnabled=true;
  ctx.drawImage(off,0,0,CAL_PSZ,CAL_PSZ,0,0,PATCH_SZ,PATCH_SZ);
}

function startCalAnim(idx) {
  if(calAnimId) clearInterval(calAnimId);
  calCurrentLm=idx; calFrameIdx=0;
  const calZOk = cellInfo[idx].length>6 ? cellInfo[idx][6] : 1;
  const cv=document.getElementById('pCalCv'), ctx=cv.getContext('2d');
  if(!calZOk) {
    ctx.clearRect(0,0,PATCH_SZ,PATCH_SZ);
    ctx.fillStyle='#000'; ctx.fillRect(0,0,PATCH_SZ,PATCH_SZ);
    ctx.fillStyle='#666'; ctx.font='9px monospace'; ctx.textAlign='center';
    ctx.fillText('No calcium',PATCH_SZ/2,PATCH_SZ/2-6);
    ctx.fillText('at z='+cellInfo[idx][3],PATCH_SZ/2,PATCH_SZ/2+6);
    return;
  }
  drawCalFrame(idx, 0);
  calAnimId=setInterval(function(){
    calFrameIdx=(calFrameIdx+1)%CAL_N_FRAMES;
    drawCalFrame(calCurrentLm, calFrameIdx);
  }, 100);
}

function stopCalAnim() {
  if(calAnimId){clearInterval(calAnimId);calAnimId=null;}
  calCurrentLm=-1;
}

function showPatch(idx) {
  if(!patchStripImg||!patchStripImg.complete) return;
  const sy=idx*PATCH_SZ;
  const ids=['pExCv','pIvCv','pMsCv'];  // calcium handled separately
  const cols=[0,1,2];
  const showCross=document.getElementById('showCross').checked;
  const ch=idx<crosshairPos.length?crosshairPos[idx]:null;
  const colors=['#ff00ff','#00ff00','#ffaa00'];
  for(let c=0;c<3;c++){
    const cv=document.getElementById(ids[c]),ctx=cv.getContext('2d');
    ctx.clearRect(0,0,PATCH_SZ,PATCH_SZ);
    ctx.drawImage(patchStripImg,cols[c]*PATCH_SZ,sy,PATCH_SZ,PATCH_SZ,0,0,PATCH_SZ,PATCH_SZ);
    if(showCross){
      const cx=PATCH_SZ/2,cy=PATCH_SZ/2;
      ctx.strokeStyle=colors[c];ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(cx-8,cy);ctx.lineTo(cx+8,cy);ctx.stroke();
      ctx.beginPath();ctx.moveTo(cx,cy-8);ctx.lineTo(cx,cy+8);ctx.stroke();
    }
  }
  // Start calcium animation
  startCalAnim(idx);
  const ci=cellInfo[idx];
  const evZ=Math.round((ci[1]+ci[2])/2), ivZ=Math.round((ci[4]+ci[5])/2);
  document.getElementById('pairInfo').innerHTML='<b>#'+idx+'</b><br>('+tileNames[idx]+')<br>'+
    '<span style="color:#f0f;font-size:9px">ev z='+evZ+'</span><br>'+
    '<span style="color:#0f0;font-size:9px">iv z='+ivZ+'</span>';
  document.getElementById('patchPanel').classList.add('show');
}

function init() {
  const VOX_DATA=[exVox, ivVox, calVox, mercVox];

  lineCv=document.getElementById('lineCanvas');
  lineCtx=lineCv.getContext('2d');

  patchStripImg=new Image();
  patchStripImg.src='data:image/png;base64,'+patchStripB64;
  calStripImg=new Image();
  calStripImg.src='data:image/png;base64,'+calStripB64;

  for(let qi=0;qi<4;qi++){
    const container=document.getElementById(CELLS[qi]);
    const w=container.clientWidth, h=container.clientHeight;
    const scene=new THREE.Scene();
    const camera=new THREE.PerspectiveCamera(50, w/h, 0.01, 100);
    camera.position.set(0,0,camZ);
    const renderer=new THREE.WebGLRenderer({antialias:false, alpha:false});
    renderer.setSize(w,h);
    renderer.setPixelRatio(1);
    renderer.setClearColor(0x000000);
    container.appendChild(renderer.domElement);
    const pivot=new THREE.Group();
    scene.add(pivot);
    scenes.push(scene); cameras.push(camera); renderers.push(renderer); pivots.push(pivot);

    const cloud=buildCloud(VOX_DATA[qi], CMAPS[qi]);
    cloudInfos.push({xc:cloud.xc, yc:cloud.yc, zc:cloud.zc, maxR:cloud.maxR});
    const mat=new THREE.PointsMaterial({
      size:DEF_PS[qi]*PS_MULT[qi], vertexColors:true, transparent:true, opacity:DEF_OPAC[qi],
      blending:THREE.AdditiveBlending, depthWrite:false
    });
    const pts=new THREE.Points(cloud.geo,mat);
    pivot.add(pts);
    clouds.push(pts);
  }

  // Controls
  for(let qi=0;qi<4;qi++){
    const opS=document.getElementById('op'+qi), opV=document.getElementById('opV'+qi);
    const psS=document.getElementById('ps'+qi), psV=document.getElementById('psV'+qi);
    opS.oninput=function(){opV.textContent=this.value;clouds[qi].material.opacity=this.value/100;};
    psS.oninput=function(){psV.textContent=this.value;clouds[qi].material.size=this.value*PS_MULT[qi];};
  }
  document.getElementById('lineOpac').oninput=function(){document.getElementById('loVal').textContent=this.value;};
  document.getElementById('autorot').onchange=function(){autoRotate=this.checked;};
  document.getElementById('tileSelect').addEventListener('change',function(){visibleIndices=getVisibleIndices();});
  document.getElementById('showLines').addEventListener('change',function(){if(!this.checked){hoveredIdx=-1;}});
  document.getElementById('closeBtn').onclick=function(){
    document.getElementById('patchPanel').classList.remove('show');
    stopCalAnim();
    selectedSet.clear();
  };

  visibleIndices=getVisibleIndices();

  // Mouse interaction
  const grid=document.getElementById('grid');
  let clickStartX=0, clickStartY=0;
  grid.addEventListener('mousedown',function(e){
    dragging=true;shiftDrag=e.shiftKey;lastMX=e.clientX;lastMY=e.clientY;
    clickStartX=e.clientX;clickStartY=e.clientY;
  });
  window.addEventListener('mouseup',function(e){
    const wasDrag=Math.abs(e.clientX-clickStartX)>3||Math.abs(e.clientY-clickStartY)>3;
    if(!wasDrag){
      // Click — find nearest line
      const idx=findNearestLine(e.clientX, e.clientY);
      if(idx>=0){
        if(selectedSet.has(idx)){
          selectedSet.delete(idx);
          if(selectedSet.size===0) document.getElementById('patchPanel').classList.remove('show');
        } else {
          selectedSet.add(idx);
          showPatch(idx);
        }
      } else {
        selectedSet.clear();
        document.getElementById('patchPanel').classList.remove('show');
      }
    }
    dragging=false;
  });
  window.addEventListener('mousemove',function(e){
    if(dragging){
      const dx=e.clientX-lastMX,dy=e.clientY-lastMY;
      lastMX=e.clientX;lastMY=e.clientY;
      if(shiftDrag||e.shiftKey){panX+=dx*0.003;panY-=dy*0.003;}
      else{rotY+=dx*0.006;rotX+=dy*0.006;rotX=Math.max(-Math.PI/2,Math.min(Math.PI/2,rotX));}
    } else {
      // Hover detection
      const idx=findNearestLine(e.clientX, e.clientY);
      if(idx!==hoveredIdx){
        hoveredIdx=idx;
        document.body.style.cursor=idx>=0?'pointer':'default';
      }
    }
  });
  grid.addEventListener('wheel',function(e){
    e.preventDefault();camZ*=1+e.deltaY*0.001;camZ=Math.max(0.5,Math.min(10,camZ));
  },{passive:false});

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  if(autoRotate) rotY+=0.003;
  for(let qi=0;qi<4;qi++){
    pivots[qi].rotation.set(rotX,rotY,0);
    pivots[qi].position.set(panX,panY,0);
    cameras[qi].position.z=camZ;
    renderers[qi].render(scenes[qi],cameras[qi]);
  }
  drawLines();
}

window.addEventListener('resize',function(){
  for(let qi=0;qi<4;qi++){
    const c=document.getElementById(CELLS[qi]);
    const w=c.clientWidth,h=c.clientHeight;
    cameras[qi].aspect=w/h; cameras[qi].updateProjectionMatrix();
    renderers[qi].setSize(w,h);
  }
});

init();
</script></body></html>
''')

html = "".join(html_parts)
with open(OUT, 'w') as f:
    f.write(html)
print(f"Done! {OUT} ({len(html)/1e6:.1f} MB)")
