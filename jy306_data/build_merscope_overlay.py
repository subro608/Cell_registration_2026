#!/usr/bin/env python3
"""Full-FOV overlay: ex-vivo GCaMP (green) + MERFISH gene dots (per-gene colour).
One PNG per region, saved to png_exports/merscope_overlay/.
Uses full-resolution nd2 tiles (4200×4200) for the GCaMP.

Coordinate pipeline:
  global microns → mosaic pixels (micron2mosaic) → fliplr → zoom(0.108)
  = MERSCOPE source coords → PKL affine inverse → ex-vivo TIF coords
  → scale by (4200/TIF_SIZE) → nd2 native coords
"""
import numpy as np
import cv2
import nd2
import tifffile
import pandas as pd
import pickle
import os, re, json

BASE    = '/Users/neurolab/neuroinformatics/margaret'
EX_DIR  = f'{BASE}/exvivo_merscope_combined'
PKL_DIR = f'{BASE}/merscope_exvivo '
VAROL   = f'{BASE}/jy306_varol'
ND2_DIR = f'{BASE}/registration_video'
OUT_DIR = f'{BASE}/png_exports/merscope_overlay'
os.makedirs(OUT_DIR, exist_ok=True)

ND2_SIZE = 4200  # native nd2 tile size

# 24 distinct gene colours
GENE_PALETTE = [
    (255, 80,  80),  (80,  200, 255), (255, 200, 50),  (180, 80,  255),
    (80,  255, 160), (255, 120, 0),   (0,   160, 255), (255, 80,  200),
    (160, 255, 80),  (255, 220, 120), (80,  80,  255), (255, 160, 80),
    (0,   220, 200), (200, 80,  255), (255, 255, 80),  (80,  255, 255),
    (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 200, 200),
    (200, 255, 200), (200, 200, 255), (255, 180, 50),  (50,  255, 180),
]
N_GENES = len(GENE_PALETTE)

# ── Find region files ──────────────────────────────────────────────────────
ex_files  = {}
pkl_files = {}

for fname in os.listdir(EX_DIR):
    m = re.match(r'(\d+_\d+)_merscope(\d+)\.tif', fname)
    if m:
        ex_files[int(m.group(2))] = (m.group(1), f'{EX_DIR}/{fname}')

for fname in os.listdir(PKL_DIR):
    m = re.match(r'[\d_]+_reg(\d+)_transformed.*\.pkl', fname)
    if m:
        mid = int(m.group(1))
        if mid not in pkl_files or fname > pkl_files[mid][0]:
            pkl_files[mid] = (fname, f'{PKL_DIR}/{fname}')

regions = sorted(set(ex_files) & set(pkl_files))
print(f'Regions: {regions}')

def get_nd2_path(tile):
    """Get nd2 file path from tile name like '2_1'."""
    row, col = tile.split('_')
    # row5 has a 'Row5' subdirectory
    if row == '5':
        return f'{ND2_DIR}/row{row}/Row{row}/{col}.nd2'
    return f'{ND2_DIR}/row{row}/{col}.nd2'

def norm8(img, p_lo=1, p_hi=99.5):
    v = img[img > 0]
    if len(v) < 10:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(v, [p_lo, p_hi])
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

def build_pkl_affine(T_dict):
    """Build the B matrix from PKL transform chain, return R_3_inv and offset_3."""
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

def gxy_to_nd2(gx_arr, gy_arr, W_mosaic, scale, tx, ty, R_3_inv, offset_3, tif_size):
    """global microns → nd2 native pixels via PKL affine + scale-up."""
    # Microns → mosaic → fliplr
    x_mos = scale * gx_arr + tx
    y_mos = scale * gy_arr + ty
    x_flip = W_mosaic - 1 - x_mos
    # Zoom by 0.108 → MERSCOPE source coords
    merc_x = x_flip * 0.108
    merc_y = y_mos * 0.108
    # Inverse affine → ex-vivo TIF coords
    adj_y = merc_y - offset_3[1]
    adj_x = merc_x - offset_3[2]
    tif_y = R_3_inv[1,1] * adj_y + R_3_inv[1,2] * adj_x
    tif_x = R_3_inv[2,1] * adj_y + R_3_inv[2,2] * adj_x
    # Scale up to nd2 native resolution
    nd2_scale = ND2_SIZE / tif_size
    return tif_x * nd2_scale, tif_y * nd2_scale

# ── Process ────────────────────────────────────────────────────────────────
for ms_id in regions:
    tile, ex_path  = ex_files[ms_id]
    _,    pkl_path = pkl_files[ms_id]
    csv_path = f'{VAROL}/region_{ms_id}_resegmentation/detected_transcripts.csv'
    mnf_path = f'{VAROL}/region_{ms_id}_resegmentation/manifest.json'
    m2m_path = f'{VAROL}/region_{ms_id}_resegmentation/micron_to_mosaic_pixel_transform.csv'

    nd2_path = get_nd2_path(tile)
    if not os.path.exists(nd2_path):
        print(f'  Region {ms_id} ({tile}): nd2 not found at {nd2_path}, skip')
        continue
    if not all(os.path.exists(p) for p in [csv_path, mnf_path, m2m_path]):
        print(f'  Region {ms_id}: missing MERSCOPE data, skip')
        continue

    print(f'  Region {ms_id} ({tile})...', end=' ', flush=True)

    # Get TIF size for scale factor
    exvivo_tif = tifffile.imread(ex_path).astype(np.float32)
    tif_size = exvivo_tif.shape[1]  # (z, H, W, c) → H
    del exvivo_tif

    # Load full-res nd2: (Z, C, H, W), ch0=GCaMP, ch1=DAPI
    nd2_data = nd2.imread(nd2_path).astype(np.float32)
    gfp_mip = nd2_data[:, 0, :, :].max(axis=0)   # GCaMP MIP
    dapi_mip = nd2_data[:, 1, :, :].max(axis=0)   # DAPI MIP
    H, W = gfp_mip.shape
    gfp_u8 = norm8(gfp_mip)
    dapi_u8 = norm8(dapi_mip)
    del nd2_data

    # PKL affine
    with open(pkl_path, 'rb') as f:
        pdat = pickle.load(f)
    R_3_inv, offset_3 = build_pkl_affine(pdat['transformations'])

    # Manifest + transform
    with open(mnf_path) as f:
        mnf = json.load(f)
    W_mosaic = mnf['mosaic_width_pixels']
    m2m = np.loadtxt(m2m_path, delimiter=' ')
    scale_m, tx_m, ty_m = m2m[0,0], m2m[0,2], m2m[1,2]

    # Load transcripts
    try:
        df = pd.read_csv(csv_path, usecols=['global_x','global_y','gene'])
    except Exception as e:
        print(f'(err: {e})')
        continue

    gx = df.global_x.values
    gy = df.global_y.values
    tx_arr, ty_arr = gxy_to_nd2(gx, gy, W_mosaic, scale_m, tx_m, ty_m,
                                 R_3_inv, offset_3, tif_size)
    df = df.copy()
    df['nd2_x'] = tx_arr
    df['nd2_y'] = ty_arr

    in_bounds = ((tx_arr >= 0) & (tx_arr < W) &
                 (ty_arr >= 0) & (ty_arr < H))
    df = df[in_bounds]

    # Assign gene colours (top 24 by count, rest grey)
    top_genes = df.gene.value_counts().head(N_GENES).index.tolist()
    gene_colour = {g: GENE_PALETTE[i] for i, g in enumerate(top_genes)}

    # Build BGR canvas: GCaMP green + DAPI blue
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:, :, 0] = dapi_u8  # blue = DAPI
    canvas[:, :, 1] = gfp_u8   # green = GCaMP

    # Rasterise single-pixel dots
    dot_layer = np.zeros((H, W, 3), dtype=np.uint8)
    for g, gdf in df.groupby('gene'):
        col = gene_colour.get(g, (120, 120, 120))
        xi = np.clip(gdf.nd2_x.values.astype(int), 0, W-1)
        yi = np.clip(gdf.nd2_y.values.astype(int), 0, H-1)
        dot_layer[yi, xi] = col

    # Dots on top of GCaMP: keep GCaMP visible, dots clearly distinct
    mask = dot_layer.max(axis=2) > 0
    canvas[mask] = dot_layer[mask]

    # Confocal-only panel: GCaMP green + DAPI blue
    gfp_panel = np.zeros((H, W, 3), dtype=np.uint8)
    gfp_panel[:, :, 0] = dapi_u8  # blue = DAPI
    gfp_panel[:, :, 1] = gfp_u8   # green = GCaMP

    # Separator + label
    sep = np.full((H, 8, 3), 40, dtype=np.uint8)
    lw = W * 2 + 8
    lbar = np.zeros((32, lw, 3), dtype=np.uint8)
    cv2.putText(lbar, f'Region {ms_id} ({tile})   LEFT: Confocal (GCaMP+DAPI)   RIGHT: Confocal + MERFISH gene dots  [nd2 {W}x{H}]',
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    side_by_side = np.hstack([gfp_panel, sep, canvas])
    combined = np.vstack([lbar, side_by_side])

    out_path = f'{OUT_DIR}/region_{ms_id}_{tile}.png'
    cv2.imwrite(out_path, combined)
    print(f'saved {combined.shape[1]}x{combined.shape[0]}  ({in_bounds.sum()}/{len(gx)} dots)')

print(f'\nDone. Images in {OUT_DIR}/')
