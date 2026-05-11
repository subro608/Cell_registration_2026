#!/usr/bin/env python3
"""Generate cell-centered patches: DAPI+GCaMP | DAPI+GCaMP+GeneDots overlay.
One contact sheet per region, saved to png_exports/merscope_overlay/.
"""
import numpy as np
import cv2
import nd2
import tifffile
import pandas as pd
import pickle, json, os, re

BASE    = '/Users/neurolab/neuroinformatics/margaret'
EX_DIR  = f'{BASE}/exvivo_merscope_combined'
PKL_DIR = f'{BASE}/merscope_exvivo '
VAROL   = f'{BASE}/jy306_varol'
ND2_DIR = f'{BASE}/registration_video'
OUT_DIR = f'{BASE}/png_exports/merscope_overlay'
os.makedirs(OUT_DIR, exist_ok=True)

PATCH_SZ = 80
CROP_R = 60
N_COLS = 10
MIN_TRANSCRIPTS = 50

GENE_PALETTE = [
    (255, 80,  80),  (80,  200, 255), (255, 200, 50),  (180, 80,  255),
    (80,  255, 160), (255, 120, 0),   (0,   160, 255), (255, 80,  200),
    (160, 255, 80),  (255, 220, 120), (80,  80,  255), (255, 160, 80),
]

def norm8(img):
    v = img[img > 0]
    if len(v) < 10:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(v, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

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
    return np.linalg.inv(R_3), offset_3

def get_nd2_path(tile):
    row, col = tile.split('_')
    if row == '5':
        return f'{ND2_DIR}/row{row}/Row{row}/{col}.nd2'
    return f'{ND2_DIR}/row{row}/{col}.nd2'

def crop_patch(img, cy, cx, r):
    h, w = img.shape[:2]
    y0, y1 = int(cy)-r, int(cy)+r
    x0, x1 = int(cx)-r, int(cx)+r
    sy0, sy1 = max(0,y0), min(h,y1)
    sx0, sx1 = max(0,x0), min(w,x1)
    out = np.zeros((2*r, 2*r), dtype=img.dtype)
    dy0, dy1 = sy0-y0, sy0-y0+(sy1-sy0)
    dx0, dx1 = sx0-x0, sx0-x0+(sx1-sx0)
    if sy1>sy0 and sx1>sx0:
        out[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return out

# Find region files
ex_files, pkl_files = {}, {}
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

for ms_id in regions:
    tile, ex_path = ex_files[ms_id]
    _, pkl_path = pkl_files[ms_id]
    csv_path = f'{VAROL}/region_{ms_id}_resegmentation/detected_transcripts.csv'
    mnf_path = f'{VAROL}/region_{ms_id}_resegmentation/manifest.json'
    m2m_path = f'{VAROL}/region_{ms_id}_resegmentation/micron_to_mosaic_pixel_transform.csv'
    nd2_path = get_nd2_path(tile)

    if not os.path.exists(nd2_path):
        print(f'  Region {ms_id} ({tile}): nd2 not found, skip')
        continue
    if not all(os.path.exists(p) for p in [csv_path, mnf_path, m2m_path]):
        print(f'  Region {ms_id}: missing MERSCOPE data, skip')
        continue

    print(f'  Region {ms_id} ({tile})...', end=' ', flush=True)

    # Load nd2
    nd2_data = nd2.imread(nd2_path).astype(np.float32)
    gfp_mip = nd2_data[:, 0].max(axis=0)
    dapi_mip = nd2_data[:, 1].max(axis=0)
    H, W = gfp_mip.shape
    gfp_u8 = norm8(gfp_mip)
    dapi_u8 = norm8(dapi_mip)
    del nd2_data

    # Scale factor
    tif_data = tifffile.imread(ex_path)
    tif_size = tif_data.shape[1]
    nd2_scale = W / tif_size
    del tif_data

    # PKL affine
    with open(pkl_path, 'rb') as f:
        pdat = pickle.load(f)
    R_3_inv, offset_3 = build_pkl_affine(pdat['transformations'])

    # MERSCOPE transform
    with open(mnf_path) as f:
        mnf = json.load(f)
    W_mos = mnf['mosaic_width_pixels']
    m2m = np.loadtxt(m2m_path, delimiter=' ')
    scale_m, tx_m, ty_m = m2m[0,0], m2m[0,2], m2m[1,2]

    # Transcripts
    try:
        df = pd.read_csv(csv_path, usecols=['global_x','global_y','gene','cell_id'])
    except Exception as e:
        print(f'err: {e}')
        continue

    df = df[df.cell_id != -1].copy()
    gx, gy = df.global_x.values, df.global_y.values
    x_mos = scale_m * gx + tx_m
    y_mos = scale_m * gy + ty_m
    merc_x = (W_mos - 1 - x_mos) * 0.108
    merc_y = y_mos * 0.108
    adj_y = merc_y - offset_3[1]
    adj_x = merc_x - offset_3[2]
    df['nd2_x'] = (R_3_inv[2,1]*adj_y + R_3_inv[2,2]*adj_x) * nd2_scale
    df['nd2_y'] = (R_3_inv[1,1]*adj_y + R_3_inv[1,2]*adj_x) * nd2_scale

    in_bounds = (df.nd2_x >= 0) & (df.nd2_x < W) & (df.nd2_y >= 0) & (df.nd2_y < H)
    df = df[in_bounds]

    # Cell centroids
    cell_stats = df.groupby('cell_id').agg(
        cx=('nd2_x','mean'), cy=('nd2_y','mean'), n=('gene','size')
    ).reset_index()
    cell_stats = cell_stats[cell_stats.n >= MIN_TRANSCRIPTS].sort_values('n', ascending=False)

    # Gene colours
    top_genes = df.gene.value_counts().head(len(GENE_PALETTE)).index.tolist()
    gene_colour = {g: GENE_PALETTE[i] for i, g in enumerate(top_genes)}

    # Build cell→transcript lookup (vectorized)
    cell_groups = {cid: gdf for cid, gdf in df.groupby('cell_id')}

    cards = []
    for _, cell in cell_stats.iterrows():
        cid = cell.cell_id
        cx, cy = int(round(cell.cx)), int(round(cell.cy))

        dapi_patch = crop_patch(dapi_u8, cy, cx, CROP_R)
        gfp_patch = crop_patch(gfp_u8, cy, cx, CROP_R)

        confocal = np.zeros((CROP_R*2, CROP_R*2, 3), dtype=np.uint8)
        confocal[:,:,0] = dapi_patch
        confocal[:,:,1] = gfp_patch

        overlay = confocal.copy()
        if cid in cell_groups:
            cdf = cell_groups[cid]
            xi = (cdf.nd2_x.values - (cx - CROP_R)).astype(int)
            yi = (cdf.nd2_y.values - (cy - CROP_R)).astype(int)
            valid = (xi >= 0) & (xi < CROP_R*2) & (yi >= 0) & (yi < CROP_R*2)
            for g in cdf.gene.values[valid]:
                pass  # handled below
            for j in np.where(valid)[0]:
                gc = gene_colour.get(cdf.gene.values[j], (200,200,200))
                cv2.circle(overlay, (int(xi[j]), int(yi[j])), 1, gc, -1)

        p1 = cv2.resize(confocal, (PATCH_SZ, PATCH_SZ), interpolation=cv2.INTER_AREA)
        p2 = cv2.resize(overlay, (PATCH_SZ, PATCH_SZ), interpolation=cv2.INTER_AREA)

        label = np.zeros((14, PATCH_SZ*2+2, 3), dtype=np.uint8)
        cv2.putText(label, f'cell {int(cid)} n={int(cell.n)}', (2, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180,180,180), 1)

        sep = np.full((PATCH_SZ, 2, 3), 40, dtype=np.uint8)
        card = np.vstack([label, np.hstack([p1, sep, p2])])
        cards.append(card)

    if not cards:
        print('no cells')
        continue

    card_h, card_w = cards[0].shape[:2]
    sep_v = np.full((card_h, 3, 3), 20, dtype=np.uint8)

    grid_rows = []
    for start in range(0, len(cards), N_COLS):
        chunk = cards[start:start+N_COLS]
        while len(chunk) < N_COLS:
            chunk.append(np.zeros_like(cards[0]))
        row_img = np.hstack([c for pair in zip(chunk, [sep_v]*N_COLS)
                               for c in pair][:-1])
        grid_rows.append(row_img)

    hdr_w = grid_rows[0].shape[1]
    hdr = np.zeros((24, hdr_w, 3), dtype=np.uint8)
    cv2.putText(hdr, f'Region {ms_id} ({tile})  |  LEFT: DAPI+GCaMP  |  RIGHT: +gene dots  |  {len(cards)} cells',
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,220,255), 1)

    sheet = np.vstack([hdr] + grid_rows)
    out_path = f'{OUT_DIR}/cell_patches_region_{ms_id}_{tile}.png'
    cv2.imwrite(out_path, sheet)
    print(f'{len(cards)} cells, {sheet.shape[1]}x{sheet.shape[0]}')

print(f'\nDone. Patches in {OUT_DIR}/')
