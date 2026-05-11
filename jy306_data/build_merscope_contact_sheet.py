#!/usr/bin/env python3
"""Alignment QC contact sheet: ex-vivo GCaMP vs MERFISH gene dots.
Shows whether the ex-vivo GCaMP tiles (warped to MERSCOPE space) align
with MERFISH transcript positions. Centered on pcd_fixed control points.

Panels: Ex-vivo GCaMP | Cell masks (filled convex hull) + gene dots | Overlay
Only segmented cells (cell_id != -1) are shown.
"""
import numpy as np
import cv2
import tifffile
import pandas as pd
import pickle
import os, re, json

BASE    = '/Users/neurolab/neuroinformatics/margaret'
EX_DIR  = f'{BASE}/exvivo_merscope_combined'
PKL_DIR = f'{BASE}/merscope_exvivo '   # note trailing space in dir name
VAROL   = f'{BASE}/jy306_varol'
OUT     = f'{BASE}/png_exports/alignment_exvivo_merfish_contact_sheet.png'

N_COLS  = 6
R       = 80    # patch half-size px (in TIF space)
SZ      = 160   # display size

GENE_PALETTE = [
    (0,   255, 80),  (255, 80,  0),  (80,  80,  255), (255, 255, 0),
    (255, 0,   200), (0,   220, 255), (180, 255, 80),  (255, 120, 200),
    (255, 180, 0),   (0,   180, 255), (200, 0,   255), (0,   255, 200),
]
N_GENES = len(GENE_PALETTE)

# Consistent per-cell colours (pastel, semi-transparent look)
CELL_PALETTE = [
    (180, 80,  80),  (80,  180, 80),  (80,  80,  180), (180, 160, 60),
    (160, 60,  180), (60,  180, 160), (180, 100, 50),  (50,  100, 180),
    (140, 180, 60),  (180, 60,  140), (60,  140, 180), (140, 60,  180),
    (100, 180, 100), (180, 100, 100), (100, 100, 180), (180, 140, 80),
]

def cell_color(cell_id):
    return CELL_PALETTE[int(cell_id) % len(CELL_PALETTE)]

# ── Find region files ──────────────────────────────────────────────────────
ex_files  = {}   # ms_id -> (tile, tif path)
pkl_files = {}   # ms_id -> (fname, pkl path)

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
print(f'Regions with exvivo TIF + PKL: {regions}')

# ── Helpers ────────────────────────────────────────────────────────────────
def norm8(img, p_lo=1, p_hi=99.5):
    v = img[img > 0]
    if len(v) < 10:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(v, [p_lo, p_hi])
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

def crop2d(img, cy, cx, r):
    h, w = img.shape[:2]
    y0, y1 = int(cy)-r, int(cy)+r
    x0, x1 = int(cx)-r, int(cx)+r
    sy0 = max(0,y0); sy1 = min(h,y1)
    sx0 = max(0,x0); sx1 = min(w,x1)
    out = np.zeros((2*r, 2*r), dtype=img.dtype)
    dy0 = sy0-y0; dy1 = dy0+(sy1-sy0)
    dx0 = sx0-x0; dx1 = dx0+(sx1-sx0)
    if sy1>sy0 and sx1>sx0:
        out[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return out

def gxy_to_tif(gx_arr, gy_arr, W_mosaic, scale, tx, ty):
    """global µm → TIF pixel (fliplr + zoom 0.108)."""
    x_m = scale * gx_arr + tx
    y_m = scale * gy_arr + ty
    return (W_mosaic - 1 - x_m) * 0.108, y_m * 0.108

def draw_cell_masks_and_dots(patch_h, patch_w, cells_in_patch, gene_colours):
    """
    cells_in_patch: dict {cell_id -> DataFrame with local tif_x, tif_y, gene columns}
    Returns BGR image with filled cell hulls + gene dots.
    """
    img = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)

    # Draw filled convex hulls (semi-transparent via addWeighted)
    hull_layer = np.zeros_like(img)
    for cid, cdf in cells_in_patch.items():
        pts = cdf[['tif_x', 'tif_y']].values.astype(np.float32)
        if len(pts) < 3:
            continue
        hull = cv2.convexHull(pts.reshape(-1,1,2).astype(np.int32))
        color = cell_color(cid)
        cv2.fillConvexPoly(hull_layer, hull, color)

    cv2.addWeighted(hull_layer, 0.4, img, 0.6, 0, img)

    # Draw cell outlines
    for cid, cdf in cells_in_patch.items():
        pts = cdf[['tif_x', 'tif_y']].values.astype(np.float32)
        if len(pts) < 3:
            continue
        hull = cv2.convexHull(pts.reshape(-1,1,2).astype(np.int32))
        color = cell_color(cid)
        cv2.polylines(img, [hull], True, color, 1)

    # Draw gene dots
    for cid, cdf in cells_in_patch.items():
        for _, row in cdf.iterrows():
            xi = int(row.tif_x)
            yi = int(row.tif_y)
            if 0 <= xi < patch_w and 0 <= yi < patch_h:
                gc = gene_colours.get(row.gene, (200, 200, 200))
                cv2.circle(img, (xi, yi), 2, gc, -1)

    return img

# ── Process each region ────────────────────────────────────────────────────
all_cards = {}

for ms_id in regions:
    tile, ex_path  = ex_files[ms_id]
    _,    pkl_path = pkl_files[ms_id]
    csv_path = f'{VAROL}/region_{ms_id}_resegmentation/detected_transcripts.csv'
    mnf_path = f'{VAROL}/region_{ms_id}_resegmentation/manifest.json'
    m2m_path = f'{VAROL}/region_{ms_id}_resegmentation/micron_to_mosaic_pixel_transform.csv'

    if not all(os.path.exists(p) for p in [csv_path, mnf_path, m2m_path]):
        print(f'  Region {ms_id}: missing transcript data, skip')
        continue

    print(f'  Region {ms_id} ({tile})...', end=' ', flush=True)

    # Ex-vivo TIF: (Z, H, W, C), ch0=GFP ch1=DAPI
    exvivo = tifffile.imread(ex_path).astype(np.float32)
    gfp_mip = exvivo[:, :, :, 0].max(axis=0)   # (H, W)
    TIF_H, TIF_W = gfp_mip.shape
    gfp_u8 = norm8(gfp_mip)

    # PKL: get pcd_fixed (control point positions in TIF space)
    with open(pkl_path, 'rb') as f:
        pdat = pickle.load(f)
    pcd_fixed = pdat['pcd_fixed']   # (N, 3): z, y, x in TIF space

    # Manifest + micron2mosaic
    with open(mnf_path) as f:
        mnf = json.load(f)
    W_mosaic = mnf['mosaic_width_pixels']
    m2m = np.loadtxt(m2m_path, delimiter=' ')
    scale_m, tx_m, ty_m = m2m[0,0], m2m[0,2], m2m[1,2]

    # Transcripts — only segmented cells (cell_id != -1)
    try:
        df = pd.read_csv(csv_path, usecols=['global_x','global_y','gene','cell_id'])
    except Exception as e:
        print(f'(csv err: {e})')
        continue

    df = df[df.cell_id != -1].copy()

    gx = df.global_x.values
    gy = df.global_y.values
    tx_arr, ty_arr = gxy_to_tif(gx, gy, W_mosaic, scale_m, tx_m, ty_m)
    df['tif_x'] = tx_arr
    df['tif_y'] = ty_arr

    in_bounds = ((tx_arr >= 0) & (tx_arr < TIF_W) &
                 (ty_arr >= 0) & (ty_arr < TIF_H))
    df = df[in_bounds]

    # Top-N genes → colours
    gene_colours = {}
    top_genes = df.gene.value_counts().head(N_GENES).index.tolist()
    for gi, gname in enumerate(top_genes):
        gene_colours[gname] = GENE_PALETTE[gi % N_GENES]

    # ── Build one card per control point ──────────────────────────────
    cards = []
    for pt in pcd_fixed:
        cy, cx = float(pt[1]), float(pt[2])
        if not (R <= cy < TIF_H-R and R <= cx < TIF_W-R):
            continue

        icy, icx = int(round(cy)), int(round(cx))

        # Find transcripts in this patch (with margin for hull cells)
        margin = R + 20
        patch_df = df[
            (df.tif_x >= icx - margin) & (df.tif_x < icx + margin) &
            (df.tif_y >= icy - margin) & (df.tif_y < icy + margin)
        ].copy()

        # Localise coordinates to patch origin
        patch_df = patch_df.copy()
        patch_df['tif_x'] = patch_df['tif_x'] - (icx - R)
        patch_df['tif_y'] = patch_df['tif_y'] - (icy - R)

        # Group by cell
        cells_in_patch = {cid: cdf for cid, cdf in patch_df.groupby('cell_id')}

        # Panel 1: ex-vivo GCaMP
        gfp_patch = crop2d(gfp_u8, icy, icx, R)
        gfp_rgb   = cv2.cvtColor(gfp_patch, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(gfp_rgb, (R,R), (0,255,0), cv2.MARKER_CROSS, 16, 1)

        # Panel 2: cell masks + gene dots
        cell_img = draw_cell_masks_and_dots(2*R, 2*R, cells_in_patch, gene_colours)
        cv2.drawMarker(cell_img, (R,R), (0,255,255), cv2.MARKER_CROSS, 16, 1)

        # Panel 3: overlay — GCaMP in green, cell masks + dots on top
        overlay = np.zeros((2*R, 2*R, 3), dtype=np.uint8)
        overlay[:,:,1] = gfp_patch  # green channel = GCaMP

        # Blend cell hull layer at 30% over GCaMP
        hull_layer2 = np.zeros((2*R, 2*R, 3), dtype=np.uint8)
        for cid, cdf in cells_in_patch.items():
            pts = cdf[['tif_x', 'tif_y']].values.astype(np.float32)
            if len(pts) < 3:
                continue
            hull = cv2.convexHull(pts.reshape(-1,1,2).astype(np.int32))
            cv2.fillConvexPoly(hull_layer2, hull, cell_color(cid))
        cv2.addWeighted(hull_layer2, 0.3, overlay, 0.7, 0, overlay)

        # Gene dots on top
        for cid, cdf in cells_in_patch.items():
            for _, row in cdf.iterrows():
                xi = int(row.tif_x)
                yi = int(row.tif_y)
                if 0 <= xi < 2*R and 0 <= yi < 2*R:
                    gc = gene_colours.get(row.gene, (200,200,200))
                    cv2.circle(overlay, (xi, yi), 2, gc, -1)

        cv2.drawMarker(overlay, (R,R), (255,255,255), cv2.MARKER_CROSS, 16, 1)

        p1 = cv2.resize(gfp_rgb,   (SZ,SZ), interpolation=cv2.INTER_AREA)
        p2 = cv2.resize(cell_img,  (SZ,SZ), interpolation=cv2.INTER_AREA)
        p3 = cv2.resize(overlay,   (SZ,SZ), interpolation=cv2.INTER_AREA)

        n_cells = len(cells_in_patch)
        label = np.zeros((16, SZ*3+4, 3), dtype=np.uint8)
        cv2.putText(label, f'R{ms_id} ({tile})  y={icy} x={icx}  cells={n_cells}', (2,11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200,200,200), 1)

        card = np.vstack([label,
                          np.hstack([p1,
                                     np.full((SZ,2,3),40,np.uint8),
                                     p2,
                                     np.full((SZ,2,3),40,np.uint8),
                                     p3])])
        cards.append(card)

    all_cards[ms_id] = cards
    print(f'{len(cards)} cards')

if not all_cards:
    print('No cards generated.')
    exit()

# ── Assemble ───────────────────────────────────────────────────────────────
sample = next(iter(all_cards.values()))[0]
sep_v  = np.full((sample.shape[0], 4, 3), 20, np.uint8)

full_rows = []
for ms_id in regions:
    cards = all_cards.get(ms_id, [])
    if not cards:
        continue
    tile = ex_files[ms_id][0]
    card_h, card_w = cards[0].shape[:2]

    hdr_w = card_w*N_COLS + 4*(N_COLS-1)
    reg_hdr = np.full((20, hdr_w, 3), (30,30,30), dtype=np.uint8)
    cv2.putText(reg_hdr,
                f'Region {ms_id} ({tile})  |  Ex-vivo GCaMP  |  Cell masks + gene dots  |  Overlay',
                (6,14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180,220,255), 1)

    grid = []
    for start in range(0, len(cards), N_COLS):
        chunk = cards[start:start+N_COLS]
        while len(chunk) < N_COLS:
            chunk.append(np.zeros_like(cards[0]))
        row_img = np.hstack([c for pair in zip(chunk, [sep_v]*N_COLS)
                               for c in pair][:-1])
        grid.append(row_img)

    block = np.vstack([reg_hdr] + grid)
    if full_rows:
        full_rows.append(np.full((6, block.shape[1], 3), 10, np.uint8))
    full_rows.append(block)

sheet = np.vstack(full_rows)
cv2.imwrite(OUT, sheet)
print(f'\nSaved: {OUT}  ({sheet.shape[1]}x{sheet.shape[0]})')
