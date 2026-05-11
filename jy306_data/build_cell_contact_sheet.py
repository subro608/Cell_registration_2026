#!/usr/bin/env python3
"""Cell candidate contact sheet: 3 columns per cell (In vivo | Ex vivo | MERSCOPE).
Shows top N cells by registration error for picking best showcase cells.
"""
import numpy as np, cv2, os, re, pickle, json
import tifffile
import pandas as pd
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/png_exports/twitter_showcase'
os.makedirs(OUT_DIR, exist_ok=True)

ND2_SIZE = 4200
CROP_ND2 = 120      # crop radius in nd2 pixels (~77µm)
CELL_DISP = 250     # display size per cell panel
GFP_DIM = 0.20
BRIGHT = 1.8
N_SHOW = 10          # show top N cells per tile

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

def make_rainbow(n):
    colors = []
    for i in range(n):
        h = int(180 * i / n)
        s = 200 + int(55 * ((i % 3) / 2))
        v = 200 + int(55 * ((i % 5) / 4))
        bgr = cv2.cvtColor(np.array([[[h, s, v]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(c) for c in bgr))
    return colors

PALETTE = make_rainbow(550)

def norm8(img, lo=1, hi=99.5):
    v = img[img > 0]
    if len(v) < 100:
        return np.zeros_like(img, dtype=np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

def crop_patch(img, cx, cy, r):
    h, w = img.shape[:2]
    x0, y0 = cx - r, cy - r
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, cx + r), min(h, cy + r)
    dx0, dy0 = sx0 - x0, sy0 - y0
    if img.ndim == 3:
        patch = np.zeros((2*r, 2*r, img.shape[2]), dtype=img.dtype)
    else:
        patch = np.zeros((2*r, 2*r), dtype=img.dtype)
    if sx1 > sx0 and sy1 > sy0:
        patch[dy0:dy0+(sy1-sy0), dx0:dx0+(sx1-sx0)] = img[sy0:sy1, sx0:sx1]
    return patch

def build_pkl_affine(T_dict):
    B = np.eye(4)
    for step in T_dict:
        for k, v in step.items():
            if k == 'bhat':
                B = B @ np.c_[v, np.array((0,0,0,1))]
            if k == 'scale':
                B[:,:3] *= v
    R_3 = np.linalg.inv(B[:3,:3]).T
    offset_3 = -B[-1,:-1] @ np.linalg.inv(B[:3,:3])
    return np.linalg.inv(R_3), offset_3

# ── MERSCOPE infrastructure ──
EX_DIR = f'{BASE}/exvivo_merscope_combined'
PKL_DIR = f'{BASE}/merscope_exvivo '
VAROL = f'{BASE}/jy306_varol'

tile_to_ms = {}
for fname in os.listdir(EX_DIR):
    m = re.match(r'(\d+_\d+)_merscope(\d+)\.tif', fname)
    if m:
        tile_short = m.group(1)
        ms_id = int(m.group(2))
        row_tile = f'row{tile_short}'
        tile_to_ms[row_tile] = (ms_id, tile_short, f'{EX_DIR}/{fname}')

pkl_files = {}
for fname in os.listdir(PKL_DIR):
    m = re.match(r'[\d_]+_reg(\d+)_transformed.*\.pkl', fname)
    if m:
        mid = int(m.group(1))
        if mid not in pkl_files or fname > pkl_files[mid][0]:
            pkl_files[mid] = (fname, f'{PKL_DIR}/{fname}')

TILES = [
    ('row2_1', 'ORIENS'),
    ('row3_6', 'PYRAMIDALE'),
]

# ── Load in-vivo ──
print("Loading JY306 in-vivo stack...")
iv_vol = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv = iv_vol.shape[0]

for tile, layer in TILES:
    print(f"\n{'='*60}")
    print(f"  {tile} — {layer}")
    print(f"{'='*60}")

    pkl_tfm_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
    tfm = np.load(pkl_tfm_path)
    M2d = tfm['M2d_jy306_to_nd2']
    ev_nd2 = tfm['ev_nd2']
    iv_nd2_c = tfm['iv_nd2']
    pcd_iv = tfm['pcd_invivo_jy306']

    errs = np.sqrt((iv_nd2_c[:,0]-ev_nd2[:,0])**2 + (iv_nd2_c[:,1]-ev_nd2[:,1])**2) * 0.645

    # Load nd2 slices
    tile_dir = f'{BASE}/png_exports/registration_video/{tile}'
    nd2_slices = []
    for zi in range(12):
        img = cv2.imread(f'{tile_dir}/GFP_z{zi:03d}.png', cv2.IMREAD_UNCHANGED)
        if img is not None:
            nd2_slices.append(img.astype(np.float32))
        else:
            nd2_slices.append(np.zeros((ND2_SIZE, ND2_SIZE), dtype=np.float32))
    nd2_slices = np.array(nd2_slices)

    # Load MERSCOPE data
    gene_nd2_x_all = gene_nd2_y_all = gene_names_all = None
    gene_colour = {}
    if tile in tile_to_ms:
        ms_id, tile_short, ex_path = tile_to_ms[tile]
        csv_path = f'{VAROL}/region_{ms_id}_resegmentation/detected_transcripts.csv'
        mnf_path = f'{VAROL}/region_{ms_id}_resegmentation/manifest.json'
        m2m_path = f'{VAROL}/region_{ms_id}_resegmentation/micron_to_mosaic_pixel_transform.csv'

        if ms_id in pkl_files and all(os.path.exists(p) for p in [csv_path, mnf_path, m2m_path]):
            exvivo_tif = tifffile.imread(ex_path).astype(np.float32)
            tif_size = exvivo_tif.shape[1]
            nd2_scale = ND2_SIZE / tif_size
            del exvivo_tif

            _, pkl_file_path = pkl_files[ms_id]
            with open(pkl_file_path, 'rb') as f:
                pdat = pickle.load(f)
            R_3_inv, offset_3 = build_pkl_affine(pdat['transformations'])

            with open(mnf_path) as f:
                mnf = json.load(f)
            W_mos = mnf['mosaic_width_pixels']
            m2m = np.loadtxt(m2m_path, delimiter=' ')
            scale_m, tx_m, ty_m = m2m[0,0], m2m[0,2], m2m[1,2]

            df = pd.read_csv(csv_path, usecols=['global_x','global_y','gene'])
            is_blank = df.gene.str.startswith('Blank', na=False)
            df = df[~is_blank].reset_index(drop=True)
            gx, gy = df.global_x.values, df.global_y.values
            x_mos = scale_m * gx + tx_m
            y_mos = scale_m * gy + ty_m
            merc_x = (W_mos - 1 - x_mos) * 0.108
            merc_y = y_mos * 0.108
            adj_y = merc_y - offset_3[1]
            adj_x = merc_x - offset_3[2]
            gene_nd2_x = (R_3_inv[2,1]*adj_y + R_3_inv[2,2]*adj_x) * nd2_scale
            gene_nd2_y = (R_3_inv[1,1]*adj_y + R_3_inv[1,2]*adj_x) * nd2_scale
            gene_names = df.gene.values
            valid = (gene_nd2_x >= 0) & (gene_nd2_x < ND2_SIZE) & \
                    (gene_nd2_y >= 0) & (gene_nd2_y < ND2_SIZE)
            gene_nd2_x_all = gene_nd2_x[valid].astype(int)
            gene_nd2_y_all = gene_nd2_y[valid].astype(int)
            gene_names_all = gene_names[valid]
            gc = Counter(gene_names_all)
            gene_colour = {g: PALETTE[i % 550] for i, g in enumerate(g for g, _ in gc.most_common())}
            print(f"  {valid.sum()} transcripts loaded")

    # Sort by error, pick top N
    order = np.argsort(errs)[:N_SHOW]

    # Build contact sheet
    GAP = 6
    CD = CELL_DISP
    HEADER_H = 40
    ROW_LABEL_H = 22
    col_labels = ['In vivo', 'Ex vivo', 'MERSCOPE', 'Registered']
    n_cols = 4
    total_w = CD * n_cols + GAP * (n_cols - 1)
    total_h = HEADER_H + N_SHOW * (ROW_LABEL_H + CD + GAP)

    canvas = np.zeros((total_h, total_w, 3), np.uint8)

    # Header
    col_colors = [(0, 255, 0), (255, 0, 255), (0, 180, 255), (255, 255, 255)]
    for i, (label, col) in enumerate(zip(col_labels, col_colors)):
        ts = 0.6
        (tw, _), _ = cv2.getTextSize(label, FONT, ts, 1)
        px = i * (CD + GAP) + (CD - tw) // 2
        cv2.putText(canvas, label, (px, 28), FONT, ts, col, 1, cv2.LINE_AA)

    cur_y = HEADER_H
    warp_cache = {}

    for rank, lm_idx in enumerate(order):
        cx_nd2 = int(round(ev_nd2[lm_idx, 0]))
        cy_nd2 = int(round(ev_nd2[lm_idx, 1]))
        z_nd2 = int(round(np.clip(ev_nd2[lm_idx, 2], 0, 11)))
        z_iv = int(round(np.clip(pcd_iv[lm_idx, 0], 0, nz_iv - 1)))
        err = errs[lm_idx]

        # Row label
        info = f'#{lm_idx}  err={err:.1f}\u00b5m  z_iv={z_iv}  z_nd2={z_nd2}'
        cv2.putText(canvas, info, (4, cur_y + 15), FONT, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
        cur_y += ROW_LABEL_H

        # In-vivo: warp correct z
        if z_iv not in warp_cache:
            warp_cache[z_iv] = cv2.warpAffine(iv_vol[z_iv], M2d, (ND2_SIZE, ND2_SIZE),
                                               flags=cv2.INTER_LINEAR, borderValue=0)
        iv_w = warp_cache[z_iv]
        iw_u8 = norm8(iv_w)
        iw_bright = np.clip(iw_u8.astype(np.float32) * BRIGHT, 0, 255).astype(np.uint8)
        iw_patch = crop_patch(iw_bright, cx_nd2, cy_nd2, CROP_ND2)
        c1 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
        c1[:,:,1] = iw_patch  # green

        # Ex-vivo: correct z
        ev_slice = nd2_slices[z_nd2]
        ev_u8 = norm8(ev_slice)
        ev_bright = np.clip(ev_u8.astype(np.float32) * BRIGHT, 0, 255).astype(np.uint8)
        ev_patch = crop_patch(ev_bright, cx_nd2, cy_nd2, CROP_ND2)
        c2 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
        c2[:,:,2] = ev_patch  # magenta
        c2[:,:,0] = ev_patch

        # MERSCOPE: dots on dim magenta background
        ev_dim = crop_patch((ev_u8.astype(np.float32) * GFP_DIM).astype(np.uint8),
                            cx_nd2, cy_nd2, CROP_ND2)
        c3 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
        c3[:,:,2] = ev_dim
        c3[:,:,0] = ev_dim

        if gene_nd2_x_all is not None:
            dx = gene_nd2_x_all - (cx_nd2 - CROP_ND2)
            dy = gene_nd2_y_all - (cy_nd2 - CROP_ND2)
            in_patch = (dx >= 0) & (dx < CROP_ND2*2) & (dy >= 0) & (dy < CROP_ND2*2)
            pxi = dx[in_patch].astype(int)
            pyi = dy[in_patch].astype(int)
            pgn = gene_names_all[in_patch]
            n_dots = int(in_patch.sum())
            max_show = 500
            if n_dots > max_show:
                rng = np.random.default_rng(lm_idx)
                sel = rng.choice(n_dots, max_show, replace=False)
                pxi, pyi, pgn = pxi[sel], pyi[sel], pgn[sel]
            for j in range(len(pxi)):
                gc_col = gene_colour.get(pgn[j], (200,200,200))
                cv2.circle(c3, (int(pxi[j]), int(pyi[j])), 3, gc_col, -1, cv2.LINE_AA)

        # Registered: all 3 blended
        IV_A, EV_A, DOT_A = 0.5, 0.5, 0.6
        c4 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.float32)
        c4[:,:,1] += iw_patch.astype(np.float32) * IV_A
        c4[:,:,2] += ev_patch.astype(np.float32) * EV_A
        c4[:,:,0] += ev_patch.astype(np.float32) * EV_A
        # Add dots from c3 (where dots were drawn on dim bg)
        c3_dot_mask = c3.max(axis=2) > (ev_dim.max() * 0.8 + 10)
        c4[c3_dot_mask] += c3[c3_dot_mask].astype(np.float32) * DOT_A
        c4 = np.clip(c4, 0, 255).astype(np.uint8)

        # Resize and place
        panels = [c1, c2, c3, c4]
        for i, p in enumerate(panels):
            p_r = cv2.resize(p, (CD, CD), interpolation=cv2.INTER_LANCZOS4)
            p_r = np.clip(p_r.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
            px = i * (CD + GAP)
            canvas[cur_y:cur_y+CD, px:px+CD] = p_r
            cv2.rectangle(canvas, (px, cur_y), (px+CD-1, cur_y+CD-1), (60, 60, 60), 1)

        # Scale bar on last column (20µm)
        sb_um = 20
        um_per_px = 0.645 * (CROP_ND2 * 2 / CD)
        sb_px = int(sb_um / um_per_px)
        sb_x0 = 3 * (CD + GAP) + CD - sb_px - 8
        sb_y = cur_y + CD - 12
        cv2.line(canvas, (sb_x0, sb_y), (sb_x0 + sb_px, sb_y), WHITE, 2)
        cv2.putText(canvas, f'{sb_um}\u00b5m', (sb_x0, sb_y - 5), FONT, 0.3, WHITE, 1, cv2.LINE_AA)

        cur_y += CD + GAP
        print(f"  #{rank+1} LM#{lm_idx}: err={err:.2f}µm, dots={n_dots if gene_nd2_x_all is not None else 0}")

    # Crop to actual content
    canvas = canvas[:cur_y]

    out_path = f'{OUT_DIR}/cell_candidates_{layer.lower()}_{tile}.png'
    cv2.imwrite(out_path, canvas)
    print(f"  Saved: {out_path} ({canvas.shape[1]}x{canvas.shape[0]})")

print(f'\nDone. Outputs in {OUT_DIR}/')
