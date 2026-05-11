#!/usr/bin/env python3
"""Twitter final layout per template:
  LEFT:   GIF frames — tissue cycling in-vivo → ex-vivo → MERSCOPE (saved as frames)
  MIDDLE: Static tissue overlay (all 3 registered)
  RIGHT:  3 cells × 3 columns (In vivo | Ex vivo | MERSCOPE)

Two outputs per tile: static PNG composite + GIF frames folder.
"""
import numpy as np, cv2, os, re, pickle, json
import tifffile
import pandas as pd
from collections import Counter
from scipy.ndimage import gaussian_filter

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/png_exports/twitter_showcase'
os.makedirs(OUT_DIR, exist_ok=True)

ND2_SIZE = 4200
GFP_DIM = 0.20
MASK_PAD = 80
BRIGHT = 1.2
EV_BRIGHT = 0.6
CROP_ND2 = 120     # cell crop radius in nd2 px

# Layout sizes
TISSUE_SIZE = 800   # tissue panel display size (left + middle)
CELL_DISP = 220     # cell patch display size
CELL_GAP = 6

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

# GIF params
GIF_FPS = 12
HOLD_FRAMES = 24     # hold each modality
FADE_FRAMES = 12     # crossfade between modalities

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

def mask_crop_resize(img, mask_bool, bbox, size):
    """Apply mask, crop to bbox, resize."""
    y0, y1, x0, x1 = bbox
    out = img.copy()
    out[~mask_bool] = 0
    cropped = out[y0:y1, x0:x1]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)

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

# ── Tiles + selected cells ──
# Per-tile brightness: (iv_tissue, ev_tissue, iv_cell, ev_cell)
TILE_BRIGHT = {
    'row2_1': (0.45, 0.55, 0.45, 0.35),
    'row3_6': (0.7, 0.8, 1.0, 0.9),
}
TILES = [
    ('row2_1', 'ORIENS',     [7, 5, 17]),
    ('row3_6', 'PYRAMIDALE', [0, 36, 8]),
]

# ── Load masks + in-vivo ──
print("Loading masks + in-vivo...")
masks_npz = np.load(f'{BASE}/registration_video/via_masks_v4.npz')
iv_vol = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv = iv_vol.shape[0]

for tile, layer, cell_lms in TILES:
    print(f"\n{'='*60}")
    print(f"  {tile} — {layer} — cells {cell_lms}")
    print(f"{'='*60}")

    # Load transforms
    pkl_tfm_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
    tfm = np.load(pkl_tfm_path)
    M2d = tfm['M2d_jy306_to_nd2']
    ev_nd2 = tfm['ev_nd2']
    iv_nd2_c = tfm['iv_nd2']
    pcd_iv = tfm['pcd_invivo_jy306']
    errs = np.sqrt((iv_nd2_c[:,0]-ev_nd2[:,0])**2 + (iv_nd2_c[:,1]-ev_nd2[:,1])**2) * 0.645

    med_z_iv = int(round(np.median(pcd_iv[:, 0])))
    med_z_iv = max(0, min(med_z_iv, nz_iv - 1))

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
    nd2_mip = nd2_slices.max(axis=0)
    ev_u8 = norm8(nd2_mip)

    # In-vivo single z-slice (median z of landmarks) with edge feathering
    iv_slice = iv_vol[med_z_iv].copy()
    # Cosine taper over 150px to soften rectangular boundary
    FEATHER = 150
    h_iv, w_iv = iv_slice.shape
    for d in range(FEATHER):
        a = 0.5 - 0.5 * np.cos(np.pi * d / FEATHER)
        iv_slice[d, :] *= a
        iv_slice[h_iv-1-d, :] *= a
        iv_slice[:, d] *= a
        iv_slice[:, w_iv-1-d] *= a
    iv_mip = iv_slice
    iv_warped = cv2.warpAffine(iv_mip, M2d, (ND2_SIZE, ND2_SIZE),
                                flags=cv2.INTER_LINEAR, borderValue=0)
    iw_u8 = norm8(iv_warped)
    tb = TILE_BRIGHT[tile]
    iv_tissue_b, ev_tissue_b, iv_cell_b, ev_cell_b = tb
    iw_bright = np.clip(iw_u8.astype(np.float32) * iv_tissue_b, 0, 255).astype(np.uint8)
    ev_bright = np.clip(ev_u8.astype(np.float32) * ev_tissue_b, 0, 255).astype(np.uint8)

    # Build full-res tissue panels
    # In-vivo green
    tissue_iv = np.zeros((ND2_SIZE, ND2_SIZE, 3), np.uint8)
    tissue_iv[:,:,1] = iw_bright
    # Ex-vivo magenta
    tissue_ev = np.zeros((ND2_SIZE, ND2_SIZE, 3), np.uint8)
    tissue_ev[:,:,2] = ev_bright
    tissue_ev[:,:,0] = ev_bright
    # MERSCOPE dots
    tissue_ms = np.zeros((ND2_SIZE, ND2_SIZE, 3), np.uint8)
    tissue_ms[:,:,2] = (ev_u8.astype(np.float32) * GFP_DIM).astype(np.uint8)
    tissue_ms[:,:,0] = (ev_u8.astype(np.float32) * GFP_DIM).astype(np.uint8)

    dot_layer = np.zeros((ND2_SIZE, ND2_SIZE, 3), np.uint8)
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
            print(f"  Filtered {is_blank.sum()} Blank probes")
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

            for j in range(len(gene_nd2_x_all)):
                col = gene_colour.get(gene_names_all[j], (200,200,200))
                cv2.circle(dot_layer, (int(gene_nd2_x_all[j]), int(gene_nd2_y_all[j])), 1, col, -1)
            print(f"  {valid.sum()} transcripts")

    dot_mask = dot_layer.max(axis=2) > 0
    tissue_ms[dot_mask] = dot_layer[dot_mask]

    # Mask + bbox
    mask = masks_npz[tile]
    mask_bool = mask > 0
    ys, xs = np.where(mask_bool)
    y0 = max(0, ys.min() - MASK_PAD)
    y1 = min(ND2_SIZE, ys.max() + MASK_PAD)
    x0 = max(0, xs.min() - MASK_PAD)
    x1 = min(ND2_SIZE, xs.max() + MASK_PAD)
    h_box, w_box = y1 - y0, x1 - x0
    side = max(h_box, w_box)
    cy_b, cx_b = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, cy_b - side // 2)
    y1 = min(ND2_SIZE, y0 + side)
    x0 = max(0, cx_b - side // 2)
    x1 = min(ND2_SIZE, x0 + side)
    bbox = (y0, y1, x0, x1)
    crop_w = x1 - x0

    TS = TISSUE_SIZE
    tissue_iv_d = mask_crop_resize(tissue_iv, mask_bool, bbox, TS)
    tissue_ev_d = mask_crop_resize(tissue_ev, mask_bool, bbox, TS)
    tissue_ms_d = mask_crop_resize(tissue_ms, mask_bool, bbox, TS)

    # Tissue overlay (all 3 blended)
    IV_A, EV_A, DOT_A = 0.5, 0.5, 0.6
    tissue_ov = np.zeros((ND2_SIZE, ND2_SIZE, 3), np.float32)
    tissue_ov[:,:,1] += iw_bright.astype(np.float32) * IV_A
    tissue_ov[:,:,2] += ev_bright.astype(np.float32) * EV_A
    tissue_ov[:,:,0] += ev_bright.astype(np.float32) * EV_A
    tissue_ov[dot_mask] += dot_layer[dot_mask].astype(np.float32) * DOT_A
    tissue_ov = np.clip(tissue_ov, 0, 255).astype(np.uint8)
    tissue_ov_d = mask_crop_resize(tissue_ov, mask_bool, bbox, TS)

    # Brightness boost on display panels
    tissue_iv_d = np.clip(tissue_iv_d.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
    tissue_ev_d = np.clip(tissue_ev_d.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
    tissue_ms_d = np.clip(tissue_ms_d.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
    tissue_ov_d = np.clip(tissue_ov_d.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)

    # ══════════════════════════════════════════════════════════
    # RIGHT PANEL: 3 cells × 3 columns
    # ══════════════════════════════════════════════════════════
    CD = CELL_DISP
    CG = CELL_GAP
    n_cells = len(cell_lms)
    n_cell_cols = 4
    cell_col_labels = ['In vivo', 'Ex vivo', 'MERSCOPE mRNA', 'Overlay']
    cell_col_colors = [(0, 255, 0), (255, 0, 255), (0, 180, 255), WHITE]

    cell_grid_w = CD * n_cell_cols + CG * (n_cell_cols - 1)
    CELL_HDR = 30
    CELL_ROW_LABEL = 18
    cell_grid_h = CELL_HDR + n_cells * (CELL_ROW_LABEL + CD + CG)

    cell_grid = np.zeros((cell_grid_h, cell_grid_w, 3), np.uint8)

    # Cell column headers
    for i, (label, col) in enumerate(zip(cell_col_labels, cell_col_colors)):
        ts = 0.5
        (tw, _), _ = cv2.getTextSize(label, FONT, ts, 1)
        px = i * (CD + CG) + (CD - tw) // 2
        cv2.putText(cell_grid, label, (px, 22), FONT, ts, col, 1, cv2.LINE_AA)

    warp_cache = {}
    cur_y = CELL_HDR

    for ci, lm_idx in enumerate(cell_lms):
        cx_nd2 = int(round(ev_nd2[lm_idx, 0]))
        cy_nd2 = int(round(ev_nd2[lm_idx, 1]))
        z_nd2 = int(round(np.clip(ev_nd2[lm_idx, 2], 0, 11)))
        z_iv = int(round(np.clip(pcd_iv[lm_idx, 0], 0, nz_iv - 1)))
        err = errs[lm_idx]

        cur_y += CELL_ROW_LABEL

        # In-vivo warped
        if z_iv not in warp_cache:
            warp_cache[z_iv] = cv2.warpAffine(iv_vol[z_iv], M2d, (ND2_SIZE, ND2_SIZE),
                                               flags=cv2.INTER_LINEAR, borderValue=0)
        iv_w = warp_cache[z_iv]
        iw_cell = np.clip(norm8(iv_w).astype(np.float32) * iv_cell_b, 0, 255).astype(np.uint8)
        iw_patch = crop_patch(iw_cell, cx_nd2, cy_nd2, CROP_ND2)
        c1 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
        c1[:,:,1] = iw_patch

        # Ex-vivo
        ev_sl = nd2_slices[z_nd2]
        ev_cell = np.clip(norm8(ev_sl, hi=99.9).astype(np.float32) * ev_cell_b, 0, 255).astype(np.uint8)
        ev_patch = crop_patch(ev_cell, cx_nd2, cy_nd2, CROP_ND2)
        c2 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
        c2[:,:,2] = ev_patch
        c2[:,:,0] = ev_patch

        # MERSCOPE (dots only, no background)
        c3 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)

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
                cv2.circle(c3, (int(pxi[j]), int(pyi[j])), 1, gc_col, -1, cv2.LINE_AA)
            print(f"  Cell #{lm_idx}: err={err:.2f}µm, {n_dots} dots")

        # Overlay (all 3 blended)
        c4 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.float32)
        c4 += c1.astype(np.float32) * 0.5
        c4 += c2.astype(np.float32) * 0.5
        c4 += c3.astype(np.float32) * 0.6
        c4 = np.clip(c4, 0, 255).astype(np.uint8)

        # Place cells
        for i, c in enumerate([c1, c2, c3, c4]):
            c_r = cv2.resize(c, (CD, CD), interpolation=cv2.INTER_LANCZOS4)
            c_r = np.clip(c_r.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
            px = i * (CD + CG)
            cell_grid[cur_y:cur_y+CD, px:px+CD] = c_r

        cur_y += CD + CG

    cell_grid = cell_grid[:cur_y]

    # ══════════════════════════════════════════════════════════
    # BUILD 2-PANEL LAYOUT: [GIF] | [Cells]
    # ══════════════════════════════════════════════════════════
    MAIN_GAP = 16

    # Scale cell grid to match tissue height
    cell_scale = TS / cell_grid.shape[0]
    cell_grid_scaled = cv2.resize(cell_grid,
                                   (int(cell_grid.shape[1] * cell_scale), TS),
                                   interpolation=cv2.INTER_LANCZOS4)
    cgw = cell_grid_scaled.shape[1]

    LABEL_H = 45
    FOOTER_H = 30
    # Total: GIF panel + gap + cell grid
    total_w = TS + MAIN_GAP + cgw
    total_h = LABEL_H + TS + FOOTER_H

    # ── Build the static parts (cells only) ──
    static_base = np.zeros((total_h, total_w, 3), np.uint8)

    # Right: cell grid
    cell_x = TS + MAIN_GAP
    static_base[LABEL_H:LABEL_H+TS, cell_x:cell_x+cgw] = cell_grid_scaled

    # Footer
    footer = f'{layer}  |  {tile.upper().replace("_"," ")}'
    ts_f = 0.5
    (tw, _), _ = cv2.getTextSize(footer, FONT, ts_f, 1)
    cv2.putText(static_base, footer, ((total_w - tw) // 2, LABEL_H + TS + 22),
                FONT, ts_f, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Save static version (with overlay in left panel) ──
    static = static_base.copy()
    static[LABEL_H:LABEL_H+TS, 0:TS] = tissue_ov_d
    static_label = 'REGISTERED OVERLAY'
    (tw, _), _ = cv2.getTextSize(static_label, FONT, 0.55, 1)
    cv2.putText(static, static_label, ((TS - tw) // 2, LABEL_H - 12),
                FONT, 0.55, WHITE, 1, cv2.LINE_AA)
    static_path = f'{OUT_DIR}/twitter_{layer.lower()}_{tile}_static.png'
    cv2.imwrite(static_path, static)
    print(f"  Static saved: {static_path} ({static.shape[1]}x{static.shape[0]})")

    # ══════════════════════════════════════════════════════════
    # GIF: Scene5-style registration morph
    # Ex-vivo (magenta) stays fixed. In-vivo (green) morphs from
    # centroid-aligned to full M2d affine registration on top.
    # Then MERSCOPE dots fade in. Same approach as scene5 Phase D→E→F.
    # ══════════════════════════════════════════════════════════
    gif_dir = f'{OUT_DIR}/gif_frames_{layer.lower()}_{tile}'
    os.makedirs(gif_dir, exist_ok=True)

    def ease(t):
        return t * t * (3 - 2 * t)

    # Build affine matrices:
    # M_start: in-vivo scaled to FILL the crop bbox (same display size as ex-vivo)
    # M_end: full M2d registration (in-vivo at correct position/scale)
    M3 = np.vstack([M2d, [0, 0, 1]])

    # In-vivo content bounds in JY306 space
    iv_h, iv_w = iv_mip.shape[:2]  # 658 × 629
    # Crop bbox center in nd2 space
    bbox_cx = (x0 + x1) / 2.0
    bbox_cy = (y0 + y1) / 2.0
    bbox_w = x1 - x0
    bbox_h = y1 - y0

    # M_start: scale in-vivo to fill the same crop bbox as ex-vivo
    fill_scale = min(bbox_w / iv_w, bbox_h / iv_h)
    iv_cx_jy = iv_w / 2.0
    iv_cy_jy = iv_h / 2.0
    M_start = np.array([
        [fill_scale, 0, bbox_cx - iv_cx_jy * fill_scale],
        [0, fill_scale, bbox_cy - iv_cy_jy * fill_scale],
        [0, 0, 1]
    ], dtype=np.float64)

    # M_end: full M2d (perfect registration)
    M_end = M3.copy()

    # Pre-render ex-vivo background (magenta, masked, cropped)
    ev_bg = tissue_ev.copy()
    ev_bg[~mask_bool] = 0
    ev_bg_crop = ev_bg[y0:y1, x0:x1]

    PH_EV_ONLY = 18      # show ex-vivo alone
    PH_IV_FADE = 12      # fade in in-vivo (full size on top)
    PH_HOLD_BOTH = 18    # hold both at same size
    PH_MORPH = 42        # morph in-vivo → registered position
    PH_HOLD_REG = 18     # hold registered
    PH_MERSCOPE = 24     # MERSCOPE dots fade in
    PH_HOLD_END = 30     # hold final
    PH_FADE_BACK = 18    # fade back for loop

    frame_idx = 0

    def render_left(iv_M, iv_alpha, ev_alpha, ms_alpha, label, label_col):
        """Render left panel: ex-vivo bg + warped in-vivo + optional dots."""
        # Warp in-vivo at full res
        iv_w = cv2.warpAffine(iv_mip, iv_M[:2].astype(np.float64),
                               (ND2_SIZE, ND2_SIZE),
                               flags=cv2.INTER_LINEAR, borderValue=0)
        iw = np.clip(norm8(iv_w).astype(np.float32) * iv_tissue_b, 0, 255).astype(np.uint8)

        # Composite: ex-vivo magenta bg + green in-vivo overlay
        comp = np.zeros((ND2_SIZE, ND2_SIZE, 3), np.float32)
        comp[:,:,2] = ev_bright.astype(np.float32) * ev_alpha   # magenta R
        comp[:,:,0] = ev_bright.astype(np.float32) * ev_alpha   # magenta B
        # Overlay green where in-vivo has signal
        iv_mask = iw > 10
        comp[iv_mask, 1] += iw[iv_mask].astype(np.float32) * iv_alpha

        # MERSCOPE dots
        if ms_alpha > 0.01:
            dm = dot_layer.max(axis=2) > 0
            comp[dm] = comp[dm] * (1 - ms_alpha * 0.5) + \
                       dot_layer[dm].astype(np.float32) * ms_alpha

        comp = np.clip(comp, 0, 255).astype(np.uint8)

        # Mask + crop + resize
        comp[~mask_bool] = 0
        left = cv2.resize(comp[y0:y1, x0:x1], (TS, TS), interpolation=cv2.INTER_AREA)
        left = np.clip(left.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)

        # Build full frame
        frame = static_base.copy()
        frame[LABEL_H:LABEL_H+TS, 0:TS] = left
        ts_f = 0.55
        (tw, _), _ = cv2.getTextSize(label, FONT, ts_f, 1)
        cv2.putText(frame, label, ((TS - tw) // 2, LABEL_H - 12),
                    FONT, ts_f, label_col, 1, cv2.LINE_AA)
        return frame

    print("  Rendering GIF frames...")

    # Phase 1: Ex-vivo alone (magenta tissue)
    for f in range(PH_EV_ONLY):
        frame = render_left(M_start, 0.0, 0.7, 0.0, 'EX VIVO', (255, 0, 255))
        cv2.imwrite(f'{gif_dir}/frame_{frame_idx:04d}.png', frame)
        frame_idx += 1

    # Phase 2: Fade in in-vivo (green, full size on top of ex-vivo)
    for f in range(PH_IV_FADE):
        t = ease(f / (PH_IV_FADE - 1))
        frame = render_left(M_start, t * 0.8, 0.7, 0.0, 'IN VIVO + EX VIVO', (0, 255, 0))
        cv2.imwrite(f'{gif_dir}/frame_{frame_idx:04d}.png', frame)
        frame_idx += 1

    # Phase 3: Hold both at same size (unregistered)
    for f in range(PH_HOLD_BOTH):
        frame = render_left(M_start, 0.8, 0.7, 0.0, 'IN VIVO + EX VIVO', (0, 255, 0))
        cv2.imwrite(f'{gif_dir}/frame_{frame_idx:04d}.png', frame)
        frame_idx += 1

    # Phase 4: Morph — in-vivo shrinks/moves to registered position
    for f in range(PH_MORPH):
        t = ease(f / (PH_MORPH - 1))
        M_t = M_start * (1 - t) + M_end * t
        frame = render_left(M_t, 0.8, 0.7, 0.0, 'AFFINE REGISTRATION', (0, 200, 200))
        cv2.imwrite(f'{gif_dir}/frame_{frame_idx:04d}.png', frame)
        frame_idx += 1

    # Phase 5: Hold registered
    for f in range(PH_HOLD_REG):
        frame = render_left(M_end, 0.8, 0.7, 0.0, 'REGISTERED', WHITE)
        cv2.imwrite(f'{gif_dir}/frame_{frame_idx:04d}.png', frame)
        frame_idx += 1

    # Phase 6: MERSCOPE dots fade in
    for f in range(PH_MERSCOPE):
        t = ease(f / (PH_MERSCOPE - 1))
        frame = render_left(M_end, 0.7, 0.6, t * 0.7, '+ MERSCOPE mRNA', (0, 180, 255))
        cv2.imwrite(f'{gif_dir}/frame_{frame_idx:04d}.png', frame)
        frame_idx += 1

    # Phase 7: Hold final (all 3 registered)
    for f in range(PH_HOLD_END):
        frame = render_left(M_end, 0.7, 0.6, 0.7, 'ALL REGISTERED', WHITE)
        cv2.imwrite(f'{gif_dir}/frame_{frame_idx:04d}.png', frame)
        frame_idx += 1

    # Phase 8: Fade back for loop
    for f in range(PH_FADE_BACK):
        t = ease(f / (PH_FADE_BACK - 1))
        M_t = M_end * (1 - t) + M_start * t
        ms_a = 0.7 * (1 - t)
        iv_a = 0.7 + 0.1 * t
        frame = render_left(M_t, iv_a * (1-t), 0.7, ms_a, 'EX VIVO', (255, 0, 255))
        cv2.imwrite(f'{gif_dir}/frame_{frame_idx:04d}.png', frame)
        frame_idx += 1

    print(f"  GIF frames: {frame_idx} frames in {gif_dir}/")

    # Build GIF with ffmpeg
    gif_path = f'{OUT_DIR}/twitter_{layer.lower()}_{tile}.gif'
    os.system(f'ffmpeg -y -framerate {GIF_FPS} -i {gif_dir}/frame_%04d.png '
              f'-vf "split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];'
              f'[s1][p]paletteuse=dither=floyd_steinberg" '
              f'-loop 0 {gif_path} 2>/dev/null')
    gif_size = os.path.getsize(gif_path) / 1024 / 1024
    print(f"  GIF saved: {gif_path} ({gif_size:.1f} MB)")

print(f'\nDone. Outputs in {OUT_DIR}/')
