#!/usr/bin/env python3
"""Twitter tissue-level showcase: 4-column contact sheets + cell crops.
  Col 1: IN VIVO (green, warped to nd2 space)
  Col 2: EX VIVO (magenta, nd2 GFP MIP)
  Col 3: MERSCOPE mRNA EXPRESSION (rainbow dots on dim magenta background)
  Col 4: REGISTERED (all 3 overlaid: green + magenta + dots)

Top row: full tissue (masked).
Below: 2 best-matching cells from that tissue, same 4 columns.
"""
import numpy as np, cv2, os, re, pickle, json
import tifffile
import pandas as pd
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/png_exports/twitter_showcase'
os.makedirs(OUT_DIR, exist_ok=True)

ND2_SIZE = 4200
DISP_SIZE = 1400   # display size per tissue panel
CELL_DISP = 350    # display size per cell panel
CROP_ND2 = 160     # crop radius in nd2 pixels (~103µm)
GFP_DIM = 0.20
MASK_PAD = 80
BRIGHT = 1.8

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

# ── Rainbow palette ──
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

# ── Tiles to showcase: (tile, layer, [(lm_idx, lm_idx), ...]) ──
TILES = [
    ('row2_1', 'ORIENS',     [5, 17]),    # best 2 spatially spread, <1µm error
    ('row3_6', 'PYRAMIDALE', [36, 19]),   # best 2 spatially spread, <1µm error
]

# ── Load masks ──
print("Loading tissue masks...")
masks_npz = np.load(f'{BASE}/registration_video/via_masks_v4.npz')

# ── Load in-vivo volume ──
print("Loading JY306 in-vivo stack...")
iv_vol = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv = iv_vol.shape[0]
print(f"  Shape: {iv_vol.shape}")

for tile, layer, cell_lms in TILES:
    print(f"\n{'='*60}")
    print(f"  {tile} — {layer}")
    print(f"{'='*60}")

    # ── Load pkl transform ──
    pkl_tfm_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
    tfm = np.load(pkl_tfm_path)
    M2d = tfm['M2d_jy306_to_nd2']
    ev_nd2 = tfm['ev_nd2']
    iv_nd2_coords = tfm['iv_nd2']
    pcd_iv = tfm['pcd_invivo_jy306']

    # Errors per landmark
    pkl_dist_um = np.sqrt((iv_nd2_coords[:,0]-ev_nd2[:,0])**2 +
                          (iv_nd2_coords[:,1]-ev_nd2[:,1])**2) * 0.645

    med_z_iv = int(round(np.median(pcd_iv[:, 0])))
    med_z_iv = max(0, min(med_z_iv, nz_iv - 1))
    print(f"  Median in-vivo z = {med_z_iv}")

    # ── Load nd2 GFP slices ──
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
    print(f"  Ex-vivo MIP built from 12 z-slices")

    # ── Tissue panels (full FOV) ──
    # Panel 1: IN VIVO (green, warped MIP ±2z)
    z_lo = max(0, med_z_iv - 2)
    z_hi = min(nz_iv, med_z_iv + 3)
    iv_mip = iv_vol[z_lo:z_hi].max(axis=0)
    iv_warped = cv2.warpAffine(iv_mip, M2d, (ND2_SIZE, ND2_SIZE),
                                flags=cv2.INTER_LINEAR, borderValue=0)
    iw_u8 = norm8(iv_warped)
    iw_bright = np.clip(iw_u8.astype(np.float32) * BRIGHT, 0, 255).astype(np.uint8)
    t1 = np.zeros((ND2_SIZE, ND2_SIZE, 3), np.uint8)
    t1[:,:,1] = iw_bright

    # Panel 2: EX VIVO (magenta)
    ev_bright = np.clip(ev_u8.astype(np.float32) * BRIGHT, 0, 255).astype(np.uint8)
    t2 = np.zeros((ND2_SIZE, ND2_SIZE, 3), np.uint8)
    t2[:,:,2] = ev_bright
    t2[:,:,0] = ev_bright

    # Panel 3: MERSCOPE mRNA (rainbow dots on dim magenta)
    t3 = np.zeros((ND2_SIZE, ND2_SIZE, 3), np.uint8)
    t3[:,:,2] = (ev_u8.astype(np.float32) * GFP_DIM).astype(np.uint8)
    t3[:,:,0] = (ev_u8.astype(np.float32) * GFP_DIM).astype(np.uint8)

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

            try:
                df = pd.read_csv(csv_path, usecols=['global_x','global_y','gene'])
                is_blank = df.gene.str.startswith('Blank', na=False)
                df = df[~is_blank].reset_index(drop=True)
                print(f"    Filtered {is_blank.sum()} Blank probes")
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
                    cv2.circle(dot_layer, (int(gene_nd2_x_all[j]), int(gene_nd2_y_all[j])),
                               2, col, -1)
                print(f"  {valid.sum()} MERSCOPE transcripts in bounds")
            except Exception as e:
                print(f"  Gene dot error: {e}")

    dot_mask = dot_layer.max(axis=2) > 0
    t3[dot_mask] = dot_layer[dot_mask]

    # Panel 4: REGISTERED (blended)
    IV_ALPHA, EV_ALPHA, DOT_ALPHA = 0.5, 0.5, 0.6
    t4 = np.zeros((ND2_SIZE, ND2_SIZE, 3), np.float32)
    t4[:,:,1] += iw_bright.astype(np.float32) * IV_ALPHA
    t4[:,:,2] += ev_bright.astype(np.float32) * EV_ALPHA
    t4[:,:,0] += ev_bright.astype(np.float32) * EV_ALPHA
    t4[dot_mask] += dot_layer[dot_mask].astype(np.float32) * DOT_ALPHA
    t4 = np.clip(t4, 0, 255).astype(np.uint8)

    # ── Apply tissue mask and crop to bbox ──
    mask = masks_npz[tile]
    mask_bool = mask > 0
    ys, xs = np.where(mask_bool)
    y0 = max(0, ys.min() - MASK_PAD)
    y1 = min(ND2_SIZE, ys.max() + MASK_PAD)
    x0 = max(0, xs.min() - MASK_PAD)
    x1 = min(ND2_SIZE, xs.max() + MASK_PAD)
    h_box, w_box = y1 - y0, x1 - x0
    side = max(h_box, w_box)
    cy_box, cx_box = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, cy_box - side // 2)
    y1 = min(ND2_SIZE, y0 + side)
    x0 = max(0, cx_box - side // 2)
    x1 = min(ND2_SIZE, x0 + side)
    crop_w, crop_h = x1 - x0, y1 - y0
    print(f"  Mask bbox: ({x0},{y0})-({x1},{y1}), {crop_w}x{crop_h}")

    tissue_panels = []
    for p in [t1, t2, t3, t4]:
        p_masked = p.copy()
        p_masked[~mask_bool] = 0
        cropped = p_masked[y0:y1, x0:x1]
        tissue_panels.append(cv2.resize(cropped, (DISP_SIZE, DISP_SIZE), interpolation=cv2.INTER_AREA))

    # ── Cell crop rows ──
    cell_rows = []
    for lm_idx in cell_lms:
        cx_nd2 = int(round(ev_nd2[lm_idx, 0]))
        cy_nd2 = int(round(ev_nd2[lm_idx, 1]))
        z_nd2 = int(round(np.clip(ev_nd2[lm_idx, 2], 0, 11)))
        z_iv = int(round(np.clip(pcd_iv[lm_idx, 0], 0, nz_iv - 1)))
        err = pkl_dist_um[lm_idx]
        print(f"  Cell LM#{lm_idx}: err={err:.2f}µm, nd2=({cx_nd2},{cy_nd2}), z_iv={z_iv}, z_nd2={z_nd2}")

        # In-vivo: warp the correct z-slice
        iv_slice = iv_vol[z_iv]
        iv_w = cv2.warpAffine(iv_slice, M2d, (ND2_SIZE, ND2_SIZE),
                               flags=cv2.INTER_LINEAR, borderValue=0)
        iw_cell = norm8(iv_w)
        iw_cell_bright = np.clip(iw_cell.astype(np.float32) * BRIGHT, 0, 255).astype(np.uint8)
        iw_patch = crop_patch(iw_cell_bright, cx_nd2, cy_nd2, CROP_ND2)
        c1 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
        c1[:,:,1] = iw_patch

        # Ex-vivo: use correct z-slice
        ev_slice = nd2_slices[z_nd2]
        ev_cell = norm8(ev_slice)
        ev_cell_bright = np.clip(ev_cell.astype(np.float32) * BRIGHT, 0, 255).astype(np.uint8)
        ev_patch = crop_patch(ev_cell_bright, cx_nd2, cy_nd2, CROP_ND2)
        c2 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
        c2[:,:,2] = ev_patch
        c2[:,:,0] = ev_patch

        # MERSCOPE: dots in cell patch
        ev_dim_patch = crop_patch(
            (ev_cell.astype(np.float32) * GFP_DIM).astype(np.uint8),
            cx_nd2, cy_nd2, CROP_ND2)
        c3 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
        c3[:,:,2] = ev_dim_patch
        c3[:,:,0] = ev_dim_patch

        if gene_nd2_x_all is not None:
            dx = gene_nd2_x_all - (cx_nd2 - CROP_ND2)
            dy = gene_nd2_y_all - (cy_nd2 - CROP_ND2)
            in_patch = (dx >= 0) & (dx < CROP_ND2*2) & (dy >= 0) & (dy < CROP_ND2*2)
            pxi = dx[in_patch].astype(int)
            pyi = dy[in_patch].astype(int)
            pgn = gene_names_all[in_patch]
            # Subsample if too many
            max_show = 600
            n_dots = int(in_patch.sum())
            if n_dots > max_show:
                rng = np.random.default_rng(lm_idx)
                sel = rng.choice(n_dots, max_show, replace=False)
                pxi, pyi, pgn = pxi[sel], pyi[sel], pgn[sel]
            for j in range(len(pxi)):
                gc_col = gene_colour.get(pgn[j], (200,200,200))
                cv2.circle(c3, (int(pxi[j]), int(pyi[j])), 3, gc_col, -1, cv2.LINE_AA)
            print(f"    {n_dots} gene dots in cell patch")

        # Registered: all 3 blended
        c4 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.float32)
        c4[:,:,1] += iw_patch.astype(np.float32) * IV_ALPHA
        c4[:,:,2] += ev_patch.astype(np.float32) * EV_ALPHA
        c4[:,:,0] += ev_patch.astype(np.float32) * EV_ALPHA
        # Add cell dots
        cell_dot_mask = c3.max(axis=2) > (ev_dim_patch * 0.8)  # dots brighter than dim bg
        dot_only = c3.copy()
        dot_only[~cell_dot_mask] = 0
        c4_dots = dot_only.astype(np.float32) * DOT_ALPHA
        c4 += c4_dots
        c4 = np.clip(c4, 0, 255).astype(np.uint8)

        # Resize all cell panels
        cell_panels = []
        for c in [c1, c2, c3, c4]:
            cp = cv2.resize(c, (CELL_DISP, CELL_DISP), interpolation=cv2.INTER_LANCZOS4)
            cp = np.clip(cp.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
            cell_panels.append(cp)

        cell_rows.append((cell_panels, lm_idx, err))

    # ══════════════════════════════════════════════════════════════
    # ── Assemble final canvas: tissue row + cell rows ──
    # ══════════════════════════════════════════════════════════════
    GAP = 10
    LABEL_H = 50
    CELL_LABEL_H = 30
    FOOTER_H = 40
    DS = DISP_SIZE
    CD = CELL_DISP

    total_w = DS * 4 + GAP * 3
    # Cell row width: 4 cells centered
    cell_row_w = CD * 4 + GAP * 3

    # Heights
    tissue_h = LABEL_H + DS
    cell_total_h = sum(CELL_LABEL_H + CD for _ in cell_rows)
    total_h = tissue_h + GAP + cell_total_h + FOOTER_H

    canvas = np.zeros((total_h, total_w, 3), np.uint8)

    # ── Tissue row ──
    labels = ['IN VIVO', 'EX VIVO', 'MERSCOPE mRNA', 'REGISTERED']
    for i, (panel, label) in enumerate(zip(tissue_panels, labels)):
        panel = np.clip(panel.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
        px = i * (DS + GAP)
        py = LABEL_H
        canvas[py:py+DS, px:px+DS] = panel

        ts = 0.7
        (tw, _), _ = cv2.getTextSize(label, FONT, ts, 1)
        cv2.putText(canvas, label, (px + (DS - tw) // 2, LABEL_H - 14),
                    FONT, ts, WHITE, 1, cv2.LINE_AA)

    # Scale bar on tissue (200µm)
    sb_um = 200
    um_per_px = 0.645 * (crop_w / DS)
    sb_px = int(sb_um / um_per_px)
    sb_x0 = 3 * (DS + GAP) + DS - sb_px - 20
    sb_y = LABEL_H + DS - 25
    cv2.line(canvas, (sb_x0, sb_y), (sb_x0 + sb_px, sb_y), WHITE, 3)
    cv2.putText(canvas, f'{sb_um} \u00b5m', (sb_x0, sb_y - 10), FONT, 0.45, WHITE, 1, cv2.LINE_AA)

    # ── Cell rows ──
    cell_x_offset = (total_w - cell_row_w) // 2  # center cells
    cur_y = tissue_h + GAP

    for cell_panels, lm_idx, err in cell_rows:
        # Cell label
        cell_label = f'CELL {lm_idx}  |  {err:.1f} \u00b5m registration error'
        ts = 0.5
        (tw, _), _ = cv2.getTextSize(cell_label, FONT, ts, 1)
        cv2.putText(canvas, cell_label, (cell_x_offset + (cell_row_w - tw) // 2, cur_y + 20),
                    FONT, ts, (200, 200, 200), 1, cv2.LINE_AA)
        cur_y += CELL_LABEL_H

        for i, cp in enumerate(cell_panels):
            px = cell_x_offset + i * (CD + GAP)
            canvas[cur_y:cur_y+CD, px:px+CD] = cp
            cv2.rectangle(canvas, (px, cur_y), (px+CD-1, cur_y+CD-1), (80, 80, 80), 1)

        # Scale bar on cell row (50µm)
        cell_sb_um = 50
        cell_um_per_px = 0.645 * (CROP_ND2 * 2 / CD)
        cell_sb_px = int(cell_sb_um / cell_um_per_px)
        cell_sb_x0 = cell_x_offset + 3 * (CD + GAP) + CD - cell_sb_px - 10
        cell_sb_y = cur_y + CD - 15
        cv2.line(canvas, (cell_sb_x0, cell_sb_y), (cell_sb_x0 + cell_sb_px, cell_sb_y), WHITE, 2)
        cv2.putText(canvas, f'{cell_sb_um} \u00b5m', (cell_sb_x0, cell_sb_y - 6),
                    FONT, 0.35, WHITE, 1, cv2.LINE_AA)

        cur_y += CD

    # ── Footer ──
    footer = f'{layer}  |  {tile.upper().replace("_"," ")}  |  Hippocampus CA1'
    ts = 0.6
    (tw, _), _ = cv2.getTextSize(footer, FONT, ts, 1)
    cv2.putText(canvas, footer, ((total_w - tw) // 2, cur_y + 28),
                FONT, ts, (180, 180, 180), 1, cv2.LINE_AA)

    out_path = f'{OUT_DIR}/tissue_{layer.lower()}_{tile}.png'
    cv2.imwrite(out_path, canvas)
    print(f"  Saved: {out_path} ({canvas.shape[1]}x{canvas.shape[0]})")

print(f'\nDone. Outputs in {OUT_DIR}/')
