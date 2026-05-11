#!/usr/bin/env python3
"""Twitter showcase: 4-panel contact sheets for Jason.
  Panel 1: IN VIVO (green, warped to nd2 space)
  Panel 2: EX VIVO (magenta, nd2 GFP native)
  Panel 3: MERSCOPE mRNA EXPRESSION (rainbow dots on dim magenta GCaMP)
  Panel 4: REGISTERED OVERLAY (green in-vivo + magenta ex-vivo)

Two cells: one oriens (row2_1), one pyramidale (row4_4).
"""
import numpy as np, cv2, os, glob, pickle, json, re
import tifffile
import pandas as pd
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/png_exports/twitter_showcase'
os.makedirs(OUT_DIR, exist_ok=True)

ND2_XY_UM = 0.645
CROP_ND2 = 160          # radius in nd2 pixels (~103µm) — generous for Twitter
PATCH_DISP = 512         # display size per panel (high-res)
GFP_DIM = 0.20           # dim GCaMP behind gene dots

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

# ── Rainbow palette for gene dots ──
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

# ── Candidates: (tile, lm_idx, layer_name) ──
CANDIDATES = [
    ('row2_1', 5,  'ORIENS'),       # err=0.5µm, dorsal
    ('row3_6', 0,  'PYRAMIDALE'),   # err=0.4µm, pyramidale ring
    ('row4_4', 21, 'PYRAMIDALE'),   # err=0.5µm, deep pyramidale (backup)
]

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

# ── Load in-vivo ──
print("Loading JY306 in-vivo stack...")
iv_vol = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv = iv_vol.shape[0]
print(f"  Shape: {iv_vol.shape}")

# ── Process each candidate ──
for tile, best_idx, layer in CANDIDATES:
    print(f"\n{'='*60}")
    print(f"  {tile} LM#{best_idx} — {layer}")
    print(f"{'='*60}")

    pkl_tfm_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
    tfm = np.load(pkl_tfm_path)
    ev_nd2 = tfm['ev_nd2']
    iv_nd2 = tfm['iv_nd2']
    pcd_iv = tfm['pcd_invivo_jy306']
    M2d = tfm['M2d_jy306_to_nd2']

    pkl_dist_um = np.sqrt((iv_nd2[:,0]-ev_nd2[:,0])**2 + (iv_nd2[:,1]-ev_nd2[:,1])**2) * ND2_XY_UM
    print(f"  Registration error: {pkl_dist_um[best_idx]:.1f} µm")

    cx_nd2 = int(round(ev_nd2[best_idx, 0]))
    cy_nd2 = int(round(ev_nd2[best_idx, 1]))
    z_nd2 = int(round(np.clip(ev_nd2[best_idx, 2], 0, 11)))
    z_iv = int(round(np.clip(pcd_iv[best_idx, 0], 0, nz_iv - 1)))

    # Load nd2 GFP slice
    tile_dir = f'{BASE}/png_exports/registration_video/{tile}'
    nd2_path = f'{tile_dir}/GFP_z{z_nd2:03d}.png'
    nd2_slice = cv2.imread(nd2_path, cv2.IMREAD_GRAYSCALE)
    if nd2_slice is None:
        print(f"  WARNING: {nd2_path} not found, skipping")
        continue
    nd2_slice = nd2_slice.astype(np.float32)

    # ── Panel 1: IN VIVO (green, warped to nd2 space) ──
    iv_slice = iv_vol[z_iv]
    iv_warped = cv2.warpAffine(iv_slice, M2d, (4200, 4200),
                                flags=cv2.INTER_LINEAR, borderValue=0)
    iw_u8 = norm8(iv_warped)
    iw_patch = crop_patch(iw_u8, cx_nd2, cy_nd2, CROP_ND2)
    p1 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
    p1[:,:,1] = iw_patch  # green

    # ── Panel 2: EX VIVO (magenta) ──
    ev_u8 = norm8(nd2_slice)
    ev_patch = crop_patch(ev_u8, cx_nd2, cy_nd2, CROP_ND2)
    p2 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
    p2[:,:,2] = ev_patch  # R (magenta)
    p2[:,:,0] = ev_patch  # B (magenta)

    # ── Panel 3: MERSCOPE mRNA EXPRESSION (rainbow dots on dim magenta GCaMP) ──
    p3 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
    # Dim magenta GCaMP background
    p3[:,:,2] = (ev_patch.astype(np.float32) * GFP_DIM).astype(np.uint8)
    p3[:,:,0] = (ev_patch.astype(np.float32) * GFP_DIM).astype(np.uint8)

    n_dots = 0
    if tile in tile_to_ms:
        ms_id, tile_short, ex_path = tile_to_ms[tile]
        csv_path = f'{VAROL}/region_{ms_id}_resegmentation/detected_transcripts.csv'
        mnf_path = f'{VAROL}/region_{ms_id}_resegmentation/manifest.json'
        m2m_path = f'{VAROL}/region_{ms_id}_resegmentation/micron_to_mosaic_pixel_transform.csv'

        if ms_id in pkl_files and all(os.path.exists(p) for p in [csv_path, mnf_path, m2m_path]):
            exvivo_tif = tifffile.imread(ex_path).astype(np.float32)
            tif_size = exvivo_tif.shape[1]
            nd2_scale = 4200 / tif_size
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

                valid = (gene_nd2_x >= 0) & (gene_nd2_x < 4200) & \
                        (gene_nd2_y >= 0) & (gene_nd2_y < 4200)
                gene_nd2_x = gene_nd2_x[valid]
                gene_nd2_y = gene_nd2_y[valid]
                gene_names = gene_names[valid]

                gc = Counter(gene_names)
                gene_colour = {g: PALETTE[i % 550] for i, g in enumerate(g for g, _ in gc.most_common())}

                dx = gene_nd2_x - (cx_nd2 - CROP_ND2)
                dy = gene_nd2_y - (cy_nd2 - CROP_ND2)
                in_patch = (dx >= 0) & (dx < CROP_ND2*2) & (dy >= 0) & (dy < CROP_ND2*2)
                pxi = dx[in_patch].astype(int)
                pyi = dy[in_patch].astype(int)
                pgn = gene_names[in_patch]
                n_dots = int(in_patch.sum())

                # Subsample to keep dots visible but not overwhelming
                max_show = 800
                if n_dots > max_show:
                    rng = np.random.default_rng(42)
                    sel = rng.choice(n_dots, max_show, replace=False)
                    pxi, pyi, pgn = pxi[sel], pyi[sel], pgn[sel]

                for j in range(len(pxi)):
                    gc_col = gene_colour.get(pgn[j], (200,200,200))
                    cv2.circle(p3, (int(pxi[j]), int(pyi[j])), 3, gc_col, -1, cv2.LINE_AA)
                print(f"  {n_dots} gene dots in patch")
            except Exception as e:
                print(f"  Gene dot error: {e}")

    # ── Panel 4: REGISTERED OVERLAY (green in-vivo + magenta ex-vivo + MERSCOPE dots) ──
    p4 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
    p4[:,:,1] = iw_patch                   # green: in-vivo
    p4[:,:,2] = ev_patch                    # R: ex-vivo magenta
    p4[:,:,0] = ev_patch                    # B: ex-vivo magenta
    # Add gene dots on top of the overlay
    if n_dots > 0 and 'pxi' in dir():
        for j in range(len(pxi)):
            gc_col = gene_colour.get(pgn[j], (200,200,200))
            cv2.circle(p4, (int(pxi[j]), int(pyi[j])), 3, gc_col, -1, cv2.LINE_AA)

    # ── Assemble 4-panel contact sheet ──
    GAP = 12
    PS = PATCH_DISP
    LABEL_H = 50
    FOOTER_H = 36

    labels = ['IN VIVO', 'EX VIVO', 'MERSCOPE mRNA', 'REGISTERED OVERLAY']
    panels_raw = [p1, p2, p3, p4]

    total_w = PS * 4 + GAP * 3
    total_h = LABEL_H + PS + FOOTER_H
    canvas = np.zeros((total_h, total_w, 3), np.uint8)

    for i, (panel, label) in enumerate(zip(panels_raw, labels)):
        resized = cv2.resize(panel, (PS, PS), interpolation=cv2.INTER_LANCZOS4)
        # Gentle brightness boost
        resized = np.clip(resized.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)

        px = i * (PS + GAP)
        py = LABEL_H
        canvas[py:py+PS, px:px+PS] = resized

        # Thin border
        cv2.rectangle(canvas, (px, py), (px+PS-1, py+PS-1), (80, 80, 80), 1)

        # Label centered above panel
        ts = 0.65
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, FONT, ts, thickness)
        lx = px + (PS - tw) // 2
        cv2.putText(canvas, label, (lx, LABEL_H - 14), FONT, ts, WHITE, thickness, cv2.LINE_AA)

    # Footer: layer + tile info
    layer_txt = f'{layer}  |  {tile.upper().replace("_"," ")}  |  Registration error: {pkl_dist_um[best_idx]:.1f} \u00b5m'
    ts = 0.55
    (tw, _), _ = cv2.getTextSize(layer_txt, FONT, ts, 1)
    cv2.putText(canvas, layer_txt, ((total_w - tw) // 2, LABEL_H + PS + 26),
                FONT, ts, (180, 180, 180), 1, cv2.LINE_AA)

    # Scale bar (50µm)
    sb_um = 50
    sb_px = int(sb_um / ND2_XY_UM * PS / (CROP_ND2 * 2))  # scale bar in display pixels
    sb_x0 = 3 * (PS + GAP) + PS - sb_px - 20  # bottom-right of last panel
    sb_y = LABEL_H + PS - 20
    cv2.line(canvas, (sb_x0, sb_y), (sb_x0 + sb_px, sb_y), WHITE, 3)
    cv2.putText(canvas, f'{sb_um} \u00b5m', (sb_x0, sb_y - 8), FONT, 0.45, WHITE, 1, cv2.LINE_AA)

    out_path = f'{OUT_DIR}/showcase_{layer.lower()}_{tile}_lm{best_idx}.png'
    cv2.imwrite(out_path, canvas)
    print(f"  Saved: {out_path} ({canvas.shape[1]}x{canvas.shape[0]})")

    # Also save as RGB for preview
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    out_rgb = f'{OUT_DIR}/showcase_{layer.lower()}_{tile}_lm{best_idx}_rgb.png'
    cv2.imwrite(out_rgb, cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR))

print(f'\nDone. Outputs in {OUT_DIR}/')
