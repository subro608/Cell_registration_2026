#!/usr/bin/env python3
"""Landmark patch strip: 4 panels per landmark row.
  Col 0: In-vivo zoomed (raw JY306)
  Col 1: In-vivo warped (pkl M2d → nd2 space)
  Col 2: Ex-vivo nd2 (GFP)
  Col 3: Ex-vivo + MERFISH gene dots

Metadata label per row: landmark#, tile, z_iv→z_nd2, error.
Output: png_exports/landmark_genedot_patches/
"""
import numpy as np
import cv2
import os, re, glob, json, pickle
import tifffile
import pandas as pd
from scipy.optimize import curve_fit
from collections import defaultdict

BASE    = '/Users/neurolab/neuroinformatics/margaret'
EX_DIR  = f'{BASE}/exvivo_merscope_combined'
PKL_DIR = f'{BASE}/merscope_exvivo '
VAROL   = f'{BASE}/jy306_varol'
OUT_DIR = f'{BASE}/png_exports/landmark_genedot_patches'
os.makedirs(OUT_DIR, exist_ok=True)

IV_XY_UM  = 0.6835
ND2_XY_UM = 0.645
PATCH_DISP = 120        # display size per panel
CROP_ND2   = 78         # ~50µm radius in nd2 pixels
CROP_JY_ZOOM = 37       # ~25µm radius in JY306 (2x zoom)
N_COLS     = 10         # patches per row in contact sheet
WARP_ERR_MAX_UM = 5.0
DOT_MIN = 20            # minimum dots in patch (skip empty)
DOT_MAX = 800           # maximum dots in patch (skip saturated)
GFP_DIM = 0.25          # dim GFP in gene dot panel so rainbow dots pop

def make_rainbow_palette(n):
    """Generate n maximally distinct bright colours cycling through HSV."""
    colors = []
    for i in range(n):
        h = int(180 * i / n)          # hue 0-179 (OpenCV HSV)
        s = 200 + int(55 * ((i % 3) / 2))  # vary saturation 200-255
        v = 200 + int(55 * ((i % 5) / 4))  # vary value 200-255
        bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(c) for c in bgr))
    return colors

N_GENE_COLORS = 550  # cover all genes with unique rainbow colours
GENE_PALETTE = make_rainbow_palette(N_GENE_COLORS)

# ── Helpers ───────────────────────────────────────────────────
def gauss(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def find_z_gaussian(intensities):
    zs = np.arange(len(intensities), dtype=np.float64)
    vals = np.array(intensities, dtype=np.float64)
    vals = vals - vals.min()
    total = vals.sum()
    if total < 1e-6:
        return float(np.argmax(intensities))
    peak_z = np.argmax(vals)
    try:
        p0 = [vals[peak_z], float(peak_z), 2.0]
        popt, _ = curve_fit(gauss, zs, vals, p0=p0,
                            bounds=([0, -1, 0.3], [vals.max()*3, 12, 8]),
                            maxfev=1000)
        if 0 <= popt[1] <= 11:
            return popt[1]
    except (RuntimeError, ValueError):
        pass
    return float(np.sum(zs * vals) / total)

def norm8(img):
    v = img[img > 0]
    if len(v) < 100:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(v, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

def crop_patch(img, cx, cy, r):
    h, w = img.shape[:2]
    x0, y0 = cx - r, cy - r
    x1, y1 = cx + r, cy + r
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    dx0, dy0 = sx0 - x0, sy0 - y0
    ndim = img.shape[2] if img.ndim == 3 else None
    if ndim:
        patch = np.zeros((2*r, 2*r, ndim), dtype=img.dtype)
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

# ── Tile ↔ MERSCOPE region mapping ───────────────────────────
tile_to_ms = {}   # 'row2_1' → (ms_id, tile_short)
for fname in os.listdir(EX_DIR):
    m = re.match(r'(\d+_\d+)_merscope(\d+)\.tif', fname)
    if m:
        tile_short = m.group(1)
        ms_id = int(m.group(2))
        row_tile = f'row{tile_short}'
        tile_to_ms[row_tile] = (ms_id, tile_short, f'{EX_DIR}/{fname}')

# PKL files per region
pkl_files = {}
for fname in os.listdir(PKL_DIR):
    m = re.match(r'[\d_]+_reg(\d+)_transformed.*\.pkl', fname)
    if m:
        mid = int(m.group(1))
        if mid not in pkl_files or fname > pkl_files[mid][0]:
            pkl_files[mid] = (fname, f'{PKL_DIR}/{fname}')

# ── Load in-vivo ──────────────────────────────────────────────
print("Loading JY306 in-vivo...")
iv_vol = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = iv_vol.shape
print(f"  {iv_vol.shape}")

# ── Discover landmarks ────────────────────────────────────────
lm_files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz'))
legacy = f'{BASE}/registration_video/landmarks_27_nd2_native.npz'

tile_lm = {}
for lf in lm_files:
    bn = os.path.basename(lf)
    tile = bn.replace('landmarks_nd2_native_', '').replace('.npz', '')
    tile_lm[tile] = lf
if os.path.exists(legacy):
    tile_lm['row2_1'] = legacy
print(f"Found {len(tile_lm)} tiles with landmarks")

# ── Process each tile ─────────────────────────────────────────
for tile in sorted(tile_lm.keys()):
    print(f"\n{'='*60}")
    print(f"  {tile}")
    print(f"{'='*60}")

    lm_path = tile_lm[tile]
    d = np.load(lm_path)
    ev_nd2 = d['ev_nd2']
    iv_nd2 = d['iv_nd2']
    pcd_iv = d['pcd_invivo_jy306']
    N_LM = len(ev_nd2)

    if N_LM < 3:
        print("  SKIP: <3 landmarks")
        continue

    # Load pkl M2d
    pkl_tfm_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
    if not os.path.exists(pkl_tfm_path):
        print(f"  SKIP: no pkl transform")
        continue
    tfm = np.load(pkl_tfm_path)
    M2d = tfm['M2d_jy306_to_nd2']

    # Load nd2 GFP slices
    tile_dir = f'{BASE}/png_exports/registration_video/{tile}'
    nd2_slices = []
    for zi in range(12):
        img = cv2.imread(f'{tile_dir}/GFP_z{zi:03d}.png', cv2.IMREAD_UNCHANGED)
        if img is None:
            nd2_slices.append(np.zeros((4200, 4200), dtype=np.float32))
        else:
            nd2_slices.append(img.astype(np.float32))
    nd2_slices = np.array(nd2_slices)
    nd2_h, nd2_w = 4200, 4200

    # Compute z for each landmark
    nd2_z_vals = []
    for i in range(N_LM):
        x, y = ev_nd2[i, 0], ev_nd2[i, 1]
        c = int(round(np.clip(x, 10, nd2_h - 11)))
        r = int(round(np.clip(y, 10, nd2_h - 11)))
        intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]
        nd2_z_vals.append(find_z_gaussian(intensities))

    # PKL 2D errors
    pkl_dx = iv_nd2[:, 0] - ev_nd2[:, 0]
    pkl_dy = iv_nd2[:, 1] - ev_nd2[:, 1]
    pkl_dist_um = np.sqrt(pkl_dx**2 + pkl_dy**2) * ND2_XY_UM

    # ── Load MERFISH gene dots for this tile ──────────────────
    gene_nd2_x, gene_nd2_y, gene_names = None, None, None
    gene_colour = {}
    if tile in tile_to_ms:
        ms_id, tile_short, ex_path = tile_to_ms[tile]
        csv_path = f'{VAROL}/region_{ms_id}_resegmentation/detected_transcripts.csv'
        mnf_path = f'{VAROL}/region_{ms_id}_resegmentation/manifest.json'
        m2m_path = f'{VAROL}/region_{ms_id}_resegmentation/micron_to_mosaic_pixel_transform.csv'

        if ms_id in pkl_files and all(os.path.exists(p) for p in [csv_path, mnf_path, m2m_path]):
            print(f"  Loading MERFISH region {ms_id}...")

            # TIF size for scale factor
            exvivo_tif = tifffile.imread(ex_path).astype(np.float32)
            tif_size = exvivo_tif.shape[1]
            nd2_scale = nd2_w / tif_size
            del exvivo_tif

            # PKL affine for MERFISH transform
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

                # In-bounds filter
                valid = (gene_nd2_x >= 0) & (gene_nd2_x < nd2_w) & \
                        (gene_nd2_y >= 0) & (gene_nd2_y < nd2_h)
                gene_nd2_x = gene_nd2_x[valid]
                gene_nd2_y = gene_nd2_y[valid]
                gene_names = gene_names[valid]

                # Assign every gene a unique rainbow colour (sorted by count)
                from collections import Counter
                gc = Counter(gene_names)
                all_genes_sorted = [g for g, _ in gc.most_common()]
                gene_colour = {g: GENE_PALETTE[i % N_GENE_COLORS] for i, g in enumerate(all_genes_sorted)}
                print(f"    {len(gene_nd2_x)} transcripts in bounds")
            except Exception as e:
                print(f"    gene load err: {e}")
                gene_nd2_x = None
        else:
            print(f"  No MERFISH data for region {ms_id}")
    else:
        print(f"  No MERSCOPE mapping for {tile}")

    # ── Z-pair grouping for warp cache ────────────────────────
    z_pair_to_lm = defaultdict(list)
    for i in range(N_LM):
        z_iv = int(round(np.clip(pcd_iv[i, 0], 0, nz_iv - 1)))
        z_nd2 = int(round(np.clip(nd2_z_vals[i], 0, 11)))
        z_pair_to_lm[(z_iv, z_nd2)].append(i)

    warp_cache = {}
    cards = []
    all_dot_counts = []

    for (z_iv, z_nd2), lm_indices in sorted(z_pair_to_lm.items()):
        # Warp in-vivo slice
        if (z_iv, z_nd2) not in warp_cache:
            warp_cache[(z_iv, z_nd2)] = cv2.warpAffine(
                iv_vol[z_iv], M2d, (nd2_w, nd2_h),
                flags=cv2.INTER_LINEAR, borderValue=0)
        iv_warped = warp_cache[(z_iv, z_nd2)]

        nd2_sl = nd2_slices[z_nd2]
        ev_u8 = norm8(nd2_sl)
        iw_u8 = norm8(iv_warped)
        ir_u8 = norm8(iv_vol[z_iv])

        for i in lm_indices:
            cx = int(round(ev_nd2[i, 0]))
            cy = int(round(ev_nd2[i, 1]))
            err_um = pkl_dist_um[i]
            passed = err_um <= WARP_ERR_MAX_UM

            # Col 0: In-vivo zoomed (raw JY306 space, 2x zoom)
            ix = int(round(pcd_iv[i, 2]))
            iy = int(round(pcd_iv[i, 1]))
            iv_zoom_patch = crop_patch(ir_u8, ix, iy, CROP_JY_ZOOM)
            iv_zoom_rgb = np.zeros((CROP_JY_ZOOM*2, CROP_JY_ZOOM*2, 3), dtype=np.uint8)
            iv_zoom_rgb[:,:,1] = iv_zoom_patch  # green

            # Col 1: In-vivo warped (nd2 space)
            iv_warp_patch = crop_patch(iw_u8, cx, cy, CROP_ND2)
            iv_warp_rgb = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), dtype=np.uint8)
            iv_warp_rgb[:,:,1] = iv_warp_patch  # green

            # Col 2: Ex-vivo nd2 (GFP)
            ev_patch = crop_patch(ev_u8, cx, cy, CROP_ND2)
            ev_rgb = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), dtype=np.uint8)
            ev_rgb[:,:,1] = ev_patch  # green

            # Col 3: Ex-vivo (dimmed) + gene dots (subsampled to ~300 max)
            n_dots = 0
            ev_dots_rgb = (ev_rgb.astype(np.float32) * GFP_DIM).astype(np.uint8)
            if gene_nd2_x is not None:
                dx = gene_nd2_x - (cx - CROP_ND2)
                dy = gene_nd2_y - (cy - CROP_ND2)
                in_patch = (dx >= 0) & (dx < CROP_ND2*2) & \
                           (dy >= 0) & (dy < CROP_ND2*2)
                pxi = dx[in_patch].astype(int)
                pyi = dy[in_patch].astype(int)
                pgn = gene_names[in_patch]
                n_dots = int(in_patch.sum())
                # Subsample to keep dots sparse and visible
                max_show = 300
                if n_dots > max_show:
                    rng = np.random.default_rng(i)
                    sel = rng.choice(n_dots, max_show, replace=False)
                    pxi, pyi, pgn = pxi[sel], pyi[sel], pgn[sel]
                for j in range(len(pxi)):
                    gc = gene_colour.get(pgn[j], (200,200,200))
                    cv2.circle(ev_dots_rgb, (int(pxi[j]), int(pyi[j])), 2, gc, -1)

            # Skip if not passing
            if not passed:
                continue
            # Track density for filtering
            all_dot_counts.append(n_dots)

            # Resize all to same display size
            p0 = cv2.resize(iv_zoom_rgb, (PATCH_DISP, PATCH_DISP), interpolation=cv2.INTER_AREA)
            p1 = cv2.resize(iv_warp_rgb, (PATCH_DISP, PATCH_DISP), interpolation=cv2.INTER_AREA)
            p2 = cv2.resize(ev_rgb, (PATCH_DISP, PATCH_DISP), interpolation=cv2.INTER_AREA)
            p3 = cv2.resize(ev_dots_rgb, (PATCH_DISP, PATCH_DISP), interpolation=cv2.INTER_AREA)

            # Label bar
            label_h = 16
            label_w = PATCH_DISP * 4 + 6
            label = np.zeros((label_h, label_w, 3), dtype=np.uint8)
            txt = f'LM#{i} {tile} z:{z_iv}->{z_nd2} err={err_um:.1f}um dots={n_dots}'
            cv2.putText(label, txt, (2, 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180,180,180), 1)

            sep = np.full((PATCH_DISP, 2, 3), 40, dtype=np.uint8)
            row_img = np.hstack([p0, sep, p1, sep, p2, sep, p3])
            card = np.vstack([label, row_img])
            cards.append((card, err_um, n_dots, i, z_iv, z_nd2))

    del warp_cache, nd2_slices

    if not cards:
        print("  No cards")
        continue

    if all_dot_counts:
        arr = np.array(all_dot_counts)
        print(f"  Dot counts: min={arr.min()} median={int(np.median(arr))} max={arr.max()} p25={int(np.percentile(arr,25))} p75={int(np.percentile(arr,75))}")

    # Filter: keep bottom 50% by dot count (least saturated)
    if cards:
        dot_counts_cards = np.array([c[2] for c in cards])
        dot_thresh = np.percentile(dot_counts_cards, 50)
        cards = [c for c in cards if c[2] <= dot_thresh]

    # Sort by dot count ascending (least saturated first)
    cards.sort(key=lambda c: c[2])
    print(f"  {len(cards)} candidates (least saturated half, <{WARP_ERR_MAX_UM}um)")

    # Build contact sheet
    card_h, card_w = cards[0][0].shape[:2]
    sep_v = np.full((card_h, 4, 3), 20, dtype=np.uint8)
    sep_h = np.full((3, card_w * N_COLS + 4 * (N_COLS - 1), 3), 20, dtype=np.uint8)

    # Header
    hdr_w = card_w * N_COLS + 4 * (N_COLS - 1)
    hdr = np.zeros((28, hdr_w, 3), dtype=np.uint8)
    col_labels = 'IV-zoomed | IV-warped | EV-nd2 | EV+genedots'
    cv2.putText(hdr, f'{tile} | {len(cards)} candidates ({DOT_MIN}-{DOT_MAX} dots, <{WARP_ERR_MAX_UM}um) | {col_labels}',
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,220,255), 1)

    grid_rows = []
    card_imgs = [c[0] for c in cards]
    for start in range(0, len(card_imgs), N_COLS):
        chunk = card_imgs[start:start+N_COLS]
        while len(chunk) < N_COLS:
            chunk.append(np.zeros_like(card_imgs[0]))
        row_img = np.hstack([c for pair in zip(chunk, [sep_v]*N_COLS)
                               for c in pair][:-1])
        grid_rows.append(row_img)

    sheet = np.vstack([hdr] + grid_rows)
    out_path = f'{OUT_DIR}/{tile}.png'
    cv2.imwrite(out_path, sheet)
    print(f"  Saved: {out_path} ({sheet.shape[1]}x{sheet.shape[0]})")

print(f'\nDone. Patches in {OUT_DIR}/')
