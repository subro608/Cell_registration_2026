#!/usr/bin/env python3
"""Generate 4-panel candidate cell contact sheets:
  Panel 1: In-vivo WARPED (JY306 warped to nd2 space via M2d)
  Panel 2: Ex-vivo (nd2 GFP native)
  Panel 3: Calcium (movie frame crop at cell location)
  Panel 4: GCaMP + MERSCOPE gene dots

For 5 candidate cells from diverse tiles.
"""
import numpy as np, cv2, os, glob, pickle, json, re
import tifffile
import pandas as pd
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/animation'

IV_XY_UM = 0.6835
ND2_XY_UM = 0.645
CROP_ND2 = 120       # radius in nd2 pixels (~77µm)
PATCH_DISP = 300     # display size per panel
GFP_DIM = 0.25       # dim GFP behind gene dots

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

# ── Candidates: (tile, landmark_index) — best "good RNA" cells ──
CANDIDATES = [
    # Batch 3: more diverse landmarks
    ('row1_3', 3),
    ('row1_3', 9),
    ('row2_1', 5),
    ('row2_1', 15),
    ('row2_2', 10),
    ('row2_3', 12),
    ('row2_4', 15),
    ('row2_5', 8),
    ('row3_1', 10),
    ('row3_2', 15),
    ('row3_3', 5),
    ('row3_4', 10),
    ('row3_5', 10),
    ('row3_6', 10),
    ('row4_1', 15),
]

# ── Load in-vivo volume ──
print("Loading JY306 in-vivo stack...")
iv_vol = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = iv_vol.shape
print(f"  Shape: {iv_vol.shape}")

# ── Load calcium movie ──
print("Loading calcium movie...")
cap = cv2.VideoCapture(f'{BASE}/movie_rolling_avg_win12_step3_short.avi')
cal_frames = []
while True:
    ret, frm = cap.read()
    if not ret: break
    cal_frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY))
cap.release()
# Average frame for contact sheet
cal_avg = np.mean(cal_frames, axis=0).astype(np.float32)
print(f"  {len(cal_frames)} frames, using average")

# Movie → JY306 affine
M_movie2jy = np.load(f'{BASE}/animation/movie_avi_to_jy306_affine.npz')['M_affine']  # 2x3
# Invert: JY306 → movie
M_full = np.eye(3)
M_full[:2, :] = M_movie2jy
M_jy2movie = np.linalg.inv(M_full)[:2, :]
print(f"  Movie→JY306 affine loaded, best_z=3")

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

# ── Process each candidate ──
for tile, best_idx in CANDIDATES:
    print(f"\n{'='*60}")
    print(f"  {tile} LM#{best_idx}")
    print(f"{'='*60}")

    # Load landmarks
    pkl_tfm_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
    tfm = np.load(pkl_tfm_path)
    ev_nd2 = tfm['ev_nd2']        # (N, 3) x,y,z in nd2 space
    pcd_iv = tfm['pcd_invivo_jy306']  # (N, 3) z,y,x in JY306 space
    M2d = tfm['M2d_jy306_to_nd2']    # 2x3 affine
    N_LM = len(ev_nd2)

    # Compute warp errors
    iv_nd2 = tfm['iv_nd2'] if 'iv_nd2' in tfm else None
    if iv_nd2 is not None:
        pkl_dist_um = np.sqrt((iv_nd2[:,0]-ev_nd2[:,0])**2 + (iv_nd2[:,1]-ev_nd2[:,1])**2) * ND2_XY_UM
    else:
        pkl_dist_um = np.zeros(N_LM)

    print(f"  err={pkl_dist_um[best_idx]:.1f}µm")

    cx_nd2 = int(round(ev_nd2[best_idx, 0]))
    cy_nd2 = int(round(ev_nd2[best_idx, 1]))
    z_nd2 = int(round(ev_nd2[best_idx, 2]))
    z_iv = int(round(np.clip(pcd_iv[best_idx, 0], 0, nz_iv - 1)))
    y_iv = int(round(pcd_iv[best_idx, 1]))
    x_iv = int(round(pcd_iv[best_idx, 2]))

    # Load nd2 GFP slices for this tile
    tile_dir = f'{BASE}/png_exports/registration_video/{tile}'
    nd2_path = f'{tile_dir}/GFP_z{z_nd2:03d}.png'
    nd2_slice = cv2.imread(nd2_path, cv2.IMREAD_GRAYSCALE)
    if nd2_slice is None:
        print(f"  WARNING: {nd2_path} not found, skipping")
        continue
    nd2_slice = nd2_slice.astype(np.float32)

    # ── Panel 1: In-vivo WARPED to nd2 space ──
    iv_slice = iv_vol[z_iv]
    iv_warped = cv2.warpAffine(iv_slice, M2d, (4200, 4200),
                                flags=cv2.INTER_LINEAR, borderValue=0)
    iw_patch = crop_patch(norm8(iv_warped), cx_nd2, cy_nd2, CROP_ND2)
    p1 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
    p1[:,:,1] = iw_patch  # green

    # ── Panel 2: Ex-vivo (nd2 GFP) ──
    ev_patch = crop_patch(norm8(nd2_slice), cx_nd2, cy_nd2, CROP_ND2)
    p2 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
    p2[:,:,2] = ev_patch  # magenta: R channel
    p2[:,:,0] = ev_patch  # magenta: B channel

    # ── Panel 3: Calcium ──
    # Map JY306 coords to movie space (movie is at z=3, flip YX)
    # Movie pixel at (mx, my) maps to JY306 at M_movie2jy @ [mx, my, 1]
    # Inverse: JY306 (x,y) → movie (mx, my)
    jy_pt = np.array([x_iv, y_iv, 1.0])
    movie_pt = M_jy2movie @ jy_pt
    mx, my = movie_pt[0], movie_pt[1]
    # Calcium crop radius (movie pixels, ~50µm)
    CAL_CROP = 40
    cal_patch = crop_patch(norm8(cal_avg), int(mx), int(my), CAL_CROP)
    cal_patch_r = cv2.resize(cal_patch, (CROP_ND2*2, CROP_ND2*2), interpolation=cv2.INTER_LANCZOS4)
    p3 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
    p3[:,:,0] = cal_patch_r  # grayscale in all channels
    p3[:,:,1] = cal_patch_r
    p3[:,:,2] = cal_patch_r

    # ── Panel 4: GCaMP + gene dots ──
    p4 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
    # Dim GCaMP background
    p4[:,:,1] = (ev_patch.astype(np.float32) * GFP_DIM).astype(np.uint8)

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

                # Dots in patch
                dx = gene_nd2_x - (cx_nd2 - CROP_ND2)
                dy = gene_nd2_y - (cy_nd2 - CROP_ND2)
                in_patch = (dx >= 0) & (dx < CROP_ND2*2) & (dy >= 0) & (dy < CROP_ND2*2)
                pxi = dx[in_patch].astype(int)
                pyi = dy[in_patch].astype(int)
                pgn = gene_names[in_patch]
                n_dots = int(in_patch.sum())

                max_show = 500
                if n_dots > max_show:
                    rng = np.random.default_rng(42)
                    sel = rng.choice(n_dots, max_show, replace=False)
                    pxi, pyi, pgn = pxi[sel], pyi[sel], pgn[sel]

                for j in range(len(pxi)):
                    gc_col = gene_colour.get(pgn[j], (200,200,200))
                    cv2.circle(p4, (int(pxi[j]), int(pyi[j])), 2, gc_col, -1)
                print(f"  {n_dots} gene dots in patch")
            except Exception as e:
                print(f"  Gene dot error: {e}")
        else:
            print(f"  No MERFISH data for region {ms_id}")
    else:
        print(f"  No MERSCOPE mapping for {tile}")

    # ── Assemble 4-panel contact sheet ──
    GAP = 8
    PS = PATCH_DISP
    labels = ['IN-VIVO WARPED', 'EX-VIVO', 'CALCIUM', 'GCaMP + GENE DOTS']
    panels_raw = [p1, p2, p3, p4]

    total_w = PS * 4 + GAP * 3
    total_h = PS + 40  # room for labels
    canvas = np.zeros((total_h, total_w, 3), np.uint8)

    for i, (panel, label) in enumerate(zip(panels_raw, labels)):
        resized = cv2.resize(panel, (PS, PS), interpolation=cv2.INTER_LANCZOS4)
        # Brightness boost for panels 1,2,4
        if i != 2:  # not calcium
            resized = np.clip(resized.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)

        px = i * (PS + GAP)
        py = 25
        canvas[py:py+PS, px:px+PS] = resized

        # Border
        cv2.rectangle(canvas, (px, py), (px+PS-1, py+PS-1), (100, 100, 100), 1)

        # Crosshair at center
        cr = PS // 2
        color = (0, 255, 255)
        cv2.circle(canvas, (px + cr, py + cr), 12, color, 1, cv2.LINE_AA)
        cv2.line(canvas, (px+cr-18, py+cr), (px+cr-7, py+cr), color, 1)
        cv2.line(canvas, (px+cr+7, py+cr), (px+cr+18, py+cr), color, 1)
        cv2.line(canvas, (px+cr, py+cr-18), (px+cr, py+cr-7), color, 1)
        cv2.line(canvas, (px+cr, py+cr+7), (px+cr, py+cr+18), color, 1)

        # Label
        ts = 0.5
        (tw, _), _ = cv2.getTextSize(label, FONT, ts, 1)
        cv2.putText(canvas, label, (px + (PS - tw) // 2, 18), FONT, ts, WHITE, 1, cv2.LINE_AA)

    # Tile info
    info = f'{tile.upper().replace("_"," ")}  |  LM#{best_idx}  z_iv={z_iv}  z_nd2={z_nd2}  err={pkl_dist_um[best_idx]:.1f}um'
    ts = 0.45
    (tw, _), _ = cv2.getTextSize(info, FONT, ts, 1)
    # put at bottom-right
    # Actually just save as filename info

    out_path = f'{OUT_DIR}/candidate_rna_{tile}_lm{best_idx}.png'
    cv2.imwrite(out_path, canvas)
    print(f"  Saved: {out_path}")

print("\nDone!")
