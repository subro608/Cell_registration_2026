#!/usr/bin/env python3
"""Pre-compute scene7 assets: 3D volume, panels, calcium warps, dot data.
Saves to scene7_assets.pkl so scene7_cell_cards.py starts fast.
"""
import numpy as np, cv2, os, glob, sys, tifffile, re, json, pickle
import pandas as pd
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
PKL_BASE = f'{BASE}/png_exports/registration_per_tile_pkl'
VAROL = f'{BASE}/jy306_varol'
PKL_MERC_DIR = f'{BASE}/merscope_exvivo '

SELECTED_CELLS = [
    ('row1_3', 6),
    ('row1_3', 9),
    ('row2_1', 5),
    ('row2_1', 0),
    ('row2_1', 4),
]

TILE_TO_REGION = {
    'row1_3': 23, 'row2_1': 17, 'row2_2': 18, 'row2_3': 19,
    'row2_4': 20, 'row2_5': 21, 'row3_1': 16, 'row3_2': 15,
    'row3_3': 14, 'row3_4': 13, 'row3_5': 12, 'row3_6': 11,
    'row4_1': 5, 'row4_2': 6, 'row4_3': 7, 'row4_4': 8,
    'row4_5': 9, 'row4_6': 10, 'row5_1': 4,
}

def make_rainbow_palette(n):
    colors = []
    for i in range(n):
        h = int(180 * i / n)
        s = 200 + int(55 * ((i % 3) / 2))
        v = 200 + int(55 * ((i % 5) / 4))
        bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(c) for c in bgr))
    return colors

N_GENE_COLORS = 550
GENE_PALETTE = make_rainbow_palette(N_GENE_COLORS)

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

def norm_u8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 50:
        return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img.astype(np.float32) - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

def crop_centered(img, cx, cy, radius):
    h, w = img.shape[:2]
    ndim = img.shape[2] if img.ndim == 3 else None
    out = np.zeros((radius * 2, radius * 2, ndim) if ndim else (radius * 2, radius * 2), dtype=img.dtype)
    sy0 = max(0, cy - radius); sy1 = min(h, cy + radius)
    sx0 = max(0, cx - radius); sx1 = min(w, cx + radius)
    dy0 = sy0 - (cy - radius); dy1 = dy0 + (sy1 - sy0)
    dx0 = sx0 - (cx - radius); dx1 = dx0 + (sx1 - sx0)
    out[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return out

CROP_SM = 130
CROP_LG = 400
PANEL_SZ = 180

needed_tiles = sorted(set(t for t, _ in SELECTED_CELLS))

# ── Load 3D stitched volume ──
print("Loading three_stacks_assets PKL...")
with open(f'{BASE}/animation/scene5b_three_stacks_assets.pkl', 'rb') as f:
    all_data = pickle.load(f)

stitched = all_data['_stitched']
SUBSAMPLE = 3
vol_sub = stitched['combined'][::SUBSAMPLE]
z_sub = stitched['z'][::SUBSAMPLE]
disp_w = stitched['width']
disp_h = stitched['height']
print(f"  Volume: {vol_sub.shape} ({len(z_sub)} slices subsampled)")

# Per-tile position data
v3_data = {k: v for k, v in all_data.items() if k != '_stitched'}
del all_data

# ── Load JY306 ──
print("Loading JY306...")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)

# ── Load calcium movie ──
print("Loading calcium movie...")
cap = cv2.VideoCapture(f'{BASE}/movie_rolling_avg_win12_step3_short.avi')
cal_frames = []
while True:
    ret, fr = cap.read()
    if not ret: break
    cal_frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr)
cap.release()
cal_movie = np.array(cal_frames, dtype=np.uint8)
n_cal = len(cal_movie)
print(f"  {n_cal} calcium frames")

M_movie_to_jy306 = np.load(f'{BASE}/animation/movie_avi_to_jy306_affine.npz')['M_affine']

# ── Process each cell ──
assets = {
    'vol_sub': vol_sub,
    'z_sub': z_sub,
    'disp_w': disp_w,
    'disp_h': disp_h,
    'cells': {},
}

for ci, (tile, local) in enumerate(SELECTED_CELLS):
    print(f"\n  Cell {ci}: {tile} #{local}")

    npz = np.load(f'{PKL_BASE}/{tile}/pkl_transform_{tile}.npz', allow_pickle=True)
    iv = npz['pcd_invivo_jy306']
    ev = npz['ev_nd2']
    M2d = npz['M2d_jy306_to_nd2']

    x_nd2 = int(round(ev[local, 0]))
    y_nd2 = int(round(ev[local, 1]))
    z_nd2 = ev[local, 2]
    z_lm = int(round(iv[local, 0]))

    # ── 3D marker position ──
    tile_v3 = v3_data[tile]
    margin_nd2 = 350
    crop_x0 = max(0, int(ev[:, 0].min() - margin_nd2))
    crop_y0 = max(0, int(ev[:, 1].min() - margin_nd2))
    crop_x1 = min(4200, int(ev[:, 0].max() + margin_nd2))
    crop_y1 = min(4200, int(ev[:, 1].max() + margin_nd2))
    scale_nd2_to_tile = tile_v3['cell_h'] / (crop_y1 - crop_y0)
    lm_tile_x = (x_nd2 - crop_x0) * scale_nd2_to_tile
    lm_tile_y = (y_nd2 - crop_y0) * scale_nd2_to_tile
    max_w = disp_w
    pad_l = (max_w - tile_v3['cell_w']) // 2
    dx = (lm_tile_x + pad_l) - disp_w / 2
    dy = lm_tile_y - disp_h / 2
    z_off = tile_v3['stitch_z_offset']
    dz_arr = tile_v3['dense_z']
    z_spacing = np.diff(dz_arr).mean() if len(dz_arr) > 1 else 1.0
    lm_z_global = z_off + z_nd2 * z_spacing
    closest_z_idx = int(np.argmin(np.abs(z_sub - lm_z_global)))
    dz = z_sub[closest_z_idx]
    print(f"    3D marker: ({dx:.0f}, {dy:.0f}, {dz:.1f})")

    # ── In-vivo warped to nd2 ──
    jy_u8 = norm_u8(jy306[z_lm])
    warped_jy = cv2.warpAffine(jy_u8, M2d, (4200, 4200))

    # ── Ex-vivo nd2 single z ──
    nd2_files = sorted(glob.glob(f'{BASE}/png_exports/registration_video/{tile}/GFP_z*.png'))
    best_z = min(len(nd2_files)-1, max(0, int(round(z_nd2))))
    nd2_slice = cv2.imread(nd2_files[best_z], cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # ── MERSCOPE dots (550 rainbow) ──
    region_id = TILE_TO_REGION.get(tile)
    dot_rel_x, dot_rel_y, dot_colors = None, None, None
    if region_id is not None:
        csv_path = f'{VAROL}/region_{region_id}_resegmentation/detected_transcripts.csv'
        mnf_path = f'{VAROL}/region_{region_id}_resegmentation/manifest.json'
        m2m_path = f'{VAROL}/region_{region_id}_resegmentation/micron_to_mosaic_pixel_transform.csv'
        pkl_merc_file = None
        for fname in os.listdir(PKL_MERC_DIR):
            if fname.endswith('.pkl'):
                mm = re.match(r'[\d_]+_reg(\d+)_transformed.*\.pkl', fname)
                if mm and int(mm.group(1)) == region_id:
                    if pkl_merc_file is None or fname > pkl_merc_file:
                        pkl_merc_file = fname
        if all(os.path.exists(p) for p in [csv_path, mnf_path, m2m_path]) and pkl_merc_file:
            m2m = np.loadtxt(m2m_path, delimiter=' ')
            scale_m, tx_m, ty_m = m2m[0, 0], m2m[0, 2], m2m[1, 2]
            with open(mnf_path) as f:
                mnf = json.load(f)
            W_mosaic = mnf['mosaic_width_pixels']
            with open(f'{PKL_MERC_DIR}/{pkl_merc_file}', 'rb') as f:
                pdat = pickle.load(f)
            R_3_inv, offset_3 = build_pkl_affine(pdat['transformations'])
            tif_size = pdat['transformed'].shape[-1]
            nd2_scale = 4200 / tif_size
            df = pd.read_csv(csv_path, usecols=['global_x', 'global_y', 'gene'])
            gx, gy = df.global_x.values, df.global_y.values
            x_mos = scale_m * gx + tx_m
            y_mos = scale_m * gy + ty_m
            merc_x = (W_mosaic - 1 - x_mos) * 0.108
            merc_y = y_mos * 0.108
            adj_y = merc_y - offset_3[1]
            adj_x = merc_x - offset_3[2]
            nd2_x_all = (R_3_inv[2, 1] * adj_y + R_3_inv[2, 2] * adj_x) * nd2_scale
            nd2_y_all = (R_3_inv[1, 1] * adj_y + R_3_inv[1, 2] * adj_x) * nd2_scale
            # Filter to CROP_LG around landmark
            in_crop = ((nd2_x_all >= x_nd2 - CROP_LG) & (nd2_x_all < x_nd2 + CROP_LG) &
                       (nd2_y_all >= y_nd2 - CROP_LG) & (nd2_y_all < y_nd2 + CROP_LG))
            dot_rel_x = (nd2_x_all[in_crop] - x_nd2).astype(np.float32)
            dot_rel_y = (nd2_y_all[in_crop] - y_nd2).astype(np.float32)
            gene_counts = Counter(df['gene'].values[in_crop])
            all_genes_sorted = [g for g, _ in gene_counts.most_common()]
            gene_to_color = {g: GENE_PALETTE[i % N_GENE_COLORS] for i, g in enumerate(all_genes_sorted)}
            dot_colors = np.array([gene_to_color[g] for g in df['gene'].values[in_crop]], dtype=np.uint8)
            print(f"    MERSCOPE: {in_crop.sum()} dots, {len(all_genes_sorted)} genes")

    # ── Build panels (iv, ev, merscope) at sm and lg crops ──
    def make_panel_set(crop_r):
        yn0 = max(0, y_nd2 - crop_r); yn1 = min(4200, y_nd2 + crop_r)
        xn0 = max(0, x_nd2 - crop_r); xn1 = min(4200, x_nd2 + crop_r)
        # IV (green)
        iv_crop = warped_jy[yn0:yn1, xn0:xn1]
        iv_r = cv2.resize(iv_crop, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
        iv_p = np.zeros((PANEL_SZ, PANEL_SZ, 3), np.uint8)
        iv_p[:, :, 1] = iv_r
        # EV (magenta)
        ev_crop = norm_u8(nd2_slice[yn0:yn1, xn0:xn1])
        ev_r = cv2.resize(ev_crop, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
        ev_p = np.zeros((PANEL_SZ, PANEL_SZ, 3), np.uint8)
        ev_p[:, :, 2] = ev_r; ev_p[:, :, 0] = ev_r
        # MERSCOPE dots only
        dot_canvas = np.zeros((crop_r * 2, crop_r * 2, 3), np.uint8)
        if dot_rel_x is not None:
            in_r = (np.abs(dot_rel_x) < crop_r) & (np.abs(dot_rel_y) < crop_r)
            px = (dot_rel_x[in_r] + crop_r).astype(int)
            py = (dot_rel_y[in_r] + crop_r).astype(int)
            c = dot_colors[in_r]
            valid = (px >= 0) & (px < crop_r*2) & (py >= 0) & (py < crop_r*2)
            dot_canvas[py[valid], px[valid]] = c[valid]
        gd_p = cv2.resize(dot_canvas, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
        return iv_p, ev_p, gd_p

    panels_sm = make_panel_set(CROP_SM)
    panels_lg = make_panel_set(CROP_LG)

    # ── Calcium warped to nd2, cropped ──
    M_m2j = np.vstack([M_movie_to_jy306, [0, 0, 1]])
    M_j2n = np.vstack([M2d, [0, 0, 1]])
    M_movie_to_nd2 = (M_j2n @ M_m2j)[:2, :]
    print(f"    Warping {n_cal} calcium frames to nd2...")
    cal_sm, cal_lg = [], []
    for fi in range(n_cal):
        warped = cv2.warpAffine(cal_movie[fi], M_movie_to_nd2, (4200, 4200), borderValue=0)
        cs = crop_centered(warped, x_nd2, y_nd2, CROP_SM)
        cal_sm.append(cv2.cvtColor(cs, cv2.COLOR_GRAY2BGR))
        cl = crop_centered(warped, x_nd2, y_nd2, CROP_LG)
        cal_lg.append(cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR))

    assets['cells'][ci] = {
        'tile': tile,
        'local': local,
        'lm_display': (dx, dy, dz),
        'iv_z': iv[local, 0],
        'panels_sm': panels_sm,
        'panels_lg': panels_lg,
        'cal_sm': cal_sm,
        'cal_lg': cal_lg,
        'dot_rel_x': dot_rel_x,
        'dot_rel_y': dot_rel_y,
        'dot_colors': dot_colors,
    }
    print(f"    Done")

del jy306, cal_movie

out_path = f'{BASE}/animation/scene7_assets.pkl'
print(f"\nSaving to {out_path}...")
with open(out_path, 'wb') as f:
    pickle.dump(assets, f)
sz = os.path.getsize(out_path) / 1e6
print(f"Done! {sz:.0f} MB")
