#!/usr/bin/env python3
"""Generate 4-panel contact sheets — ALL panels in nd2 (ex-vivo) space:
  Panel 1: Calcium (warped movie→JY306→nd2)
  Panel 2: In-vivo warped (JY306→nd2 via M2d)
  Panel 3: Ex-vivo (nd2 GFP native)
  Panel 4: GCaMP + MERSCOPE gene dots

3 selected cells: row1_3 #6, row1_3 #9, row2_1 #5
"""
import numpy as np, cv2, os, re, pickle, json
import tifffile
import pandas as pd
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/animation'

ND2_XY_UM = 0.645
CROP_ND2 = 130       # radius in nd2 pixels (same as scene7 CROP_SM)
PATCH_DISP = 300     # display size per panel
GFP_DIM = 0.25

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

SELECTED = [
    ('row2_2', 35),   # z_iv=3, z_nd2=0, 23k dots, 2.2µm err, bright=150
    ('row2_3', 34),   # z_iv=3, z_nd2=1, 19k dots, 0.9µm err, bright=202
    ('row2_5', 17),   # z_iv=3, z_nd2=0, 18k dots, 2.8µm err
    ('row3_1', 36),   # z_iv=5, z_nd2=0, 22k dots, 2.0µm err
    ('row3_4', 4),    # z_iv=9, z_nd2=0, 28k dots, 3.9µm err
    ('row3_5', 27),   # z_iv=10, z_nd2=1, 68k dots, 1.0µm err
    ('row3_6', 24),   # z_iv=10, z_nd2=0, 100k dots, 2.0µm err
    ('row3_2', 33),   # z_iv=4, z_nd2=1, 16k dots, 1.7µm err
    ('row3_3', 11),   # z_iv=9, z_nd2=0, 19k dots, 3.8µm err
    ('row2_4', 2),    # z_iv=5, z_nd2=0, 17k dots, 0.9µm err
]

def norm8(img, lo=1, hi=99.5):
    v = img[img > 0]
    if len(v) < 100:
        return np.zeros_like(img, dtype=np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

def crop_patch(img, cx, cy, r):
    h, w = img.shape[:2]
    ndim = img.shape[2] if img.ndim == 3 else None
    out = np.zeros((2*r, 2*r, ndim) if ndim else (2*r, 2*r), dtype=img.dtype)
    sx0, sy0 = max(0, cx-r), max(0, cy-r)
    sx1, sy1 = min(w, cx+r), min(h, cy+r)
    dx0, dy0 = sx0-(cx-r), sy0-(cy-r)
    if sx1 > sx0 and sy1 > sy0:
        out[dy0:dy0+(sy1-sy0), dx0:dx0+(sx1-sx0)] = img[sy0:sy1, sx0:sx1]
    return out

# ── Load in-vivo ──
print("Loading JY306...")
iv_vol = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv = iv_vol.shape[0]

# ── Load calcium movie ──
print("Loading calcium movie...")
cap = cv2.VideoCapture(f'{BASE}/movie_rolling_avg_win12_step3_short.avi')
cal_frames = []
while True:
    ret, frm = cap.read()
    if not ret: break
    cal_frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY).astype(np.float32))
cap.release()
cal_avg = np.mean(cal_frames, axis=0).astype(np.float32)

# Movie→JY306 affine
M_movie2jy = np.load(f'{BASE}/animation/movie_avi_to_jy306_affine.npz')['M_affine']

# ── MERSCOPE overlay dir ──
OVERLAY_DIR = f'{BASE}/png_exports/merscope_overlay'
EX_DIR = f'{BASE}/exvivo_merscope_combined'
tile_to_ms = {}
for fname in os.listdir(EX_DIR):
    m = re.match(r'(\d+_\d+)_merscope(\d+)\.tif', fname)
    if m:
        tile_to_ms[f'row{m.group(1)}'] = int(m.group(2))

# ── Process each cell ──
for tile, lm_idx in SELECTED:
    print(f"\n{'='*60}")
    print(f"  {tile} LM#{lm_idx}")
    print(f"{'='*60}")

    pkl_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
    tfm = np.load(pkl_path)
    ev_nd2 = tfm['ev_nd2']
    pcd_iv = tfm['pcd_invivo_jy306']
    M2d = tfm['M2d_jy306_to_nd2']

    cx_nd2 = int(round(ev_nd2[lm_idx, 0]))
    cy_nd2 = int(round(ev_nd2[lm_idx, 1]))
    z_nd2 = int(round(ev_nd2[lm_idx, 2]))
    z_iv = int(round(np.clip(pcd_iv[lm_idx, 0], 0, nz_iv - 1)))

    # ── Panel 1: Calcium warped to nd2 space ──
    # Compose: movie → JY306 → nd2
    M_m2j = np.vstack([M_movie2jy, [0, 0, 1]])
    M_j2n = np.vstack([M2d, [0, 0, 1]])
    M_movie2nd2 = (M_j2n @ M_m2j)[:2, :]
    cal_warped = cv2.warpAffine(cal_avg, M_movie2nd2, (4200, 4200),
                                 flags=cv2.INTER_LINEAR, borderValue=0)
    cal_patch = crop_patch(norm8(cal_warped), cx_nd2, cy_nd2, CROP_ND2)
    p1 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
    p1[:,:,0] = cal_patch; p1[:,:,1] = cal_patch; p1[:,:,2] = cal_patch

    # ── Panel 2: In-vivo warped to nd2 (single z) ──
    iv_slice = iv_vol[z_iv]
    iv_warped = cv2.warpAffine(norm8(iv_slice), M2d, (4200, 4200), borderValue=0)
    iv_patch = crop_patch(iv_warped, cx_nd2, cy_nd2, CROP_ND2)
    p2 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
    p2[:,:,1] = iv_patch  # green

    # ── Panel 3: Ex-vivo nd2 (single z) ──
    tile_dir = f'{BASE}/png_exports/registration_video/{tile}'
    nd2_path = f'{tile_dir}/GFP_z{z_nd2:03d}.png'
    nd2_slice = cv2.imread(nd2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    ev_patch = crop_patch(norm8(nd2_slice), cx_nd2, cy_nd2, CROP_ND2)
    p3 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
    p3[:,:,2] = ev_patch; p3[:,:,0] = ev_patch  # magenta

    # ── Panel 4: GCaMP + gene dots (from overlay PNGs) ──
    p4 = np.zeros((CROP_ND2*2, CROP_ND2*2, 3), np.uint8)
    ms_id = tile_to_ms.get(tile)
    if ms_id is not None:
        row_col = tile.replace('row', '')
        ov_path = f'{OVERLAY_DIR}/region_{ms_id}_{row_col}.png'
        if os.path.exists(ov_path):
            full_ov = cv2.imread(ov_path)
            left = full_ov[32:, :4200]
            right = full_ov[32:, 4208:]
            gc = left[:, :, 1]  # GCaMP green channel
            diff = np.clip(right.astype(np.int16) - left.astype(np.int16), 0, 255).astype(np.uint8)
            # Build: dimmed GCaMP (magenta) + gene dots
            gc_crop = crop_patch(gc, cx_nd2, cy_nd2, CROP_ND2)
            dots_crop = crop_patch(diff, cx_nd2, cy_nd2, CROP_ND2)
            p4[:,:,0] = (gc_crop.astype(np.float32) * GFP_DIM).astype(np.uint8)
            p4[:,:,2] = (gc_crop.astype(np.float32) * GFP_DIM).astype(np.uint8)
            dot_px = dots_crop.max(axis=2) > 0
            p4[dot_px] = dots_crop[dot_px]
            print(f"  Gene dots: {dot_px.sum()} pixels")
        else:
            print(f"  Overlay not found: {ov_path}")

    # ── Assemble ──
    GAP = 8
    PS = PATCH_DISP
    labels = ['CALCIUM', 'IN-VIVO WARPED', 'EX-VIVO', 'GCaMP + GENE DOTS']
    panels = [p1, p2, p3, p4]

    total_w = PS * 4 + GAP * 3
    total_h = PS + 40
    canvas = np.zeros((total_h, total_w, 3), np.uint8)

    for i, (panel, label) in enumerate(zip(panels, labels)):
        resized = cv2.resize(panel, (PS, PS), interpolation=cv2.INTER_LANCZOS4)
        if i != 0:  # brightness boost for non-calcium
            resized = np.clip(resized.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
        px = i * (PS + GAP)
        py = 25
        canvas[py:py+PS, px:px+PS] = resized
        cv2.rectangle(canvas, (px, py), (px+PS-1, py+PS-1), (100,100,100), 1)
        # Crosshair
        cr = PS // 2
        color = (0, 255, 255)
        cv2.circle(canvas, (px+cr, py+cr), 12, color, 1, cv2.LINE_AA)
        cv2.line(canvas, (px+cr-18, py+cr), (px+cr-7, py+cr), color, 1)
        cv2.line(canvas, (px+cr+7, py+cr), (px+cr+18, py+cr), color, 1)
        cv2.line(canvas, (px+cr, py+cr-18), (px+cr, py+cr-7), color, 1)
        cv2.line(canvas, (px+cr, py+cr+7), (px+cr, py+cr+18), color, 1)
        # Label
        ts = 0.5
        (tw, _), _ = cv2.getTextSize(label, FONT, ts, 1)
        cv2.putText(canvas, label, (px + (PS-tw)//2, 18), FONT, ts, WHITE, 1, cv2.LINE_AA)

    out_path = f'{OUT_DIR}/cell_nd2_{tile}_lm{lm_idx}.png'
    cv2.imwrite(out_path, canvas)
    print(f"  Saved: {out_path}")

print("\nDone!")
