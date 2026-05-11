#!/usr/bin/env python3
"""
Registration animation (v1):
  Phase 0 : Calcium movie plays in JY306 space                (90 fr, 3s)
  Phase 1 : Pause → max-projection highlight                  (30 fr, 1s)
  Phase 2 : Max-frame warps into JY306 z=3 reference          (60 fr, 2s)
  Phase 3 : Z-scan through JY306 in-vivo stack                (48 fr, 1.6s)
  Phase 4 : JY306 MIP warps onto ex-vivo tile (row2_1)        (75 fr, 2.5s)
  Phase 5 : Zoom out → full stitched MIP + all landmarks      (60 fr, 2s)
  Phase 6–8: 3 focal-plane cells → calcium / iv / ev panels  (3×120 fr)
  Total ≈ 21s at 30 fps
"""
import numpy as np
import cv2
import tifffile
import json
import os
import glob
import math
import subprocess

BASE = '/Users/neurolab/neuroinformatics/margaret'
W, H   = 1920, 1080
FPS    = 30
TMP_MP4 = f'{BASE}/png_exports/registration_animation_raw.mp4'
OUT_MP4 = f'{BASE}/png_exports/registration_animation_v1.mp4'
os.makedirs(f'{BASE}/png_exports', exist_ok=True)

IV_XY_UM  = 0.6835
IV_Z_UM   = 3.0
ND2_XY_UM = 0.645
CROP_JY   = 73       # 50 µm radius in JY306 px
SKIP_TILES = {'row3_1', 'row3_5'}

# BGR colours
GREEN   = (  80, 255,  80)
MAGENTA = ( 200,  60, 255)
CYAN    = ( 255, 220,   0)
ORANGE  = (  20, 140, 255)
WHITE   = ( 255, 255, 255)
YELLOW  = (  20, 220, 255)
LBLUE   = ( 255, 180,  80)

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
def norm_u8(img, lo=2, hi=99.5):
    v = img[img > 0] if img.ndim == 2 else img.ravel()
    v = v[v > 0]
    if len(v) < 100: return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img.astype(np.float32) - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

def ease(t): return 0.5 - 0.5 * math.cos(math.pi * t)

def blend_f(a, b, t):
    return np.clip((1-t)*a.astype(np.float32) + t*b.astype(np.float32), 0, 255).astype(np.uint8)

def lbl(frame, text, y, x, col=WHITE, scale=0.65, thick=1):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)

def to_bgr(gray, col_bgr):
    """Colourize a grayscale u8 image with a single BGR colour."""
    rgb = np.zeros(gray.shape + (3,), dtype=np.float32)
    norm = gray.astype(np.float32) / 255.0
    for c, v in enumerate(col_bgr):
        rgb[:, :, c] = norm * v
    return np.clip(rgb, 0, 255).astype(np.uint8)

def fit_square(img_u8, size):
    """Fit image into (size, size) black square, return (canvas, scale, y_off, x_off)."""
    if img_u8.ndim == 2:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    h, w = img_u8.shape[:2]
    s = size / max(h, w)
    nh, nw = int(h * s), int(w * s)
    rs = cv2.resize(img_u8, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    c = np.zeros((size, size, 3), np.uint8)
    yo, xo = (size - nh) // 2, (size - nw) // 2
    c[yo:yo+nh, xo:xo+nw] = rs
    return c, s, yo, xo

def place_center(frame, img, cy, cx):
    ih, iw = img.shape[:2]
    y0, x0 = cy - ih//2, cx - iw//2
    y1, x1 = y0+ih, x0+iw
    sy0 = max(0, -y0); sx0 = max(0, -x0)
    fy0 = max(0, y0);  fx0 = max(0, x0)
    fy1 = min(H, y1);  fx1 = min(W, x1)
    if fy1 > fy0 and fx1 > fx0:
        frame[fy0:fy1, fx0:fx1] = img[sy0:sy0+(fy1-fy0), sx0:sx0+(fx1-fx0)]

def title_bar(frame, text, sub='', alpha=1.0):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 60), (0, 0, 0), -1)
    lbl(overlay, text, 30, 30, YELLOW, 0.85, 2)
    if sub:
        lbl(overlay, sub, 52, 32, (180,180,180), 0.45, 1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
print("Loading JY306 stack…")
jy306  = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = jy306.shape  # (16,658,629)
jy306_mip_u8  = norm_u8(np.max(jy306, axis=0))
jy306_z_u8    = [norm_u8(jy306[z]) for z in range(nz_iv)]

print("Loading calcium movie (warped to JY306)…")
cap = cv2.VideoCapture(f'{BASE}/png_exports/native_invivo/movie_warped_h264.mp4')
movie_frames = []
while True:
    ret, frm = cap.read()
    if not ret: break
    movie_frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY))
cap.release()
movie_frames = np.array(movie_frames, dtype=np.uint8)   # (663, 658, 629)
n_movie = len(movie_frames)
movie_max_u8 = norm_u8(np.max(movie_frames.astype(np.float32), axis=0))
print(f"  {n_movie} frames {movie_frames.shape[1:]}")

print("Loading nd2 row2_1 tile…")
nd2_slices = []
for zi in range(12):
    p = cv2.imread(f'{BASE}/png_exports/registration_video/row2_1/GFP_z{zi:03d}.png', cv2.IMREAD_UNCHANGED)
    if p is not None: nd2_slices.append(p.astype(np.float32))
nd2_mip_u8 = norm_u8(np.max(nd2_slices, axis=0))

# row2_1 landmarks
lm27 = np.load(f'{BASE}/registration_video/landmarks_27_nd2_native.npz')
ev_nd2_27  = lm27['ev_nd2']            # (27,2) x,y in nd2 px
pcd_iv_27  = lm27['pcd_invivo_jy306']  # (27,3) z,y,x in JY306 px

# nd2→JY306 affine (existing file)
M_nd2_to_jy = np.load(f'{BASE}/registration_video/affine_nd2_to_exvivo.npy')  # (2,3)
M3 = np.vstack([M_nd2_to_jy, [0,0,1]])
M_jy_to_nd2 = np.linalg.inv(M3)[:2, :]   # JY306 → nd2

# SIFT affine: movie_flipped(512×512) → JY306(658×629)
# From memory: scale=0.881, rotation=0.045°, tx=61.7, ty=89.3
theta = math.radians(0.045)
s = 0.881
M_sift = np.array([
    [s*math.cos(theta), -s*math.sin(theta), 61.7],
    [s*math.sin(theta),  s*math.cos(theta), 89.3]], dtype=np.float64)
jy306_z3_u8 = jy306_z_u8[3]

print("Loading patch strip + cell info…")
patch_strip = cv2.imread(f'{BASE}/3d_viewer/patch_strip_v4.png')
with open(f'{BASE}/3d_viewer/cell_info_v4.json') as f:
    raw_ci = json.load(f)
cell_info = [json.loads(x) if isinstance(x,str) else x for x in raw_ci]
PATCH_SZ = 80

# Build ordered (y,x) per landmark (same iteration as viewer build)
all_lm_jy = []     # (y,x) in JY306
tile_lm_map = {}
for lf in sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz')):
    tile = os.path.basename(lf).replace('landmarks_nd2_native_','').replace('.npz','')
    if tile not in SKIP_TILES:
        tile_lm_map[tile] = lf
legacy = f'{BASE}/registration_video/landmarks_27_nd2_native.npz'
if os.path.exists(legacy) and 'row2_1' not in tile_lm_map:
    tile_lm_map['row2_1'] = legacy

for tile in sorted(tile_lm_map):
    d = np.load(tile_lm_map[tile])
    for row in d['pcd_invivo_jy306']:
        all_lm_jy.append((int(round(row[1])), int(round(row[2]))))

# Pick 3 focal-plane cells (z_iv ≈ 3 ± 2)
focal_cells = [ci for ci,info in enumerate(cell_info)
               if 1 <= info[3] <= 5][:3]
print(f"  {len(all_lm_jy)} landmarks loaded, focal cells: {focal_cells}")

print("Building stitched ex-vivo MIP (downsampled)…")
EX_TIFF = f'{BASE}/registration_video/stitched/stitched_gfp_fullres_v5_1um_isotropic.tif'
with tifffile.TiffFile(EX_TIFF) as tif:
    n_pages = len(tif.pages)
    step = max(1, n_pages // 64)
    stitched_mip = np.max(
        [tif.pages[zi].asarray().astype(np.float32) for zi in range(0, n_pages, step)], axis=0)
stitched_mip_u8 = norm_u8(stitched_mip)
sm_h, sm_w = stitched_mip_u8.shape
del stitched_mip
print(f"  Stitched MIP: {sm_h}×{sm_w}")

# All landmark positions on stitched MIP (DS_EX=4)
all_st_y, all_st_x = [], []
for lf in sorted(glob.glob(f'{BASE}/registration_video/landmarks_stitched_v5_*.npz')):
    tile = os.path.basename(lf).replace('landmarks_stitched_v5_','').replace('.npz','')
    if tile in SKIP_TILES: continue
    sc = np.load(lf)['stitched_coords']   # (N,3) [z_um,y_um,x_um]
    for row in sc:
        py, px = int(row[1]/4), int(row[2]/4)
        if 0 <= py < sm_h and 0 <= px < sm_w:
            all_st_y.append(py); all_st_x.append(px)
all_st_y, all_st_x = np.array(all_st_y), np.array(all_st_x)

# Focal-cell stitched positions
focal_st_yx = []
for ci in focal_cells:
    # find tile+index in same sorted order
    offset = 0
    for tile in sorted(tile_lm_map):
        d = np.load(tile_lm_map[tile])
        n = len(d['pcd_invivo_jy306'])
        if offset + n > ci:
            local_i = ci - offset
            sv = np.load(f'{BASE}/registration_video/landmarks_stitched_v5_{tile}.npz')
            sc = sv['stitched_coords'][local_i]
            focal_st_yx.append((int(sc[1]/4), int(sc[2]/4)))
            break
        offset += n
    else:
        focal_st_yx.append((sm_h//2, sm_w//2))

print("Data ready. Opening VideoWriter…")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vw = cv2.VideoWriter(TMP_MP4, fourcc, FPS, (W, H))

# ═══════════════════════════════════════════════════════════════════════
# PHASE 0  Calcium movie plays  (90 frames)
# ═══════════════════════════════════════════════════════════════════════
SZ = 680   # display size for main image
print("Phase 0: calcium movie…")
for fi in range(90):
    frame = np.zeros((H, W, 3), np.uint8)
    idx = fi * 7 % n_movie      # play at 7× speed to show dynamics
    img = movie_frames[idx]
    sq, _, _, _ = fit_square(to_bgr(img, MAGENTA), SZ)
    place_center(frame, sq, H//2, W//2)
    alpha_fade = min(1.0, fi/15)
    title_bar(frame, '2-Photon Calcium Imaging  |  In Vivo',
              f'JY306 hippocampus — {n_movie} frames — warped to JY306 space', alpha_fade)
    lbl(frame, f'Frame {idx}/{n_movie-1}', H-25, W-180, (120,120,120), 0.4)
    vw.write(frame)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1  Pause → max-projection  (30 frames)
# ═══════════════════════════════════════════════════════════════════════
print("Phase 1: pause + max projection…")
last_movie = movie_frames[(89*7) % n_movie]
for fi in range(30):
    t = ease(fi / 29)
    frame = np.zeros((H, W, 3), np.uint8)
    img_blend = blend_f(last_movie, movie_max_u8, t)
    glow_boost = 1.0 + 0.4 * t     # slight brightness boost
    sq, _, _, _ = fit_square(to_bgr(img_blend, MAGENTA), SZ)
    sq = np.clip(sq.astype(np.float32) * glow_boost, 0, 255).astype(np.uint8)
    place_center(frame, sq, H//2, W//2)
    title_bar(frame, 'Max Projection  |  Temporal Max Over 663 Frames', alpha=t)
    # draw "PAUSE" text fading out
    if fi < 15:
        a = 1 - fi/15
        lbl(frame, '[ PAUSED ]', H//2 - SZ//2 - 30, W//2 - 60, tuple(int(v*a) for v in YELLOW), 0.6)
    vw.write(frame)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2  Max-frame warp → JY306 z=3  (60 frames)
# ═══════════════════════════════════════════════════════════════════════
# The calcium movie is already in JY306 space (warped). So we just
# cross-fade the max-projection into JY306 z=3, showing how they align.
print("Phase 2: warp → JY306 z=3 reference…")
jy_z3_bgr = to_bgr(jy306_z3_u8, GREEN)

for fi in range(60):
    t = ease(fi / 59)
    frame = np.zeros((H, W, 3), np.uint8)

    # Composited: movie_max (magenta) + jy306_z3 (green)
    # align widths (movie=628, JY306=629) — crop to common size
    mh = min(movie_max_u8.shape[0], jy306_z3_u8.shape[0])
    mw = min(movie_max_u8.shape[1], jy306_z3_u8.shape[1])
    mv_col  = to_bgr(movie_max_u8[:mh, :mw], MAGENTA).astype(np.float32) * (1-t*0.7)
    jy_col  = to_bgr(jy306_z3_u8[:mh, :mw],  GREEN).astype(np.float32) * t
    comp = np.clip(mv_col + jy_col, 0, 255).astype(np.uint8)
    sq, scale, yo, xo = fit_square(comp, SZ)
    place_center(frame, sq, H//2, W//2)

    # Draw landmark dots at JY306 positions (fade in)
    if t > 0.3:
        a_lm = min(1.0, (t-0.3)/0.4)
        cy_sq, cx_sq = H//2 - SZ//2, W//2 - SZ//2   # top-left corner of square
        for i in range(len(pcd_iv_27)):
            ry = int(pcd_iv_27[i,1] * scale) + yo + cy_sq
            rx = int(pcd_iv_27[i,2] * scale) + xo + cx_sq
            col = tuple(int(v*a_lm) for v in CYAN)
            cv2.circle(frame, (rx, ry), 5, col, -1, cv2.LINE_AA)

    title_bar(frame, 'Aligning Calcium Movie → In-Vivo Stack  (JY306 z=3)',
              'Magenta: calcium max-projection  |  Green: two-photon z=3')
    lbl(frame, f'{int(t*100)}% aligned', H-30, W-160, (160,160,160), 0.5)
    vw.write(frame)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3  Z-scan through JY306 in-vivo stack  (48 frames)
# ═══════════════════════════════════════════════════════════════════════
print("Phase 3: JY306 z-scan…")
for fi in range(48):
    # 3 full z-cycles in 48 frames → 16 frames/cycle
    z = fi % nz_iv
    frame = np.zeros((H, W, 3), np.uint8)
    sq, scale, yo, xo = fit_square(to_bgr(jy306_z_u8[z], GREEN), SZ)
    place_center(frame, sq, H//2, W//2)
    # Draw all 27 landmarks at z ± 1
    cx0, cy0 = W//2 - SZ//2, H//2 - SZ//2
    for i in range(len(pcd_iv_27)):
        dz = abs(pcd_iv_27[i,0] - z)
        if dz > 2: continue
        alpha_lm = max(0.2, 1.0 - dz*0.4)
        ry = int(pcd_iv_27[i,1] * scale) + yo + cy0
        rx = int(pcd_iv_27[i,2] * scale) + xo + cx0
        col = tuple(int(v*alpha_lm) for v in CYAN)
        cv2.circle(frame, (rx, ry), 6, col, -1, cv2.LINE_AA)
        cv2.circle(frame, (rx, ry), 6, WHITE, 1,  cv2.LINE_AA)
    title_bar(frame, 'In-Vivo Stack  (JY306)',
              f'z = {z:2d} / {nz_iv-1}  |  0.68 µm/px XY  |  3 µm z-step  |  27 matched landmarks')
    # Z-depth bar on right
    bar_x, bar_y0, bar_h2 = W-80, H//2-SZ//2+20, SZ-40
    cv2.rectangle(frame, (bar_x, bar_y0), (bar_x+12, bar_y0+bar_h2), (60,60,60), -1)
    zp = int(z / (nz_iv-1) * bar_h2)
    cv2.rectangle(frame, (bar_x, bar_y0), (bar_x+12, bar_y0+zp), GREEN, -1)
    lbl(frame, f'z={z}', bar_y0+bar_h2+15, bar_x-5, (160,160,160), 0.4)
    vw.write(frame)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 4  JY306 → ex-vivo tile (row2_1)  (75 frames)
# ═══════════════════════════════════════════════════════════════════════
# Layout: JY306 MIP left | ex-vivo tile MIP right
# Animate: draw matching lines tile by tile, then warp overlay
print("Phase 4: JY306 → ex-vivo tile…")
SZ4 = 480
jy_sq, jy_sc, jy_yo, jy_xo = fit_square(to_bgr(jy306_mip_u8, GREEN), SZ4)
nd2_sq, nd2_sc, nd2_yo, nd2_xo = fit_square(to_bgr(nd2_mip_u8, (80,220,80)), SZ4)

jy_left_cx, jy_left_cy  = W//4,     H//2
nd2_right_cx, nd2_right_cy = 3*W//4, H//2

# Warp JY306 MIP into nd2 space for cross-fade
jy_mip_float = jy306_mip_u8.astype(np.float32)
jy_warped_in_nd2 = cv2.warpAffine(jy_mip_float, M_jy_to_nd2, (4200, 4200),
                                    flags=cv2.INTER_LINEAR, borderValue=0)
jy_warped_nd2_u8 = norm_u8(jy_warped_in_nd2[:nd2_mip_u8.shape[0], :nd2_mip_u8.shape[1]])

for fi in range(75):
    t_global = fi / 74
    frame = np.zeros((H, W, 3), np.uint8)

    # Phase 4a (0-25 frames): show both images side by side, draw lines
    # Phase 4b (25-55 frames): overlay warp animation
    # Phase 4c (55-75 frames): hold on overlay

    # Draw left (JY306)
    place_center(frame, jy_sq, jy_left_cy, jy_left_cx)
    lbl(frame, 'In-Vivo (JY306)',  jy_left_cy + SZ4//2 + 22, jy_left_cx - 80, GREEN, 0.55)
    lbl(frame, '(658×629 px)', jy_left_cy + SZ4//2 + 42, jy_left_cx - 55, (130,130,130), 0.4)

    if t_global < 25/74:
        # Draw nd2 tile on right
        t_lines = t_global / (25/74)
        place_center(frame, nd2_sq, nd2_right_cy, nd2_right_cx)
        lbl(frame, 'Ex-Vivo Tile (row2_1)', nd2_right_cy+SZ4//2+22, nd2_right_cx-100, (80,220,80), 0.55)
        lbl(frame, '(4200×4200 px 0.645 µm/px)', nd2_right_cy+SZ4//2+42, nd2_right_cx-120, (130,130,130), 0.4)

        # Fade-in landmark lines one by one
        n_shown = int(t_lines * 27)
        jy_top = jy_left_cy - SZ4//2;  jy_lft = jy_left_cx - SZ4//2
        nd2_top = nd2_right_cy - SZ4//2; nd2_lft = nd2_right_cx - SZ4//2
        for i in range(n_shown):
            # JY306 point
            ry_jy = int(pcd_iv_27[i,1] * jy_sc) + jy_yo + jy_top
            rx_jy = int(pcd_iv_27[i,2] * jy_sc) + jy_xo + jy_lft
            # nd2 point
            rx_nd2 = int(ev_nd2_27[i,0] * nd2_sc) + nd2_xo + nd2_lft
            ry_nd2 = int(ev_nd2_27[i,1] * nd2_sc) + nd2_yo + nd2_top
            cv2.circle(frame, (rx_jy,  ry_jy),  5, CYAN,  -1, cv2.LINE_AA)
            cv2.circle(frame, (rx_nd2, ry_nd2), 5, ORANGE, -1, cv2.LINE_AA)
            cv2.line(frame, (rx_jy, ry_jy), (rx_nd2, ry_nd2), (60,60,60), 1, cv2.LINE_AA)

    else:
        # Cross-fade JY306 → warped overlay on nd2 tile
        t_warp = ease(min(1.0, (t_global - 25/74) / (30/74)))
        overlay_col = blend_f(to_bgr(nd2_mip_u8, (80,220,80)),
                              to_bgr(jy_warped_nd2_u8, MAGENTA), t_warp * 0.55)
        ov_sq, ov_sc, ov_yo, ov_xo = fit_square(overlay_col, SZ4)
        place_center(frame, ov_sq, nd2_right_cy, nd2_right_cx)
        lbl(frame, 'In-Vivo warped → Ex-Vivo tile', nd2_right_cy+SZ4//2+22, nd2_right_cx-120,
            MAGENTA if t_warp > 0.5 else (80,220,80), 0.55)

    title_bar(frame, 'JY306 In-Vivo  →  Ex-Vivo Confocal Tile (row2_1)',
              '3D affine registration  |  0.645 µm/px  |  27 matched landmarks')
    vw.write(frame)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 5  Zoom out to full stitched MIP  (60 frames)
# ═══════════════════════════════════════════════════════════════════════
print("Phase 5: zoom out to stitched…")
sm_bgr = to_bgr(stitched_mip_u8, (80, 220, 80))
sm_sq, sm_scale, sm_yo, sm_xo = fit_square(sm_bgr, SZ)
sm_top = H//2 - SZ//2;  sm_lft = W//2 - SZ//2

# Compute row2_1 tile bounding box in stitched space
sv21  = np.load(f'{BASE}/registration_video/landmarks_stitched_v5_row2_1.npz')
sc21  = sv21['stitched_coords']   # (N,3) z,y,x in µm
tile_y_min = int(sc21[:,1].min() / 4 * sm_scale) + sm_yo + sm_top
tile_y_max = int(sc21[:,1].max() / 4 * sm_scale) + sm_yo + sm_top
tile_x_min = int(sc21[:,2].min() / 4 * sm_scale) + sm_xo + sm_lft
tile_x_max = int(sc21[:,2].max() / 4 * sm_scale) + sm_xo + sm_lft

for fi in range(60):
    t = ease(fi / 59)
    frame = np.zeros((H, W, 3), np.uint8)
    place_center(frame, sm_sq, H//2, W//2)

    # Fade in all landmark dots
    if t > 0.2:
        a_lm = min(1.0, (t-0.2)/0.5)
        for py, px in zip(all_st_y, all_st_x):
            ry = int(py * sm_scale) + sm_yo + sm_top
            rx = int(px * sm_scale) + sm_xo + sm_lft
            col = tuple(int(v*a_lm) for v in CYAN)
            cv2.circle(frame, (rx, ry), 2, col, -1)

    # Fade in row2_1 tile bounding box
    if t > 0.4:
        a_box = min(1.0, (t-0.4)/0.4)
        col_box = tuple(int(v*a_box) for v in ORANGE)
        cv2.rectangle(frame, (tile_x_min-4, tile_y_min-4),
                      (tile_x_max+4, tile_y_max+4), col_box, 2, cv2.LINE_AA)
        lbl(frame, 'row2_1', tile_y_min-12, tile_x_min, col_box, 0.4)

    n_lm = int(len(all_st_y) * min(1.0, t*2))
    title_bar(frame, 'Full Stitched Ex-Vivo  |  All Tiles + Landmarks',
              f'{len(all_st_y)} total landmarks across 17 tiles  |  1 µm isotropic  |  {n_lm} shown')
    vw.write(frame)

# ═══════════════════════════════════════════════════════════════════════
# PHASE 6–8  Per-cell panels  (3 × 120 frames)
# ═══════════════════════════════════════════════════════════════════════
PANEL_SZ = 200
PANEL_Y  = H - PANEL_SZ - 55
GAP      = 24
PX0      = (W - 3*PANEL_SZ - 2*GAP) // 2   # left edge of panel row
MAIN_SZ  = H - PANEL_SZ - 110              # main image area height

def draw_panels(frame, calcium_frame_gray, piv_patch, ev_patch, alpha=1.0):
    """Bottom 3-panel: calcium | in-vivo | ex-vivo."""
    panels = [
        (to_bgr(calcium_frame_gray if calcium_frame_gray is not None else
                np.zeros((PANEL_SZ, PANEL_SZ), np.uint8), MAGENTA), MAGENTA, '2P Calcium'),
        (cv2.resize(piv_patch, (PANEL_SZ, PANEL_SZ)), GREEN,   'In-Vivo'),
        (cv2.resize(ev_patch,  (PANEL_SZ, PANEL_SZ)), (80,220,80), 'Ex-Vivo'),
    ]
    for i, (img, col, name) in enumerate(panels):
        x0 = PX0 + i * (PANEL_SZ + GAP)
        y0 = PANEL_Y
        if alpha < 1.0:
            overlay = frame.copy()
            overlay[y0:y0+PANEL_SZ, x0:x0+PANEL_SZ] = img
            cv2.rectangle(overlay, (x0,y0),(x0+PANEL_SZ-1,y0+PANEL_SZ-1), col, 2)
            lbl(overlay, name, y0+PANEL_SZ+18, x0+4, col, 0.45, 1)
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        else:
            frame[y0:y0+PANEL_SZ, x0:x0+PANEL_SZ] = img
            cv2.rectangle(frame, (x0,y0),(x0+PANEL_SZ-1,y0+PANEL_SZ-1), col, 2)
            lbl(frame, name, y0+PANEL_SZ+18, x0+4, col, 0.45, 1)

print("Phases 6-8: cell panels…")
for cell_idx, ci in enumerate(focal_cells):
    info = cell_info[ci]             # [z_nd2, ez_lo, ez_hi, z_iv, ivz_lo, ivz_hi]
    lm_y, lm_x = all_lm_jy[ci]     # JY306 (y,x)
    st_y, st_x  = focal_st_yx[cell_idx]   # stitched MIP pixel coords

    # Pre-crop patches from patch strip
    ev_patch  = cv2.resize(patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ, 0:PATCH_SZ],
                            (PANEL_SZ, PANEL_SZ))
    piv_patch = cv2.resize(patch_strip[ci*PATCH_SZ:(ci+1)*PATCH_SZ, PATCH_SZ*4:PATCH_SZ*5],
                            (PANEL_SZ, PANEL_SZ))

    # Zoomed stitched MIP around this cell
    pad = 120
    zoom_y0 = max(0, st_y - pad); zoom_y1 = min(sm_h, st_y + pad)
    zoom_x0 = max(0, st_x - pad); zoom_x1 = min(sm_w, st_x + pad)
    cell_crop_sm = sm_bgr[zoom_y0:zoom_y1, zoom_x0:zoom_x1].copy()
    cv2.circle(cell_crop_sm,
               (st_x - zoom_x0, st_y - zoom_y0), 8, YELLOW, 2, cv2.LINE_AA)

    for fi in range(120):
        frame = np.zeros((H, W, 3), np.uint8)

        # ── main area: stitched MIP (upper portion)
        main_top = 70; main_h = PANEL_Y - main_top - 20
        main_w   = min(int(main_h * sm_w / sm_h), W - 40)

        t_zoom = ease(min(1.0, fi / 20))         # zoom in first 20 fr

        # Blend full stitched → zoomed cell crop
        full_rs  = cv2.resize(sm_bgr,       (main_w, main_h))
        cell_rs  = cv2.resize(cell_crop_sm, (main_w, main_h))
        main_img = blend_f(full_rs, cell_rs, t_zoom)

        # Draw all landmarks on main
        for py_s, px_s in zip(all_st_y, all_st_x):
            rx = int((px_s - (zoom_x0 if t_zoom > 0.5 else 0)) /
                     ((zoom_x1-zoom_x0) if t_zoom > 0.5 else sm_w) * main_w)
            ry = int((py_s - (zoom_y0 if t_zoom > 0.5 else 0)) /
                     ((zoom_y1-zoom_y0) if t_zoom > 0.5 else sm_h) * main_h)
            if 0 <= rx < main_w and 0 <= ry < main_h:
                cv2.circle(main_img, (rx, ry), 2, CYAN, -1)

        # Highlight this cell's dot
        if t_zoom > 0.3:
            cx_dot = main_w // 2
            cy_dot = main_h // 2
            pulse = 8 + int(4 * math.sin(fi * 0.3))
            cv2.circle(main_img, (cx_dot, cy_dot), pulse, YELLOW, 2, cv2.LINE_AA)
            cv2.circle(main_img, (cx_dot, cy_dot), 5, YELLOW, -1, cv2.LINE_AA)

        x0_main = (W - main_w) // 2
        frame[main_top:main_top+main_h, x0_main:x0_main+main_w] = main_img

        # ── calcium crop from warped movie (animated playback)
        movie_fi   = fi % n_movie
        movie_crop_raw = movie_frames[movie_fi]
        y0c = max(0, lm_y - PANEL_SZ//2); y1c = min(ny_iv, lm_y + PANEL_SZ//2)
        x0c = max(0, lm_x - PANEL_SZ//2); x1c = min(nx_iv, lm_x + PANEL_SZ//2)
        crop = movie_crop_raw[y0c:y1c, x0c:x1c]
        crop_sq = np.zeros((PANEL_SZ, PANEL_SZ), np.uint8)
        crop_sq[:crop.shape[0], :crop.shape[1]] = crop
        crop_u8 = norm_u8(crop_sq.astype(np.float32))

        # ── panels (fade in over first 15 frames)
        panel_alpha = ease(min(1.0, fi / 15))
        draw_panels(frame, crop_u8, piv_patch, ev_patch, panel_alpha)

        title_bar(frame,
                  f'Cell {cell_idx+1} of {len(focal_cells)}  —  Landmark #{ci}',
                  f'JY306 z={info[3]}  |  nd2 z={info[0]}  |  '
                  f'Magenta=calcium movie  Green=in-vivo MIP  Lime=ex-vivo MIP')

        # frame counter for calcium panel
        lbl(frame, f'▶ {movie_fi}/{n_movie-1}', PANEL_Y + PANEL_SZ + 38,
            PX0 + 4, MAGENTA, 0.38, 1)

        vw.write(frame)

    print(f"  Cell {cell_idx+1} done")

vw.release()
print(f"\nRaw MP4 written: {TMP_MP4}")

# ─── Re-encode to H.264 (browser-compatible)
print("Re-encoding to H.264…")
cmd = ['ffmpeg', '-y', '-i', TMP_MP4,
       '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
       '-preset', 'fast', OUT_MP4]
subprocess.run(cmd, check=True)
print(f"\n✓ Done: {OUT_MP4}")
print(f"  Duration: ~{(90+30+60+48+75+60+3*120)/FPS:.1f}s  ({W}×{H} @ {FPS}fps)")
