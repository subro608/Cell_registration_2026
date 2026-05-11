"""
Scene 1+2: Calcium movie → pause → warp deformation to JY306 z=3 → overlay.

1920×1080, 24fps, grayscale. Matches v4 animation style.

Step 1: SIFT-match flipped movie max-proj to JY306 z=3 → get affine M
Step 2: Animate the warp by interpolating identity → M at t=0→1
Step 3: Overlay red (warped calcium) + green (confocal z=3)

Timeline (at 24fps):
  0.0 – 5.0s  : Calcium movie playback, native flipped, /max normalization (120 fr)
  5.0 – 6.0s  : Freeze on max-proj (24 fr)
  6.0 – 9.0s  : Warp deformation: identity → affine M (72 fr, slow dramatic)
  9.0 – 11.0s : Overlay red+green hold (48 fr)
"""

import numpy as np, cv2, tifffile, math, subprocess, os

BASE = '/Users/neurolab/neuroinformatics/margaret'
TMP  = f'{BASE}/animation/scene1_2_raw.mp4'
OUT  = f'{BASE}/animation/scene1_2_h264.mp4'

W, H = 1920, 1080
FPS  = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GRAY  = (160, 160, 160)

from text_utils import put_text_mixed, text_width_mixed

# ── helpers (from v4) ──
def norm_u8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 50:
        return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img.astype(np.float32) - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

def fit_into(img, tw, th):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(w * s), int(h * s)
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    c = np.zeros((th, tw, 3), np.uint8)
    yo, xo = (th - nh) // 2, (tw - nw) // 2
    c[yo:yo + nh, xo:xo + nw] = rs
    return c, s, yo, xo

def place(frame, img, cy, cx):
    ih, iw = img.shape[:2]
    y0, x0 = cy - ih // 2, cx - iw // 2
    fy0, fy1 = max(0, y0), min(H, y0 + ih)
    fx0, fx1 = max(0, x0), min(W, x0 + iw)
    sy0, sx0 = fy0 - y0, fx0 - x0
    if fy1 > fy0 and fx1 > fx0:
        frame[fy0:fy1, fx0:fx1] = img[sy0:sy0 + (fy1 - fy0), sx0:sx0 + (fx1 - fx0)]

def zoom_crop(img, scale, cx=None, cy=None):
    h, w = img.shape[:2]
    if cx is None: cx = w // 2
    if cy is None: cy = h // 2
    nw, nh = int(w / scale), int(h / scale)
    x0 = max(0, min(w - nw, cx - nw // 2))
    y0 = max(0, min(h - nh, cy - nh // 2))
    crop = img[y0:y0 + nh, x0:x0 + nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)

def caption(frame, text, alpha=1.0):
    if alpha < 0.01: return
    ts = 0.72
    tw = text_width_mixed(text, ts)
    x = (W - tw) // 2
    y = H - 42
    col = tuple(int(v * alpha) for v in WHITE)
    put_text_mixed(frame, text, (x, y), FONT, ts, col, 1)

def small_label(frame, text, y, x, col=GRAY, alpha=1.0):
    col2 = tuple(int(v * alpha) for v in col)
    cv2.putText(frame, text, (x, y), FONT, 0.38, col2, 1, cv2.LINE_AA)

# ── Scale bar ──
# Pixel sizes from microscope metadata (README.md):
#   JY306 confocal: 1.30 µm/px (820µm FOV / 629px, downscaled from 0.82 native)
#   Calcium movie: same FOV, 512px → 820/512 = 1.60 µm/px
IV_XY_UM = 1.30
MOVIE_XY_UM = 1.60

def draw_scale_bar(frame, um_per_disp_px, alpha=1.0, x_right=None, y_bottom=None):
    """Draw a scale bar with physical distance label."""
    return  # disabled
    if x_right is None:
        x_right = W - 50
    if y_bottom is None:
        y_bottom = H - 75
    # Fixed bar width, compute exact µm from FOV
    bar_px = 120  # fixed display pixels
    bar_um = bar_px * um_per_disp_px
    # Round to nearest integer if >= 10, else 1 decimal
    if bar_um >= 10:
        bar_um_label = f'{int(round(bar_um))} um'
    else:
        bar_um_label = f'{bar_um:.1f} um'
    x_left = x_right - bar_px
    y_bar = y_bottom
    # Black outline + white bar
    col = tuple(int(255 * alpha) for _ in range(3))
    bg = (0, 0, 0)
    cv2.rectangle(frame, (x_left - 1, y_bar - 4), (x_right + 1, y_bar + 4), bg, -1)
    cv2.rectangle(frame, (x_left, y_bar - 3), (x_right, y_bar + 3), col, -1)
    # Label
    label = bar_um_label
    ts = 0.45
    (tw, _), _ = cv2.getTextSize(label, FONT, ts, 1)
    tx = x_left + (bar_px - tw) // 2
    ty = y_bar - 8
    cv2.putText(frame, label, (tx, ty), FONT, ts, bg, 3, cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty), FONT, ts, col, 1, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("Loading calcium movie AVI (flipped YX)...")
avi_path = f'{BASE}/movie_rolling_avg_win12_step3_short.avi'
cap = cv2.VideoCapture(avi_path)
avi_frames = []
while True:
    ret, fr = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
    avi_frames.append(gray)  # AVI already in correct orientation
cap.release()
movie_tif = np.array(avi_frames, dtype=np.float32)
print(f"  raw: {movie_tif.shape}, max={movie_tif.max():.0f}")
# Simple normalization: /max * 255
movie_u8 = (movie_tif / movie_tif.max() * 255).astype(np.uint8)
n_movie = len(movie_u8)
# Max projection
movie_maxproj = np.max(movie_tif, axis=0)
movie_maxproj_u8 = (movie_maxproj / movie_maxproj.max() * 255).astype(np.uint8)
print(f"  {n_movie} frames {movie_u8.shape[1:]}, maxproj=/max")

print("Loading JY306 z=3…")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
jy306_z3 = jy306[3]
jy306_z3_u8 = norm_u8(jy306_z3)
print(f"  z=3 shape: {jy306_z3_u8.shape}")

# ══════════════════════════════════════════════════════════════════
# SIFT: flipped movie max-proj → JY306 z=3  →  affine M
# ══════════════════════════════════════════════════════════════════
_aff_path = f'{BASE}/animation/movie_avi_to_jy306_affine.npz'
h_z3, w_z3 = jy306_z3_u8.shape

if os.path.exists(_aff_path):
    print("Loading saved affine from movie_to_jy306_affine.npz…")
    _aff = np.load(_aff_path)
    M_affine = _aff['M_affine']
    ncc = float(_aff['ncc'])
else:
    print("Computing SIFT affine (flipped movie → JY306 z=3)…")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    sift = cv2.SIFT_create(5000)
    kp1, des1 = sift.detectAndCompute(clahe.apply(movie_maxproj_u8), None)
    kp2, des2 = sift.detectAndCompute(clahe.apply(jy306_z3_u8), None)
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    M_affine, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    warped = cv2.warpAffine(movie_maxproj_u8, M_affine, (w_z3, h_z3))
    mask = (warped > 0) & (jy306_z3_u8 > 0)
    a = warped[mask].astype(np.float64); b = jy306_z3_u8[mask].astype(np.float64)
    a -= a.mean(); b -= b.mean()
    ncc = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    np.savez(_aff_path, M_affine=M_affine, best_z=3, ncc=ncc)
print(f"  M = {M_affine}")
print(f"  NCC = {ncc:.3f}")

# ══════════════════════════════════════════════════════════════════
# PRE-RENDER: compute display-space affines for smooth warp
# ══════════════════════════════════════════════════════════════════
print("Pre-rendering warped frames + overlay…")
DISPLAY_SZ = 860  # same as v4

# All warping done in 1920×1080 canvas space to avoid jarring resizes.
#
# M_start: movie pixel → canvas pixel (same size as Scene 1 display)
#   scale = 860/512 = 1.68, centered at (960, 540)
mov_h, mov_w = movie_maxproj_u8.shape  # 512, 512
s_start = DISPLAY_SZ / max(mov_h, mov_w)  # 1.68
tx_start = (W - mov_w * s_start) / 2.0
ty_start = (H - mov_h * s_start) / 2.0
M_start = np.array([[s_start, 0, tx_start],
                     [0, s_start, ty_start]], dtype=np.float64)

# M_end: movie pixel → JY306 pixel → canvas pixel
s_fit = min(DISPLAY_SZ / w_z3, DISPLAY_SZ / h_z3)  # ~1.307
tx_fit = (W - w_z3 * s_fit) / 2.0
ty_fit = (H - h_z3 * s_fit) / 2.0
M_fit_3x3 = np.array([[s_fit, 0, tx_fit],
                       [0, s_fit, ty_fit],
                       [0, 0, 1]], dtype=np.float64)
M_affine_3x3 = np.vstack([M_affine, [0, 0, 1]])
M_end_3x3 = M_fit_3x3 @ M_affine_3x3
M_end = M_end_3x3[:2, :]

# Also compute M for JY306 z=3 in canvas space (for overlay)
M_z3_to_canvas = np.array([[s_fit, 0, tx_fit],
                            [0, s_fit, ty_fit]], dtype=np.float64)

# Pre-render final overlay in canvas space
warped_canvas = cv2.warpAffine(movie_maxproj_u8, M_end, (W, H))
z3_canvas = cv2.warpAffine(jy306_z3_u8, M_z3_to_canvas, (W, H))

# Confocal z=3 in green (in-vivo confocal emerges as green)
z3_green = np.zeros((H, W, 3), np.uint8)
z3_green[:, :, 1] = z3_canvas  # green

# Overlay: grayscale calcium + green confocal on top
overlay_final = np.zeros((H, W, 3), np.uint8)
overlay_final[:, :, 0] = warped_canvas  # grayscale calcium
overlay_final[:, :, 1] = np.minimum(255, warped_canvas.astype(np.int16) + z3_canvas.astype(np.int16)).astype(np.uint8)
overlay_final[:, :, 2] = warped_canvas

cv2.imwrite(f'{BASE}/animation/overlay_hot_gray.png', overlay_final)
print(f"  Overlay saved: animation/overlay_hot_gray.png")
print(f"  Display scale: movie={s_start:.3f}  jy306_fit={s_fit:.3f}")

# ══════════════════════════════════════════════════════════════════
# VIDEO WRITER
# ══════════════════════════════════════════════════════════════════
vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

# ─────────────────────────────────────────────────────────────────
# SCENE 1: Calcium movie playback (120 frames = 5s)
# ─────────────────────────────────────────────────────────────────
print("Scene 1: calcium movie (5s)…")
N_PLAY = 120
MOV_STEP = max(1, n_movie // N_PLAY)

um_disp_movie = MOVIE_XY_UM / s_start   # calcium display: µm per display pixel
um_disp_jy306 = IV_XY_UM / s_fit        # JY306 overlay display: µm per display pixel

for fi in range(N_PLAY):
    frame = np.zeros((H, W, 3), np.uint8)
    idx = (fi * MOV_STEP) % n_movie
    gray = movie_u8[idx]
    sq, _, _, _ = fit_into(gray, DISPLAY_SZ, DISPLAY_SZ)
    place(frame, sq, H // 2, W // 2)
    if fi >= 60:
        a = ease((fi - 60) / 30)
        caption(frame, 'IN VIVO  TWO-PHOTON  CALCIUM  IMAGING', alpha=a)
        draw_scale_bar(frame, um_disp_movie, alpha=a)
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────
# SCENE 1b: Freeze on max-proj (24 frames = 1s)
# ─────────────────────────────────────────────────────────────────
print("Scene 1b: freeze on max-proj (1s)…")
# Crossfade last movie frame → max-proj over first 12 frames, then hold
last_idx = ((N_PLAY - 1) * MOV_STEP) % n_movie
last_frame_u8 = movie_u8[last_idx]

for fi in range(24):
    frame = np.zeros((H, W, 3), np.uint8)
    t_max = ease(fi / 12)
    blended = cv2.addWeighted(last_frame_u8, 1 - t_max, movie_maxproj_u8, t_max, 0)
    sq, _, _, _ = fit_into(blended, DISPLAY_SZ, DISPLAY_SZ)
    place(frame, sq, H // 2, W // 2)
    caption(frame, 'IN VIVO  TWO-PHOTON  CALCIUM  IMAGING')
    draw_scale_bar(frame, um_disp_movie)
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────
# SCENE 2a: Warp calcium from centered → registered position (48 fr = 2s)
# then fade in confocal overlay (48 fr = 2s)
# ─────────────────────────────────────────────────────────────────
print("Scene 2a: warp + overlay (4s)...")
N_WARP = 48   # smooth warp animation
N_FADE = 48   # fade in confocal

# Calcium in registered position (grayscale — calcium stays grayscale)
cal_registered = np.zeros((H, W, 3), np.uint8)
cal_reg_gray = cv2.warpAffine(movie_maxproj_u8, M_end, (W, H))
cal_registered[:, :, 0] = cal_reg_gray
cal_registered[:, :, 1] = cal_reg_gray
cal_registered[:, :, 2] = cal_reg_gray

# Phase 1: smoothly warp calcium from centered to registered position
for fi in range(N_WARP):
    t = ease(fi / (N_WARP - 1))
    # Interpolate affine: M_start*(1-t) + M_end*t
    M_t = M_start * (1 - t) + M_end * t
    warped_t = cv2.warpAffine(movie_maxproj_u8, M_t, (W, H))
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:, :, 0] = warped_t
    frame[:, :, 1] = warped_t
    frame[:, :, 2] = warped_t
    a_old = 1.0 - ease(fi / 20)
    caption(frame, 'IN VIVO  TWO-PHOTON  CALCIUM  IMAGING', alpha=float(max(0, a_old)))
    # Scale bar fades out during warp (scale is changing)
    draw_scale_bar(frame, um_disp_movie, alpha=float(max(0, 1.0 - ease(fi / 12))))
    vw.write(frame)

# Phase 2: fade in confocal over the registered calcium
for fi in range(N_FADE):
    t = ease(fi / (N_FADE - 1))
    frame = cv2.addWeighted(cal_registered, 1 - t, overlay_final, t, 0)
    a_new = ease((fi - 10) / 20)
    caption(frame, 'MATCHING  CELLS  TO  IN VIVO  Z-STACK', alpha=float(max(0, a_new)))
    # Scale bar fades in with new JY306 scale
    draw_scale_bar(frame, um_disp_jy306, alpha=float(max(0, a_new)))
    vw.write(frame)

N_XFADE = N_WARP + N_FADE

# ─────────────────────────────────────────────────────────────────
# SCENE 2c: Hold overlay (36 frames = 1.5s)
# ─────────────────────────────────────────────────────────────────
print("Scene 2c: overlay hold (1.5s)…")
N_HOLD = 36

for fi in range(N_HOLD):
    frame = overlay_final.copy()
    caption(frame, 'IMAGED  FIELD  OF  VIEW  +  IN VIVO  Z-STACK  (Z = 3)')
    draw_scale_bar(frame, um_disp_jy306)
    vw.write(frame)

# ── finalize ──
vw.release()
print("Re-encoding to H.264…")
subprocess.run([
    'ffmpeg', '-y', '-i', TMP,
    '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
    OUT
], capture_output=True)

os.remove(TMP)

total_fr = N_PLAY + 24 + N_XFADE + N_HOLD
total_s = total_fr / FPS
print(f"Done! {total_fr} frames, {total_s:.1f}s @ {FPS}fps → {OUT}")
