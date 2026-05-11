"""
Combined Scenes 1+2, 3, 5 — Single unified video.

Merges:
  scene1_2_movie_to_zstack.py  (calcium movie → warp → overlay)
  scene3_zstack_3d.py          (z-stack 3D rotation, highlight z=3)
  scene5_all_tiles_v2.py       (all-tiles registration animation)

Output: animation/merged_scenes_1_3_5_h264.mp4
Also:   animation/merged_scenes_1_3_5_6x_h264.mp4  (6x speedup mid-section)
"""

import numpy as np, cv2, tifffile, math, subprocess, os, glob
from collections import Counter
from text_utils import put_text_mixed, text_width_mixed

# ══════════════════════════════════════════════════════════════════
# SHARED CONSTANTS
# ══════════════════════════════════════════════════════════════════
BASE = '/Users/neurolab/neuroinformatics/margaret'
TMP  = f'{BASE}/animation/merged_scenes_1_3_5_raw.mp4'
OUT  = f'{BASE}/animation/merged_scenes_1_3_5_h264.mp4'
OUT_6X = f'{BASE}/animation/merged_scenes_1_3_5_6x_h264.mp4'

W, H = 1920, 1080
FPS  = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GRAY  = (160, 160, 160)
GREEN = (0, 220, 0)

# Pixel sizes from microscope metadata
IV_XY_UM    = 0.82   # JY306 in-vivo confocal (µm/px) — used by scenes 3 & 5
MOVIE_XY_UM = 1.60   # Calcium movie: 820µm FOV / 512px
IV_XY_UM_S1 = 1.30   # JY306 confocal downscaled (scene 1-2 uses this)
ND2_XY_UM   = 0.65   # nd2 ex-vivo confocal (µm/px)

# 3D stack rendering constants (shared by scene 3 and scene 5)
Z_SPACING = 4
INTERP_PER_GAP = 3
INIT_ROT_X = -0.3

# ══════════════════════════════════════════════════════════════════
# SHARED UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

def norm8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 50:
        return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img.astype(np.float32) - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

# Alias used by scene 1-2
norm_u8 = norm8

def make_green(u8):
    g = np.zeros((*u8.shape, 3), np.uint8)
    g[:, :, 1] = u8
    return g

def make_magenta(u8):
    m = np.zeros((*u8.shape, 3), np.uint8)
    m[:, :, 0] = u8  # B
    m[:, :, 2] = u8  # R
    return m

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

def small_label(frame, text, pos, col=GRAY, alpha=1.0, scale=0.38):
    col2 = tuple(int(v * alpha) for v in col)
    cv2.putText(frame, text, pos, FONT, scale, col2, 1, cv2.LINE_AA)

def draw_scale_bar(frame, um_per_disp_px, alpha=1.0, x_right=None, y_bottom=None):
    """Draw a scale bar with physical distance label."""
    return  # disabled
    if x_right is None:
        x_right = W - 50
    if y_bottom is None:
        y_bottom = H - 75
    for target_um in [10, 20, 50, 100, 200, 500]:
        bar_px = int(round(target_um / um_per_disp_px))
        if 80 <= bar_px <= 200:
            break
    x_left = x_right - bar_px
    y_bar = y_bottom
    col = tuple(int(255 * alpha) for _ in range(3))
    bg = (0, 0, 0)
    cv2.rectangle(frame, (x_left - 1, y_bar - 4), (x_right + 1, y_bar + 4), bg, -1)
    cv2.rectangle(frame, (x_left, y_bar - 3), (x_right, y_bar + 3), col, -1)
    label = f'{target_um} um'
    ts = 0.45
    (tw, _), _ = cv2.getTextSize(label, FONT, ts, 1)
    tx = x_left + (bar_px - tw) // 2
    ty = y_bar - 8
    cv2.putText(frame, label, (tx, ty), FONT, ts, bg, 3, cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty), FONT, ts, col, 1, cv2.LINE_AA)

def draw_arrow(frame, pt1, pt2, color, thickness=2, tip=0.025):
    cv2.arrowedLine(frame, pt1, pt2, color, thickness, cv2.LINE_AA, tipLength=tip)

# ── Axis widget (static 2D view — no rotation) ──
AX_CX, AX_CY, AX_LEN = 100, H - 100, 50
AX_AXES = [(1, 0, 'ML', (0, 0, 180)),    # right
           (0, -1, 'AP', (40, 40, 40)),   # up
           ]

def draw_axes(frame, alpha=1.0):
    """Draw static ML/AP axis widget in bottom-left corner."""
    if alpha < 0.01:
        return
    cx, cy = AX_CX, AX_CY
    for ux, uy, label, color in AX_AXES:
        px, py = int(cx + ux * AX_LEN), int(cy + uy * AX_LEN)
        col = tuple(int(c * alpha) for c in color)
        cv2.arrowedLine(frame, (cx, cy), (px, py), col, 3, cv2.LINE_AA, tipLength=0.15)
        dx = ux * 18
        dy = uy * 18
        lx, ly = int(px + dx), int(py + dy)
        cv2.putText(frame, label, (lx - 10, ly + 5), FONT, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (lx - 10, ly + 5), FONT, 0.55, col, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 4, tuple(int(200 * alpha) for _ in range(3)), -1, cv2.LINE_AA)

def write_frame(vw, frame, alpha=1.0):
    """Write frame with axis widget overlay (used by scene 5)."""
    draw_axes(frame, alpha)
    vw.write(frame)


# ══════════════════════════════════════════════════════════════════
# SCENE 1+2: Calcium movie → pause → warp → overlay
# ══════════════════════════════════════════════════════════════════

def render_scene1_2(vw):
    """
    Timeline (at 24fps):
      0.0 – 5.0s  : Calcium movie playback (120 fr)
      5.0 – 6.0s  : Freeze on max-proj (24 fr)
      6.0 – 9.0s  : Warp deformation: identity → affine M (48 fr warp + 48 fr fade)
      9.0 – 11.5s : Overlay hold (36 fr)
    Total: 276 frames = 11.5s
    """
    # ── Load data ──
    print("Loading calcium movie AVI (flipped YX)...")
    avi_path = f'{BASE}/movie_rolling_avg_win12_step3_short.avi'
    cap = cv2.VideoCapture(avi_path)
    avi_frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
        avi_frames.append(gray)
    cap.release()
    movie_tif = np.array(avi_frames, dtype=np.float32)
    print(f"  raw: {movie_tif.shape}, max={movie_tif.max():.0f}")
    movie_u8 = (movie_tif / movie_tif.max() * 255).astype(np.uint8)
    n_movie = len(movie_u8)
    movie_maxproj = np.max(movie_tif, axis=0)
    movie_maxproj_u8 = (movie_maxproj / movie_maxproj.max() * 255).astype(np.uint8)
    print(f"  {n_movie} frames {movie_u8.shape[1:]}, maxproj=/max")

    print("Loading JY306 z=3…")
    jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
    jy306_z3 = jy306[3]
    jy306_z3_u8 = norm8(jy306_z3)
    print(f"  z=3 shape: {jy306_z3_u8.shape}")

    # ── SIFT affine ──
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

    # ── Pre-render display-space affines ──
    print("Pre-rendering warped frames + overlay…")
    DISPLAY_SZ = 860

    mov_h, mov_w = movie_maxproj_u8.shape
    s_start = DISPLAY_SZ / max(mov_h, mov_w)
    tx_start = (W - mov_w * s_start) / 2.0
    ty_start = (H - mov_h * s_start) / 2.0
    M_start = np.array([[s_start, 0, tx_start],
                         [0, s_start, ty_start]], dtype=np.float64)

    s_fit = min(DISPLAY_SZ / w_z3, DISPLAY_SZ / h_z3)
    tx_fit = (W - w_z3 * s_fit) / 2.0
    ty_fit = (H - h_z3 * s_fit) / 2.0
    M_fit_3x3 = np.array([[s_fit, 0, tx_fit],
                           [0, s_fit, ty_fit],
                           [0, 0, 1]], dtype=np.float64)
    M_affine_3x3 = np.vstack([M_affine, [0, 0, 1]])
    M_end_3x3 = M_fit_3x3 @ M_affine_3x3
    M_end = M_end_3x3[:2, :]

    M_z3_to_canvas = np.array([[s_fit, 0, tx_fit],
                                [0, s_fit, ty_fit]], dtype=np.float64)

    warped_canvas = cv2.warpAffine(movie_maxproj_u8, M_end, (W, H))
    z3_canvas = cv2.warpAffine(jy306_z3_u8, M_z3_to_canvas, (W, H))

    z3_green = np.zeros((H, W, 3), np.uint8)
    z3_green[:, :, 1] = z3_canvas

    overlay_final = np.zeros((H, W, 3), np.uint8)
    overlay_final[:, :, 0] = warped_canvas
    overlay_final[:, :, 1] = np.minimum(255, warped_canvas.astype(np.int16) + z3_canvas.astype(np.int16)).astype(np.uint8)
    overlay_final[:, :, 2] = warped_canvas

    cv2.imwrite(f'{BASE}/animation/overlay_hot_gray.png', overlay_final)
    print(f"  Overlay saved: animation/overlay_hot_gray.png")
    print(f"  Display scale: movie={s_start:.3f}  jy306_fit={s_fit:.3f}")

    um_disp_movie = MOVIE_XY_UM / s_start
    um_disp_jy306 = IV_XY_UM_S1 / s_fit

    # ── SCENE 1: Calcium movie playback (120 frames = 5s) ──
    print("Scene 1: calcium movie (5s)…")
    N_PLAY = 120
    MOV_STEP = max(1, n_movie // N_PLAY)

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

    # ── SCENE 1b: Freeze on max-proj (24 frames = 1s) ──
    print("Scene 1b: freeze on max-proj (1s)…")
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

    # ── SCENE 2a: Warp + fade overlay (4s) ──
    print("Scene 2a: warp + overlay (4s)...")
    N_WARP = 48
    N_FADE = 48

    cal_registered = np.zeros((H, W, 3), np.uint8)
    cal_reg_gray = cv2.warpAffine(movie_maxproj_u8, M_end, (W, H))
    cal_registered[:, :, 0] = cal_reg_gray
    cal_registered[:, :, 1] = cal_reg_gray
    cal_registered[:, :, 2] = cal_reg_gray

    for fi in range(N_WARP):
        t = ease(fi / (N_WARP - 1))
        M_t = M_start * (1 - t) + M_end * t
        warped_t = cv2.warpAffine(movie_maxproj_u8, M_t, (W, H))
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:, :, 0] = warped_t
        frame[:, :, 1] = warped_t
        frame[:, :, 2] = warped_t
        a_old = 1.0 - ease(fi / 20)
        caption(frame, 'IN VIVO  TWO-PHOTON  CALCIUM  IMAGING', alpha=float(max(0, a_old)))
        draw_scale_bar(frame, um_disp_movie, alpha=float(max(0, 1.0 - ease(fi / 12))))
        vw.write(frame)

    for fi in range(N_FADE):
        t = ease(fi / (N_FADE - 1))
        frame = cv2.addWeighted(cal_registered, 1 - t, overlay_final, t, 0)
        a_new = ease((fi - 10) / 20)
        caption(frame, 'MATCHING  CELLS  TO  IN VIVO  Z-STACK', alpha=float(max(0, a_new)))
        draw_scale_bar(frame, um_disp_jy306, alpha=float(max(0, a_new)))
        vw.write(frame)

    N_XFADE = N_WARP + N_FADE

    # ── SCENE 2c: Hold overlay (36 frames = 1.5s) ──
    print("Scene 2c: overlay hold (1.5s)…")
    N_HOLD = 36

    for fi in range(N_HOLD):
        frame = overlay_final.copy()
        caption(frame, 'IMAGED  FIELD  OF  VIEW  +  IN VIVO  Z-STACK  (Z = 3)')
        draw_scale_bar(frame, um_disp_jy306)
        vw.write(frame)

    total_fr = N_PLAY + 24 + N_XFADE + N_HOLD
    print(f"Scene 1+2 done: {total_fr} frames ({total_fr/FPS:.1f}s)")
    return total_fr


# ══════════════════════════════════════════════════════════════════
# SCENE 3: In-vivo z-stack 3D volume rotation
# ══════════════════════════════════════════════════════════════════

def render_scene3(vw):
    """
    Timeline (at 24fps):
      0.0 – 3.0s  : Transition from overlay → shrinking z=3 + slices emerge (72 fr)
      3.0 – 8.0s  : 3D rotation (120 fr)
      8.0 – 9.0s  : Pause front-facing (24 fr)
      9.0 – 11.0s : Highlight z=3 (48 fr)
      11.0 – 12.5s: Slide z=3 to left (36 fr)
    Total: 300 frames = 12.5s
    """
    # ── Load z-stack ──
    print("Loading JY306 z-stack…")
    jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
    nz, hy, wx = jy306.shape

    slices_u8 = []
    for z in range(nz):
        slices_u8.append(norm8(jy306[z]))
    slices_u8 = np.array(slices_u8)

    _max_dim = 400
    _scale = _max_dim / max(wx, hy)
    SLICE_W, SLICE_H = int(wx * _scale), int(hy * _scale)
    slices_small = []
    for z in range(nz):
        s = cv2.resize(slices_u8[z], (SLICE_W, SLICE_H), interpolation=cv2.INTER_AREA)
        slices_small.append(s)
    slices_small = np.array(slices_small)
    print(f"  Slice display size: {SLICE_W}×{SLICE_H} (aspect preserved)")

    # ── Gaussian-interpolate between slices ──
    print(f"Gaussian-interpolating z-stack: {nz} slices → {nz + (nz-1)*INTERP_PER_GAP} sub-slices...")
    dense_slices = []
    dense_z_positions = []
    dense_is_real = []
    dense_real_idx = []

    for z in range(nz):
        dense_slices.append(slices_small[z])
        dense_z_positions.append(z * Z_SPACING)
        dense_is_real.append(True)
        dense_real_idx.append(z)

        if z < nz - 1:
            for sub in range(1, INTERP_PER_GAP + 1):
                t = sub / (INTERP_PER_GAP + 1)
                interp = (slices_small[z].astype(np.float32) * (1-t) +
                          slices_small[z+1].astype(np.float32) * t)
                dense_slices.append(interp.astype(np.uint8))
                dense_z_positions.append(z * Z_SPACING + t * Z_SPACING)
                dense_is_real.append(False)
                dense_real_idx.append(-1)

    dense_slices = np.array(dense_slices)
    dense_z_positions = np.array(dense_z_positions)
    n_dense = len(dense_slices)
    STACK_CENTER_Z = (dense_z_positions[-1] + dense_z_positions[0]) / 2.0
    print(f"  {n_dense} total sub-slices, Z range: 0-{dense_z_positions[-1]:.0f}px")

    def render_3d_stack_scene3(rot_y, rot_x, slice_alphas):
        """Render the z-stack as a 3D rotated stack of planes (green channel)."""
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        cx, cy = W // 2, H // 2

        cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
        cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

        z_depths = []
        for i in range(n_dense):
            dz = dense_z_positions[i] - STACK_CENTER_Z
            rz = cos_y * dz
            rz2 = cos_x * rz
            z_depths.append((rz2, i))
        z_depths.sort(key=lambda x: x[0])

        for depth, i in z_depths:
            real_idx = dense_real_idx[i]
            if real_idx >= 0:
                alpha = slice_alphas[real_idx]
            else:
                zp = dense_z_positions[i]
                z_below = int(zp / Z_SPACING)
                z_above = min(nz-1, z_below + 1)
                t = (zp - z_below * Z_SPACING) / Z_SPACING if Z_SPACING > 0 else 0
                alpha = slice_alphas[z_below] * (1-t) + slice_alphas[z_above] * t

            if alpha < 0.01:
                continue

            sl = dense_slices[i].astype(np.float32) / 255.0
            sh, sw = sl.shape

            hw, hh = sw / 2, sh / 2
            dz = dense_z_positions[i] - STACK_CENTER_Z

            corners_3d = np.array([
                [-hw, -hh, dz],
                [ hw, -hh, dz],
                [ hw,  hh, dz],
                [-hw,  hh, dz],
            ], dtype=np.float64)

            rot_corners = []
            for c in corners_3d:
                rx = cos_y * c[0] + sin_y * c[2]
                ry = c[1]
                rz = -sin_y * c[0] + cos_y * c[2]
                ry2 = cos_x * ry - sin_x * rz
                rz2 = sin_x * ry + cos_x * rz
                rot_corners.append([rx + cx, ry2 + cy])

            rot_corners = np.array(rot_corners, dtype=np.float32)
            src_corners = np.array([[0,0],[sw,0],[sw,sh],[0,sh]], dtype=np.float32)

            M = cv2.getPerspectiveTransform(src_corners, rot_corners)
            warped = cv2.warpPerspective(sl, M, (W, H))

            green = np.zeros((H, W, 3), dtype=np.float32)
            green[:, :, 1] = warped

            mask = warped > 0.01
            mask3 = np.stack([mask]*3, axis=-1)
            canvas = np.where(mask3,
                             np.maximum(canvas, green * alpha),
                             canvas)

        return np.clip(canvas * 255, 0, 255).astype(np.uint8)

    # ── Load overlay data matching scene 1-2 ending ──
    z3_u8 = norm8(jy306[3])
    z3_green_full = np.zeros((*z3_u8.shape, 3), np.uint8)
    z3_green_full[:, :, 1] = z3_u8

    _aff_path = f'{BASE}/animation/movie_avi_to_jy306_affine.npz'
    _aff = np.load(_aff_path)
    M_affine = _aff['M_affine']
    avi_path = f'{BASE}/movie_rolling_avg_win12_step3_short.avi'
    cap = cv2.VideoCapture(avi_path)
    avi_frames = []
    while True:
        ret, fr = cap.read()
        if not ret: break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
        avi_frames.append(gray)
    cap.release()
    movie_tif = np.array(avi_frames, dtype=np.float32)
    movie_maxproj_u8 = (np.max(movie_tif, axis=0) / movie_tif.max() * 255).astype(np.uint8)

    DISPLAY_SZ = 860
    s_fit = min(DISPLAY_SZ / wx, DISPLAY_SZ / hy)

    tx_fit = (W - wx * s_fit) / 2.0
    ty_fit = (H - hy * s_fit) / 2.0
    M_fit_3x3 = np.array([[s_fit, 0, tx_fit], [0, s_fit, ty_fit], [0, 0, 1]], dtype=np.float64)
    M_affine_3x3 = np.vstack([M_affine, [0, 0, 1]])
    M_end = (M_fit_3x3 @ M_affine_3x3)[:2, :]
    M_z3_to_canvas = np.array([[s_fit, 0, tx_fit], [0, s_fit, ty_fit]], dtype=np.float64)

    warped_canvas = cv2.warpAffine(movie_maxproj_u8, M_end, (W, H))
    z3_canvas = cv2.warpAffine(z3_u8, M_z3_to_canvas, (W, H))

    overlay_start = np.zeros((H, W, 3), np.uint8)
    overlay_start[:, :, 0] = warped_canvas
    overlay_start[:, :, 1] = np.minimum(255, warped_canvas.astype(np.int16) + z3_canvas.astype(np.int16)).astype(np.uint8)
    overlay_start[:, :, 2] = warped_canvas

    z3_display = np.zeros((H, W, 3), np.uint8)
    z3_display[:, :, 1] = z3_canvas

    nw_start, nh_start = int(wx * s_fit), int(hy * s_fit)
    nw_end, nh_end = SLICE_W, SLICE_H

    # ─── SCENE 3a: slide + shrink + slices emerge (72 fr = 3s) ───
    print("Scene 3a: slide + shrink + slices emerge (3s)…")

    N_TRANS = 72
    N_CALCIUM_FADE = 24

    z3_green_frames = []
    for fi in range(N_TRANS):
        t = ease(fi / (N_TRANS - 1))
        nw_t = int(round(nw_start * (1 - t) + nw_end * t))
        nh_t = int(round(nh_start * (1 - t) + nh_end * t))
        nw_t = max(1, nw_t)
        nh_t = max(1, nh_t)
        z3_green_frames.append(cv2.resize(z3_green_full, (nw_t, nh_t), interpolation=cv2.INTER_LANCZOS4))

    um_disp_flat = IV_XY_UM / s_fit

    for fi in range(N_TRANS):
        t = ease(fi / (N_TRANS - 1))

        if fi < N_CALCIUM_FADE:
            t_fade = ease(fi / (N_CALCIUM_FADE - 1))
            frame = cv2.addWeighted(overlay_start, 1 - t_fade, z3_display, t_fade, 0)
        else:
            frame = np.zeros((H, W, 3), np.uint8)
            z3_resized = z3_green_frames[fi]
            nh_t, nw_t = z3_resized.shape[:2]
            x0 = (W - nw_t) // 2
            y0 = (H - nh_t) // 2
            fx0, fy0 = max(0, x0), max(0, y0)
            sx0, sy0 = fx0 - x0, fy0 - y0
            fw = min(nw_t - sx0, W - fx0)
            fh = min(nh_t - sy0, H - fy0)
            if fw > 0 and fh > 0:
                frame[fy0:fy0+fh, fx0:fx0+fw] = z3_resized[sy0:sy0+fh, sx0:sx0+fw]

            t_emerge = max(0.0, (t - 0.5) / 0.5)
            if t_emerge > 0.01:
                for z in range(nz):
                    if z == 3: continue
                    dist = abs(z - 3)
                    max_dist = ease(t_emerge) * 13
                    if dist > max_dist: continue
                    alpha = min(1.0, (max_dist - dist + 1) / 2.0) * 0.6
                    sl_u8_z = norm8(jy306[z])
                    sl_green = np.zeros((*sl_u8_z.shape, 3), np.uint8)
                    sl_green[:, :, 1] = sl_u8_z
                    sl_resized = cv2.resize(sl_green, (nw_t, nh_t), interpolation=cv2.INTER_AREA)
                    z_offset = int((z - 3) * Z_SPACING * (nw_t / SLICE_W) * 0.3 * ease(t_emerge))
                    sy = y0 + z_offset
                    sx = (W - nw_t) // 2
                    fy0s, fx0s = max(0, sy), max(0, sx)
                    sy0s, sx0s = fy0s - sy, fx0s - sx
                    fws = min(nw_t - sx0s, W - fx0s)
                    fhs = min(nh_t - sy0s, H - fy0s)
                    if fws > 0 and fhs > 0:
                        region = sl_resized[sy0s:sy0s+fhs, sx0s:sx0s+fws]
                        existing = frame[fy0s:fy0s+fhs, fx0s:fx0s+fws]
                        blended = np.maximum(existing, (region.astype(np.float32) * alpha).astype(np.uint8))
                        frame[fy0s:fy0s+fhs, fx0s:fx0s+fws] = blended

        caption(frame, 'IN VIVO  Z-STACK')
        sb_alpha = max(0.0, 1.0 - ease(fi / (N_CALCIUM_FADE + 6)))
        draw_scale_bar(frame, um_disp_flat, alpha=sb_alpha)
        vw.write(frame)

    # ─── SCENE 3c: 3D rotation (120 fr = 5s) ───
    print("Scene 3c: 3D rotation (5s)…")

    slice_alphas_full = np.ones(nz, dtype=np.float32) * 0.7
    slice_alphas_full[3] = 1.0

    for fi in range(120):
        t = fi / 119.0
        rot_y = t * math.pi * 2.0
        rot_x = INIT_ROT_X + 0.15 * math.sin(t * math.pi)

        frame = render_3d_stack_scene3(rot_y, rot_x, slice_alphas_full)
        caption(frame, 'IN VIVO  Z-STACK')
        vw.write(frame)

    # ─── SCENE 3d: Pause front-facing (24 fr = 1s) ───
    print("Scene 3d: pause front-facing (1s)…")

    final_rot_y = math.pi * 2.0

    for fi in range(24):
        t = ease(fi / 20)
        rot_y = final_rot_y + 0.0 * t
        rot_x = INIT_ROT_X

        frame = render_3d_stack_scene3(rot_y, rot_x, slice_alphas_full)
        caption(frame, 'IN VIVO  Z-STACK')
        vw.write(frame)

    # ─── SCENE 3e: Highlight z=3 (48 fr = 2s) ───
    print("Scene 3e: highlight z=3 (2s)…")

    for fi in range(48):
        t = ease(fi / 35)

        slice_alphas = np.ones(nz, dtype=np.float32) * 0.7 * (1 - t * 0.8)
        slice_alphas[3] = 1.0

        frame = render_3d_stack_scene3(0.0, INIT_ROT_X, slice_alphas)

        a_old = 1 - ease(fi / 15)
        a_new = ease((fi - 15) / 20)
        caption(frame, 'IN VIVO  Z-STACK', alpha=float(max(0, a_old)))
        caption(frame, 'BEST ALIGNMENT:  Z = 3  --  EX VIVO  TILE  ROW2_1', alpha=float(max(0, a_new)))
        vw.write(frame)

    # ─── SCENE 3f: Slide z=3 to left (36 fr = 1.5s) ───
    print("Scene 3f: slide z=3 to left (1.5s)…")

    z3_flat = np.zeros((H, W, 3), np.uint8)
    z3_green_2d = cv2.resize(z3_green_full, (nw_end, nh_end), interpolation=cv2.INTER_LANCZOS4)

    cx_start = (W - nw_end) // 2
    cy_start = (H - nh_end) // 2

    cx_end = W // 6 - nw_end // 2
    cy_end = cy_start

    N_SLIDE = 36
    for fi in range(N_SLIDE):
        t = ease(fi / (N_SLIDE - 1))
        frame = np.zeros((H, W, 3), np.uint8)

        cx = int(cx_start + t * (cx_end - cx_start))
        cy = int(cy_start + t * (cy_end - cy_start))

        py0 = max(0, cy); py1 = min(H, cy + nh_end)
        px0 = max(0, cx); px1 = min(W, cx + nw_end)
        sy0 = py0 - cy; sx0 = px0 - cx
        if py1 > py0 and px1 > px0:
            frame[py0:py1, px0:px1] = z3_green_2d[sy0:sy0+(py1-py0), sx0:sx0+(px1-px0)]

        a_old = max(0, 1 - ease(fi / 12))
        caption(frame, 'BEST ALIGNMENT:  Z = 3  --  EX VIVO  TILE  ROW2_1', alpha=a_old)
        vw.write(frame)

    total_fr = N_TRANS + 120 + 24 + 48 + N_SLIDE
    print(f"Scene 3 done: {total_fr} frames ({total_fr/FPS:.1f}s)")
    return total_fr


# ══════════════════════════════════════════════════════════════════
# SCENE 5: All tiles registration animation
# ══════════════════════════════════════════════════════════════════

def render_scene5(vw):
    """
    - row2_1: full animation (phases A-F, no 3D in this version)
    - Other tiles: phases A-F, faster
    Returns total frame count.
    """
    # ── Load shared data ──
    print("Loading JY306 z-stack (shared across all tiles)...")
    jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
    nz_jy, hy_jy, wx_jy = jy306.shape
    jy_p1, jy_p2 = np.percentile(jy306[jy306 > 0], [1, 99.5])

    # ── Tile order ──
    ALL_TILES = sorted([d for d in os.listdir(f'{BASE}/png_exports/registration_per_tile_pkl')
                        if os.path.isdir(f'{BASE}/png_exports/registration_per_tile_pkl/{d}')])
    FIRST_TILE = 'row2_1'
    OTHER_TILES = [t for t in ALL_TILES if t != FIRST_TILE]
    TILE_ORDER = [FIRST_TILE] + OTHER_TILES
    print(f"{len(TILE_ORDER)} tiles: {TILE_ORDER}")

    def render_tile(vw, tile, jy306, full_3d=False, is_first_tile=False):
        """Render all phases for one tile."""

        # ── Timing ──
        if full_3d:
            N_A, N_B, N_C, N_D, N_E, N_F = 36, 216, 36, 48, 96, 72
            N_TOP_LM = 9
        else:
            N_A, N_B, N_C, N_D, N_E, N_F = 12, 96, 12, 36, 72, 48
            N_TOP_LM = min(7, 9)

        # ── Load tile data ──
        nd2_files = sorted(glob.glob(f'{BASE}/png_exports/registration_video/{tile}/GFP_z*.png'))
        nd2_stack = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in nd2_files])

        pkl = np.load(f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz')
        M2d = pkl['M2d_jy306_to_nd2']
        M3 = np.vstack([M2d, [0, 0, 1]])
        iv = pkl['pcd_invivo_jy306']
        ev = pkl['ev_nd2']
        n_lm = len(iv)

        MODE_Z = int(round(np.median(iv[:, 0])))
        MODE_Z = max(0, min(nz_jy - 1, MODE_Z))

        z_lms = [i for i in range(n_lm) if int(round(iv[i, 0])) == MODE_Z]
        if z_lms:
            nd2_z_mode = Counter([int(round(ev[i, 2])) for i in z_lms]).most_common(1)[0][0]
        else:
            nd2_z_mode = Counter([int(round(ev[i, 2])) for i in range(n_lm)]).most_common(1)[0][0]
        nd2_z_mode = max(0, min(len(nd2_stack) - 1, nd2_z_mode))

        # ── Display images ──
        z_u8 = norm8(jy306[MODE_Z])
        z_green = make_green(z_u8)

        nd2_u8 = norm8(nd2_stack[nd2_z_mode])
        nd2_magenta_full = np.zeros((4200, 4200, 3), dtype=np.uint8)
        nd2_magenta_full[:, :, 0] = nd2_u8
        nd2_magenta_full[:, :, 2] = nd2_u8

        # ── Layout ──
        DISP_H = int(H * 0.72)
        IMG_GAP = 100

        scale_jy = DISP_H / hy_jy
        disp_jy_w = int(wx_jy * scale_jy)
        disp_jy_h = DISP_H
        jy_disp = cv2.resize(z_green, (disp_jy_w, disp_jy_h), interpolation=cv2.INTER_LANCZOS4)

        margin_nd2 = 350
        crop_x0 = max(0, int(ev[:, 0].min() - margin_nd2))
        crop_y0 = max(0, int(ev[:, 1].min() - margin_nd2))
        crop_x1 = min(4200, int(ev[:, 0].max() + margin_nd2))
        crop_y1 = min(4200, int(ev[:, 1].max() + margin_nd2))
        nd2_crop = nd2_magenta_full[crop_y0:crop_y1, crop_x0:crop_x1]
        scale_nd2 = DISP_H / nd2_crop.shape[0]
        disp_nd2_w = int(nd2_crop.shape[1] * scale_nd2)
        disp_nd2_h = DISP_H
        nd2_disp = cv2.resize(nd2_crop, (disp_nd2_w, disp_nd2_h), interpolation=cv2.INTER_LANCZOS4)

        total_w = disp_jy_w + IMG_GAP + disp_nd2_w
        jy_x0 = (W - total_w) // 2
        jy_y0 = (H - DISP_H) // 2 - 20
        nd2_x0 = jy_x0 + disp_jy_w + IMG_GAP
        nd2_y0 = jy_y0

        um_disp_jy = IV_XY_UM / scale_jy
        um_disp_nd2 = ND2_XY_UM / scale_nd2
        sb_jy_xr = jy_x0 + disp_jy_w - 15
        sb_nd2_xr = nd2_x0 + disp_nd2_w - 15
        sb_yb = jy_y0 + disp_jy_h - 15

        lm_jy_disp = []
        lm_nd2_disp = []
        for i in range(n_lm):
            dx = int(iv[i, 2] * scale_jy) + jy_x0
            dy = int(iv[i, 1] * scale_jy) + jy_y0
            lm_jy_disp.append((dx, dy))
            dx2 = int((ev[i, 0] - crop_x0) * scale_nd2) + nd2_x0
            dy2 = int((ev[i, 1] - crop_y0) * scale_nd2) + nd2_y0
            lm_nd2_disp.append((dx2, dy2))

        z_matched = [i for i in range(n_lm)
                     if int(round(iv[i, 0])) == MODE_Z and int(round(ev[i, 2])) == nd2_z_mode]
        if not z_matched:
            z_matched = [i for i in range(n_lm) if int(round(iv[i, 0])) == MODE_Z]
        if not z_matched:
            z_matched = list(range(n_lm))

        # ── Zoom panels ──
        CROP_R_JY = 40; CROP_R_ND2 = 100; PANEL_SZ = 110
        lm_scores = []
        for i in z_matched:
            y_i = int(round(iv[i, 1])); x_i = int(round(iv[i, 2]))
            r = CROP_R_JY
            crop = jy306[MODE_Z, max(0,y_i-r):min(hy_jy,y_i+r), max(0,x_i-r):min(wx_jy,x_i+r)]
            lm_scores.append(float(crop.std()))
        SELECTED = sorted(range(len(z_matched)), key=lambda i: -lm_scores[i])[:N_TOP_LM + 1]
        if len(SELECTED) > 5:
            SELECTED.pop(5)
        SELECTED = SELECTED[:N_TOP_LM]
        SELECTED_GLOBAL = [z_matched[s] for s in SELECTED]

        zoom_panels = []
        for idx in SELECTED_GLOBAL:
            y_lm = int(round(iv[idx, 1])); x_lm = int(round(iv[idx, 2]))
            jy_slice = jy306[MODE_Z]
            y0 = max(0, y_lm - CROP_R_JY); y1 = min(hy_jy, y_lm + CROP_R_JY)
            x0 = max(0, x_lm - CROP_R_JY); x1 = min(wx_jy, x_lm + CROP_R_JY)
            crop_n = np.clip((jy_slice[y0:y1, x0:x1] - jy_p1) / max(jy_p2 - jy_p1, 1) * 255, 0, 255).astype(np.uint8)
            crop_n = cv2.resize(crop_n, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
            green_p_iv = np.zeros((PANEL_SZ, PANEL_SZ, 3), np.uint8); green_p_iv[:, :, 1] = crop_n

            x_nd2 = int(round(ev[idx, 0])); y_nd2 = int(round(ev[idx, 1]))
            nd2_slice = nd2_stack[nd2_z_mode]
            yn0 = max(0, y_nd2 - CROP_R_ND2); yn1 = min(4200, y_nd2 + CROP_R_ND2)
            xn0 = max(0, x_nd2 - CROP_R_ND2); xn1 = min(4200, x_nd2 + CROP_R_ND2)
            crop_nd2 = norm8(nd2_slice[yn0:yn1, xn0:xn1])
            crop_nd2 = cv2.resize(crop_nd2, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
            magenta_p = np.zeros((PANEL_SZ, PANEL_SZ, 3), np.uint8)
            magenta_p[:, :, 0] = crop_nd2; magenta_p[:, :, 2] = crop_nd2
            zoom_panels.append((green_p_iv, magenta_p))

        def draw_base(frame):
            frame[jy_y0:jy_y0 + disp_jy_h, jy_x0:jy_x0 + disp_jy_w] = jy_disp
            frame[nd2_y0:nd2_y0 + disp_nd2_h, nd2_x0:nd2_x0 + disp_nd2_w] = nd2_disp
            put_text_mixed(frame, f'IN VIVO  z={MODE_Z}', (jy_x0 + 10, jy_y0 - 12),
                        FONT, 0.5, (0, 255, 0), 1)
            put_text_mixed(frame, f'EX VIVO  {tile}  z={nd2_z_mode}', (nd2_x0 + 10, nd2_y0 - 12),
                        FONT, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

        frame_count = 0

        # ═══ PHASE A: Side-by-side appear ═══
        for fi in range(N_A):
            t = ease(fi / max(1, N_A - 8))
            frame = np.zeros((H, W, 3), np.uint8)
            jy_alpha = 1.0 if is_first_tile else t
            jy_d = (jy_disp.astype(np.float32) * jy_alpha).astype(np.uint8)
            nd2_d = (nd2_disp.astype(np.float32) * t).astype(np.uint8)
            frame[jy_y0:jy_y0 + disp_jy_h, jy_x0:jy_x0 + disp_jy_w] = jy_d
            slide = int((1 - t) * 200)
            rx = nd2_x0 + slide; rw = min(disp_nd2_w, W - rx)
            if rw > 0:
                frame[nd2_y0:nd2_y0 + disp_nd2_h, rx:rx + rw] = nd2_d[:, :rw]
            jy_lbl_alpha = 1.0 if is_first_tile else t
            put_text_mixed(frame, f'IN VIVO  z={MODE_Z}', (jy_x0 + 10, jy_y0 - 12),
                        FONT, 0.5, tuple(int(v * jy_lbl_alpha) for v in (0, 255, 0)), 1)
            put_text_mixed(frame, f'EX VIVO  {tile}  z={nd2_z_mode}', (nd2_x0 + 10, nd2_y0 - 12),
                        FONT, 0.5, tuple(int(v * t) for v in (255, 0, 255)), 1)
            caption(frame, f'TILE  {tile.upper()}  --  NATIVE  SPACES', alpha=t)
            draw_scale_bar(frame, um_disp_jy, alpha=jy_lbl_alpha, x_right=sb_jy_xr, y_bottom=sb_yb)
            draw_scale_bar(frame, um_disp_nd2, alpha=t, x_right=sb_nd2_xr, y_bottom=sb_yb)
            write_frame(vw, frame); frame_count += 1

        # ═══ PHASE B: Landmarks + zoom panels ═══
        n_sel = len(SELECTED_GLOBAL)
        frames_per_lm = N_B // max(1, n_sel)
        for fi in range(N_B):
            frame = np.zeros((H, W, 3), np.uint8)
            draw_base(frame)

            current_lm = -1
            for li in range(n_sel):
                idx = SELECTED_GLOBAL[li]
                appear = li * frames_per_lm
                age = fi - appear
                if age < 0: continue

                jpt = lm_jy_disp[idx]; npt = lm_nd2_disp[idx]
                cv2.circle(frame, jpt, 8, GREEN, 2, cv2.LINE_AA)
                cv2.circle(frame, npt, 8, GREEN, 2, cv2.LINE_AA)

                progress = min(1.0, age / 14.0)
                if progress >= 1.0:
                    draw_arrow(frame, jpt, npt, GREEN, 2, 0.025)
                else:
                    mid_x = int(jpt[0] * (1 - progress) + npt[0] * progress)
                    mid_y = int(jpt[1] * (1 - progress) + npt[1] * progress)
                    cv2.line(frame, jpt, (mid_x, mid_y), GREEN, 2, cv2.LINE_AA)

                if age < frames_per_lm: current_lm = li

            if current_lm >= 0:
                zh, zg = zoom_panels[current_lm]
                appear = current_lm * frames_per_lm
                age = fi - appear
                p_alpha = ease(min(1.0, age / 8.0)) * ease(max(0, 1 - (age - frames_per_lm + 8) / 8.0))

                if p_alpha > 0.01:
                    pw = PANEL_SZ; gap = 30
                    total_pw = pw * 2 + gap
                    px_s = (W - total_pw) // 2
                    py_s = H - PANEL_SZ - 85
                    border = 2

                    hp = (zh.astype(np.float32) * p_alpha).astype(np.uint8)
                    frame[py_s:py_s + pw, px_s:px_s + pw] = hp
                    cv2.rectangle(frame, (px_s - border, py_s - border),
                                  (px_s + pw + border, py_s + pw + border), GREEN, border)

                    gp = (zg.astype(np.float32) * p_alpha).astype(np.uint8)
                    gx = px_s + pw + gap
                    frame[py_s:py_s + pw, gx:gx + pw] = gp
                    cv2.rectangle(frame, (gx - border, py_s - border),
                                  (gx + pw + border, py_s + pw + border), GREEN, border)

                    idx_sel = SELECTED_GLOBAL[current_lm]
                    jpt2 = lm_jy_disp[idx_sel]; npt2 = lm_nd2_disp[idx_sel]
                    lcol = tuple(int(v * p_alpha) for v in GREEN)
                    cv2.line(frame, jpt2, (px_s + pw // 2, py_s), lcol, 1, cv2.LINE_AA)
                    cv2.line(frame, npt2, (gx + pw // 2, py_s), lcol, 1, cv2.LINE_AA)

                    put_text_mixed(frame, 'IN VIVO', (px_s, py_s - 8), FONT, 0.38,
                                tuple(int(v * p_alpha) for v in (0, 255, 0)), 1)
                    put_text_mixed(frame, 'EX VIVO', (gx, py_s - 8), FONT, 0.38,
                                tuple(int(v * p_alpha) for v in (255, 0, 255)), 1)

            n_shown = sum(1 for li in range(n_sel) if fi >= li * frames_per_lm)
            caption(frame, f'MATCHED  LANDMARKS  ({n_shown}/{n_sel})')
            draw_scale_bar(frame, um_disp_jy, x_right=sb_jy_xr, y_bottom=sb_yb)
            draw_scale_bar(frame, um_disp_nd2, x_right=sb_nd2_xr, y_bottom=sb_yb)
            write_frame(vw, frame); frame_count += 1

        # ═══ PHASE C: Hold all arrows ═══
        for fi in range(N_C):
            frame = np.zeros((H, W, 3), np.uint8)
            draw_base(frame)
            for li in range(n_sel):
                idx = SELECTED_GLOBAL[li]
                draw_arrow(frame, lm_jy_disp[idx], lm_nd2_disp[idx], GREEN, 2, 0.025)
                cv2.circle(frame, lm_jy_disp[idx], 8, GREEN, 2, cv2.LINE_AA)
                cv2.circle(frame, lm_nd2_disp[idx], 8, GREEN, 2, cv2.LINE_AA)
            caption(frame, f'{n_sel}  MATCHED  CELLS  (z={MODE_Z})')
            draw_scale_bar(frame, um_disp_jy, x_right=sb_jy_xr, y_bottom=sb_yb)
            draw_scale_bar(frame, um_disp_nd2, x_right=sb_nd2_xr, y_bottom=sb_yb)
            write_frame(vw, frame); frame_count += 1

        # ═══ PHASE D: Centroid alignment ═══
        iv_cx = np.mean([lm_jy_disp[i][0] for i in range(n_lm)])
        iv_cy = np.mean([lm_jy_disp[i][1] for i in range(n_lm)])
        ev_cx = np.mean([lm_nd2_disp[i][0] for i in range(n_lm)])
        ev_cy = np.mean([lm_nd2_disp[i][1] for i in range(n_lm)])

        M_start_d = np.array([[scale_jy, 0, jy_x0],
                             [0, scale_jy, jy_y0],
                             [0, 0, 1]], dtype=np.float64)

        iv_cx_px = np.mean(iv[:, 2])
        iv_cy_px = np.mean(iv[:, 1])
        cx_start = iv_cx_px * scale_jy + jy_x0
        cy_start = iv_cy_px * scale_jy + jy_y0
        shift_x = ev_cx - cx_start
        shift_y = ev_cy - cy_start

        M_centroid = np.array([[scale_jy, 0, jy_x0 + shift_x],
                                [0, scale_jy, jy_y0 + shift_y],
                                [0, 0, 1]], dtype=np.float64)

        nd2_bg = np.zeros((H, W, 3), np.uint8)
        nd2_bg[nd2_y0:nd2_y0 + disp_nd2_h, nd2_x0:nd2_x0 + disp_nd2_w] = nd2_disp

        for fi in range(N_D):
            t = ease(fi / max(1, N_D - 8))
            frame = np.zeros((H, W, 3), np.uint8)
            frame[:, :, :] = nd2_bg

            M_t = M_start_d * (1 - t) + M_centroid * t
            warped = cv2.warpAffine(z_u8, M_t[:2].astype(np.float64), (W, H),
                                     flags=cv2.INTER_LANCZOS4, borderValue=0)
            w_hot = make_green(warped)
            mask = warped > 0
            frame[mask] = cv2.addWeighted(frame[mask], 0.5, w_hot[mask], 0.5, 0)

            arrow_alpha = 1 - ease(t * 3)
            if arrow_alpha > 0.01:
                for li in range(n_sel):
                    idx = SELECTED_GLOBAL[li]
                    acol = tuple(int(v * arrow_alpha) for v in GREEN)
                    cv2.circle(frame, lm_jy_disp[idx], 6, acol, 1, cv2.LINE_AA)
                    draw_arrow(frame, lm_jy_disp[idx], lm_nd2_disp[idx], acol, 1, 0.025)

            a_old = 1 - ease(fi / max(1, N_D * 0.3))
            a_new = ease((fi - N_D * 0.3) / max(1, N_D * 0.4))
            caption(frame, f'{n_sel}  MATCHED  CELLS  (z={MODE_Z})', alpha=max(0, a_old))
            caption(frame, 'CENTROID  ALIGNMENT', alpha=max(0, a_new))
            draw_scale_bar(frame, um_disp_nd2, x_right=sb_nd2_xr, y_bottom=sb_yb)
            write_frame(vw, frame); frame_count += 1

        # ═══ PHASE E: Affine warp M2d ═══
        M_end_e = np.array([
            [scale_nd2, 0, nd2_x0 - crop_x0 * scale_nd2],
            [0, scale_nd2, nd2_y0 - crop_y0 * scale_nd2],
            [0, 0, 1]
        ], dtype=np.float64) @ M3

        for fi in range(N_E):
            t = ease(fi / max(1, N_E - 11))
            frame = np.zeros((H, W, 3), np.uint8)
            frame[:, :, :] = nd2_bg

            M_t = M_centroid * (1 - t) + M_end_e * t
            warped = cv2.warpAffine(z_u8, M_t[:2].astype(np.float64), (W, H),
                                     flags=cv2.INTER_LANCZOS4, borderValue=0)
            w_hot = make_green(warped)
            mask = warped > 0
            frame[mask] = cv2.addWeighted(frame[mask], 0.5, w_hot[mask], 0.5, 0)

            a_new = ease((fi - 10) / 30)
            caption(frame, 'AFFINE  REGISTRATION  (M2d)', alpha=max(0, a_new))
            draw_scale_bar(frame, um_disp_nd2, x_right=sb_nd2_xr, y_bottom=sb_yb)
            write_frame(vw, frame); frame_count += 1

        # ═══ PHASE F: Final overlay + cell confirmation ═══
        warped_final = cv2.warpAffine(z_u8, M_end_e[:2].astype(np.float64), (W, H),
                                       flags=cv2.INTER_LANCZOS4, borderValue=0)
        w_final_hot = make_green(warped_final)

        final_base = nd2_bg.copy()
        mask_f = warped_final > 0
        final_base[mask_f] = cv2.addWeighted(final_base[mask_f], 0.5, w_final_hot[mask_f], 0.5, 0)

        lm_registered = []
        for i in range(n_lm):
            src = np.array([iv[i, 2], iv[i, 1], 1.0])
            dst_nd2 = M2d @ src
            dx = int((dst_nd2[0] - crop_x0) * scale_nd2) + nd2_x0
            dy = int((dst_nd2[1] - crop_y0) * scale_nd2) + nd2_y0
            lm_registered.append((dx, dy))

        for fi in range(N_F):
            frame = final_base.copy()
            n_show = min(n_sel, 1 + int(fi * n_sel / max(1, N_F * 0.55)))
            for ii in range(n_show):
                i = SELECTED_GLOBAL[ii]
                rpt = lm_registered[i]
                ept = lm_nd2_disp[i]
                cv2.circle(frame, ept, 8, GREEN, 2, cv2.LINE_AA)
                cv2.circle(frame, rpt, 6, (0, 140, 255), 2, cv2.LINE_AA)
            caption(frame, 'GREEN = IN VIVO    MAGENTA = EX VIVO')
            draw_scale_bar(frame, um_disp_nd2, x_right=sb_nd2_xr, y_bottom=sb_yb)
            write_frame(vw, frame); frame_count += 1

        # ═══ PHASE G: 3D rotation (only for full_3d tiles) ═══
        if full_3d:
            iv_z_min = max(0, int(iv[:, 0].min()) - 1)
            iv_z_max = min(nz_jy - 1, int(iv[:, 0].max()) + 1)
            z_range_3d = list(range(iv_z_min, iv_z_max + 1))

            overlay_slices_3d = []
            overlay_z_labels_3d = []

            for z_iv in z_range_3d:
                iv_u8_z = norm8(jy306[z_iv])
                warped_iv_z = cv2.warpAffine(iv_u8_z, M2d, (4200, 4200),
                                              flags=cv2.INTER_LINEAR, borderValue=0)
                warped_crop_z = warped_iv_z[crop_y0:crop_y1, crop_x0:crop_x1]

                best_ncc_z, best_nd2_z = -1, 0
                for zi in range(len(nd2_stack)):
                    nd2_full_z = nd2_stack[zi].astype(np.uint8)
                    nd2_c_z = nd2_full_z[crop_y0:min(crop_y1, nd2_full_z.shape[0]),
                                         crop_x0:min(crop_x1, nd2_full_z.shape[1])]
                    wc_z = warped_crop_z[:nd2_c_z.shape[0], :nd2_c_z.shape[1]]
                    wn_z = norm8(wc_z); nn_z = norm8(nd2_c_z)
                    mask_z = (wn_z > 5) & (nn_z > 5)
                    if mask_z.sum() < 100: continue
                    a_v = wn_z[mask_z].astype(np.float32); a_v -= a_v.mean()
                    b_v = nn_z[mask_z].astype(np.float32); b_v -= b_v.mean()
                    ncc_v = float(np.sum(a_v * b_v) / (np.sqrt(np.sum(a_v**2) * np.sum(b_v**2)) + 1e-8))
                    if ncc_v > best_ncc_z: best_ncc_z, best_nd2_z = ncc_v, zi

                nd2_best_z = nd2_stack[best_nd2_z].astype(np.uint8)
                nd2_c_best = nd2_best_z[crop_y0:min(crop_y1, nd2_best_z.shape[0]),
                                        crop_x0:min(crop_x1, nd2_best_z.shape[1])]
                wc_best = warped_crop_z[:nd2_c_best.shape[0], :nd2_c_best.shape[1]]

                ov_3d = np.zeros((nd2_c_best.shape[0], nd2_c_best.shape[1], 3), np.uint8)
                ev_u8 = norm8(nd2_c_best)
                ov_3d[:, :, 0] = ev_u8
                ov_3d[:, :, 2] = ev_u8
                ov_hot = make_green(norm8(wc_best))
                ov_3d = cv2.addWeighted(ov_3d, 0.5, ov_hot[:nd2_c_best.shape[0], :nd2_c_best.shape[1]], 0.5, 0)

                ov_small = cv2.resize(ov_3d, (disp_nd2_w, disp_nd2_h), interpolation=cv2.INTER_AREA)
                overlay_slices_3d.append(ov_small)
                overlay_z_labels_3d.append((z_iv, best_nd2_z))

            n_slices_3d = len(overlay_slices_3d)
            mid_idx_3d = z_range_3d.index(MODE_Z) if MODE_Z in z_range_3d else n_slices_3d // 2

            dense_slices_g = []
            dense_z_pos_g = []
            dense_real_idx_g = []
            for i in range(n_slices_3d):
                dense_slices_g.append(overlay_slices_3d[i])
                dense_z_pos_g.append(i * Z_SPACING)
                dense_real_idx_g.append(i)
                if i < n_slices_3d - 1:
                    for sub in range(1, INTERP_PER_GAP + 1):
                        t_sub = sub / (INTERP_PER_GAP + 1)
                        interp = (overlay_slices_3d[i].astype(np.float32) * (1 - t_sub) +
                                  overlay_slices_3d[i + 1].astype(np.float32) * t_sub)
                        dense_slices_g.append(interp.astype(np.uint8))
                        dense_z_pos_g.append(i * Z_SPACING + t_sub * Z_SPACING)
                        dense_real_idx_g.append(-1)

            dense_slices_g = np.array(dense_slices_g)
            dense_z_pos_g = np.array(dense_z_pos_g, dtype=np.float64)
            n_dense_g = len(dense_slices_g)
            STACK_CENTER_Z_G = (dense_z_pos_g[-1] + dense_z_pos_g[0]) / 2.0

            def render_3d_stack_g(rot_y, rot_x, slice_alphas, center=None):
                canvas = np.zeros((H, W, 3), dtype=np.float32)
                cx_r, cy_r = center if center else (W // 2, H // 2)
                cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
                cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

                z_depths = []
                for ii in range(n_dense_g):
                    dz = dense_z_pos_g[ii] - STACK_CENTER_Z_G
                    rz = cos_y * dz
                    rz2 = cos_x * rz
                    z_depths.append((rz2, ii))
                z_depths.sort(key=lambda x: x[0])

                for depth, ii in z_depths:
                    real_idx = dense_real_idx_g[ii]
                    if real_idx >= 0:
                        alpha = slice_alphas[real_idx] if real_idx < len(slice_alphas) else 0.5
                    else:
                        zp = dense_z_pos_g[ii]
                        z_below = int(zp / Z_SPACING)
                        z_above = min(n_slices_3d - 1, z_below + 1)
                        t_a = (zp - z_below * Z_SPACING) / Z_SPACING if Z_SPACING > 0 else 0
                        a_b = slice_alphas[z_below] if z_below < len(slice_alphas) else 0.5
                        a_a = slice_alphas[z_above] if z_above < len(slice_alphas) else 0.5
                        alpha = a_b * (1 - t_a) + a_a * t_a

                    if alpha < 0.01: continue

                    sl = dense_slices_g[ii].astype(np.float32) / 255.0
                    sh_, sw_ = sl.shape[:2]
                    hw, hh = sw_ / 2, sh_ / 2
                    dz = dense_z_pos_g[ii] - STACK_CENTER_Z_G

                    corners_3d = np.array([
                        [-hw, -hh, dz], [hw, -hh, dz],
                        [hw, hh, dz], [-hw, hh, dz],
                    ], dtype=np.float64)

                    rot_corners = []
                    for c in corners_3d:
                        rx_ = cos_y * c[0] + sin_y * c[2]
                        ry_ = c[1]
                        rz_ = -sin_y * c[0] + cos_y * c[2]
                        ry2 = cos_x * ry_ - sin_x * rz_
                        rot_corners.append([rx_ + cx_r, ry2 + cy_r])

                    rot_corners = np.array(rot_corners, dtype=np.float32)
                    src_corners = np.array([[0, 0], [sw_, 0], [sw_, sh_], [0, sh_]], dtype=np.float32)

                    M_persp = cv2.getPerspectiveTransform(src_corners, rot_corners)
                    warped_sl = cv2.warpPerspective(sl, M_persp, (W, H))

                    mask_sl = np.max(warped_sl, axis=2) > 0.01
                    mask3 = np.stack([mask_sl] * 3, axis=-1)
                    canvas = np.where(mask3, np.maximum(canvas, warped_sl * alpha), canvas)

                return np.clip(canvas * 255, 0, 255).astype(np.uint8)

            # G0: Slide to center
            start_cx = nd2_x0 + disp_nd2_w // 2
            start_cy = nd2_y0 + disp_nd2_h // 2
            end_cx = W // 2
            end_cy = H // 2
            mid_overlay = overlay_slices_3d[mid_idx_3d]

            for fi in range(36):
                t = ease(fi / 30)
                frame = np.zeros((H, W, 3), np.uint8)
                if t < 1.0:
                    frame = (final_base.astype(np.float32) * (1 - t)).astype(np.uint8)
                cur_cx = int(start_cx * (1 - t) + end_cx * t)
                cur_cy = int(start_cy * (1 - t) + end_cy * t)
                sh_, sw_ = mid_overlay.shape[:2]
                px = cur_cx - sw_ // 2; py = cur_cy - sh_ // 2
                src_x0_ = max(0, -px); src_y0_ = max(0, -py)
                dst_x0_ = max(0, px); dst_y0_ = max(0, py)
                dst_x1_ = min(W, px + sw_); dst_y1_ = min(H, py + sh_)
                if dst_x1_ > dst_x0_ and dst_y1_ > dst_y0_:
                    region = mid_overlay[src_y0_:src_y0_ + (dst_y1_ - dst_y0_),
                                         src_x0_:src_x0_ + (dst_x1_ - dst_x0_)]
                    alpha_sl = max(0.5, t)
                    existing = frame[dst_y0_:dst_y1_, dst_x0_:dst_x1_]
                    frame[dst_y0_:dst_y1_, dst_x0_:dst_x1_] = cv2.addWeighted(
                        existing, 1 - alpha_sl, region, alpha_sl, 0)
                caption(frame, 'GREEN = IN VIVO    MAGENTA = EX VIVO')
                write_frame(vw, frame); frame_count += 1

            # G1: Emerge
            for fi in range(48):
                t = ease(fi / 40)
                alphas = np.zeros(n_slices_3d, dtype=np.float32)
                alphas[mid_idx_3d] = 0.8
                for si in range(n_slices_3d):
                    if si == mid_idx_3d: continue
                    dist = abs(si - mid_idx_3d)
                    max_dist = t * (n_slices_3d - 1)
                    if dist <= max_dist:
                        alphas[si] = min(0.7, (max_dist - dist + 1) / 2.0) * t
                rot_x = INIT_ROT_X * t
                frame = render_3d_stack_g(0.0, rot_x, alphas)
                a_new = ease((fi - 10) / 20)
                caption(frame, f'3D DEPTH:  IN VIVO Z = {iv_z_min}  TO  Z = {iv_z_max}  ALIGNED',
                        alpha=max(0, a_new))
                write_frame(vw, frame); frame_count += 1

            # G2: Rotation
            alphas_full = np.ones(n_slices_3d, dtype=np.float32) * 0.7
            for fi in range(120):
                t = fi / 119.0
                rot_y = t * math.pi * 1.5
                rot_x = INIT_ROT_X + 0.15 * math.sin(t * math.pi)
                frame = render_3d_stack_g(rot_y, rot_x, alphas_full)
                caption(frame, f'MULTI-SLICE  ALIGNMENT  --  IN VIVO  Z = {iv_z_min}  TO  {iv_z_max}')
                write_frame(vw, frame); frame_count += 1

            # G3: Settle
            final_rot_y = math.pi * 1.5
            for fi in range(36):
                t = ease(fi / 30)
                rot_y = final_rot_y * (1 - t)
                rot_x = INIT_ROT_X
                frame = render_3d_stack_g(rot_y, rot_x, alphas_full)
                caption(frame, f'MULTI-SLICE  ALIGNMENT  --  IN VIVO  Z = {iv_z_min}  TO  {iv_z_max}')
                write_frame(vw, frame); frame_count += 1

            # G4: Hold
            for fi in range(24):
                frame = render_3d_stack_g(0.0, INIT_ROT_X, alphas_full)
                label_alpha = ease(fi / 12)
                for si, (z_iv_l, z_nd2_l) in enumerate(overlay_z_labels_3d):
                    ly = H // 2 - int((si - n_slices_3d / 2) * 28)
                    col = tuple(int(v * label_alpha) for v in WHITE)
                    cv2.putText(frame, f'z={z_iv_l} -- nd2 z={z_nd2_l}', (W - 280, ly),
                                FONT, 0.38, col, 1, cv2.LINE_AA)
                caption(frame, f'MULTI-SLICE  ALIGNMENT  --  {tile.upper()}')
                write_frame(vw, frame); frame_count += 1

        return frame_count

    # ── Main tile loop ──
    total = 0
    for ti, tile in enumerate(TILE_ORDER):
        is_first = (tile == FIRST_TILE)
        label = "FULL (with 3D)" if is_first else "fast"
        print(f"\n{'='*60}")
        print(f"[{ti+1}/{len(TILE_ORDER)}] {tile} — {label}")
        print(f"{'='*60}")
        n = render_tile(vw, tile, jy306, full_3d=False, is_first_tile=is_first)
        total += n
        print(f"  -> {n} frames ({n/FPS:.1f}s)")

    print(f"Scene 5 done: {total} frames ({total/FPS:.1f}s)")
    return total


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

    print("\n" + "=" * 70)
    print("RENDERING SCENE 1+2")
    print("=" * 70)
    n_s12 = render_scene1_2(vw)

    print("\n" + "=" * 70)
    print("RENDERING SCENE 3")
    print("=" * 70)
    n_s3 = render_scene3(vw)

    print("\n" + "=" * 70)
    print("RENDERING SCENE 5")
    print("=" * 70)
    n_s5 = render_scene5(vw)

    vw.release()

    total_fr = n_s12 + n_s3 + n_s5
    total_s = total_fr / FPS
    print(f"\nTotal: {total_fr} frames, {total_s:.1f}s @ {FPS}fps")
    print(f"  Scene 1+2: {n_s12} frames ({n_s12/FPS:.1f}s)")
    print(f"  Scene 3:   {n_s3} frames ({n_s3/FPS:.1f}s)")
    print(f"  Scene 5:   {n_s5} frames ({n_s5/FPS:.1f}s)")

    # ── Re-encode to H.264 ──
    print(f"\nRe-encoding to H.264… → {OUT}")
    subprocess.run([
        'ffmpeg', '-y', '-i', TMP,
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
        OUT
    ], capture_output=True)
    if os.path.exists(TMP):
        os.remove(TMP)
    print(f"Done: {OUT}")

    # ── 6x speedup version ──
    # Scene 5 starts at frame (n_s12 + n_s3). Each tile = 276 frames.
    # Normal: first 2 tiles (row2_1, row1_3) + scenes 1-3
    # 6x fast: tiles 3-17 (15 tiles)
    # Normal: last 2 tiles (row4_6, row5_1)
    scene5_start = n_s12 + n_s3
    frames_per_tile = 276
    n_tiles_total = 19  # from the tile order

    # Normal speed: scenes 1-3 + first 2 tiles of scene 5
    normal_start_frames = scene5_start + 2 * frames_per_tile
    # 6x: next 15 tiles
    fast_frames = 15 * frames_per_tile
    # Normal: last 2 tiles
    normal_end_start = normal_start_frames + fast_frames

    t_normal1_end = normal_start_frames / FPS
    t_fast_end = normal_end_start / FPS
    t_total = total_fr / FPS

    print(f"\nCreating 6x speedup version…")
    print(f"  Normal: 0 – {t_normal1_end:.1f}s (frames 0-{normal_start_frames-1})")
    print(f"  6x:     {t_normal1_end:.1f}s – {t_fast_end:.1f}s (frames {normal_start_frames}-{normal_end_start-1})")
    print(f"  Normal: {t_fast_end:.1f}s – {t_total:.1f}s (frames {normal_end_start}-{total_fr-1})")

    # Use ffmpeg filter_complex to:
    # 1. Split into 3 segments
    # 2. Speed up middle segment by 6x
    # 3. Concatenate
    TMP_6X = f'{BASE}/animation/merged_scenes_1_3_5_6x_raw.mp4'

    # trim uses seconds
    t1 = normal_start_frames / FPS
    t2 = normal_end_start / FPS

    cmd = [
        'ffmpeg', '-y', '-i', OUT,
        '-filter_complex',
        f'[0:v]split=3[v1][v2][v3];'
        f'[v1]trim=0:{t1},setpts=PTS-STARTPTS[p1];'
        f'[v2]trim={t1}:{t2},setpts=(PTS-STARTPTS)/6[p2];'
        f'[v3]trim={t2},setpts=PTS-STARTPTS[p3];'
        f'[p1][p2][p3]concat=n=3:v=1:a=0[out]',
        '-map', '[out]',
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
        OUT_6X
    ]
    print(f"Running ffmpeg for 6x version…")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr[:500]}")
    else:
        fast_dur = fast_frames / FPS / 6
        total_6x = t1 + fast_dur + (t_total - t2)
        print(f"Done: {OUT_6X}")
        print(f"  6x version duration: ~{total_6x:.1f}s (vs {t_total:.1f}s original)")
