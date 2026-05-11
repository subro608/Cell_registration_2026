"""
Scene 5 — All tiles registration animation.

- row2_1: full animation with 3D rotation (phases A-G, ~32s)
- Other tiles: phases A-F only, faster (~10s each)
- Final: stitched 3D overlay rotation from scene5b assets (~10s)

Output: animation/scene5_all_tiles_v2_h264.mp4
"""

import numpy as np, cv2, tifffile, math, subprocess, os, glob
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
TMP  = f'{BASE}/animation/scene5_all_v2_raw.mp4'
OUT  = f'{BASE}/animation/scene5_all_tiles_v2_h264.mp4'

W, H = 1920, 1080
FPS  = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (0, 220, 0)

from text_utils import put_text_mixed, text_width_mixed

Z_SPACING = 4
INTERP_PER_GAP = 3
INIT_ROT_X = -0.3

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
    """Write frame with axis widget overlay."""
    draw_axes(frame, alpha)
    vw.write(frame)

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

def norm8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 50: return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

def make_green(u8):
    g = np.zeros((*u8.shape, 3), np.uint8)
    g[:, :, 1] = u8
    return g

def make_magenta(u8):
    m = np.zeros((*u8.shape, 3), np.uint8)
    m[:, :, 0] = u8  # B
    m[:, :, 2] = u8  # R
    return m

def caption(frame, text, alpha=1.0):
    if alpha < 0.01: return
    ts = 0.72
    tw = text_width_mixed(text, ts)
    x = (W - tw) // 2; y = H - 42
    col = tuple(int(v * alpha) for v in WHITE)
    put_text_mixed(frame, text, (x, y), FONT, ts, col, 1)

# ── Scale bar ──
# Pixel sizes from microscope metadata (README.md)
IV_XY_UM = 0.82     # JY306 in-vivo confocal (µm/px)
ND2_XY_UM = 0.65    # nd2 ex-vivo confocal (µm/px)

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

# ── Load shared data ──
print("Loading JY306 z-stack (shared across all tiles)...")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_jy, hy_jy, wx_jy = jy306.shape
jy_p1, jy_p2 = np.percentile(jy306[jy306 > 0], [1, 99.5])

# ── Tile order ──
ALL_TILES = sorted([d for d in os.listdir(f'{BASE}/png_exports/registration_per_tile_pkl')
                    if os.path.isdir(f'{BASE}/png_exports/registration_per_tile_pkl/{d}')])
# row2_1 first, then rest in order
FIRST_TILE = 'row2_1'
OTHER_TILES = [t for t in ALL_TILES if t != FIRST_TILE]
TILE_ORDER = [FIRST_TILE] + OTHER_TILES
print(f"{len(TILE_ORDER)} tiles: {TILE_ORDER}")


def render_tile(vw, tile, jy306, full_3d=False, is_first_tile=False):
    """Render all phases for one tile. full_3d=True adds 3D rotation phases.
    is_first_tile=True: in-vivo already visible from frame 0 (matches scene 4 ending)."""

    # ── Timing (fast for non-first tiles) ──
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

    # Mode z for this tile
    MODE_Z = int(round(np.median(iv[:, 0])))
    MODE_Z = max(0, min(nz_jy - 1, MODE_Z))

    # Best nd2 z
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
    nd2_magenta_full[:, :, 0] = nd2_u8  # B } ex-vivo = magenta
    nd2_magenta_full[:, :, 2] = nd2_u8  # R }

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

    # Scale bar parameters (µm per display pixel)
    um_disp_jy = IV_XY_UM / scale_jy
    um_disp_nd2 = ND2_XY_UM / scale_nd2
    # Position: bottom of each image panel
    sb_jy_xr = jy_x0 + disp_jy_w - 15
    sb_nd2_xr = nd2_x0 + disp_nd2_w - 15
    sb_yb = jy_y0 + disp_jy_h - 15

    # Landmark display positions
    lm_jy_disp = []
    lm_nd2_disp = []
    for i in range(n_lm):
        dx = int(iv[i, 2] * scale_jy) + jy_x0
        dy = int(iv[i, 1] * scale_jy) + jy_y0
        lm_jy_disp.append((dx, dy))
        dx2 = int((ev[i, 0] - crop_x0) * scale_nd2) + nd2_x0
        dy2 = int((ev[i, 1] - crop_y0) * scale_nd2) + nd2_y0
        lm_nd2_disp.append((dx2, dy2))

    # ── Filter landmarks to displayed z-slices ──
    # Only show landmarks present in the current invivo z (MODE_Z) and exvivo z (nd2_z_mode)
    z_matched = [i for i in range(n_lm)
                 if int(round(iv[i, 0])) == MODE_Z and int(round(ev[i, 2])) == nd2_z_mode]
    if not z_matched:
        # Fallback: landmarks matching just the invivo z
        z_matched = [i for i in range(n_lm) if int(round(iv[i, 0])) == MODE_Z]
    if not z_matched:
        # Last fallback: all landmarks
        z_matched = list(range(n_lm))

    # ── Zoom panels (exact z-slice, not MIP) ──
    CROP_R_JY = 40; CROP_R_ND2 = 100; PANEL_SZ = 110
    lm_scores = []
    for i in z_matched:
        y_i = int(round(iv[i, 1])); x_i = int(round(iv[i, 2]))
        r = CROP_R_JY
        crop = jy306[MODE_Z, max(0,y_i-r):min(hy_jy,y_i+r), max(0,x_i-r):min(wx_jy,x_i+r)]
        lm_scores.append(float(crop.std()))
    SELECTED = sorted(range(len(z_matched)), key=lambda i: -lm_scores[i])[:N_TOP_LM + 1]
    # Remove the 6th landmark (index 5) — user-requested exclusion
    if len(SELECTED) > 5:
        SELECTED.pop(5)
    SELECTED = SELECTED[:N_TOP_LM]
    # Map SELECTED indices back to global landmark indices
    SELECTED_GLOBAL = [z_matched[s] for s in SELECTED]

    zoom_panels = []
    for idx in SELECTED_GLOBAL:
        y_lm = int(round(iv[idx, 1])); x_lm = int(round(iv[idx, 2]))
        # Exact z-slice, not MIP
        jy_slice = jy306[MODE_Z]
        y0 = max(0, y_lm - CROP_R_JY); y1 = min(hy_jy, y_lm + CROP_R_JY)
        x0 = max(0, x_lm - CROP_R_JY); x1 = min(wx_jy, x_lm + CROP_R_JY)
        crop_n = np.clip((jy_slice[y0:y1, x0:x1] - jy_p1) / max(jy_p2 - jy_p1, 1) * 255, 0, 255).astype(np.uint8)
        crop_n = cv2.resize(crop_n, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
        green_p_iv = np.zeros((PANEL_SZ, PANEL_SZ, 3), np.uint8); green_p_iv[:, :, 1] = crop_n

        x_nd2 = int(round(ev[idx, 0])); y_nd2 = int(round(ev[idx, 1]))
        # Exact z-slice for exvivo too
        nd2_slice = nd2_stack[nd2_z_mode]
        yn0 = max(0, y_nd2 - CROP_R_ND2); yn1 = min(4200, y_nd2 + CROP_R_ND2)
        xn0 = max(0, x_nd2 - CROP_R_ND2); xn1 = min(4200, x_nd2 + CROP_R_ND2)
        crop_nd2 = norm8(nd2_slice[yn0:yn1, xn0:xn1])
        crop_nd2 = cv2.resize(crop_nd2, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
        magenta_p = np.zeros((PANEL_SZ, PANEL_SZ, 3), np.uint8)
        magenta_p[:, :, 0] = crop_nd2; magenta_p[:, :, 2] = crop_nd2
        zoom_panels.append((green_p_iv, magenta_p))

    # ── Helpers ──
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
        # First tile: in-vivo already visible (matches scene 4 ending)
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

        # Zoom panel
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

    M_start = np.array([[scale_jy, 0, jy_x0],
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

        M_t = M_start * (1 - t) + M_centroid * t
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
    M_end = np.array([
        [scale_nd2, 0, nd2_x0 - crop_x0 * scale_nd2],
        [0, scale_nd2, nd2_y0 - crop_y0 * scale_nd2],
        [0, 0, 1]
    ], dtype=np.float64) @ M3

    for fi in range(N_E):
        t = ease(fi / max(1, N_E - 11))
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:, :, :] = nd2_bg

        M_t = M_centroid * (1 - t) + M_end * t
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
    warped_final = cv2.warpAffine(z_u8, M_end[:2].astype(np.float64), (W, H),
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
        # Build multi-z overlay slices
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
            ov_3d[:, :, 0] = ev_u8  # B } ex-vivo = magenta
            ov_3d[:, :, 2] = ev_u8  # R }
            ov_hot = make_green(norm8(wc_best))
            ov_3d = cv2.addWeighted(ov_3d, 0.5, ov_hot[:nd2_c_best.shape[0], :nd2_c_best.shape[1]], 0.5, 0)

            ov_small = cv2.resize(ov_3d, (disp_nd2_w, disp_nd2_h), interpolation=cv2.INTER_AREA)
            overlay_slices_3d.append(ov_small)
            overlay_z_labels_3d.append((z_iv, best_nd2_z))

        n_slices_3d = len(overlay_slices_3d)
        mid_idx_3d = z_range_3d.index(MODE_Z) if MODE_Z in z_range_3d else n_slices_3d // 2

        # Gaussian interpolation
        dense_slices = []
        dense_z_pos = []
        dense_real_idx = []
        for i in range(n_slices_3d):
            dense_slices.append(overlay_slices_3d[i])
            dense_z_pos.append(i * Z_SPACING)
            dense_real_idx.append(i)
            if i < n_slices_3d - 1:
                for sub in range(1, INTERP_PER_GAP + 1):
                    t_sub = sub / (INTERP_PER_GAP + 1)
                    interp = (overlay_slices_3d[i].astype(np.float32) * (1 - t_sub) +
                              overlay_slices_3d[i + 1].astype(np.float32) * t_sub)
                    dense_slices.append(interp.astype(np.uint8))
                    dense_z_pos.append(i * Z_SPACING + t_sub * Z_SPACING)
                    dense_real_idx.append(-1)

        dense_slices = np.array(dense_slices)
        dense_z_pos = np.array(dense_z_pos, dtype=np.float64)
        n_dense = len(dense_slices)
        STACK_CENTER_Z = (dense_z_pos[-1] + dense_z_pos[0]) / 2.0

        def render_3d_stack(rot_y, rot_x, slice_alphas, center=None):
            canvas = np.zeros((H, W, 3), dtype=np.float32)
            cx, cy = center if center else (W // 2, H // 2)
            cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
            cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

            z_depths = []
            for ii in range(n_dense):
                dz = dense_z_pos[ii] - STACK_CENTER_Z
                rz = cos_y * dz
                rz2 = cos_x * rz
                z_depths.append((rz2, ii))
            z_depths.sort(key=lambda x: x[0])

            for depth, ii in z_depths:
                real_idx = dense_real_idx[ii]
                if real_idx >= 0:
                    alpha = slice_alphas[real_idx] if real_idx < len(slice_alphas) else 0.5
                else:
                    zp = dense_z_pos[ii]
                    z_below = int(zp / Z_SPACING)
                    z_above = min(n_slices_3d - 1, z_below + 1)
                    t_a = (zp - z_below * Z_SPACING) / Z_SPACING if Z_SPACING > 0 else 0
                    a_b = slice_alphas[z_below] if z_below < len(slice_alphas) else 0.5
                    a_a = slice_alphas[z_above] if z_above < len(slice_alphas) else 0.5
                    alpha = a_b * (1 - t_a) + a_a * t_a

                if alpha < 0.01: continue

                sl = dense_slices[ii].astype(np.float32) / 255.0
                sh_, sw_ = sl.shape[:2]
                hw, hh = sw_ / 2, sh_ / 2
                dz = dense_z_pos[ii] - STACK_CENTER_Z

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
                    rot_corners.append([rx_ + cx, ry2 + cy])

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
            frame = render_3d_stack(0.0, rot_x, alphas)
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
            frame = render_3d_stack(rot_y, rot_x, alphas_full)
            caption(frame, f'MULTI-SLICE  ALIGNMENT  --  IN VIVO  Z = {iv_z_min}  TO  {iv_z_max}')
            write_frame(vw, frame); frame_count += 1

        # G3: Settle
        final_rot_y = math.pi * 1.5
        for fi in range(36):
            t = ease(fi / 30)
            rot_y = final_rot_y * (1 - t)
            rot_x = INIT_ROT_X
            frame = render_3d_stack(rot_y, rot_x, alphas_full)
            caption(frame, f'MULTI-SLICE  ALIGNMENT  --  IN VIVO  Z = {iv_z_min}  TO  {iv_z_max}')
            write_frame(vw, frame); frame_count += 1

        # G4: Hold
        for fi in range(24):
            frame = render_3d_stack(0.0, INIT_ROT_X, alphas_full)
            label_alpha = ease(fi / 12)
            for si, (z_iv_l, z_nd2_l) in enumerate(overlay_z_labels_3d):
                ly = H // 2 - int((si - n_slices_3d / 2) * 28)
                col = tuple(int(v * label_alpha) for v in WHITE)
                cv2.putText(frame, f'z={z_iv_l} -- nd2 z={z_nd2_l}', (W - 280, ly),
                            FONT, 0.38, col, 1, cv2.LINE_AA)
            caption(frame, f'MULTI-SLICE  ALIGNMENT  --  {tile.upper()}')
            write_frame(vw, frame); frame_count += 1

    return frame_count


# ══════════════════════════════════════════════════════════════════
# MAIN: Render all tiles + final stitched 3D
# ══════════════════════════════════════════════════════════════════
vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
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

# ── Finalize ──
vw.release()
print(f"\nRe-encoding to H.264... ({total} frames, {total / FPS:.1f}s)")
subprocess.run([
    'ffmpeg', '-y', '-i', TMP,
    '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
    OUT
], capture_output=True)
os.remove(TMP)
print(f"Done! {total} frames, {total / FPS:.1f}s @ {FPS}fps -- {OUT}")