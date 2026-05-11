"""
Scene 7: Cell identity cards — continues from scene5b last frame.

Uses scene5b's exact 3D rendering pipeline (crop, scale, gaussian interp).
Marks landmarks in the rotating dual volume, shows 4 panels below.

Step 1: Generate frames to animation/frames_scene7/ for QC
Step 2: Run with --encode to assemble video

Output: animation/scene7_h264.mp4
"""

import numpy as np, cv2, math, subprocess, os, glob, sys, tifffile, re, json
import pandas as pd
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
PATCH_DIR = '/Users/neurolab/neuroinformatics/invivo-exvivo-cell-registration/patches'
FRAME_DIR = f'{BASE}/animation/frames_scene7_v5'
OUT = f'{BASE}/animation/scene7_v5_h264.mp4'
# (overlay PNGs no longer used — dots rendered directly from CSV)

W, H = 1920, 1080
FPS = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

from text_utils import put_text_mixed, text_width_mixed

# Selected cells: (tile, local_landmark_index)
SELECTED_CELLS = [
    ('row1_3', 6),
    ('row1_3', 9),
    ('row2_1', 5),
    ('row2_1', 0),   # original global #13
    ('row2_1', 4),   # original global #17
]

VAROL = f'{BASE}/jy306_varol'
PKL_MERC_DIR = f'{BASE}/merscope_exvivo '
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

ENCODE_ONLY = '--encode' in sys.argv

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

def norm_u8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 50:
        return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img.astype(np.float32) - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

# ── Scale bar ──
ND2_XY_UM = 0.65   # nd2 ex-vivo pixel size (µm/px)
IV_XY_UM = 0.82    # in-vivo pixel size (µm/px)

def draw_scale_bar(frame, um_per_disp_px, alpha=1.0, x_right=None, y_bottom=None):
    """Draw a scale bar with physical distance label."""
    return  # disabled
    if x_right is None:
        x_right = W - 50
    if y_bottom is None:
        y_bottom = H - 75
    bar_px = 120
    bar_um = bar_px * um_per_disp_px
    if bar_um >= 10:
        bar_um_label = f'{int(round(bar_um))} um'
    else:
        bar_um_label = f'{bar_um:.1f} um'
    x_left = x_right - bar_px
    y_bar = y_bottom
    col = tuple(int(255 * alpha) for _ in range(3))
    bg = (0, 0, 0)
    cv2.rectangle(frame, (x_left - 1, y_bar - 4), (x_right + 1, y_bar + 4), bg, -1)
    cv2.rectangle(frame, (x_left, y_bar - 3), (x_right, y_bar + 3), col, -1)
    label = bar_um_label
    ts = 0.45
    (tw, _), _ = cv2.getTextSize(label, FONT, ts, 1)
    tx = x_left + (bar_px - tw) // 2
    ty = y_bar - 8
    cv2.putText(frame, label, (tx, ty), FONT, ts, bg, 3, cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty), FONT, ts, col, 1, cv2.LINE_AA)

if ENCODE_ONLY:
    n_frames = len(glob.glob(f'{FRAME_DIR}/frame_*.png'))
    print(f"Encoding {n_frames} frames...")
    subprocess.run([
        'ffmpeg', '-y', '-framerate', str(FPS),
        '-i', f'{FRAME_DIR}/frame_%05d.png',
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
        OUT
    ], capture_output=True)
    print(f"Done! -> {OUT}")
    sys.exit(0)

# ══════════════════════════════════════════════════════════════
# LOAD PRE-COMPUTED ASSETS
# ══════════════════════════════════════════════════════════════
import pickle
print("Loading pre-computed assets...")
with open(f'{BASE}/animation/scene7_assets.pkl', 'rb') as f:
    assets = pickle.load(f)

INIT_ROT_X = -0.3
CANVAS_UM_PER_PX = 0.65

dense_slices = assets['vol_sub']
dense_z_pos = assets['z_sub']
disp_w = assets['disp_w']
disp_h = assets['disp_h']

n_dense = len(dense_slices)
CENTER_Z = (dense_z_pos[-1] + dense_z_pos[0]) / 2.0
print(f"  {n_dense} slices, {disp_w}x{disp_h}")

um_disp_3d = CANVAS_UM_PER_PX

RENDER_SCALE = 0.5
RW, RH = int(W * RENDER_SCALE), int(H * RENDER_SCALE)

dense_f32 = [sl.astype(np.float32) / 255.0 for sl in dense_slices]

def render_3d(rot_y, rot_x, marker_display_xyz=None, alpha_val=0.85):
    """Render full stitched 3D volume with optional landmark marker."""
    canvas = np.zeros((RH, RW, 3), dtype=np.float32)
    cx, cy = RW / 2, RH / 2
    s = RENDER_SCALE
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    z_depths = []
    for i in range(len(dense_slices)):
        dz = dense_z_pos[i] - CENTER_Z
        rz2 = cos_x * (cos_y * dz)
        z_depths.append((rz2, i))
    z_depths.sort(key=lambda x: x[0])

    for depth, i in z_depths:
        sl = dense_f32[i]
        sh, sw = sl.shape[:2]
        hw, hh = sw / 2, sh / 2
        dz = dense_z_pos[i] - CENTER_Z

        corners_3d = np.array([[-hw,-hh,dz],[hw,-hh,dz],[hw,hh,dz],[-hw,hh,dz]], dtype=np.float64)
        rot_corners = np.empty((4, 2), dtype=np.float32)
        for j, c in enumerate(corners_3d):
            rx = cos_y * c[0] + sin_y * c[2]
            ry = c[1]
            rz = -sin_y * c[0] + cos_y * c[2]
            ry2 = cos_x * ry - sin_x * rz
            rot_corners[j] = [rx * s + cx, ry2 * s + cy]

        src_corners = np.array([[0,0],[sw,0],[sw,sh],[0,sh]], dtype=np.float32)
        M_persp = cv2.getPerspectiveTransform(src_corners, rot_corners)
        warped = cv2.warpPerspective(sl, M_persp, (RW, RH))
        mask = np.max(warped, axis=2) > 0.01
        mask3 = np.stack([mask] * 3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, warped * alpha_val), canvas)

    # Upscale to full res
    result_f = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_LINEAR)
    result = np.clip(result_f * 255, 0, 255).astype(np.uint8)

    # Draw 3-axis trident (bottom-left corner, rotates with volume)
    # Each axis extends both directions from origin, arrow tip on positive end
    ax_cx, ax_cy = 150, H - 160  # center of axis widget
    ax_len = 100  # half-length in pixels (extends ±ax_len from center)
    # 3D unit vectors: X=ML (horizontal), Y=AP (vertical), Z=DV (depth)
    axes_3d = [
        (1, 0, 0),     # ML — horizontal in slice
        (0, -1, 0),    # AP — vertical in slice (negative = up)
        (0, 0, 1),     # DV — depth (z-stack, positive = down/ventral)
    ]
    ax_colors = [(0, 0, 180), (40, 40, 40), (200, 80, 0)]  # ML=dark red, AP=black, DV=blue
    ax_labels = ['ML', 'AP', 'DV']
    for ai, (ux, uy, uz) in enumerate(axes_3d):
        # Rotate unit vector same as volume
        rx = cos_y * ux + sin_y * uz
        ry = uy
        rz = -sin_y * ux + cos_y * uz
        ry2 = cos_x * ry - sin_x * rz
        # Positive end (with arrow tip + label)
        px, py = int(ax_cx + rx * ax_len), int(ax_cy + ry2 * ax_len)
        # Negative end (thin line, no arrow)
        nx, ny = int(ax_cx - rx * ax_len), int(ax_cy - ry2 * ax_len)
        # Draw full line through origin
        cv2.line(result, (nx, ny), (ax_cx, ax_cy), ax_colors[ai], 2, cv2.LINE_AA)
        # Arrow on positive half
        cv2.arrowedLine(result, (ax_cx, ax_cy), (px, py), ax_colors[ai], 3, cv2.LINE_AA, tipLength=0.15)
        # Label at positive tip
        dx_tip = rx / max(abs(rx), abs(ry2), 1) * 18 if abs(rx) > 3 else 0
        dy_tip = ry2 / max(abs(rx), abs(ry2), 1) * 18 if abs(ry2) > 3 else -12
        lx, ly = int(px + dx_tip), int(py + dy_tip)
        cv2.putText(result, ax_labels[ai], (lx - 10, ly + 5), FONT, 0.65, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(result, ax_labels[ai], (lx - 10, ly + 5), FONT, 0.65, ax_colors[ai], 1, cv2.LINE_AA)
    # Origin dot
    cv2.circle(result, (ax_cx, ax_cy), 4, (200, 200, 200), -1, cv2.LINE_AA)

    # Draw landmark marker (at full res)
    if marker_display_xyz is not None:
        mx, my, mz = marker_display_xyz
        dz = mz - CENTER_Z
        rx = cos_y * mx + sin_y * dz
        ry = my
        rz = -sin_y * mx + cos_y * dz
        ry2 = cos_x * ry - sin_x * rz
        sx, sy = int(rx + W/2), int(ry2 + H/2)
        if 10 < sx < W - 10 and 10 < sy < H - 10:
            for r, c, t in [(18, (0,180,180), 1), (12, (0,255,255), 2), (6, (0,255,255), -1)]:
                cv2.circle(result, (sx, sy), r, c, t, cv2.LINE_AA)

    return result

# ══════════════════════════════════════════════════════════════
# UNPACK PRE-COMPUTED CELL DATA
# ══════════════════════════════════════════════════════════════
CROP_SM = 130
CROP_LG = 400
PANEL_SZ = 180

panels_sm = {}
panels_lg = {}
lm_display = {}
cell_dot_store = {}
cal_patches_sm = {}
cal_patches_lg = {}
cell_iv_z = {}

for ci, cell_data in assets['cells'].items():
    tile = cell_data['tile']
    local = cell_data['local']
    lm_display[ci] = cell_data['lm_display']
    cell_iv_z[ci] = cell_data['iv_z']
    panels_sm[ci] = cell_data['panels_sm']
    panels_lg[ci] = cell_data['panels_lg']
    cal_patches_sm[ci] = cell_data['cal_sm']
    cal_patches_lg[ci] = cell_data['cal_lg']
    dot_rx = cell_data['dot_rel_x']
    dot_ry = cell_data['dot_rel_y']
    dot_cols = cell_data['dot_colors']
    cell_dot_store[ci] = (dot_rx, dot_ry, dot_cols) if dot_rx is not None else None
    print(f"  Cell {ci}: {tile} #{local}, marker={lm_display[ci]}")

del assets['cells']
print("Assets loaded.")

# ══════════════════════════════════════════════════════════════
# RENDER FRAMES
# ══════════════════════════════════════════════════════════════
print("Rendering frames...")
os.makedirs(FRAME_DIR, exist_ok=True)

ROT_Y_START = 0.0
# Phase durations (frames)
N_FADE_IN = 18       # fade in from black
N_ROTATE = 96        # 3D rotation with small panels
N_HOLD = 36          # hold (3D + panels, calcium keeps playing)
N_ZOOM_IN = 36       # panels zoom in together (1x → 3x) to show gene dot diversity
N_ZOOM_HOLD = 48     # hold zoomed in
N_ZOOM_OUT = 36      # panels zoom back out (3x → 1x)
N_CONVERGE = 24      # 4 panels slide horizontally to center (same y)
N_MERGE = 36         # stacked panels merge into 1 overlay (3D visible)
N_HOLD_MERGE = 48    # hold merged overlay (3D visible)
N_FADE_OUT = 18      # fade to black

PANEL_SM = PANEL_SZ  # small panel size (180)
PANEL_LG = 400       # large panel size
GAP_SM = 20
GAP_LG = 30
LABELS = ['GCaMP IN VIVO FUNCTIONAL', 'GCaMP IN VIVO STATIC', 'GCaMP EX VIVO STATIC', 'MERSCOPE mRNA EXPRESSION']

def get_cal_frame(cal_frames, fi_total):
    """Get calcium frame at given counter."""
    cal_idx = (fi_total * 4) % len(cal_frames)
    return cal_frames[cal_idx]

def draw_panels(frame, panel_list, labels, psz, gap, alpha=1.0, title=None, y_pos=None):
    """Draw 4 panels on screen at given size. y_pos overrides vertical position."""
    total_pw = psz * 4 + gap * 3
    x_start = (W - total_pw) // 2
    y_bottom = H - psz - 60          # bottom position (during 3D rotation)
    y_center = (H - psz) // 2        # center position (enlarged)
    y_start = y_pos if y_pos is not None else (y_center if psz >= 300 else y_bottom)

    for pi, (panel, label) in enumerate(zip(panel_list, labels)):
        px = x_start + pi * (psz + gap)
        p_resized = cv2.resize(panel, (psz, psz), interpolation=cv2.INTER_LANCZOS4)
        if alpha < 0.99:
            p_resized = (p_resized.astype(np.float32) * alpha).astype(np.uint8)
        fy = y_start
        # Clip to frame bounds
        py0 = max(0, fy); py1 = min(H, fy + psz)
        px0 = max(0, px); px1 = min(W, px + psz)
        sy0 = py0 - fy; sy1 = sy0 + (py1 - py0)
        sx0 = px0 - px; sx1 = sx0 + (px1 - px0)
        if py1 > py0 and px1 > px0:
            frame[py0:py1, px0:px1] = p_resized[sy0:sy1, sx0:sx1]
        bc = int(50 * alpha)
        cv2.rectangle(frame, (px0, py0), (px1 - 1, py1 - 1), (bc,bc,bc), 1)
        lc = int(180 * alpha)
        font_scale = 0.38 if psz < 300 else 0.55
        (tw, _), _ = cv2.getTextSize(label, FONT, font_scale, 1)
        ly = min(H - 5, py1 + 22)
        put_text_mixed(frame, label, (px + (psz - tw) // 2, ly),
                    FONT, font_scale, (lc, lc, lc), 1)

    if title:
        tc = int(240 * alpha)
        font_scale = 0.5 if psz < 300 else 0.7
        (tw, _), _ = cv2.getTextSize(title, FONT, font_scale, 1)
        ty = max(20, y_start - 15)
        put_text_mixed(frame, title, ((W - tw) // 2, ty),
                    FONT, font_scale, (tc, tc, tc), 1)

def render_dots_zoomed(cell_idx, crop_r, target_sz, dot_radius=1):
    """Re-render MERSCOPE dots for a cell at given crop/radius. Dots only, no background."""
    cdd = cell_dot_store.get(cell_idx)
    canvas = np.zeros((crop_r * 2, crop_r * 2, 3), np.uint8)
    if cdd is not None:
        rel_x, rel_y, cols = cdd
        in_r = (np.abs(rel_x) < crop_r) & (np.abs(rel_y) < crop_r)
        px = (rel_x[in_r] + crop_r).astype(int)
        py = (rel_y[in_r] + crop_r).astype(int)
        c = cols[in_r]
        if dot_radius <= 1:
            valid = (px >= 0) & (px < crop_r*2) & (py >= 0) & (py < crop_r*2)
            canvas[py[valid], px[valid]] = c[valid]
        else:
            for j in range(len(px)):
                if 0 <= px[j] < crop_r*2 and 0 <= py[j] < crop_r*2:
                    cv2.circle(canvas, (int(px[j]), int(py[j])), dot_radius,
                               tuple(int(v) for v in c[j]), -1, cv2.LINE_AA)
    return cv2.resize(canvas, (target_sz, target_sz), interpolation=cv2.INTER_LANCZOS4)

frame_idx = 0

def blend_panels(sm_set, lg_set, t):
    """Crossfade between small-crop and large-crop panels (t=0 → sm, t=1 → lg)."""
    result = []
    for s, l in zip(sm_set, lg_set):
        blended = cv2.addWeighted(s, 1.0 - t, l, t, 0)
        result.append(blended)
    return result

for ci, (tile, local) in enumerate(SELECTED_CELLS):
    if ci not in cal_patches_sm or ci not in panels_sm:
        print(f"  Skipping cell {ci} ({tile} #{local}) (missing data)")
        continue

    sm = panels_sm[ci]  # (iv, ev, gd) small crop
    lg = panels_lg[ci]  # (iv, ev, gd) large crop
    cal_sm = cal_patches_sm[ci]
    cal_lg = cal_patches_lg[ci]
    mx, my, mz = lm_display[ci]
    title = f'{tile.upper()}  --  z={cell_iv_z[ci]:.0f}'

    cal_counter = 0

    print(f"  Cell {ci+1}/{len(SELECTED_CELLS)} ({tile} local#{local})")

    # ── Phase A: Fade in + 3D rotation with small panels ──
    for fi in range(N_FADE_IN + N_ROTATE):
        if fi < N_FADE_IN:
            alpha = ease(fi / (N_FADE_IN - 1))
        else:
            alpha = 1.0

        rot_fi = fi - N_FADE_IN if fi >= N_FADE_IN else 0
        rot_y = ROT_Y_START + (rot_fi / N_ROTATE) * math.pi * 2

        frame = render_3d(rot_y, INIT_ROT_X, marker_display_xyz=(mx, my, mz))
        if alpha < 0.99:
            frame = (frame.astype(np.float32) * alpha).astype(np.uint8)

        cal_disp = get_cal_frame(cal_sm, cal_counter)
        panel_list = [cal_disp, sm[0], sm[1], sm[2]]
        draw_panels(frame, panel_list, LABELS, PANEL_SM, GAP_SM, alpha, title)

        draw_scale_bar(frame, um_disp_3d, alpha=alpha, x_right=W-30, y_bottom=H - PANEL_SM - 110)
        cv2.imwrite(f'{FRAME_DIR}/frame_{frame_idx:05d}.png', frame)
        frame_idx += 1
        cal_counter += 1

    # ── Phase B: Hold — 3D keeps rotating slowly + 4 panels at bottom ──
    rot_after_A = ROT_Y_START + math.pi * 2
    for fi in range(N_HOLD):
        rot_y = rot_after_A + (fi / max(1, N_HOLD - 1)) * math.pi * 0.5
        frame = render_3d(rot_y, INIT_ROT_X, marker_display_xyz=(mx, my, mz))

        cal_disp = get_cal_frame(cal_sm, cal_counter)
        panel_list = [cal_disp, sm[0], sm[1], sm[2]]
        draw_panels(frame, panel_list, LABELS, PANEL_SM, GAP_SM, 1.0, title)

        draw_scale_bar(frame, um_disp_3d, x_right=W-30, y_bottom=H - PANEL_SM - 110)
        cv2.imwrite(f'{FRAME_DIR}/frame_{frame_idx:05d}.png', frame)
        frame_idx += 1
        cal_counter += 1

    # ── Phase B2: Zoom in — all 4 panels zoom into center together ──
    # MERSCOPE panel re-rendered with bigger dots at zoom; others just crop center
    rot_zoom_start = rot_after_A + math.pi * 0.5
    ZOOM_MAX = 3.0

    def draw_zoomed_panels(frame, zoom, cal_counter_val):
        """Draw 4 panels with zoom crop. MERSCOPE panel re-rendered with bigger dots."""
        cal_disp = get_cal_frame(cal_sm, cal_counter_val)
        panel_list = [cal_disp, sm[0], sm[1], sm[2]]
        total_pw = PANEL_SM * 4 + GAP_SM * 3
        x_start = (W - total_pw) // 2
        y_start = H - PANEL_SM - 60
        for pi, (panel, label) in enumerate(zip(panel_list, LABELS)):
            if pi == 3 and zoom > 1.05:
                # MERSCOPE: re-render at zoomed crop (dots stay 1px, zoom makes them bigger naturally)
                zoom_crop_r = max(10, int(CROP_SM / zoom))
                resized = render_dots_zoomed(ci, zoom_crop_r, PANEL_SM, dot_radius=1)
            else:
                # Other panels: crop center
                ph, pw = panel.shape[:2]
                crop_h = int(ph / zoom); crop_w = int(pw / zoom)
                y0 = (ph - crop_h) // 2; x0 = (pw - crop_w) // 2
                cropped = panel[y0:y0+crop_h, x0:x0+crop_w]
                resized = cv2.resize(cropped, (PANEL_SM, PANEL_SM), interpolation=cv2.INTER_LANCZOS4)
            px = x_start + pi * (PANEL_SM + GAP_SM)
            frame[y_start:y_start+PANEL_SM, px:px+PANEL_SM] = resized
            cv2.rectangle(frame, (px, y_start), (px+PANEL_SM-1, y_start+PANEL_SM-1), (50,50,50), 1)
            fs = 0.38
            (tw, _), _ = cv2.getTextSize(label, FONT, fs, 1)
            put_text_mixed(frame, label, (px + (PANEL_SM-tw)//2, min(H-5, y_start+PANEL_SM+22)),
                        FONT, fs, (180,180,180), 1)
        (tw, _), _ = cv2.getTextSize(title, FONT, 0.5, 1)
        put_text_mixed(frame, title, ((W-tw)//2, max(20, y_start-15)), FONT, 0.5, (240,240,240), 1)
        draw_scale_bar(frame, um_disp_3d, x_right=W-30, y_bottom=H - PANEL_SM - 110)

    for fi in range(N_ZOOM_IN):
        t = ease(fi / max(1, N_ZOOM_IN - 1))
        zoom = 1.0 + (ZOOM_MAX - 1.0) * t
        rot_y = rot_zoom_start + (fi / max(1, N_ZOOM_IN - 1)) * math.pi * 0.1
        frame = render_3d(rot_y, INIT_ROT_X, marker_display_xyz=(mx, my, mz))
        draw_zoomed_panels(frame, zoom, cal_counter)
        cv2.imwrite(f'{FRAME_DIR}/frame_{frame_idx:05d}.png', frame)
        frame_idx += 1; cal_counter += 1

    # ── Phase B3: Hold zoomed ──
    rot_zoom_hold = rot_zoom_start + math.pi * 0.1
    for fi in range(N_ZOOM_HOLD):
        rot_y = rot_zoom_hold + (fi / max(1, N_ZOOM_HOLD - 1)) * math.pi * 0.1
        frame = render_3d(rot_y, INIT_ROT_X, marker_display_xyz=(mx, my, mz))
        draw_zoomed_panels(frame, ZOOM_MAX, cal_counter)
        cv2.imwrite(f'{FRAME_DIR}/frame_{frame_idx:05d}.png', frame)
        frame_idx += 1; cal_counter += 1

    # ── Phase B4: Zoom out ──
    rot_zoom_out_start = rot_zoom_hold + math.pi * 0.1
    for fi in range(N_ZOOM_OUT):
        t = ease(fi / max(1, N_ZOOM_OUT - 1))
        zoom = ZOOM_MAX + (1.0 - ZOOM_MAX) * t
        rot_y = rot_zoom_out_start + (fi / max(1, N_ZOOM_OUT - 1)) * math.pi * 0.1
        frame = render_3d(rot_y, INIT_ROT_X, marker_display_xyz=(mx, my, mz))
        draw_zoomed_panels(frame, zoom, cal_counter)
        cv2.imwrite(f'{FRAME_DIR}/frame_{frame_idx:05d}.png', frame)
        frame_idx += 1; cal_counter += 1

    # ── Phase C: Converge — 4 panels slide horizontally to center at bottom ──
    total_pw_sm = PANEL_SM * 4 + GAP_SM * 3
    x_start_spread = (W - total_pw_sm) // 2
    x_center_panel = (W - PANEL_SM) // 2
    y_bottom_sm = H - PANEL_SM - 60
    spread_xs = [x_start_spread + pi * (PANEL_SM + GAP_SM) for pi in range(4)]

    rot_converge_start = rot_zoom_out_start + math.pi * 0.1  # continue from B4 end
    for fi in range(N_CONVERGE):
        t = ease(fi / (N_CONVERGE - 1))
        rot_y = rot_converge_start + (fi / max(1, N_CONVERGE - 1)) * math.pi * 0.15
        frame = render_3d(rot_y, INIT_ROT_X, marker_display_xyz=(mx, my, mz))

        cal_disp = get_cal_frame(cal_sm, cal_counter)
        panel_list = [cal_disp, sm[0], sm[1], sm[2]]

        for pi, (panel, label) in enumerate(zip(panel_list, LABELS)):
            cur_x = int(spread_xs[pi] + t * (x_center_panel - spread_xs[pi]))
            p_resized = cv2.resize(panel, (PANEL_SM, PANEL_SM), interpolation=cv2.INTER_LANCZOS4)
            py0 = y_bottom_sm; py1 = py0 + PANEL_SM
            px0 = max(0, cur_x); px1 = min(W, cur_x + PANEL_SM)
            sx0 = px0 - cur_x; sx1 = sx0 + (px1 - px0)
            if py1 > py0 and px1 > px0:
                roi = frame[py0:py1, px0:px1].astype(np.float32)
                new = p_resized[:, sx0:sx1].astype(np.float32)
                frame[py0:py1, px0:px1] = np.clip(roi + new, 0, 255).astype(np.uint8)
            bc = 50
            cv2.rectangle(frame, (px0, py0), (px1 - 1, py1 - 1), (bc,bc,bc), 1)
            lbl_scale = 0.38
            (tw, _), _ = cv2.getTextSize(label, FONT, lbl_scale, 1)
            lx = cur_x + (PANEL_SM - tw) // 2
            ly = min(H - 5, py1 + 22)
            lc = int(180 * (1.0 - t))
            put_text_mixed(frame, label, (lx, ly), FONT, lbl_scale, (lc, lc, lc), 1)

        (tw, _), _ = cv2.getTextSize(title, FONT, 0.5, 1)
        put_text_mixed(frame, title, ((W - tw) // 2, max(20, y_bottom_sm - 15)),
                    FONT, 0.5, (240, 240, 240), 1)

        draw_scale_bar(frame, um_disp_3d, x_right=W-30, y_bottom=H - PANEL_SM - 110)
        cv2.imwrite(f'{FRAME_DIR}/frame_{frame_idx:05d}.png', frame)
        frame_idx += 1
        cal_counter += 1

    # ── Phase D: Merge — stacked panels crossfade to 1 overlay at bottom ──
    cal_static = get_cal_frame(cal_sm, cal_counter)
    iv_m = cv2.resize(sm[0], (PANEL_SM, PANEL_SM), interpolation=cv2.INTER_LANCZOS4)
    ev_m = cv2.resize(sm[1], (PANEL_SM, PANEL_SM), interpolation=cv2.INTER_LANCZOS4)
    ms_m = cv2.resize(sm[2], (PANEL_SM, PANEL_SM), interpolation=cv2.INTER_LANCZOS4)
    cal_m = cv2.resize(cal_static, (PANEL_SM, PANEL_SM), interpolation=cv2.INTER_LANCZOS4)

    merged = np.zeros((PANEL_SM, PANEL_SM, 3), np.uint8)
    merged[:, :, 1] = iv_m[:, :, 1]       # green from invivo
    merged[:, :, 2] = ev_m[:, :, 2]       # red from exvivo (magenta)
    merged[:, :, 0] = ev_m[:, :, 0]       # blue from exvivo (magenta)
    dot_px = ms_m.max(axis=2) > 0
    merged[dot_px] = ms_m[dot_px]
    cal_gray = cv2.cvtColor(cal_m, cv2.COLOR_BGR2GRAY)
    cal_boost = (cal_gray.astype(np.float32) / 255 * 0.3)
    for c in range(3):
        merged[:, :, c] = np.clip(merged[:, :, c].astype(np.float32) +
                                   cal_boost * 255, 0, 255).astype(np.uint8)

    # Merged panel at bottom center (same y as the 4 panels)
    x_merged = (W - PANEL_SM) // 2
    y_merged = H - PANEL_SM - 60

    rot_merge_start = rot_converge_start + math.pi * 0.15
    for fi in range(N_MERGE):
        t = ease(fi / (N_MERGE - 1))
        rot_y = rot_merge_start + (fi / max(1, N_MERGE - 1)) * math.pi * 0.2
        frame = render_3d(rot_y, INIT_ROT_X, marker_display_xyz=(mx, my, mz))

        # Stacked panels fading out at bottom center
        if t < 0.99:
            cal_disp = get_cal_frame(cal_sm, cal_counter)
            panel_list = [cal_disp, sm[0], sm[1], sm[2]]
            fade = 1.0 - t
            for panel in panel_list:
                p_r = cv2.resize(panel, (PANEL_SM, PANEL_SM), interpolation=cv2.INTER_LANCZOS4)
                p_r = (p_r.astype(np.float32) * fade).astype(np.uint8)
                roi = frame[y_merged:y_merged+PANEL_SM, x_merged:x_merged+PANEL_SM].astype(np.float32)
                frame[y_merged:y_merged+PANEL_SM, x_merged:x_merged+PANEL_SM] = \
                    np.clip(roi + p_r.astype(np.float32), 0, 255).astype(np.uint8)

        # Merged panel fading in at bottom center
        merged_show = (merged.astype(np.float32) * t).astype(np.uint8)
        py0 = y_merged; py1 = min(H, py0 + PANEL_SM)
        roi = frame[py0:py1, x_merged:x_merged+PANEL_SM].astype(np.float32)
        frame[py0:py1, x_merged:x_merged+PANEL_SM] = \
            np.clip(roi + merged_show[:py1-py0].astype(np.float32), 0, 255).astype(np.uint8)

        lc = int(180 * t)
        merge_label = 'ALIGNED OVERLAY'
        (tw, _), _ = cv2.getTextSize(merge_label, FONT, 0.38, 1)
        cv2.putText(frame, merge_label, (x_merged + (PANEL_SM - tw) // 2, min(H - 5, py1 + 22)),
                    FONT, 0.38, (lc, lc, lc), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(title, FONT, 0.5, 1)
        cv2.putText(frame, title, ((W - tw) // 2, max(20, py0 - 15)),
                    FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

        draw_scale_bar(frame, um_disp_3d, x_right=W-30, y_bottom=H - PANEL_SM - 110)
        cv2.imwrite(f'{FRAME_DIR}/frame_{frame_idx:05d}.png', frame)
        frame_idx += 1
        cal_counter += 1

    # ── Phase D: Hold merged overlay at bottom + 3D ──
    rot_hold_start = rot_merge_start + math.pi * 0.2
    for fi in range(N_HOLD_MERGE):
        rot_y = rot_hold_start + (fi / max(1, N_HOLD_MERGE - 1)) * math.pi * 0.2
        frame = render_3d(rot_y, INIT_ROT_X, marker_display_xyz=(mx, my, mz))
        py0 = y_merged; py1 = min(H, py0 + PANEL_SM)
        frame[py0:py1, x_merged:x_merged+PANEL_SM] = merged[:py1-py0]
        cv2.rectangle(frame, (x_merged, py0), (x_merged+PANEL_SM-1, py1-1), (50,50,50), 1)
        (tw, _), _ = cv2.getTextSize('ALIGNED OVERLAY', FONT, 0.38, 1)
        cv2.putText(frame, 'ALIGNED OVERLAY',
                    (x_merged + (PANEL_SM - tw) // 2, min(H - 5, py1 + 22)),
                    FONT, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(title, FONT, 0.5, 1)
        cv2.putText(frame, title, ((W - tw) // 2, max(20, py0 - 15)),
                    FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        draw_scale_bar(frame, um_disp_3d, x_right=W-30, y_bottom=H - PANEL_SM - 110)
        cv2.imwrite(f'{FRAME_DIR}/frame_{frame_idx:05d}.png', frame)
        frame_idx += 1

    # ── Phase E: Fade to black ──
    rot_final = rot_hold_start + math.pi * 0.2
    for fi in range(N_FADE_OUT):
        alpha = 1.0 - ease(fi / (N_FADE_OUT - 1))
        frame = render_3d(rot_final, INIT_ROT_X, marker_display_xyz=(mx, my, mz))
        frame = (frame.astype(np.float32) * alpha).astype(np.uint8)
        py0 = y_merged; py1 = min(H, py0 + PANEL_SM)
        merged_show = (merged[:py1-py0].astype(np.float32) * alpha).astype(np.uint8)
        frame[py0:py1, x_merged:x_merged+PANEL_SM] = merged_show
        bc = int(50 * alpha)
        cv2.rectangle(frame, (x_merged, py0), (x_merged+PANEL_SM-1, py1-1), (bc,bc,bc), 1)
        lc = int(180 * alpha)
        (tw, _), _ = cv2.getTextSize('ALIGNED OVERLAY', FONT, 0.38, 1)
        cv2.putText(frame, 'ALIGNED OVERLAY',
                    (x_merged + (PANEL_SM - tw) // 2, min(H - 5, py1 + 22)),
                    FONT, 0.38, (lc, lc, lc), 1, cv2.LINE_AA)
        tc = int(240 * alpha)
        (tw, _), _ = cv2.getTextSize(title, FONT, 0.5, 1)
        cv2.putText(frame, title, ((W - tw) // 2, max(20, py0 - 15)),
                    FONT, 0.5, (tc, tc, tc), 1, cv2.LINE_AA)
        draw_scale_bar(frame, um_disp_3d, alpha=alpha, x_right=W-30, y_bottom=H - PANEL_SM - 110)
        cv2.imwrite(f'{FRAME_DIR}/frame_{frame_idx:05d}.png', frame)
        frame_idx += 1

n_per = N_FADE_IN + N_ROTATE + N_HOLD + N_ZOOM_IN + N_ZOOM_HOLD + N_ZOOM_OUT + N_CONVERGE + N_MERGE + N_HOLD_MERGE + N_FADE_OUT
print(f"\nDone! {frame_idx} frames ({n_per} per cell) saved to {FRAME_DIR}/")
print(f"QC the frames, then run: python3 {__file__} --encode")
