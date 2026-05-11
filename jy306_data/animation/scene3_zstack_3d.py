"""
Scene 3: Show entire in-vivo z-stack as 3D volume, rotate, then focus on z=3.

Continues from scene1_2 (which ends with confocal z=3 separated to the right).
This scene starts by bringing the confocal back to center and then revealing
the full z-stack depth, rotates to show 3D structure, then pauses and highlights z=3.

Output: animation/scene3_h264.mp4

Timeline (at 24fps):
  0.0 – 1.5s  : Confocal z=3 (hot) returns to center (36 fr)
  1.5 – 3.0s  : Other z-slices emerge above/below z=3, building the stack (36 fr)
  3.0 – 8.0s  : 3D rotation showing depth (120 fr, ~270° rotation)
  8.0 – 9.0s  : Pause rotation, front-facing (24 fr)
  9.0 – 11.0s : Highlight z=3 slice (other slices fade, z=3 pulses) (48 fr)
"""

import numpy as np, cv2, tifffile, math, subprocess, os

BASE = '/Users/neurolab/neuroinformatics/margaret'
TMP  = f'{BASE}/animation/scene3_raw.mp4'
OUT  = f'{BASE}/animation/scene3_h264.mp4'

W, H = 1920, 1080
FPS  = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GRAY  = (160, 160, 160)

from text_utils import put_text_mixed, text_width_mixed

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

def norm_u8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 50:
        return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img.astype(np.float32) - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

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

# ── Scale bar ──
IV_XY_UM = 0.82  # JY306 in-vivo pixel size (µm/px)

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

# ── Load z-stack ──
print("Loading JY306 z-stack…")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz, hy, wx = jy306.shape  # 16, 658, 629

# Normalize each slice to u8
slices_u8 = []
for z in range(nz):
    slices_u8.append(norm_u8(jy306[z]))
slices_u8 = np.array(slices_u8)

# Downscale slices for 3D rendering — keep correct aspect ratio
_max_dim = 400
_scale = _max_dim / max(wx, hy)
SLICE_W, SLICE_H = int(wx * _scale), int(hy * _scale)  # 382, 400 (preserves aspect)
slices_small = []
for z in range(nz):
    s = cv2.resize(slices_u8[z], (SLICE_W, SLICE_H), interpolation=cv2.INTER_AREA)
    slices_small.append(s)
slices_small = np.array(slices_small)
print(f"  Slice display size: {SLICE_W}×{SLICE_H} (aspect preserved)")

# ── Gaussian-interpolate between slices to fill gaps ──
# Real dimensions: 1229×1177 µm XY, 189 µm Z → Z is 15.4% of XY
# At 400px display: total Z = 62px, spacing = 4.1px per slice
# With 16 original slices spaced 4px apart, we gaussian-interpolate to ~3 sub-slices
# between each pair to get a continuous volume

Z_SPACING = 4  # pixels between original slices — real physical ratio
INTERP_PER_GAP = 3  # interpolated sub-slices between each real slice

print(f"Gaussian-interpolating z-stack: {nz} slices → {nz + (nz-1)*INTERP_PER_GAP} sub-slices...")
# Build dense stack with gaussian interpolation
dense_slices = []
dense_z_positions = []  # z position in display pixels
dense_is_real = []  # True if this is a real slice (for highlighting z=3)
dense_real_idx = []  # which original z-index (-1 for interpolated)

for z in range(nz):
    dense_slices.append(slices_small[z])
    dense_z_positions.append(z * Z_SPACING)
    dense_is_real.append(True)
    dense_real_idx.append(z)

    if z < nz - 1:
        # Gaussian-weighted interpolation between z and z+1
        for sub in range(1, INTERP_PER_GAP + 1):
            t = sub / (INTERP_PER_GAP + 1)
            # Gaussian weighting: closer to the nearer slice
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

def render_3d_stack(rot_y, rot_x, slice_alphas):
    """Render the z-stack as a 3D rotated stack of planes.

    rot_y: rotation around Y axis (radians)
    rot_x: tilt around X axis (radians)
    slice_alphas: (nz,) alpha per ORIGINAL slice (16 values)
    """
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cx, cy = W // 2, H // 2

    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    # Sort sub-slices by depth (back to front for painter's algorithm)
    z_depths = []
    for i in range(n_dense):
        dz = dense_z_positions[i] - STACK_CENTER_Z
        rz = cos_y * dz
        rz2 = cos_x * rz
        z_depths.append((rz2, i))
    z_depths.sort(key=lambda x: x[0])  # far to near

    for depth, i in z_depths:
        # Get alpha from nearest original slice
        real_idx = dense_real_idx[i]
        if real_idx >= 0:
            alpha = slice_alphas[real_idx]
        else:
            # Interpolated: blend alphas of neighboring real slices
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

        # In-vivo = green
        green = np.zeros((H, W, 3), dtype=np.float32)
        green[:, :, 1] = warped  # green channel only

        mask = warped > 0.01
        mask3 = np.stack([mask]*3, axis=-1)
        canvas = np.where(mask3,
                         np.maximum(canvas, green * alpha),
                         canvas)

    return np.clip(canvas * 255, 0, 255).astype(np.uint8)

# ── Video writer ──
vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

# ─────────────────────────────────────────────────────────────────
# SCENE 3a: z=3 returns to center + shrinks + slices emerge (72 fr = 3s)
# Smooth combined transition:
#   - Slides from right offset → center
#   - Shrinks from scene1+2 display size → 3D stack size
#   - Other z-slices emerge in second half
#   - Slight tilt develops
# ─────────────────────────────────────────────────────────────────
print("Scene 3a: slide + shrink + slices emerge (3s)…")

z3_u8 = norm_u8(jy306[3])
z3_green_full = np.zeros((*z3_u8.shape, 3), np.uint8)
z3_green_full[:, :, 1] = z3_u8  # in-vivo = green

# Build overlay matching scene 1-2 ending: grayscale calcium + green confocal
# Load calcium affine and maxproj to reconstruct the overlay
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
INIT_ROT_X = -0.3

# Canvas-space transforms (matching scene 1-2)
tx_fit = (W - wx * s_fit) / 2.0
ty_fit = (H - hy * s_fit) / 2.0
M_fit_3x3 = np.array([[s_fit, 0, tx_fit], [0, s_fit, ty_fit], [0, 0, 1]], dtype=np.float64)
M_affine_3x3 = np.vstack([M_affine, [0, 0, 1]])
M_end = (M_fit_3x3 @ M_affine_3x3)[:2, :]
M_z3_to_canvas = np.array([[s_fit, 0, tx_fit], [0, s_fit, ty_fit]], dtype=np.float64)

warped_canvas = cv2.warpAffine(movie_maxproj_u8, M_end, (W, H))
z3_canvas = cv2.warpAffine(z3_u8, M_z3_to_canvas, (W, H))

# Overlay matching scene 1-2 last frame: grayscale calcium + green confocal
overlay_start = np.zeros((H, W, 3), np.uint8)
overlay_start[:, :, 0] = warped_canvas
overlay_start[:, :, 1] = np.minimum(255, warped_canvas.astype(np.int16) + z3_canvas.astype(np.int16)).astype(np.uint8)
overlay_start[:, :, 2] = warped_canvas

# Just green confocal z=3 at display size (target after calcium fades)
z3_display = np.zeros((H, W, 3), np.uint8)
z3_display[:, :, 1] = z3_canvas

# Start size (from scene 1+2 ending)
nw_start, nh_start = int(wx * s_fit), int(hy * s_fit)
# End size (what render_3d_stack uses for z=3 at rot=0)
nw_end, nh_end = SLICE_W, SLICE_H

N_TRANS = 72
N_CALCIUM_FADE = 24  # frames to fade out the grayscale calcium
print(f"  Transition: {N_TRANS} frames, calcium fade={N_CALCIUM_FADE}fr, shrink {nw_start}×{nh_start} → {nw_end}×{nh_end}")

# Pre-render green z=3 at every frame size
z3_green_frames = []
for fi in range(N_TRANS):
    t = ease(fi / (N_TRANS - 1))
    nw_t = int(round(nw_start * (1 - t) + nw_end * t))
    nh_t = int(round(nh_start * (1 - t) + nh_end * t))
    nw_t = max(1, nw_t)
    nh_t = max(1, nh_t)
    z3_green_frames.append(cv2.resize(z3_green_full, (nw_t, nh_t), interpolation=cv2.INTER_LANCZOS4))

um_disp_flat = IV_XY_UM / s_fit  # µm per display pixel when z=3 is at full s_fit scale

for fi in range(N_TRANS):
    t = ease(fi / (N_TRANS - 1))

    # First N_CALCIUM_FADE frames: crossfade from overlay (gray+green) to pure green
    if fi < N_CALCIUM_FADE:
        t_fade = ease(fi / (N_CALCIUM_FADE - 1))
        frame = cv2.addWeighted(overlay_start, 1 - t_fade, z3_display, t_fade, 0)
    else:
        frame = np.zeros((H, W, 3), np.uint8)
        # Place shrinking z=3 green
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

        # Other slices emerge in second half
        t_emerge = max(0.0, (t - 0.5) / 0.5)
        if t_emerge > 0.01:
            for z in range(nz):
                if z == 3: continue
                dist = abs(z - 3)
                max_dist = ease(t_emerge) * 13
                if dist > max_dist: continue
                alpha = min(1.0, (max_dist - dist + 1) / 2.0) * 0.6
                sl_u8_z = norm_u8(jy306[z])
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
    # Scale bar: show during flat phase, fade out as image shrinks into 3D
    sb_alpha = max(0.0, 1.0 - ease(fi / (N_CALCIUM_FADE + 6)))
    draw_scale_bar(frame, um_disp_flat, alpha=sb_alpha)
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────
# SCENE 3c: 3D rotation (120 fr = 5s, ~270° rotation)
# ─────────────────────────────────────────────────────────────────
print("Scene 3c: 3D rotation (5s)…")

slice_alphas_full = np.ones(nz, dtype=np.float32) * 0.7
slice_alphas_full[3] = 1.0  # z=3 slightly brighter

for fi in range(120):
    t = fi / 119.0
    rot_y = t * math.pi * 2.0  # full 360° rotation — avoids edge-on stall
    rot_x = INIT_ROT_X + 0.15 * math.sin(t * math.pi)  # gentle tilt oscillation

    frame = render_3d_stack(rot_y, rot_x, slice_alphas_full)
    caption(frame, 'IN VIVO  Z-STACK')
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────
# SCENE 3d: Pause rotation, front-facing (24 fr = 1s)
# ─────────────────────────────────────────────────────────────────
print("Scene 3d: pause front-facing (1s)…")

# End rotation at 360° = front-facing
final_rot_y = math.pi * 2.0

for fi in range(24):
    # Already front-facing, just hold with gentle settle
    t = ease(fi / 20)
    rot_y = final_rot_y + 0.0 * t  # hold at front
    rot_x = INIT_ROT_X

    frame = render_3d_stack(rot_y, rot_x, slice_alphas_full)
    caption(frame, 'IN VIVO  Z-STACK')
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────
# SCENE 3e: Highlight z=3 — others fade, z=3 pulses hot (48 fr = 2s)
# ─────────────────────────────────────────────────────────────────
print("Scene 3e: highlight z=3 (2s)…")

for fi in range(48):
    t = ease(fi / 35)

    # Other slices fade out
    slice_alphas = np.ones(nz, dtype=np.float32) * 0.7 * (1 - t * 0.8)
    slice_alphas[3] = 1.0  # z=3 stays full

    frame = render_3d_stack(0.0, INIT_ROT_X, slice_alphas)

    a_old = 1 - ease(fi / 15)
    a_new = ease((fi - 15) / 20)
    caption(frame, 'IN VIVO  Z-STACK', alpha=float(max(0, a_old)))
    caption(frame, 'BEST ALIGNMENT:  Z = 3  --  EX VIVO  TILE  ROW2_1', alpha=float(max(0, a_new)))
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────
# SCENE 3f: Slide z=3 from center to left (36 fr = 1.5s)
# ─────────────────────────────────────────────────────────────────
print("Scene 3f: slide z=3 to left (1.5s)…")

# Render z=3 only (flat, no 3D)
z3_flat = np.zeros((H, W, 3), np.uint8)
z3_green_2d = cv2.resize(z3_green_full, (nw_end, nh_end), interpolation=cv2.INTER_LANCZOS4)

# Start position: center of screen
cx_start = (W - nw_end) // 2
cy_start = (H - nh_end) // 2

# End position: left side (where scene 5 expects it)
# Scene 5 uses: jy_x0 = (W - total_w) // 2, jy_y0 = (H - DISP_H) // 2 - 20
# For first tile, DISP_H ~ 0.72*H ~ 778, disp_jy_w ~ 629*scale_jy ~ 629*(778/629) ~ 778
# Approximate: left third of screen, vertically centered
cx_end = W // 6 - nw_end // 2
cy_end = cy_start

N_SLIDE = 36
for fi in range(N_SLIDE):
    t = ease(fi / (N_SLIDE - 1))
    frame = np.zeros((H, W, 3), np.uint8)

    cx = int(cx_start + t * (cx_end - cx_start))
    cy = int(cy_start + t * (cy_end - cy_start))

    # Place z=3 green image
    py0 = max(0, cy); py1 = min(H, cy + nh_end)
    px0 = max(0, cx); px1 = min(W, cx + nw_end)
    sy0 = py0 - cy; sx0 = px0 - cx
    if py1 > py0 and px1 > px0:
        frame[py0:py1, px0:px1] = z3_green_2d[sy0:sy0+(py1-py0), sx0:sx0+(px1-px0)]

    a_old = max(0, 1 - ease(fi / 12))
    caption(frame, 'BEST ALIGNMENT:  Z = 3  --  EX VIVO  TILE  ROW2_1', alpha=a_old)
    vw.write(frame)

# ── Finalize ──
vw.release()
print("Re-encoding to H.264…")
subprocess.run([
    'ffmpeg', '-y', '-i', TMP,
    '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
    OUT
], capture_output=True)
if os.path.exists(TMP):
    os.remove(TMP)

total_fr = 36 + 36 + 120 + 24 + 48 + N_SLIDE
total_s = total_fr / FPS
print(f"Done! {total_fr} frames, {total_s:.1f}s @ {FPS}fps → {OUT}")
