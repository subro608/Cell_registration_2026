"""
Animation: In-vivo being affine-transformed to fit into the ex-vivo tissue.
Shows the spatial deformation process — in-vivo warps into its correct position
within the larger ex-vivo tissue section.

Pink landmarks = in-vivo cell positions
Green landmarks = ex-vivo cell positions (same 27 neurons)
"""

import numpy as np
import pickle
import tifffile
import cv2

# ============================================================
# 1. Load data
# ============================================================
print("Loading data...")

with open('2_1_merscope17transformed_20250424104024.pkl', 'rb') as f:
    pkl = pickle.load(f)

pcd_iv = pkl['pcd_invivo'][:, 1:]    # (27,2) r,c in invivo coords
pcd_ev = pkl['pcd_exvivo'][:, 1:]    # (27,2) r,c in registered coords
reg_ev = pkl['transformed'][0]        # (16, 1704, 1704)

iv_stack = tifffile.imread('JY306_in_Vivo_stack_flipped_s80.tif')  # (16, 658, 629)

def norm(img):
    img = img.astype(np.float64)
    mask = img > 0
    if mask.any():
        p1, p99 = np.percentile(img[mask], [2, 99.5])
    else:
        p1, p99 = 0, 1
    return np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)

iv_mip = norm(np.max(iv_stack, axis=0))   # (658, 629)
ev_mip = norm(np.max(reg_ev, axis=0))     # (1704, 1704)

# ============================================================
# 2. Set up canvas — crop to region of interest
# ============================================================

# Exvivo tissue extent
nz_r = np.any(ev_mip > 0.03, axis=1)
nz_c = np.any(ev_mip > 0.03, axis=0)
r1 = np.where(nz_r)[0][-1]
c1 = np.where(nz_c)[0][-1]

margin = 40
CH = min(1704, max(700, r1 + margin))  # canvas height
CW = min(1704, max(700, c1 + margin))  # canvas width
print(f"Canvas: {CH}x{CW}")

# Crop images to canvas
ev_canvas = ev_mip[:CH, :CW]

# In-vivo embedded at origin in the registered space
iv_canvas = np.zeros((CH, CW))
iv_canvas[:658, :629] = iv_mip

# Landmarks in canvas coords (already correct since canvas starts at 0,0)
iv_lm = pcd_iv.copy()   # (27,2) r,c
ev_lm = pcd_ev.copy()   # (27,2) r,c — nearly same as iv_lm in registered space

# ============================================================
# 3. Compute the "zoom out" affine
#
# Animation concept:
#   PHASE 1: Start zoomed into the invivo region (filling the frame)
#   PHASE 2: Zoom out + shift to reveal invivo sitting inside the exvivo tissue
#   PHASE 3: Crossfade to exvivo with green landmarks
#
# We need an affine that maps the invivo region (0:658, 0:629)
# to fill the output frame at t=0, and maps to its true position at t=1
# ============================================================

# At t=0: we want the invivo region (658x629) to fill the output frame
# At t=1: we want to see the full canvas (CH x CW)

# Compute zoom-to-fill matrix for the invivo region
# Scale to fill: max(CH/658, CW/629) — or use a fit that shows it nicely
iv_h, iv_w = 658, 629
scale_fill = min(CH / iv_h, CW / iv_w) * 0.85  # 85% fill
# Center the invivo in the output at t=0
tx_fill = (CW - iv_w * scale_fill) / 2
ty_fill = (CH - iv_h * scale_fill) / 2

M_start = np.array([
    [scale_fill, 0, tx_fill],
    [0, scale_fill, ty_fill]
], dtype=np.float64)

# At t=1: identity (showing the full canvas as-is)
M_end = np.array([
    [1.0, 0, 0],
    [0, 1.0, 0]
], dtype=np.float64)

print(f"Start affine (zoom in): scale={scale_fill:.3f}, tx={tx_fill:.1f}, ty={ty_fill:.1f}")

# ============================================================
# 4. Video parameters
# ============================================================

FPS = 30
HOLD_IV = 2.0        # hold on invivo (zoomed in)
DEFORM_DUR = 4.0     # zoom out / deformation
HOLD_EV = 2.5        # hold on exvivo (full view)
TOTAL_FRAMES = int((HOLD_IV + DEFORM_DUR + HOLD_EV) * FPS)

SCALE = 2
out_h, out_w = CH * SCALE, CW * SCALE

PINK = (180, 105, 255)      # BGR
GREEN = (80, 255, 80)       # BGR

out_path = 'invivo_to_exvivo_transform.mp4'
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (out_w, out_h))

print(f"Rendering {TOTAL_FRAMES} frames at {FPS}fps, {out_w}x{out_h}...")

# ============================================================
# 5. Render
# ============================================================

for fi in range(TOTAL_FRAMES):
    time_s = fi / FPS

    if time_s < HOLD_IV:
        t = 0.0
    elif time_s < HOLD_IV + DEFORM_DUR:
        lin = (time_s - HOLD_IV) / DEFORM_DUR
        t = 0.5 - 0.5 * np.cos(np.pi * lin)  # ease in-out
    else:
        t = 1.0

    # Interpolate affine: M_start -> M_end
    M_t = M_start + t * (M_end - M_start)

    # Warp in-vivo through current affine
    warped_iv = cv2.warpAffine(iv_canvas, M_t, (CW, CH),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)

    # Warp ex-vivo through same affine (it reveals as we zoom out)
    warped_ev = cv2.warpAffine(ev_canvas, M_t, (CW, CH),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)

    # Crossfade: invivo dominates early, exvivo dominates late
    # Use a slightly delayed crossfade so invivo is visible longer
    blend_t = np.clip((t - 0.2) / 0.6, 0, 1)  # starts at t=0.2, ends at t=0.8

    # Build RGB
    rgb = np.zeros((CH, CW, 3), dtype=np.float64)

    # Invivo: warm pink/white
    iv_w8 = 1.0 - blend_t
    rgb[:, :, 0] += iv_w8 * warped_iv * 1.0
    rgb[:, :, 1] += iv_w8 * warped_iv * 0.75
    rgb[:, :, 2] += iv_w8 * warped_iv * 0.82

    # Exvivo: green
    ev_w8 = blend_t
    rgb[:, :, 0] += ev_w8 * warped_ev * 0.15
    rgb[:, :, 1] += ev_w8 * warped_ev * 1.0
    rgb[:, :, 2] += ev_w8 * warped_ev * 0.15

    rgb = np.clip(rgb, 0, 1)

    # To BGR uint8 + upscale
    frame = (rgb[:, :, ::-1] * 255).astype(np.uint8)
    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    # Transform landmark positions through current affine
    # M_t maps canvas coords to output coords
    def transform_pts(pts, M):
        """Transform (N,2) row,col points through 2x3 affine M (which operates on x,y)"""
        xy = pts[:, ::-1]  # col, row -> x, y
        ones = np.ones((len(xy), 1))
        xy_h = np.hstack([xy, ones])  # (N, 3)
        out_xy = (M @ xy_h.T).T       # (N, 2)
        return out_xy[:, ::-1]         # back to row, col

    iv_lm_t = transform_pts(iv_lm, M_t) * SCALE
    ev_lm_t = transform_pts(ev_lm, M_t) * SCALE

    # Interpolate between IV and EV landmark positions
    curr_lm = iv_lm_t + blend_t * (ev_lm_t - iv_lm_t)

    # Draw landmarks
    for i in range(27):
        r, c = int(curr_lm[i, 0]), int(curr_lm[i, 1])
        if r < 0 or r >= out_h or c < 0 or c >= out_w:
            continue
        rad = int(6 * SCALE)
        thick = max(1, int(1.5 * SCALE))

        # Pink fading out
        a_p = 1.0 - blend_t
        if a_p > 0.05:
            col = tuple(int(v * a_p) for v in PINK)
            cv2.circle(frame, (c, r), rad, col, thick, cv2.LINE_AA)
            arm = int(3 * SCALE)
            cv2.line(frame, (c-arm, r), (c+arm, r), col, max(1, thick-1), cv2.LINE_AA)
            cv2.line(frame, (c, r-arm), (c, r+arm), col, max(1, thick-1), cv2.LINE_AA)

        # Green fading in
        a_g = blend_t
        if a_g > 0.05:
            col = tuple(int(v * a_g) for v in GREEN)
            cv2.circle(frame, (c, r), rad + int(2*SCALE), col, thick, cv2.LINE_AA)
            arm = int(3 * SCALE)
            cv2.line(frame, (c-arm, r), (c+arm, r), col, max(1, thick-1), cv2.LINE_AA)
            cv2.line(frame, (c, r-arm), (c, r+arm), col, max(1, thick-1), cv2.LINE_AA)

    # Draw invivo FOV border (fades in during zoom-out to show where it fits)
    if t > 0.1:
        border_alpha = min(1.0, (t - 0.1) * 2)
        corners = np.array([[0, 0], [0, 629], [658, 629], [658, 0]], dtype=np.float64)
        corners_t = transform_pts(corners, M_t) * SCALE
        pts_cv = corners_t[:, ::-1].astype(np.int32).reshape((-1, 1, 2))
        border_col = tuple(int(v * border_alpha) for v in (100, 180, 255))
        cv2.polylines(frame, [pts_cv], isClosed=True, color=border_col,
                      thickness=max(1, int(1.5 * SCALE)), lineType=cv2.LINE_AA)

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.5 * SCALE
    th = max(1, int(1.2 * SCALE))
    mx = int(15 * SCALE)
    ty_text = int(25 * SCALE)

    if blend_t < 0.5:
        a = 1.0 - blend_t * 2
        cv2.putText(frame, "In Vivo (2-Photon)", (mx, ty_text),
                    font, fs, tuple(int(v*a) for v in (200, 180, 255)), th, cv2.LINE_AA)
    if blend_t > 0.5:
        a = (blend_t - 0.5) * 2
        cv2.putText(frame, "Ex Vivo (MERSCOPE Tissue)", (mx, ty_text),
                    font, fs, tuple(int(v*a) for v in (80, 255, 80)), th, cv2.LINE_AA)

    # Count + FOV label
    info_y = ty_text + int(18 * SCALE)
    cv2.putText(frame, "27 matched neurons", (mx, info_y),
                font, 0.38 * SCALE, (160, 160, 160), max(1, SCALE), cv2.LINE_AA)

    if t > 0.3:
        a = min(1, (t - 0.3) * 2)
        col = tuple(int(v*a) for v in (100, 180, 255))
        cv2.putText(frame, "In Vivo FOV", (mx, info_y + int(18*SCALE)),
                    font, 0.35 * SCALE, col, max(1, SCALE), cv2.LINE_AA)

    # Legend
    ly = out_h - int(20 * SCALE)
    lfs = 0.35 * SCALE
    lth = max(1, SCALE)
    if (1-blend_t) > 0.15:
        cv2.circle(frame, (mx, ly), int(4*SCALE), PINK, -1, cv2.LINE_AA)
        cv2.putText(frame, "In Vivo Cells", (mx+int(10*SCALE), ly+int(3*SCALE)),
                    font, lfs, PINK, lth, cv2.LINE_AA)
    if blend_t > 0.15:
        xoff = mx + int(160*SCALE) if (1-blend_t) > 0.15 else mx
        cv2.circle(frame, (xoff, ly), int(4*SCALE), GREEN, -1, cv2.LINE_AA)
        cv2.putText(frame, "Ex Vivo Cells", (xoff+int(10*SCALE), ly+int(3*SCALE)),
                    font, lfs, GREEN, lth, cv2.LINE_AA)

    writer.write(frame)

    if fi % 60 == 0:
        print(f"  Frame {fi}/{TOTAL_FRAMES} t={t:.3f} blend={blend_t:.3f}")

writer.release()
print(f"\nDone! {out_path}")
print(f"Duration: {TOTAL_FRAMES/FPS:.1f}s, {out_w}x{out_h} @ {FPS}fps")
