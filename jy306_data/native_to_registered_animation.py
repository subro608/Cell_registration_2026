"""
3-phase animation:
  Phase 1: Native ex-vivo MERSCOPE tile (2_1_merscope17, raw unregistered)
  Phase 2: Affine warp — tissue physically moves into exvivo_total/JY316 coordinate space
           27 matched neurons appear as green landmarks
  Phase 3: Crossfade to exvivo_total (green) + JY316 (pink) overlay
           Green landmarks (ex-vivo) → pink landmarks (in-vivo)

Coordinate systems:
  - Native:     2_1_merscope17.pkl ch0 MIP  (1627x1627)
  - Registered: exvivo_total.tif MIP        (578x599)
  - In-vivo:    JY316_in_Vivo_stack_flipped (578x599)
  - Landmarks:  pcd_invivo / pcd_exvivo both in (578x599) space
"""

import numpy as np
import pickle
import tifffile
import cv2

# ============================================================
# 1. Load data
# ============================================================
print("Loading native ex-vivo (2_1_merscope17)...")
with open('jy316_images_transformationfiles/2_1_merscope17.pkl', 'rb') as f:
    native_ev = np.array(pickle.load(f))  # (3, 1627, 1627, 3)

print("Loading registration pkl...")
with open('2_1_merscope17transformed_20250424104024.pkl', 'rb') as f:
    pkl = pickle.load(f)

pcd_iv = pkl['pcd_invivo'][:, 1:]   # (27,2) r,c in 578x599 in-vivo space
pcd_ev = pkl['pcd_exvivo'][:, 1:]   # (27,2) r,c in 578x599 registered space

print("Loading exvivo_total...")
ev_total = tifffile.imread('exvivo/exvivo_total.tif')          # (19, 578, 599)

print("Loading JY316 in-vivo stack...")
iv_stack = tifffile.imread('JY316_in_Vivo_stack_flipped.tif')  # (19, 578, 599)

# ============================================================
# 2. MIPs
# ============================================================
def norm(img):
    img = img.astype(np.float64)
    mask = img > 0
    if mask.any():
        p2, p99 = np.percentile(img[mask], [2, 99.5])
    else:
        p2, p99 = 0, 1
    return np.clip((img - p2) / (p99 - p2 + 1e-8), 0, 1)

native_mip  = norm(np.max(native_ev[:, :, :, 0], axis=0))  # (1627, 1627)
ev_tot_mip  = norm(np.max(ev_total, axis=0))                # (578, 599)
iv_mip      = norm(np.max(iv_stack, axis=0))                # (578, 599)

# ============================================================
# 3. Compute native-space landmark positions
#
# pcd_exvivo is in (578x599) space.
# We need corresponding positions in native (1627x1627) space.
# Use the pkl composite affine (native->1704 canvas) to get the
# native<->canvas mapping, then find native<->578 mapping via landmarks.
# ============================================================
transforms = pkl['transformations']

def to_4x4(bhat):
    M = np.eye(4)
    M[:4, :3] = bhat
    return M

def scale_4x4(s):
    return np.diag([float(s), float(s), float(s), 1.0])

composite = np.eye(4)
for t in transforms:
    if 'bhat' in t:
        composite = composite @ to_4x4(t['bhat'])
    elif 'scale' in t:
        composite = composite @ scale_4x4(t['scale'])

# Extract 2D affine: native (r,c) -> 1704-canvas (r,c) at z_mean=1.5
c = composite
z_m = 1.5
# For row (y): native_r * c[1,1] + native_c * c[2,1] + z_m*c[0,1] + c[3,1]
# For col (x): native_r * c[1,2] + native_c * c[2,2] + z_m*c[0,2] + c[3,2]
# cv2 convention (x=col, y=row):
M_nat_to_canvas = np.array([
    [c[2, 2], c[1, 2], z_m * c[0, 2] + c[3, 2]],  # col' (x')
    [c[2, 1], c[1, 1], z_m * c[0, 1] + c[3, 1]]   # row' (y')
], dtype=np.float64)

# Invert to get canvas -> native
M3 = np.vstack([M_nat_to_canvas, [0, 0, 1]])
M3_inv = np.linalg.inv(M3)

def apply_M3(pts_rc, M):
    """Apply 3x3 homogeneous matrix (col,row convention) to (N,2) r,c -> (N,2) r,c"""
    xy = pts_rc[:, ::-1]
    xy_h = np.hstack([xy, np.ones((len(xy), 1))])
    out = (M @ xy_h.T).T[:, :2]
    return out[:, ::-1]

# canvas positions of pcd_ev (these are in 578x599 space — NOT 1704 canvas)
# So we need the direct native->578x599 affine estimated from landmark correspondences.
# Step: pcd_ev is in 578x599, pcd_ev_in_canvas is approximately pcd_ev * (1704/578)
# But that's not exact. Instead, estimate the affine from corresponding pairs directly.

# We DO have: native<->canvas affine from pkl, and we know pcd_ev is in 578x599.
# The canvas-to-578 mapping: canvas coords / scale where scale ~ 1704/578 ≈ 2.95
# Let's estimate canvas positions of pcd_ev landmarks using scale factor
canvas_scale = 1704.0 / 578.0   # approximate scale canvas<->578 space
pcd_ev_canvas = pcd_ev * canvas_scale   # approx pcd_ev in 1704 canvas space

# Get native positions by inverting canvas <- native affine
pcd_ev_native = apply_M3(pcd_ev_canvas, M3_inv)   # (27,2) in native (1627x1627) space

print(f"pcd_ev native range: r=[{pcd_ev_native[:,0].min():.0f},{pcd_ev_native[:,0].max():.0f}], "
      f"c=[{pcd_ev_native[:,1].min():.0f},{pcd_ev_native[:,1].max():.0f}]")
print(f"pcd_ev 578-space range: r=[{pcd_ev[:,0].min():.0f},{pcd_ev[:,0].max():.0f}], "
      f"c=[{pcd_ev[:,1].min():.0f},{pcd_ev[:,1].max():.0f}]")

# ============================================================
# 4. Estimate forward affine: native (1627x1627) -> 578x599 space
#    Using the landmark correspondences
# ============================================================
src_pts = pcd_ev_native[:, ::-1].astype(np.float32)   # (27,2) as (x,y)=(col,row)
dst_pts = pcd_ev[:, ::-1].astype(np.float32)           # (27,2) as (x,y)=(col,row)

M_fwd, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC,
                                       ransacReprojThreshold=5.0)
n_inliers = int(inliers.sum()) if inliers is not None else 0
print(f"Forward affine (native->578 space): {n_inliers}/27 inliers")
print(M_fwd)

M_identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

# ============================================================
# 5. Output parameters
# ============================================================
OUT = 900
FPS = 30

HOLD_NATIVE = 2.0
WARP_DUR    = 4.0
HOLD_REG    = 1.5
TRANS_OVR   = 3.5
HOLD_OVR    = 2.5

T = [0]
for d in [HOLD_NATIVE, WARP_DUR, HOLD_REG, TRANS_OVR, HOLD_OVR]:
    T.append(T[-1] + d)

TOTAL_FRAMES = int(T[-1] * FPS)

PINK  = (180, 105, 255)
GREEN = (80,  255, 80)

# Pre-scale target images to OUT x OUT
ev_tot_out = cv2.resize(ev_tot_mip.astype(np.float32), (OUT, OUT), interpolation=cv2.INTER_LANCZOS4)
iv_out     = cv2.resize(iv_mip.astype(np.float32),     (OUT, OUT), interpolation=cv2.INTER_LANCZOS4)

# Native image as float32 (1627x1627)
native_f = native_mip.astype(np.float32)

# For warping: output at OUT x OUT, but native is 1627x1627
# warpAffine output size = (OUT, OUT), with scaled M
# Scale M by OUT/578 to map from 578-space to OUT-space
out_scale = OUT / 578.0
M_fwd_out = M_fwd.copy()
M_fwd_out[0, 2] *= out_scale / 1.0    # scale tx
M_fwd_out[1, 2] *= out_scale / 1.0    # scale ty
M_fwd_out[0, :2] *= out_scale          # scale linear part
M_fwd_out[1, :2] *= out_scale

# Also need to scale input coords from 1627 native -> OUT
# Actually: warpAffine(native_f, M_scaled, (OUT,OUT)) treats native_f pixels as-is
# M_fwd maps native(col,row) -> 578-space(col',row')
# For OUT output, we want M that maps native(col,row) -> OUT(col'',row'')
# = M_fwd scaled by (OUT/578) in output
M_full_scaled = np.array([
    [M_fwd[0, 0] * out_scale, M_fwd[0, 1] * out_scale, M_fwd[0, 2] * out_scale],
    [M_fwd[1, 0] * out_scale, M_fwd[1, 1] * out_scale, M_fwd[1, 2] * out_scale]
], dtype=np.float64)

# Identity in same output space: native pixel (x,y) maps to output (x*out_scale/native_scale, ...)
# At t=0 show native centered/fitted in OUT frame
nat_h, nat_w = native_f.shape
fit_s = OUT / max(nat_h, nat_w)   # scale to fit native in OUT frame
tx_fit = (OUT - nat_w * fit_s) / 2
ty_fit = (OUT - nat_h * fit_s) / 2
M_identity_scaled = np.array([
    [fit_s, 0,     tx_fit],
    [0,     fit_s, ty_fit]
], dtype=np.float64)

out_path = 'native_exvivo_to_invivo_overlay.mp4'
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (OUT, OUT))
print(f"Rendering {TOTAL_FRAMES} frames ({T[-1]:.1f}s) at {FPS}fps ({OUT}x{OUT})")

font = cv2.FONT_HERSHEY_SIMPLEX

def ease(x):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))

def draw_lm(frame, pts_rc_578, color, radius, alpha):
    """Draw landmarks. pts_rc_578 is in 578x599 space; scale to OUT."""
    if alpha < 0.04:
        return
    col = tuple(int(v * alpha) for v in color)
    for i in range(len(pts_rc_578)):
        r = pts_rc_578[i, 0] * out_scale
        c_ = pts_rc_578[i, 1] * (OUT / 599.0)
        ri, ci = int(r), int(c_)
        if 0 <= ri < OUT and 0 <= ci < OUT:
            cv2.circle(frame, (ci, ri), radius, col, 2, cv2.LINE_AA)
            arm = 4
            cv2.line(frame, (ci-arm, ri), (ci+arm, ri), col, 1, cv2.LINE_AA)
            cv2.line(frame, (ci, ri-arm), (ci, ri+arm), col, 1, cv2.LINE_AA)

def draw_lm_warped(frame, pts_rc_native, M_cv, color, radius, alpha):
    """Draw landmarks using current warp matrix (native->OUT)."""
    if alpha < 0.04:
        return
    col = tuple(int(v * alpha) for v in color)
    for i in range(len(pts_rc_native)):
        x_n, y_n = pts_rc_native[i, 1], pts_rc_native[i, 0]   # col, row
        x_out = M_cv[0, 0] * x_n + M_cv[0, 1] * y_n + M_cv[0, 2]
        y_out = M_cv[1, 0] * x_n + M_cv[1, 1] * y_n + M_cv[1, 2]
        ri, ci = int(y_out), int(x_out)
        if 0 <= ri < OUT and 0 <= ci < OUT:
            cv2.circle(frame, (ci, ri), radius, col, 2, cv2.LINE_AA)
            arm = 4
            cv2.line(frame, (ci-arm, ri), (ci+arm, ri), col, 1, cv2.LINE_AA)
            cv2.line(frame, (ci, ri-arm), (ci, ri+arm), col, 1, cv2.LINE_AA)

# ============================================================
# 6. Render
# ============================================================
for fi in range(TOTAL_FRAMES):
    ts = fi / FPS

    if ts < T[1]:
        phase = 1
        t_warp = 0.0
        a_ev_lm = 0.0
        a_iv_lm = 0.0

    elif ts < T[2]:
        lin = (ts - T[1]) / WARP_DUR
        t_warp = ease(lin)
        phase = 2
        a_ev_lm = ease(max(0, (lin - 0.7) / 0.3))
        a_iv_lm = 0.0

    elif ts < T[3]:
        phase = 3
        t_warp = 1.0
        a_ev_lm = 1.0
        a_iv_lm = 0.0

    elif ts < T[4]:
        lin = (ts - T[3]) / TRANS_OVR
        t_blend = ease(lin)
        phase = 4
        t_warp = 1.0
        a_ev_lm = 1.0 - ease(min(1, lin / 0.5))
        a_iv_lm = ease(max(0, (lin - 0.4) / 0.6))

    else:
        phase = 5
        t_warp = 1.0
        t_blend = 1.0
        a_ev_lm = 0.0
        a_iv_lm = 1.0

    # Interpolate affine: identity (fit native to frame) -> full warp (native->578 space->OUT)
    M_t = (1 - t_warp) * M_identity_scaled + t_warp * M_full_scaled

    # Warp native image
    warped = cv2.warpAffine(native_f, M_t, (OUT, OUT),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)

    # Build frame
    if phase <= 3:
        frame_f = np.stack([warped, warped, warped], axis=2)

    else:
        t_blend_val = t_blend
        # exvivo_total (grey->green) + JY316 (pink) overlay
        frame_f = np.zeros((OUT, OUT, 3), dtype=np.float32)

        ev_w = 1.0 - t_blend_val * 0.35
        iv_w = t_blend_val

        # Warped native transitions to exvivo_total
        nat_to_ev = (1 - t_blend_val)
        frame_f[:, :, 0] += nat_to_ev * warped + ev_w * ev_tot_out * 0.15
        frame_f[:, :, 1] += nat_to_ev * warped + ev_w * ev_tot_out
        frame_f[:, :, 2] += nat_to_ev * warped + ev_w * ev_tot_out * 0.15

        frame_f[:, :, 0] += iv_w * iv_out
        frame_f[:, :, 1] += iv_w * iv_out * 0.75
        frame_f[:, :, 2] += iv_w * iv_out * 0.85

    frame_f = np.clip(frame_f, 0, 1)
    frame = (frame_f[:, :, ::-1] * 255).astype(np.uint8)

    # Landmarks
    if phase <= 3:
        draw_lm_warped(frame, pcd_ev_native, M_t, GREEN, radius=6, alpha=a_ev_lm)
    else:
        draw_lm(frame, pcd_ev, GREEN, radius=6, alpha=a_ev_lm)
        draw_lm(frame, pcd_iv, PINK,  radius=6, alpha=a_iv_lm)

    # Labels
    if phase == 1:
        cv2.putText(frame, "Ex Vivo  (native MERSCOPE space)", (12, 28),
                    font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    elif phase == 2:
        cv2.putText(frame, "Applying registration transform...", (12, 28),
                    font, 0.55, (80, 220, 80), 1, cv2.LINE_AA)
        if a_ev_lm > 0.1:
            cv2.putText(frame, "27 matched neurons", (12, 52),
                        font, 0.4, tuple(int(v * a_ev_lm) for v in GREEN), 1, cv2.LINE_AA)

    elif phase == 3:
        cv2.putText(frame, "Ex Vivo  (registered)", (12, 28),
                    font, 0.55, (80, 255, 80), 1, cv2.LINE_AA)
        cv2.putText(frame, "27 matched neurons", (12, 52),
                    font, 0.4, GREEN, 1, cv2.LINE_AA)

    else:
        lin_p4 = np.clip((ts - T[3]) / TRANS_OVR, 0, 1)
        a_lbl = ease(min(1, lin_p4 * 2))
        cv2.putText(frame, "Ex Vivo + In Vivo  (aligned)", (12, 28),
                    font, 0.55, tuple(int(v * a_lbl) for v in (210, 210, 210)), 1, cv2.LINE_AA)
        ly = OUT - 22
        if a_ev_lm > 0.05:
            cv2.circle(frame, (12, ly), 5, tuple(int(v * a_ev_lm) for v in GREEN), -1)
            cv2.putText(frame, "Ex Vivo cells", (25, ly + 4), font, 0.38,
                        tuple(int(v * a_ev_lm) for v in GREEN), 1, cv2.LINE_AA)
        if a_iv_lm > 0.05:
            xo = 160 if a_ev_lm > 0.05 else 12
            cv2.circle(frame, (xo, ly), 5, tuple(int(v * a_iv_lm) for v in PINK), -1)
            cv2.putText(frame, "In Vivo cells", (xo + 13, ly + 4), font, 0.38,
                        tuple(int(v * a_iv_lm) for v in PINK), 1, cv2.LINE_AA)

    writer.write(frame)

    if fi % 60 == 0:
        print(f"  Frame {fi}/{TOTAL_FRAMES}  t={ts:.1f}s  phase={phase}  t_warp={t_warp:.2f}")

writer.release()
print(f"\nDone!  {out_path}")
print(f"Duration: {T[-1]:.1f}s,  {OUT}x{OUT} @ {FPS}fps")
