"""
Scene 5b — Stitched 3D overlay rotation (all tiles combined).

Starts from last frame of scenes_1_to_5, transitions to a 3D rotating
stack showing all tiles' invivo (hot) aligned onto exvivo (green).

Uses scene5b_assets_v2.npz overlay slices with physical z-spacing.
Gaussian interpolation between slices for smooth volume.

Output: animation/scene5b_h264.mp4
"""

import numpy as np, cv2, math, subprocess, os

BASE = '/Users/neurolab/neuroinformatics/margaret'
TMP  = f'{BASE}/animation/scene5b_raw.mp4'
OUT  = f'{BASE}/animation/scene5b_h264.mp4'
SCENES_1_5 = f'{BASE}/animation/scenes_1_to_5.mp4'

W, H = 1920, 1080
FPS  = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

INIT_ROT_X = -0.3
INTERP_PER_GAP = 3

# Physical dimensions
ND2_Z_STEP_UM = 2.0
CANVAS_UM_PER_PX = 0.645

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

def caption(frame, text, alpha=1.0):
    if alpha < 0.01: return
    ts, th = 0.72, 1
    (tw, _), _ = cv2.getTextSize(text, FONT, ts, th)
    x = (W - tw) // 2; y = H - 42
    col = tuple(int(v * alpha) for v in WHITE)
    cv2.putText(frame, text, (x, y), FONT, ts, col, th, cv2.LINE_AA)

# ── Extract last frame of scenes_1_to_5 ──
print("Extracting last frame of scenes_1_to_5...")
last_frame_path = '/tmp/scene5b_start_frame.png'
subprocess.run([
    'ffmpeg', '-y', '-sseof', '-0.1', '-i', SCENES_1_5,
    '-frames:v', '1', '-update', '1', last_frame_path
], capture_output=True)
last_frame = cv2.imread(last_frame_path)
assert last_frame is not None, f"Failed to extract last frame from {SCENES_1_5}"
if last_frame.shape[:2] != (H, W):
    last_frame = cv2.resize(last_frame, (W, H))
print(f"  Last frame: {last_frame.shape}")

# ── Load scene5b assets ──
print("Loading scene5b assets...")
assets = np.load(f'{BASE}/animation/scene5b_assets_v2.npz')
ov_slices_raw = assets['overlay_slices']
z_indices = assets['z_indices']
n_slices = len(ov_slices_raw)
raw_h, raw_w = ov_slices_raw.shape[1:3]
print(f"  {n_slices} slices, {raw_w}x{raw_h}, z={z_indices[0]}-{z_indices[-1]}")

# ── Crop to content bounding box (with padding) ──
print("Cropping to content region...")
mask = np.any(ov_slices_raw > 3, axis=-1)
any_slice = np.any(mask, axis=0)
rows = np.where(any_slice.any(axis=1))[0]
cols = np.where(any_slice.any(axis=0))[0]

PAD = 50
r0 = max(0, rows[0] - PAD)
r1 = min(raw_h, rows[-1] + PAD)
c0 = max(0, cols[0] - PAD)
c1 = min(raw_w, cols[-1] + PAD)
print(f"  Content: rows {r0}-{r1}, cols {c0}-{c1} ({r1-r0}x{c1-c0})")

ov_cropped = ov_slices_raw[:, r0:r1, c0:c1, :]
crop_h, crop_w = ov_cropped.shape[1:3]

# ── Scale cropped slices to fill ~90% of screen height ──
DISP_H = int(H * 0.88)
scale_up = DISP_H / crop_h
disp_w = int(crop_w * scale_up)
disp_h = DISP_H
# Cap width to screen
if disp_w > int(W * 0.95):
    disp_w = int(W * 0.95)
    scale_up = disp_w / crop_w
    disp_h = int(crop_h * scale_up)
print(f"  Display size: {disp_w}x{disp_h} (scale={scale_up:.3f})")

sel_slices = np.array([cv2.resize(ov_cropped[i], (disp_w, disp_h),
                                   interpolation=cv2.INTER_LANCZOS4)
                        for i in range(n_slices)])
del ov_slices_raw, ov_cropped

# ── Physical z-positions in display pixels ──
z_um = z_indices.astype(np.float64) * ND2_Z_STEP_UM
# XY extent of the CROPPED region in µm
crop_xy_um = crop_w * CANVAS_UM_PER_PX
z_display = (z_um - z_um[0]) / crop_xy_um * disp_w
print(f"  Physical z range: {z_um[-1]-z_um[0]:.0f} um, crop XY: {crop_xy_um:.0f} um")
print(f"  Z display range: {z_display[-1]:.1f} px (ratio {(z_um[-1]-z_um[0])/crop_xy_um*100:.1f}%)")

# ── Gaussian interpolation between slices ──
print(f"Gaussian interpolation ({INTERP_PER_GAP} sub-slices per gap)...")
dense_slices = []
dense_z_pos = []
dense_real_idx = []

for i in range(n_slices):
    dense_slices.append(sel_slices[i])
    dense_z_pos.append(z_display[i])
    dense_real_idx.append(i)
    if i < n_slices - 1:
        z0 = z_display[i]
        z1 = z_display[i + 1]
        for sub in range(1, INTERP_PER_GAP + 1):
            t_sub = sub / (INTERP_PER_GAP + 1)
            sigma = 0.4
            w1 = math.exp(-0.5 * ((t_sub - 1) / sigma) ** 2)
            w0 = math.exp(-0.5 * (t_sub / sigma) ** 2)
            w_total = w0 + w1
            w0 /= w_total; w1 /= w_total
            interp = (sel_slices[i].astype(np.float32) * w0 +
                      sel_slices[i + 1].astype(np.float32) * w1)
            dense_slices.append(interp.astype(np.uint8))
            dense_z_pos.append(z0 + t_sub * (z1 - z0))
            dense_real_idx.append(-1)

dense_slices = np.array(dense_slices)
dense_z_pos = np.array(dense_z_pos, dtype=np.float64)
n_dense = len(dense_slices)
CENTER_Z = (dense_z_pos[-1] + dense_z_pos[0]) / 2.0
print(f"  {n_dense} total sub-slices")

# ── 3D rendering ──
def render_3d(rot_y, rot_x, alpha_val=0.85):
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cx, cy = W // 2, H // 2
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    z_depths = []
    for i in range(n_dense):
        dz = dense_z_pos[i] - CENTER_Z
        rz = cos_y * dz
        rz2 = cos_x * rz
        z_depths.append((rz2, i))
    z_depths.sort(key=lambda x: x[0])

    for depth, i in z_depths:
        sl = dense_slices[i].astype(np.float32) / 255.0
        sh, sw = sl.shape[:2]
        hw, hh = sw / 2, sh / 2
        dz = dense_z_pos[i] - CENTER_Z

        corners_3d = np.array([
            [-hw, -hh, dz], [hw, -hh, dz],
            [hw, hh, dz], [-hw, hh, dz],
        ], dtype=np.float64)

        rot_corners = []
        for c in corners_3d:
            rx = cos_y * c[0] + sin_y * c[2]
            ry = c[1]
            rz = -sin_y * c[0] + cos_y * c[2]
            ry2 = cos_x * ry - sin_x * rz
            rot_corners.append([rx + cx, ry2 + cy])

        rot_corners = np.array(rot_corners, dtype=np.float32)
        src_corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)

        M_persp = cv2.getPerspectiveTransform(src_corners, rot_corners)
        warped = cv2.warpPerspective(sl, M_persp, (W, H))

        mask = np.max(warped, axis=2) > 0.01
        mask3 = np.stack([mask] * 3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, warped * alpha_val), canvas)

    return np.clip(canvas * 255, 0, 255).astype(np.uint8)

# ── Mid-slice for transition ──
mid_idx = n_slices // 2
mid_slice = sel_slices[mid_idx]

# ── Video ──
vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
total = 0

# ═══════════════════════════════════════════════════════════
# PHASE 1a: Fade out scene5 last frame (1s = 24 frames)
# ═══════════════════════════════════════════════════════════
print("Phase 1a: fade out last frame (1s)...")

for fi in range(24):
    t = ease(fi / 20)
    frame = (last_frame.astype(np.float32) * (1 - t)).astype(np.uint8)
    a_old = 1 - t
    caption(frame, 'HOT = IN-VIVO    GREEN = EX-VIVO    YELLOW = OVERLAP', alpha=max(0, a_old))
    vw.write(frame); total += 1

# ═══════════════════════════════════════════════════════════
# PHASE 1b: Fade in centered overlay (1.5s = 36 frames)
# ═══════════════════════════════════════════════════════════
print("Phase 1b: fade in centered overlay (1.5s)...")

mid_centered = np.zeros((H, W, 3), np.uint8)
mx = (W - disp_w) // 2; my = (H - disp_h) // 2
mid_centered[my:my + disp_h, mx:mx + disp_w] = mid_slice

for fi in range(36):
    t = ease(fi / 30)
    frame = (mid_centered.astype(np.float32) * t).astype(np.uint8)
    a_new = ease((fi - 10) / 20)
    caption(frame, 'ALL  TILES  STITCHED  --  3D  ALIGNMENT', alpha=max(0, a_new))
    vw.write(frame); total += 1

# ═══════════════════════════════════════════════════════════
# PHASE 2: Hold centered overlay (1.5s = 36 frames)
# ═══════════════════════════════════════════════════════════
print("Phase 2: hold centered overlay (1.5s)...")

for fi in range(36):
    frame = mid_centered.copy()
    caption(frame, 'ALL  TILES  STITCHED  --  3D  ALIGNMENT')
    vw.write(frame); total += 1

# ═══════════════════════════════════════════════════════════
# PHASE 3: Slices emerge + gentle tilt (3s = 72 frames)
# ═══════════════════════════════════════════════════════════
print("Phase 3: slices emerge (3s)...")

for fi in range(72):
    t = ease(fi / 62)

    alphas = np.zeros(n_slices, dtype=np.float32)
    alphas[mid_idx] = 0.85
    for si in range(n_slices):
        if si == mid_idx: continue
        dist = abs(si - mid_idx) / max(1, n_slices - 1)
        if t > dist:
            alphas[si] = min(0.85, (t - dist) * 2.0)

    rot_x = INIT_ROT_X * t
    frame = np.zeros((H, W, 3), dtype=np.float32)
    cx, cy = W // 2, H // 2
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    z_depths = []
    for i in range(n_dense):
        dz = dense_z_pos[i] - CENTER_Z
        rz2 = cos_x * dz
        z_depths.append((rz2, i))
    z_depths.sort(key=lambda x: x[0])

    for depth, i in z_depths:
        ri = dense_real_idx[i]
        if ri >= 0:
            alpha = alphas[ri]
        else:
            zp = dense_z_pos[i]
            below = max(0, min(n_slices - 1, int(np.searchsorted(z_display, zp)) - 1))
            above = min(n_slices - 1, below + 1)
            if z_display[above] > z_display[below]:
                frac = (zp - z_display[below]) / (z_display[above] - z_display[below])
            else:
                frac = 0.5
            alpha = alphas[below] * (1 - frac) + alphas[above] * frac
        if alpha < 0.01: continue

        sl = dense_slices[i].astype(np.float32) / 255.0
        sh, sw = sl.shape[:2]
        hw, hh = sw / 2, sh / 2
        dz = dense_z_pos[i] - CENTER_Z

        corners_3d = np.array([[-hw,-hh,dz],[hw,-hh,dz],[hw,hh,dz],[-hw,hh,dz]], dtype=np.float64)
        rot_corners = []
        for c in corners_3d:
            ry2 = cos_x * c[1] - sin_x * c[2]
            rot_corners.append([c[0] + cx, ry2 + cy])
        rot_corners = np.array(rot_corners, dtype=np.float32)
        src_corners = np.array([[0,0],[sw,0],[sw,sh],[0,sh]], dtype=np.float32)
        M_persp = cv2.getPerspectiveTransform(src_corners, rot_corners)
        warped = cv2.warpPerspective(sl, M_persp, (W, H))
        mask = np.max(warped, axis=2) > 0.01
        mask3 = np.stack([mask]*3, axis=-1)
        frame = np.where(mask3, np.maximum(frame, warped * alpha), frame)

    frame_u8 = np.clip(frame * 255, 0, 255).astype(np.uint8)
    caption(frame_u8, 'ALL  TILES  STITCHED  --  3D  ALIGNMENT')
    vw.write(frame_u8); total += 1

# ═══════════════════════════════════════════════════════════
# PHASE 4: 3D rotation — gentle, avoid side-on (7s = 168 frames)
# Only rotate ~270deg with gentle x-tilt, never fully edge-on
# ═══════════════════════════════════════════════════════════
print("Phase 4: 3D rotation (7s)...")

for fi in range(168):
    t = fi / 167.0
    # Slow start, steady middle, slow end
    t_eased = ease(t)
    rot_y = t_eased * math.pi * 1.25  # ~225 deg, avoids full edge-on
    # Keep tilt modest so slices never look paper-thin
    rot_x = INIT_ROT_X + 0.1 * math.sin(t * math.pi)
    frame = render_3d(rot_y, rot_x, 0.85)
    caption(frame, 'FULL  STITCHED  3D  ALIGNMENT  --  ALL  TILES')
    vw.write(frame); total += 1

# ═══════════════════════════════════════════════════════════
# PHASE 5: Settle front-facing (2s = 48 frames)
# ═══════════════════════════════════════════════════════════
print("Phase 5: settle (2s)...")
final_rot_y = math.pi * 1.25

for fi in range(48):
    t = ease(fi / 40)
    rot_y = final_rot_y * (1 - t)
    rot_x = INIT_ROT_X * (1 - t)
    frame = render_3d(rot_y, rot_x, 0.85)
    caption(frame, 'FULL  STITCHED  3D  ALIGNMENT  --  ALL  TILES')
    vw.write(frame); total += 1

# ═══════════════════════════════════════════════════════════
# PHASE 6: Hold (2s = 48 frames)
# ═══════════════════════════════════════════════════════════
print("Phase 6: hold (2s)...")

for fi in range(48):
    frame = render_3d(0.0, 0.0, 0.85)
    caption(frame, 'FULL  STITCHED  3D  ALIGNMENT  --  ALL  TILES')
    vw.write(frame); total += 1

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
