"""
Scene 4b: Reveal 3D depth of row2_1 alignment.

Starts from scene 4's last frame (2D overlay: red=warped invivo z=3, green=exvivo nd2).
Transitions to 3D view showing multiple z-slices aligned. Gaussian interpolation
for continuous volume. Real physical z-spacing.

Timeline (~10s at 24fps = 240 frames):
  0.0 – 2.5s : Last frame → shrink overlay + other z-levels emerge (60 fr)
  2.5 – 7.5s : 3D rotation showing depth (120 fr)
  7.5 – 9.0s : Settle front-facing (36 fr)
  9.0 – 10.0s: Hold with text (24 fr)

Output: animation/scene4b_h264.mp4
"""

import numpy as np, cv2, tifffile, math, subprocess, os, glob
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
TMP  = f'{BASE}/animation/scene4b_raw.mp4'
OUT  = f'{BASE}/animation/scene4b_h264.mp4'

W, H = 1920, 1080
FPS  = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

TILE = 'row2_1'
Z_SPACING = 4  # physical ratio (same as scene 3)
INTERP_PER_GAP = 3
INIT_ROT_X = -0.3

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

def norm_u8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 50: return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img.astype(np.float32) - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

def caption(frame, text, alpha=1.0):
    if alpha < 0.01: return
    ts, th2 = 0.72, 1
    (tw2, _), _ = cv2.getTextSize(text, FONT, ts, th2)
    x = (W - tw2) // 2; y = H - 42
    col = tuple(int(v * alpha) for v in WHITE)
    cv2.putText(frame, text, (x, y), FONT, ts, col, th2, cv2.LINE_AA)

# ── Load data (same as scene 4) ──
print("Loading JY306 z-stack...")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz, hy, wx = jy306.shape

print(f"Loading {TILE} PKL and nd2 stack...")
pkl = np.load(f'{BASE}/png_exports/registration_per_tile_pkl/{TILE}/pkl_transform_{TILE}.npz', allow_pickle=True)
M2d = pkl['M2d_jy306_to_nd2']
M3 = np.vstack([M2d, [0, 0, 1]])
iv = pkl['pcd_invivo_jy306']  # (z, y, x)
ev = pkl['ev_nd2']            # (x, y, z)

nd2_files = sorted(glob.glob(f'{BASE}/png_exports/registration_video/{TILE}/GFP_z*.png'))
nd2_stack = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in nd2_files])
nd2_nz = len(nd2_files)

# ── Reconstruct scene 4's last frame layout ──
# (must match scene4_landmarks.py exactly)
DISP_H = int(H * 0.72)
IMG_GAP = 100
scale_jy = DISP_H / hy
disp_jy_w = int(wx * scale_jy); disp_jy_h = DISP_H

margin_nd2 = 350
lm_x_nd2 = ev[:,0]; lm_y_nd2 = ev[:,1]
crop_x0 = max(0, int(lm_x_nd2.min()-margin_nd2))
crop_y0 = max(0, int(lm_y_nd2.min()-margin_nd2))
crop_x1 = min(4200, int(lm_x_nd2.max()+margin_nd2))
crop_y1 = min(4200, int(lm_y_nd2.max()+margin_nd2))

# Best nd2 z for mode invivo z
z_mode = Counter(iv[:,0].astype(int)).most_common(1)[0][0]
z3_u8 = norm_u8(jy306[z_mode])

# Find best nd2 z for z_mode
best_ncc_init, best_z_init = -1, 0
M_inv = np.linalg.inv(M3)[:2]
for zi in range(nd2_nz):
    nd2_z = nd2_stack[zi].astype(np.uint8)
    warped = cv2.warpAffine(nd2_z, M_inv, (wx, hy), flags=cv2.INTER_LINEAR, borderValue=0)
    wn = norm_u8(warped)
    mask = (z3_u8 > 5) & (wn > 5)
    if mask.sum() < 100: continue
    a = z3_u8[mask].astype(np.float32); a -= a.mean()
    b = wn[mask].astype(np.float32); b -= b.mean()
    ncc = float(np.sum(a*b) / (np.sqrt(np.sum(a**2)*np.sum(b**2)) + 1e-8))
    if ncc > best_ncc_init: best_ncc_init, best_z_init = ncc, zi

nd2_best_u8 = norm_u8(nd2_stack[best_z_init].astype(np.uint8))
nd2_magenta_full = np.zeros((4200, 4200, 3), dtype=np.uint8)
nd2_magenta_full[:,:,0] = nd2_best_u8  # B } ex-vivo = magenta
nd2_magenta_full[:,:,2] = nd2_best_u8  # R }
nd2_crop = nd2_magenta_full[crop_y0:crop_y1, crop_x0:crop_x1]
scale_nd2 = DISP_H / nd2_crop.shape[0]
disp_nd2_w = int(nd2_crop.shape[1] * scale_nd2); disp_nd2_h = DISP_H
nd2_disp = cv2.resize(nd2_crop, (disp_nd2_w, disp_nd2_h), interpolation=cv2.INTER_LANCZOS4)

total_w = disp_jy_w + IMG_GAP + disp_nd2_w
jy_x0 = (W - total_w) // 2
jy_y0 = (H - DISP_H) // 2 - 20
nd2_x0 = jy_x0 + disp_jy_w + IMG_GAP
nd2_y0 = jy_y0

# Build scene 4's final frame (the 2D overlay)
M_end_canvas = np.array([
    [scale_nd2, 0, nd2_x0 - crop_x0*scale_nd2],
    [0, scale_nd2, nd2_y0 - crop_y0*scale_nd2],
    [0, 0, 1]
], dtype=np.float64) @ M3

nd2_bg = np.zeros((H, W, 3), np.uint8)
nd2_bg[nd2_y0:nd2_y0+disp_nd2_h, nd2_x0:nd2_x0+disp_nd2_w] = nd2_disp

scene4_last = np.zeros((H, W, 3), np.uint8)
warped_z3 = cv2.warpAffine(z3_u8, M_end_canvas[:2].astype(np.float64), (W, H),
                            flags=cv2.INTER_LANCZOS4, borderValue=0)
scene4_last[:,:,2] = warped_z3
scene4_last[:,:,1] = nd2_bg[:,:,1]
print(f"Scene 4 last frame reconstructed (z_mode={z_mode}, nd2_z={best_z_init})")

# ── Build multi-z overlay slices for 3D rendering ──
iv_z_min = max(0, int(iv[:,0].min()) - 1)
iv_z_max = min(nz - 1, int(iv[:,0].max()) + 1)
z_range = list(range(iv_z_min, iv_z_max + 1))
print(f"Building overlays for z={iv_z_min}-{iv_z_max} ({len(z_range)} slices)...")

crop_w = crop_x1 - crop_x0
crop_h = crop_y1 - crop_y0

# Use same display size as scene 4's nd2 panel
slice_w = disp_nd2_w
slice_h = disp_nd2_h

overlay_slices = []
overlay_z_labels = []

for z_iv in z_range:
    iv_u8 = norm_u8(jy306[z_iv])
    warped_iv = cv2.warpAffine(iv_u8, M2d, (4200, 4200),
                                flags=cv2.INTER_LINEAR, borderValue=0)
    warped_crop = warped_iv[crop_y0:crop_y1, crop_x0:crop_x1]

    # Find best nd2 z
    best_ncc_z, best_nd2_z = -1, 0
    for zi in range(nd2_nz):
        nd2_full = nd2_stack[zi].astype(np.uint8)
        nd2_c = nd2_full[crop_y0:min(crop_y1, nd2_full.shape[0]),
                         crop_x0:min(crop_x1, nd2_full.shape[1])]
        wc = warped_crop[:nd2_c.shape[0], :nd2_c.shape[1]]
        wn = norm_u8(wc); nn = norm_u8(nd2_c)
        mask = (wn > 5) & (nn > 5)
        if mask.sum() < 100: continue
        a_v = wn[mask].astype(np.float32); a_v -= a_v.mean()
        b_v = nn[mask].astype(np.float32); b_v -= b_v.mean()
        ncc_v = float(np.sum(a_v*b_v) / (np.sqrt(np.sum(a_v**2)*np.sum(b_v**2)) + 1e-8))
        if ncc_v > best_ncc_z: best_ncc_z, best_nd2_z = ncc_v, zi

    nd2_best = nd2_stack[best_nd2_z].astype(np.uint8)
    nd2_c = nd2_best[crop_y0:min(crop_y1, nd2_best.shape[0]),
                     crop_x0:min(crop_x1, nd2_best.shape[1])]
    wc = warped_crop[:nd2_c.shape[0], :nd2_c.shape[1]]

    overlay = np.zeros((nd2_c.shape[0], nd2_c.shape[1], 3), np.uint8)
    overlay[:,:,1] = norm_u8(nd2_c)
    overlay[:,:,2] = norm_u8(wc)

    overlay_small = cv2.resize(overlay, (slice_w, slice_h), interpolation=cv2.INTER_AREA)
    overlay_slices.append(overlay_small)
    overlay_z_labels.append((z_iv, best_nd2_z))
    print(f"  z_iv={z_iv} -> nd2_z={best_nd2_z}, NCC={best_ncc_z:.3f}")

n_slices = len(overlay_slices)
mid_idx = z_range.index(z_mode) if z_mode in z_range else len(z_range)//2

# ── Gaussian interpolation ──
print(f"Gaussian interpolation: {n_slices} -> {n_slices + (n_slices-1)*INTERP_PER_GAP} sub-slices...")
dense_slices = []
dense_z_pos = []
dense_real_idx = []

for i in range(n_slices):
    dense_slices.append(overlay_slices[i])
    dense_z_pos.append(i * Z_SPACING)
    dense_real_idx.append(i)
    if i < n_slices - 1:
        for sub in range(1, INTERP_PER_GAP + 1):
            t = sub / (INTERP_PER_GAP + 1)
            interp = (overlay_slices[i].astype(np.float32) * (1-t) +
                      overlay_slices[i+1].astype(np.float32) * t)
            dense_slices.append(interp.astype(np.uint8))
            dense_z_pos.append(i * Z_SPACING + t * Z_SPACING)
            dense_real_idx.append(-1)

dense_slices = np.array(dense_slices)
dense_z_pos = np.array(dense_z_pos, dtype=np.float64)
n_dense = len(dense_slices)
STACK_CENTER_Z = (dense_z_pos[-1] + dense_z_pos[0]) / 2.0

# ── 3D rendering ──
def render_3d_overlay_stack(rot_y, rot_x, slice_alphas):
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cx, cy = W // 2, H // 2

    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    z_depths = []
    for i in range(n_dense):
        dz = dense_z_pos[i] - STACK_CENTER_Z
        rz = -sin_y * 0 + cos_y * dz
        rz2 = sin_x * 0 + cos_x * rz
        z_depths.append((rz2, i))
    z_depths.sort(key=lambda x: x[0])

    for depth, i in z_depths:
        real_idx = dense_real_idx[i]
        if real_idx >= 0:
            alpha = slice_alphas[real_idx] if real_idx < len(slice_alphas) else 0.5
        else:
            zp = dense_z_pos[i]
            z_below = int(zp / Z_SPACING)
            z_above = min(n_slices-1, z_below + 1)
            t = (zp - z_below * Z_SPACING) / Z_SPACING if Z_SPACING > 0 else 0
            a_b = slice_alphas[z_below] if z_below < len(slice_alphas) else 0.5
            a_a = slice_alphas[z_above] if z_above < len(slice_alphas) else 0.5
            alpha = a_b * (1-t) + a_a * t

        if alpha < 0.01: continue

        sl = dense_slices[i].astype(np.float32) / 255.0
        sh, sw = sl.shape[:2]
        hw, hh = sw / 2, sh / 2
        dz = dense_z_pos[i] - STACK_CENTER_Z

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
        src_corners = np.array([[0,0],[sw,0],[sw,sh],[0,sh]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_corners, rot_corners)
        warped = cv2.warpPerspective(sl, M, (W, H))

        mask = np.max(warped, axis=2) > 0.01
        mask3 = np.stack([mask]*3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, warped * alpha), canvas)

    return np.clip(canvas * 255, 0, 255).astype(np.uint8)

# ── Video writer ──
vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

# ─────────────────────────────────────────────────────────────────
# Phase 1: Scene 4 last frame → transition to 3D stack (60 fr = 2.5s)
# The 2D overlay shrinks and other z-level overlays emerge around it
# ─────────────────────────────────────────────────────────────────
print("Phase 1: scene4 last frame -> 3D emerge (2.5s)...")

# Pre-render the mid-slice (z_mode) in 3D at front-facing + tilt as the "target"
# During transition, crossfade from scene4_last to the 3D view with slices emerging

for fi in range(60):
    t = ease(fi / 50)

    # Blend: (1-t) = scene4 last frame, t = 3D stack with emerging slices
    alphas = np.zeros(n_slices, dtype=np.float32)
    alphas[mid_idx] = 0.8 * t  # middle slice visible from start of 3D
    # Other slices emerge in second half
    t_emerge = max(0, (t - 0.4) / 0.6)
    for si in range(n_slices):
        if si == mid_idx: continue
        dist = abs(si - mid_idx)
        max_dist = t_emerge * (n_slices - 1)
        if dist <= max_dist:
            alphas[si] = min(0.7, (max_dist - dist + 1) / 2.0) * t_emerge

    rot_x = INIT_ROT_X * t
    frame_3d = render_3d_overlay_stack(0.0, rot_x, alphas)

    # Crossfade from scene4 last frame to 3D
    frame = cv2.addWeighted(scene4_last, 1 - t, frame_3d, t, 0)

    a_old = 1 - ease(fi / 20)
    a_new = ease((fi - 20) / 25)
    caption(frame, 'RED = IN-VIVO    GREEN = EX-VIVO    YELLOW = MATCH', alpha=max(0, a_old))
    caption(frame, f'3D DEPTH:  IN-VIVO Z = {iv_z_min}  TO  Z = {iv_z_max}  ALIGNED', alpha=max(0, a_new))
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────
# Phase 2: 3D rotation (120 fr = 5s)
# ─────────────────────────────────────────────────────────────────
print("Phase 2: 3D rotation (5s)...")
alphas_full = np.ones(n_slices, dtype=np.float32) * 0.7

for fi in range(120):
    t = fi / 119.0
    rot_y = t * math.pi * 1.5
    rot_x = INIT_ROT_X + 0.15 * math.sin(t * math.pi)

    frame = render_3d_overlay_stack(rot_y, rot_x, alphas_full)
    caption(frame, f'MULTI-SLICE  ALIGNMENT  --  IN-VIVO  Z = {iv_z_min}  TO  {iv_z_max}')
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────
# Phase 3: Settle front-facing (36 fr = 1.5s)
# ─────────────────────────────────────────────────────────────────
print("Phase 3: settle (1.5s)...")
final_rot_y = math.pi * 1.5

for fi in range(36):
    t = ease(fi / 30)
    rot_y = final_rot_y * (1 - t)
    rot_x = INIT_ROT_X

    frame = render_3d_overlay_stack(rot_y, rot_x, alphas_full)
    caption(frame, f'MULTI-SLICE  ALIGNMENT  --  IN-VIVO  Z = {iv_z_min}  TO  {iv_z_max}')
    vw.write(frame)

# ─────────────────────────────────────────────────────────────────
# Phase 4: Hold with z-level labels (24 fr = 1s)
# ─────────────────────────────────────────────────────────────────
print("Phase 4: hold with labels (1s)...")

for fi in range(24):
    frame = render_3d_overlay_stack(0.0, INIT_ROT_X, alphas_full)
    label_alpha = ease(fi / 12)
    for si, (z_iv, z_nd2) in enumerate(overlay_z_labels):
        ly = H//2 - int((si - n_slices/2) * 28)
        col = tuple(int(v * label_alpha) for v in WHITE)
        cv2.putText(frame, f'z={z_iv} -- nd2 z={z_nd2}', (W - 280, ly),
                    FONT, 0.38, col, 1, cv2.LINE_AA)
    caption(frame, f'MULTI-SLICE  ALIGNMENT  --  {TILE.upper()}')
    vw.write(frame)

# ── Finalize ──
vw.release()
total_fr = 60 + 120 + 36 + 24
print(f"Re-encoding to H.264... ({total_fr} frames, {total_fr/FPS:.1f}s)")
subprocess.run([
    'ffmpeg', '-y', '-i', TMP,
    '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
    OUT
], capture_output=True)
os.remove(TMP)
print(f"Done! {total_fr} frames, {total_fr/FPS:.1f}s @ {FPS}fps -- {OUT}")
