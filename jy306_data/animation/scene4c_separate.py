"""
Scene 4c: Separate the 3D overlay back into in-vivo and ex-vivo side-by-side.

Starts from scene 4b's last frame (3D multi-z overlay stack, front-facing).
Collapses depth to 2D overlay, then splits red/green apart into side-by-side
panels, ending in the same layout scene 5 expects.

Timeline (~4s at 24fps = 96 frames):
  0.0 – 1.5s : 3D stack collapses to single 2D overlay (36 fr)
  1.5 – 3.5s : Overlay splits into left (in-vivo hot) + right (ex-vivo green) (48 fr)
  3.5 – 4.0s : Hold side-by-side (12 fr)

Output: animation/scene4c_h264.mp4
"""

import numpy as np, cv2, tifffile, math, subprocess, os, glob
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
TMP  = f'{BASE}/animation/scene4c_raw.mp4'
OUT  = f'{BASE}/animation/scene4c_h264.mp4'

W, H = 1920, 1080
FPS  = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
INIT_ROT_X = -0.3
Z_SPACING = 4
INTERP_PER_GAP = 3

TILE = 'row2_1'

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

# ── Load data (same as scene 4/4b) ──
print("Loading JY306 z-stack...")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz, hy, wx = jy306.shape

print(f"Loading {TILE} PKL and nd2 stack...")
pkl = np.load(f'{BASE}/png_exports/registration_per_tile_pkl/{TILE}/pkl_transform_{TILE}.npz', allow_pickle=True)
M2d = pkl['M2d_jy306_to_nd2']
M3 = np.vstack([M2d, [0, 0, 1]])
iv = pkl['pcd_invivo_jy306']
ev = pkl['ev_nd2']

nd2_files = sorted(glob.glob(f'{BASE}/png_exports/registration_video/{TILE}/GFP_z*.png'))
nd2_stack = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in nd2_files])
nd2_nz = len(nd2_files)

# ── Layout (must match scene 4/4b/5) ──
DISP_H = int(H * 0.72)
IMG_GAP = 100
scale_jy = DISP_H / hy
disp_jy_w = int(wx * scale_jy); disp_jy_h = DISP_H

margin_nd2 = 350
crop_x0 = max(0, int(ev[:,0].min()-margin_nd2))
crop_y0 = max(0, int(ev[:,1].min()-margin_nd2))
crop_x1 = min(4200, int(ev[:,0].max()+margin_nd2))
crop_y1 = min(4200, int(ev[:,1].max()+margin_nd2))
crop_w = crop_x1 - crop_x0
crop_h = crop_y1 - crop_y0

z_mode = Counter(iv[:,0].astype(int)).most_common(1)[0][0]
z3_u8 = norm_u8(jy306[z_mode])

# Best nd2 z
M_inv = np.linalg.inv(M3)[:2]
best_ncc_init, best_z_init = -1, 0
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
nd2_green_full = np.zeros((4200, 4200, 3), dtype=np.uint8)
nd2_green_full[:,:,1] = nd2_best_u8
nd2_crop = nd2_green_full[crop_y0:crop_y1, crop_x0:crop_x1]
scale_nd2 = DISP_H / nd2_crop.shape[0]
disp_nd2_w = int(nd2_crop.shape[1] * scale_nd2); disp_nd2_h = DISP_H
nd2_disp = cv2.resize(nd2_crop, (disp_nd2_w, disp_nd2_h), interpolation=cv2.INTER_LANCZOS4)

total_w = disp_jy_w + IMG_GAP + disp_nd2_w
jy_x0 = (W - total_w) // 2
jy_y0 = (H - DISP_H) // 2 - 20
nd2_x0 = jy_x0 + disp_jy_w + IMG_GAP
nd2_y0 = jy_y0

# In-vivo hot display
z3_hot = cv2.applyColorMap(z3_u8, cv2.COLORMAP_HOT)
z3_hot[z3_u8 == 0] = 0
jy_disp = cv2.resize(z3_hot, (disp_jy_w, disp_jy_h), interpolation=cv2.INTER_LANCZOS4)

# Side-by-side final frame (what scene 5 expects)
side_by_side = np.zeros((H, W, 3), np.uint8)
side_by_side[jy_y0:jy_y0+disp_jy_h, jy_x0:jy_x0+disp_jy_w] = jy_disp
side_by_side[nd2_y0:nd2_y0+disp_nd2_h, nd2_x0:nd2_x0+disp_nd2_w] = nd2_disp

# ── Build 3D overlay stack (same as scene 4b) to render the starting frame ──
iv_z_min = max(0, int(iv[:,0].min()) - 1)
iv_z_max = min(nz - 1, int(iv[:,0].max()) + 1)
z_range = list(range(iv_z_min, iv_z_max + 1))
mid_idx = z_range.index(z_mode) if z_mode in z_range else len(z_range)//2

slice_w = disp_nd2_w
slice_h = disp_nd2_h

overlay_slices = []
for z_iv in z_range:
    iv_u8 = norm_u8(jy306[z_iv])
    warped_iv = cv2.warpAffine(iv_u8, M2d, (4200, 4200),
                                flags=cv2.INTER_LINEAR, borderValue=0)
    warped_crop = warped_iv[crop_y0:crop_y1, crop_x0:crop_x1]

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

n_slices = len(overlay_slices)

# Gaussian interpolation
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

def render_3d_overlay_stack(rot_y, rot_x, slice_alphas):
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cx, cy = W // 2, H // 2
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    z_depths = []
    for i in range(n_dense):
        dz = dense_z_pos[i] - STACK_CENTER_Z
        rz = cos_y * dz
        rz2 = cos_x * rz
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
        corners_3d = np.array([[-hw,-hh,dz],[hw,-hh,dz],[hw,hh,dz],[-hw,hh,dz]], dtype=np.float64)
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
# Phase 1: 3D stack collapses to single 2D overlay (36 fr = 1.5s)
# Other z-levels fade, tilt reduces to 0, left with middle slice flat
# ─────────────────────────────────────────────────────────────────
print("Phase 1: collapse 3D to 2D (1.5s)...")
alphas_full = np.ones(n_slices, dtype=np.float32) * 0.7

for fi in range(36):
    t = ease(fi / 30)
    # Other slices fade out, middle stays
    alphas = np.ones(n_slices, dtype=np.float32) * 0.7 * (1 - t)
    alphas[mid_idx] = 0.7
    # Tilt reduces to 0
    rot_x = INIT_ROT_X * (1 - t)
    frame = render_3d_overlay_stack(0.0, rot_x, alphas)
    caption(frame, f'MULTI-SLICE  ALIGNMENT  --  {TILE.upper()}')
    vw.write(frame)

# The 2D overlay (middle slice, flat, centered)
overlay_2d = render_3d_overlay_stack(0.0, 0.0, np.array([0.0]*mid_idx + [0.8] + [0.0]*(n_slices-mid_idx-1), dtype=np.float32))

# ─────────────────────────────────────────────────────────────────
# Phase 2: Overlay splits into side-by-side (48 fr = 2s)
# Red channel moves left → becomes in-vivo hot on left
# Green channel stays/moves right → becomes ex-vivo green on right
# ─────────────────────────────────────────────────────────────────
print("Phase 2: split overlay to side-by-side (2s)...")

for fi in range(48):
    t = ease(fi / 40)
    frame = np.zeros((H, W, 3), np.uint8)

    # Crossfade from 2D overlay to side-by-side
    frame = cv2.addWeighted(overlay_2d, 1 - t, side_by_side, t, 0)

    a_old = 1 - ease(fi / 15)
    a_new = ease((fi - 15) / 20)
    caption(frame, f'MULTI-SLICE  ALIGNMENT  --  {TILE.upper()}', alpha=max(0, a_old))

    # Labels fade in
    if t > 0.3:
        la = ease((t - 0.3) / 0.5)
        col_iv = tuple(int(v * la) for v in (100, 180, 255))
        col_ev = tuple(int(v * la) for v in (100, 255, 100))
        cv2.putText(frame, f'IN-VIVO  z = {z_mode}', (jy_x0+10, jy_y0-12),
                    FONT, 0.5, col_iv, 1, cv2.LINE_AA)
        cv2.putText(frame, f'EX-VIVO  {TILE}', (nd2_x0+10, nd2_y0-12),
                    FONT, 0.5, col_ev, 1, cv2.LINE_AA)

    vw.write(frame)

# ─────────────────────────────────────────────────────────────────
# Phase 3: Hold side-by-side (12 fr = 0.5s)
# ─────────────────────────────────────────────────────────────────
print("Phase 3: hold side-by-side (0.5s)...")

for fi in range(12):
    frame = side_by_side.copy()
    cv2.putText(frame, f'IN-VIVO  z = {z_mode}', (jy_x0+10, jy_y0-12),
                FONT, 0.5, (100,180,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f'EX-VIVO  {TILE}', (nd2_x0+10, nd2_y0-12),
                FONT, 0.5, (100,255,100), 1, cv2.LINE_AA)
    caption(frame, f'TILE  {TILE.upper()}  --  ROW2_1  COMPLETE')
    vw.write(frame)

# ── Finalize ──
vw.release()
total_fr = 36 + 48 + 12
print(f"Re-encoding to H.264... ({total_fr} frames, {total_fr/FPS:.1f}s)")
subprocess.run([
    'ffmpeg', '-y', '-i', TMP,
    '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
    OUT
], capture_output=True)
os.remove(TMP)
print(f"Done! {total_fr} frames, {total_fr/FPS:.1f}s @ {FPS}fps -- {OUT}")