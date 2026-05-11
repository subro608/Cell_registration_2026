"""
Scene 5b — Per-tile 3D (row5_1 + others) then stitched 3D overlay.

Flow:
  Phase 1: Start from row5_1's overlay (right-side) → slide to center (1.5s)
  Phase 2: row5_1 per-tile 3D emerge with thickness (2s)
  Phase 3: row5_1 3D rotation (3.5s)
  Phase 4: row5_1 settle + hold with label (1.5s)
  Phase 5: Brief montage of 3 other tiles' overlays + "19 tiles total" (3s)
  Phase 6: Crossfade to stitched mid-slice + "Combining all 19 tiles" (1.5s)
  Phase 7: Stitched 3D emerge (2.5s)
  Phase 8: Stitched 3D rotation (6s)
  Phase 9: Settle (1.5s)
  Phase 10: Hold (1.5s)

Output: animation/scene5b_v2_h264.mp4
"""

import numpy as np, cv2, math, subprocess, os, glob
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
FRAMES_DIR = f'{BASE}/animation/frames_scene5b_v2'
OUT  = f'{BASE}/animation/scene5b_v2_h264.mp4'
SCENES_1_5 = f'{BASE}/animation/scenes_1_to_5.mp4'

W, H = 1920, 1080
FPS  = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (0, 220, 0)

Z_SPACING = 4
INTERP_PER_GAP = 3
INIT_ROT_X = -0.3

# Physical dimensions
ND2_Z_STEP_UM = 2.0
CANVAS_UM_PER_PX = 0.645

# Tiles to show in montage (spread across grid)
MONTAGE_TILES = ['row2_3', 'row3_4', 'row4_2']

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

def norm8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 50: return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

def make_hot(u8):
    hot = cv2.applyColorMap(u8, cv2.COLORMAP_HOT)
    hot[u8 == 0] = 0
    return hot

def caption(frame, text, alpha=1.0):
    if alpha < 0.01: return
    ts, th = 0.72, 1
    (tw, _), _ = cv2.getTextSize(text, FONT, ts, th)
    x = (W - tw) // 2; y = H - 42
    col = tuple(int(v * alpha) for v in WHITE)
    cv2.putText(frame, text, (x, y), FONT, ts, col, th, cv2.LINE_AA)

def ncc(a, b):
    mask = (a > 5) & (b > 5)
    if mask.sum() < 100: return -1
    af = a[mask].astype(np.float32); af -= af.mean()
    bf = b[mask].astype(np.float32); bf -= bf.mean()
    return float(np.sum(af * bf) / (np.sqrt(np.sum(af**2) * np.sum(bf**2)) + 1e-8))


# ──────────────────────────────────────────────────────────────
# STEP 1: Extract row5_1's Phase F last frame from scene5 video
# ──────────────────────────────────────────────────────────────
print("Extracting last frame of scenes_1_to_5...")

# Extract last frame using ffmpeg (more reliable than VideoCapture seek)
_tmp_start = '/tmp/scene5b_start_frame.png'
subprocess.run([
    'ffmpeg', '-y', '-sseof', '-0.1', '-i', SCENES_1_5,
    '-frames:v', '1', '-update', '1', _tmp_start
], capture_output=True)
start_frame = cv2.imread(_tmp_start)
assert start_frame is not None, f"Failed to extract last frame from {SCENES_1_5}"
if start_frame.shape[:2] != (H, W):
    start_frame = cv2.resize(start_frame, (W, H))
print(f"  Start frame shape: {start_frame.shape}")


# ──────────────────────────────────────────────────────────────
# STEP 2: Build row5_1 per-tile 3D overlay slices
# ──────────────────────────────────────────────────────────────
print("\nBuilding row5_1 per-tile 3D overlay slices...")
import tifffile

jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_jy = jy306.shape[0]

tile = 'row5_1'
nd2_files = sorted(glob.glob(f'{BASE}/png_exports/registration_video/{tile}/GFP_z*.png'))
nd2_stack = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in nd2_files])
pkl = np.load(f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz')
M2d = pkl['M2d_jy306_to_nd2']
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

# Crop region
margin_nd2 = 350
crop_x0 = max(0, int(ev[:, 0].min() - margin_nd2))
crop_y0 = max(0, int(ev[:, 1].min() - margin_nd2))
crop_x1 = min(4200, int(ev[:, 0].max() + margin_nd2))
crop_y1 = min(4200, int(ev[:, 1].max() + margin_nd2))

# Display size (same as scene5)
DISP_H_TILE = int(H * 0.72)
scale_nd2 = DISP_H_TILE / (crop_y1 - crop_y0)
disp_nd2_w = int((crop_x1 - crop_x0) * scale_nd2)
disp_nd2_h = DISP_H_TILE

# Layout position — nd2 sits on the RIGHT in scene5
disp_jy_w = int(jy306.shape[2] * (DISP_H_TILE / jy306.shape[1]))
IMG_GAP = 100
total_w = disp_jy_w + IMG_GAP + disp_nd2_w
nd2_x0 = (W - total_w) // 2 + disp_jy_w + IMG_GAP
nd2_y0 = (H - DISP_H_TILE) // 2 - 20

# Build multi-z overlays
iv_z_min = max(0, int(iv[:, 0].min()) - 1)
iv_z_max = min(nz_jy - 1, int(iv[:, 0].max()) + 1)
z_range_3d = list(range(iv_z_min, iv_z_max + 1))

overlay_slices_tile = []
overlay_z_labels = []

print(f"  z_range: {iv_z_min}-{iv_z_max}, nd2 nz: {len(nd2_stack)}")

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
        score = ncc(wn_z, nn_z)
        if score > best_ncc_z: best_ncc_z, best_nd2_z = score, zi

    nd2_best_z = nd2_stack[best_nd2_z].astype(np.uint8)
    nd2_c_best = nd2_best_z[crop_y0:min(crop_y1, nd2_best_z.shape[0]),
                            crop_x0:min(crop_x1, nd2_best_z.shape[1])]
    wc_best = warped_crop_z[:nd2_c_best.shape[0], :nd2_c_best.shape[1]]

    ov_3d = np.zeros((nd2_c_best.shape[0], nd2_c_best.shape[1], 3), np.uint8)
    ov_3d[:, :, 1] = norm8(nd2_c_best)
    ov_hot = make_hot(norm8(wc_best))
    ov_3d = cv2.addWeighted(ov_3d, 0.5, ov_hot[:nd2_c_best.shape[0], :nd2_c_best.shape[1]], 0.5, 0)

    ov_small = cv2.resize(ov_3d, (disp_nd2_w, disp_nd2_h), interpolation=cv2.INTER_AREA)
    overlay_slices_tile.append(ov_small)
    overlay_z_labels.append((z_iv, best_nd2_z))
    print(f"    z_iv={z_iv} -> nd2_z={best_nd2_z} (NCC={best_ncc_z:.3f})")

n_tile_slices = len(overlay_slices_tile)
mid_idx_tile = z_range_3d.index(MODE_Z) if MODE_Z in z_range_3d else n_tile_slices // 2
print(f"  {n_tile_slices} slices, mid_idx={mid_idx_tile}")

# Free invivo volume (needed only for row5_1 3D)
del jy306

# Gaussian interpolation for tile 3D
dense_tile = []
dense_tile_z = []
dense_tile_real = []
for i in range(n_tile_slices):
    dense_tile.append(overlay_slices_tile[i])
    dense_tile_z.append(i * Z_SPACING)
    dense_tile_real.append(i)
    if i < n_tile_slices - 1:
        for sub in range(1, INTERP_PER_GAP + 1):
            t_sub = sub / (INTERP_PER_GAP + 1)
            interp = (overlay_slices_tile[i].astype(np.float32) * (1 - t_sub) +
                      overlay_slices_tile[i + 1].astype(np.float32) * t_sub)
            dense_tile.append(interp.astype(np.uint8))
            dense_tile_z.append(i * Z_SPACING + t_sub * Z_SPACING)
            dense_tile_real.append(-1)

dense_tile = np.array(dense_tile)
dense_tile_z = np.array(dense_tile_z, dtype=np.float64)
n_dense_tile = len(dense_tile)
TILE_CENTER_Z = (dense_tile_z[-1] + dense_tile_z[0]) / 2.0
print(f"  {n_dense_tile} dense sub-slices")


def render_tile_3d(rot_y, rot_x, slice_alphas, center=None):
    """Render per-tile 3D stack (same approach as row2_1 in scene5)."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cx, cy = center if center else (W // 2, H // 2)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    z_depths = []
    for ii in range(n_dense_tile):
        dz = dense_tile_z[ii] - TILE_CENTER_Z
        rz = cos_y * dz
        rz2 = cos_x * rz
        z_depths.append((rz2, ii))
    z_depths.sort(key=lambda x: x[0])

    for depth, ii in z_depths:
        real_idx = dense_tile_real[ii]
        if real_idx >= 0:
            alpha = slice_alphas[real_idx] if real_idx < len(slice_alphas) else 0.5
        else:
            zp = dense_tile_z[ii]
            z_below = int(zp / Z_SPACING)
            z_above = min(n_tile_slices - 1, z_below + 1)
            t_a = (zp - z_below * Z_SPACING) / Z_SPACING if Z_SPACING > 0 else 0
            a_b = slice_alphas[z_below] if z_below < len(slice_alphas) else 0.5
            a_a = slice_alphas[z_above] if z_above < len(slice_alphas) else 0.5
            alpha = a_b * (1 - t_a) + a_a * t_a

        if alpha < 0.01: continue

        sl = dense_tile[ii].astype(np.float32) / 255.0
        sh_, sw_ = sl.shape[:2]
        hw, hh = sw_ / 2, sh_ / 2
        dz = dense_tile_z[ii] - TILE_CENTER_Z

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


# ──────────────────────────────────────────────────────────────
# STEP 3: Extract montage tile frames from scene5 video
# ──────────────────────────────────────────────────────────────
print("\nExtracting montage tile frames from scenes_1_to_5...")

# scenes_1_to_5 has: frames 1-703 = scenes 1-4, frames 704+ = scene 5
# Scene 5 tile order: row2_1 first (744 frames with 3D), then other tiles (276 each)
# In scenes_1_to_5: scene5 starts at frame 704
SCENE5_START = 703  # 0-indexed
ALL_TILES = ['row2_1', 'row1_1', 'row1_2', 'row1_3', 'row2_2', 'row2_3', 'row2_4',
             'row2_5', 'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
             'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6', 'row5_1']

tile_frame_ends = {}
cumulative = SCENE5_START
for t in ALL_TILES:
    if t == 'row2_1':
        n_frames = 36 + 216 + 36 + 48 + 96 + 72 + 240  # 744
    else:
        n_frames = 12 + 96 + 12 + 36 + 72 + 48  # 276
    cumulative += n_frames
    if t == 'row2_1':
        phase_f_end = cumulative - 240  # before 3D
    else:
        phase_f_end = cumulative
    tile_frame_ends[t] = phase_f_end

montage_frames = {}
cap = cv2.VideoCapture(SCENES_1_5)
for mt in MONTAGE_TILES:
    fnum = tile_frame_ends.get(mt, 0) - 1  # 0-indexed, last Phase F frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
    ret, fr = cap.read()
    if ret:
        if fr.shape[:2] != (H, W):
            fr = cv2.resize(fr, (W, H))
        montage_frames[mt] = fr
        print(f"  {mt}: frame {fnum+1}")
    else:
        print(f"  {mt}: FAILED to read frame {fnum+1}")
cap.release()


# ──────────────────────────────────────────────────────────────
# STEP 4: Load stitched 3D assets
# ──────────────────────────────────────────────────────────────
print("\nLoading stitched 3D assets...")
assets = np.load(f'{BASE}/animation/scene5b_assets_v2.npz')
ov_slices_raw = assets['overlay_slices']
z_indices = assets['z_indices']
n_stitch_slices = len(ov_slices_raw)
raw_h, raw_w = ov_slices_raw.shape[1:3]
print(f"  {n_stitch_slices} slices, {raw_w}x{raw_h}")

# Crop to content bounding box
mask = np.any(ov_slices_raw > 3, axis=-1)
any_slice = np.any(mask, axis=0)
rows = np.where(any_slice.any(axis=1))[0]
cols = np.where(any_slice.any(axis=0))[0]
PAD = 50
r0 = max(0, rows[0] - PAD)
r1 = min(raw_h, rows[-1] + PAD)
c0 = max(0, cols[0] - PAD)
c1 = min(raw_w, cols[-1] + PAD)
ov_cropped = ov_slices_raw[:, r0:r1, c0:c1, :]
crop_h, crop_w = ov_cropped.shape[1:3]
print(f"  Cropped: {crop_w}x{crop_h}")

# Scale to display
DISP_H_STITCH = int(H * 0.88)
scale_up = DISP_H_STITCH / crop_h
disp_stitch_w = int(crop_w * scale_up)
disp_stitch_h = DISP_H_STITCH
if disp_stitch_w > int(W * 0.95):
    disp_stitch_w = int(W * 0.95)
    scale_up = disp_stitch_w / crop_w
    disp_stitch_h = int(crop_h * scale_up)
print(f"  Display: {disp_stitch_w}x{disp_stitch_h}")

sel_stitch = np.array([cv2.resize(ov_cropped[i], (disp_stitch_w, disp_stitch_h),
                                   interpolation=cv2.INTER_LANCZOS4)
                        for i in range(n_stitch_slices)])
del ov_slices_raw, ov_cropped

# Physical z-positions
z_um = z_indices.astype(np.float64) * ND2_Z_STEP_UM
crop_xy_um = crop_w * CANVAS_UM_PER_PX
z_display_stitch = (z_um - z_um[0]) / crop_xy_um * disp_stitch_w

# Gaussian interpolation for stitched
dense_stitch = []
dense_stitch_z = []
dense_stitch_real = []
for i in range(n_stitch_slices):
    dense_stitch.append(sel_stitch[i])
    dense_stitch_z.append(z_display_stitch[i])
    dense_stitch_real.append(i)
    if i < n_stitch_slices - 1:
        z0_ = z_display_stitch[i]
        z1_ = z_display_stitch[i + 1]
        for sub in range(1, INTERP_PER_GAP + 1):
            t_sub = sub / (INTERP_PER_GAP + 1)
            sigma = 0.4
            w1 = math.exp(-0.5 * ((t_sub - 1) / sigma) ** 2)
            w0 = math.exp(-0.5 * (t_sub / sigma) ** 2)
            w_total = w0 + w1; w0 /= w_total; w1 /= w_total
            interp = (sel_stitch[i].astype(np.float32) * w0 +
                      sel_stitch[i + 1].astype(np.float32) * w1)
            dense_stitch.append(interp.astype(np.uint8))
            dense_stitch_z.append(z0_ + t_sub * (z1_ - z0_))
            dense_stitch_real.append(-1)

dense_stitch = np.array(dense_stitch)
dense_stitch_z = np.array(dense_stitch_z, dtype=np.float64)
n_dense_stitch = len(dense_stitch)
STITCH_CENTER_Z = (dense_stitch_z[-1] + dense_stitch_z[0]) / 2.0
print(f"  {n_dense_stitch} dense stitched sub-slices")


def render_stitch_3d(rot_y, rot_x, alpha_val=0.85):
    """Render stitched 3D (all tiles combined)."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cx, cy = W // 2, H // 2
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    z_depths = []
    for i in range(n_dense_stitch):
        dz = dense_stitch_z[i] - STITCH_CENTER_Z
        rz = cos_y * dz
        rz2 = cos_x * rz
        z_depths.append((rz2, i))
    z_depths.sort(key=lambda x: x[0])

    for depth, i in z_depths:
        sl = dense_stitch[i].astype(np.float32) / 255.0
        sh, sw = sl.shape[:2]
        hw, hh = sw / 2, sh / 2
        dz = dense_stitch_z[i] - STITCH_CENTER_Z

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
        M_persp = cv2.getPerspectiveTransform(src_corners, rot_corners)
        warped = cv2.warpPerspective(sl, M_persp, (W, H))
        mask = np.max(warped, axis=2) > 0.01
        mask3 = np.stack([mask]*3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, warped * alpha_val), canvas)

    return np.clip(canvas * 255, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════
os.makedirs(FRAMES_DIR, exist_ok=True)
total = 0

def save_frame(frame):
    global total
    total += 1
    cv2.imwrite(f'{FRAMES_DIR}/frame_{total:05d}.png', frame)

# Precompute mid-slice overlay for transitions
mid_overlay_tile = overlay_slices_tile[mid_idx_tile]
mid_stitch_idx = n_stitch_slices // 2
mid_stitch = sel_stitch[mid_stitch_idx]

# ═════════════════════════════════════════════════════════════
# PHASE 1: Slide row5_1 overlay from right to center (1.5s = 36fr)
# ═════════════════════════════════════════════════════════════
print("\nPhase 1: slide row5_1 to center (1.5s)...")

# Starting position: center of nd2 panel in scene5 layout
start_cx = nd2_x0 + disp_nd2_w // 2
start_cy = nd2_y0 + disp_nd2_h // 2
end_cx = W // 2
end_cy = H // 2

for fi in range(36):
    t = ease(fi / 30)
    frame = np.zeros((H, W, 3), np.uint8)
    # Fade out scene5 background
    if t < 1.0:
        frame = (start_frame.astype(np.float32) * (1 - t)).astype(np.uint8)
    # Slide overlay
    cur_cx = int(start_cx * (1 - t) + end_cx * t)
    cur_cy = int(start_cy * (1 - t) + end_cy * t)
    sh_, sw_ = mid_overlay_tile.shape[:2]
    px = cur_cx - sw_ // 2; py = cur_cy - sh_ // 2
    src_x0_ = max(0, -px); src_y0_ = max(0, -py)
    dst_x0_ = max(0, px); dst_y0_ = max(0, py)
    dst_x1_ = min(W, px + sw_); dst_y1_ = min(H, py + sh_)
    if dst_x1_ > dst_x0_ and dst_y1_ > dst_y0_:
        region = mid_overlay_tile[src_y0_:src_y0_ + (dst_y1_ - dst_y0_),
                                  src_x0_:src_x0_ + (dst_x1_ - dst_x0_)]
        alpha_sl = max(0.5, t)
        existing = frame[dst_y0_:dst_y1_, dst_x0_:dst_x1_]
        frame[dst_y0_:dst_y1_, dst_x0_:dst_x1_] = cv2.addWeighted(
            existing, 1 - alpha_sl, region, alpha_sl, 0)
    caption(frame, 'HOT = IN-VIVO    GREEN = EX-VIVO    YELLOW = OVERLAP')
    save_frame(frame)

# ═════════════════════════════════════════════════════════════
# PHASE 2: row5_1 3D emerge (2s = 48fr)
# ═════════════════════════════════════════════════════════════
print("Phase 2: row5_1 3D emerge (2s)...")

for fi in range(48):
    t = ease(fi / 40)
    alphas = np.zeros(n_tile_slices, dtype=np.float32)
    alphas[mid_idx_tile] = 0.8
    for si in range(n_tile_slices):
        if si == mid_idx_tile: continue
        dist = abs(si - mid_idx_tile)
        max_dist = t * (n_tile_slices - 1)
        if dist <= max_dist:
            alphas[si] = min(0.7, (max_dist - dist + 1) / 2.0) * t
    rot_x = INIT_ROT_X * t
    frame = render_tile_3d(0.0, rot_x, alphas)
    a_new = ease((fi - 10) / 20)
    caption(frame, f'3D  DEPTH:  ROW5_1  --  IN-VIVO  Z = {iv_z_min}  TO  Z = {iv_z_max}',
            alpha=max(0, a_new))
    save_frame(frame)

# ═════════════════════════════════════════════════════════════
# PHASE 3: row5_1 3D rotation (3.5s = 84fr)
# ═════════════════════════════════════════════════════════════
print("Phase 3: row5_1 3D rotation (3.5s)...")

alphas_full_tile = np.ones(n_tile_slices, dtype=np.float32) * 0.7
for fi in range(84):
    t = fi / 83.0
    rot_y = t * math.pi * 1.25  # ~225 deg
    rot_x = INIT_ROT_X + 0.1 * math.sin(t * math.pi)
    frame = render_tile_3d(rot_y, rot_x, alphas_full_tile)
    caption(frame, f'MULTI-SLICE  ALIGNMENT  --  ROW5_1  ({n_tile_slices} z-slices)')
    save_frame(frame)

# ═════════════════════════════════════════════════════════════
# PHASE 4: row5_1 settle + hold (1.5s = 36fr)
# ═════════════════════════════════════════════════════════════
print("Phase 4: row5_1 settle + hold (1.5s)...")

final_rot_y_tile = math.pi * 1.25
for fi in range(24):
    t = ease(fi / 20)
    rot_y = final_rot_y_tile * (1 - t)
    rot_x = INIT_ROT_X
    frame = render_tile_3d(rot_y, rot_x, alphas_full_tile)
    caption(frame, f'MULTI-SLICE  ALIGNMENT  --  ROW5_1')
    save_frame(frame)

# Hold with z-labels
for fi in range(12):
    frame = render_tile_3d(0.0, INIT_ROT_X, alphas_full_tile)
    label_alpha = ease(fi / 8)
    for si, (z_iv_l, z_nd2_l) in enumerate(overlay_z_labels):
        ly = H // 2 - int((si - n_tile_slices / 2) * 28)
        col = tuple(int(v * label_alpha) for v in WHITE)
        cv2.putText(frame, f'z={z_iv_l} -- nd2 z={z_nd2_l}', (W - 280, ly),
                    FONT, 0.38, col, 1, cv2.LINE_AA)
    caption(frame, f'ROW5_1  --  1 OF 19 TILES')
    save_frame(frame)

# ═════════════════════════════════════════════════════════════
# PHASE 5: Brief montage of other tiles' overlays (3s = 72fr)
# Show 3 other tiles from scene5 with labels
# ═════════════════════════════════════════════════════════════
print("Phase 5: montage of other tiles (3s)...")

# Get last row5_1 frame
last_tile_frame = render_tile_3d(0.0, INIT_ROT_X, alphas_full_tile)

# Each tile gets ~0.8s display + crossfade between them
frames_per_tile = 24
n_montage = len(MONTAGE_TILES)

for mi, mt in enumerate(MONTAGE_TILES):
    if mt not in montage_frames:
        continue
    mt_frame = montage_frames[mt]
    tile_num = ALL_TILES.index(mt) + 1 if mt in ALL_TILES else mi + 2

    for fi in range(frames_per_tile):
        if mi == 0 and fi < 12:
            # Crossfade from row5_1 3D to first montage tile
            t = ease(fi / 10)
            frame = cv2.addWeighted(last_tile_frame, 1 - t, mt_frame, t, 0)
        elif fi < 6 and mi > 0:
            # Quick crossfade between montage tiles
            prev_mt = MONTAGE_TILES[mi - 1]
            if prev_mt in montage_frames:
                t = ease(fi / 5)
                frame = cv2.addWeighted(montage_frames[prev_mt], 1 - t, mt_frame, t, 0)
            else:
                frame = mt_frame.copy()
        else:
            frame = mt_frame.copy()

        # Tile label
        label = f'{mt.upper()}  --  TILE {tile_num} OF 19'
        caption(frame, label)
        save_frame(frame)

# ═════════════════════════════════════════════════════════════
# PHASE 6: "Combining all 19 tiles" → crossfade to stitched mid-slice (1.5s = 36fr)
# ═════════════════════════════════════════════════════════════
print("Phase 6: combine caption + crossfade to stitched (1.5s)...")

# Get last montage frame
last_montage = montage_frames.get(MONTAGE_TILES[-1], start_frame).copy()

# Stitched mid-slice centered
stitch_centered = np.zeros((H, W, 3), np.uint8)
sx = (W - disp_stitch_w) // 2; sy = (H - disp_stitch_h) // 2
stitch_centered[sy:sy + disp_stitch_h, sx:sx + disp_stitch_w] = mid_stitch

for fi in range(36):
    t = ease(fi / 30)
    frame = cv2.addWeighted(last_montage, 1 - t, stitch_centered, t, 0)
    caption(frame, 'COMBINING  ALL  19  TILES  --  STITCHED  3D  OVERLAY')
    save_frame(frame)

# ═════════════════════════════════════════════════════════════
# PHASE 7: Stitched 3D emerge (2.5s = 60fr)
# ═════════════════════════════════════════════════════════════
print("Phase 7: stitched 3D emerge (2.5s)...")

for fi in range(60):
    t = ease(fi / 50)
    # Progressively reveal slices from center
    alphas_s = np.zeros(n_stitch_slices, dtype=np.float32)
    alphas_s[mid_stitch_idx] = 0.85
    for si in range(n_stitch_slices):
        if si == mid_stitch_idx: continue
        dist = abs(si - mid_stitch_idx) / max(1, n_stitch_slices - 1)
        if t > dist:
            alphas_s[si] = min(0.85, (t - dist) * 2.5)

    rot_x = INIT_ROT_X * t
    # Custom render with per-slice alphas
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cx, cy = W // 2, H // 2
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    z_depths = []
    for i in range(n_dense_stitch):
        dz = dense_stitch_z[i] - STITCH_CENTER_Z
        rz2 = cos_x * dz
        z_depths.append((rz2, i))
    z_depths.sort(key=lambda x: x[0])

    for depth, i in z_depths:
        ri = dense_stitch_real[i]
        if ri >= 0:
            alpha = alphas_s[ri]
        else:
            zp = dense_stitch_z[i]
            below = max(0, min(n_stitch_slices - 1, int(np.searchsorted(z_display_stitch, zp)) - 1))
            above = min(n_stitch_slices - 1, below + 1)
            if z_display_stitch[above] > z_display_stitch[below]:
                frac = (zp - z_display_stitch[below]) / (z_display_stitch[above] - z_display_stitch[below])
            else:
                frac = 0.5
            alpha = alphas_s[below] * (1 - frac) + alphas_s[above] * frac
        if alpha < 0.01: continue

        sl = dense_stitch[i].astype(np.float32) / 255.0
        sh, sw = sl.shape[:2]
        hw, hh = sw / 2, sh / 2
        dz = dense_stitch_z[i] - STITCH_CENTER_Z

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
        canvas = np.where(mask3, np.maximum(canvas, warped * alpha), canvas)

    frame_u8 = np.clip(canvas * 255, 0, 255).astype(np.uint8)
    caption(frame_u8, 'ALL  TILES  STITCHED  --  3D  ALIGNMENT')
    save_frame(frame_u8)

# ═════════════════════════════════════════════════════════════
# PHASE 8: Stitched 3D rotation (6s = 144fr)
# ═════════════════════════════════════════════════════════════
print("Phase 8: stitched 3D rotation (6s)...")

for fi in range(144):
    t = fi / 143.0
    t_eased = ease(t)
    rot_y = t_eased * math.pi * 1.25  # 225 deg, no ugly edge-on
    rot_x = INIT_ROT_X + 0.1 * math.sin(t * math.pi)
    frame = render_stitch_3d(rot_y, rot_x, 0.85)
    caption(frame, 'FULL  STITCHED  3D  ALIGNMENT  --  ALL  19  TILES')
    save_frame(frame)

# ═════════════════════════════════════════════════════════════
# PHASE 9: Settle (1.5s = 36fr)
# ═════════════════════════════════════════════════════════════
print("Phase 9: settle (1.5s)...")

final_rot_y_s = math.pi * 1.25
for fi in range(36):
    t = ease(fi / 30)
    rot_y = final_rot_y_s * (1 - t)
    rot_x = INIT_ROT_X + (0 - INIT_ROT_X) * t
    frame = render_stitch_3d(rot_y, rot_x, 0.85)
    caption(frame, 'FULL  STITCHED  3D  ALIGNMENT  --  ALL  19  TILES')
    save_frame(frame)

# ═════════════════════════════════════════════════════════════
# PHASE 10: Hold (1.5s = 36fr)
# ═════════════════════════════════════════════════════════════
print("Phase 10: hold (1.5s)...")

for fi in range(36):
    frame = render_stitch_3d(0.0, 0.0, 0.85)
    caption(frame, 'FULL  STITCHED  3D  ALIGNMENT  --  ALL  19  TILES')
    save_frame(frame)

# ── Encode frames to video ──
print(f"\nEncoding {total} frames to H.264...")
subprocess.run([
    'ffmpeg', '-y', '-framerate', str(FPS),
    '-i', f'{FRAMES_DIR}/frame_%05d.png',
    '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast',
    OUT
], capture_output=True)
print(f"Done! {total} frames, {total / FPS:.1f}s @ {FPS}fps -- {OUT}")
print(f"Frames saved in: {FRAMES_DIR}")