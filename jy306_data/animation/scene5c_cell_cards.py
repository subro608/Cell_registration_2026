"""
Scene 5c: Cell identity cards — continues from scene5b.

For 6 selected landmarks from row2_1:
  - Top: stitched 3D volume (same as scene5b) with cyan marker at landmark
  - Bottom 4 panels: calcium activity | in-vivo crop | registration overlay | ex-vivo crop

Cross-fades between cells. Outputs frames to frames_scene5c/.

Output: animation/scene5c_h264.mp4
"""

import numpy as np, cv2, math, subprocess, os, glob, sys, tifffile, json

BASE = '/Users/neurolab/neuroinformatics/margaret'
PATCH_DIR = '/Users/neurolab/neuroinformatics/invivo-exvivo-cell-registration/patches'
FRAME_DIR = f'{BASE}/animation/frames_scene5c'
OUT = f'{BASE}/animation/scene5c_h264.mp4'

W, H = 1920, 1080
FPS = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)  # BGR

SELECTED = [1, 3, 6, 11, 13, 15]

INIT_ROT_X = -0.3
INTERP_PER_GAP = 3
ND2_Z_STEP_UM = 2.0
CANVAS_UM_PER_PX = 0.645

N_PER_CELL = 120  # 5s per cell
N_CROSSFADE = 12  # 0.5s crossfade between cells
PANEL_SZ = 220

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
    x = (W - tw) // 2; y = H - 20
    col = tuple(int(v * alpha) for v in WHITE)
    cv2.putText(frame, text, (x, y), FONT, ts, col, th, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════
# LOAD 3D VOLUME (same as scene5b)
# ══════════════════════════════════════════════════════════════
print("Loading scene5b assets...")
assets = np.load(f'{BASE}/animation/scene5b_assets_v2.npz')
ov_slices_raw = assets['overlay_slices']
z_indices = assets['z_indices']
n_slices = len(ov_slices_raw)
raw_h, raw_w = ov_slices_raw.shape[1:3]

# Crop to content
mask = np.any(ov_slices_raw > 3, axis=-1)
any_slice = np.any(mask, axis=0)
rows = np.where(any_slice.any(axis=1))[0]
cols = np.where(any_slice.any(axis=0))[0]
PAD = 50
r0 = max(0, rows[0] - PAD); r1 = min(raw_h, rows[-1] + PAD)
c0 = max(0, cols[0] - PAD); c1 = min(raw_w, cols[-1] + PAD)
ov_cropped = ov_slices_raw[:, r0:r1, c0:c1, :]
crop_h, crop_w = ov_cropped.shape[1:3]
print(f"  Content: {crop_w}x{crop_h}, {n_slices} slices")

# Scale — 3D volume takes top 60% of screen
VOL_H = int(H * 0.55)
scale_up = VOL_H / crop_h
disp_w = int(crop_w * scale_up)
disp_h = VOL_H
if disp_w > int(W * 0.95):
    disp_w = int(W * 0.95)
    scale_up = disp_w / crop_w
    disp_h = int(crop_h * scale_up)
print(f"  Volume display: {disp_w}x{disp_h}")

sel_slices = np.array([cv2.resize(ov_cropped[i], (disp_w, disp_h),
                                   interpolation=cv2.INTER_LANCZOS4)
                        for i in range(n_slices)])
del ov_slices_raw, ov_cropped

# Physical z-positions
z_um = z_indices.astype(np.float64) * ND2_Z_STEP_UM
crop_xy_um = crop_w * CANVAS_UM_PER_PX
z_display = (z_um - z_um[0]) / crop_xy_um * disp_w

# Gaussian interpolation
print("Gaussian interpolation...")
dense_slices = []
dense_z_pos = []
for i in range(n_slices):
    dense_slices.append(sel_slices[i])
    dense_z_pos.append(z_display[i])
    if i < n_slices - 1:
        z0, z1 = z_display[i], z_display[i + 1]
        for sub in range(1, INTERP_PER_GAP + 1):
            t_sub = sub / (INTERP_PER_GAP + 1)
            sigma = 0.4
            w1 = math.exp(-0.5 * ((t_sub - 1) / sigma) ** 2)
            w0 = math.exp(-0.5 * (t_sub / sigma) ** 2)
            w_total = w0 + w1; w0 /= w_total; w1 /= w_total
            interp = (sel_slices[i].astype(np.float32) * w0 +
                      sel_slices[i + 1].astype(np.float32) * w1)
            dense_slices.append(interp.astype(np.uint8))
            dense_z_pos.append(z0 + t_sub * (z1 - z0))

dense_slices = np.array(dense_slices)
dense_z_pos = np.array(dense_z_pos, dtype=np.float64)
n_dense = len(dense_slices)
CENTER_Z = (dense_z_pos[-1] + dense_z_pos[0]) / 2.0
print(f"  {n_dense} sub-slices")

# Volume center on canvas (shifted up to make room for panels)
VOL_CY = disp_h // 2 + 10  # near top
VOL_CX = W // 2


def render_3d(rot_y, rot_x, marker_xyz=None, alpha_val=0.85):
    """Render 3D volume with optional cyan landmark marker."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cx, cy = VOL_CX, VOL_CY
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    z_depths = []
    for i in range(n_dense):
        dz = dense_z_pos[i] - CENTER_Z
        rz2 = cos_x * (cos_y * dz)
        z_depths.append((rz2, i))
    z_depths.sort(key=lambda x: x[0])

    # Track marker depth for proper ordering
    marker_depth = None
    if marker_xyz is not None:
        mx, my, mz = marker_xyz
        dz_m = mz - CENTER_Z
        rz_m = cos_y * dz_m
        marker_depth = cos_x * rz_m

    for depth, i in z_depths:
        sl = dense_slices[i].astype(np.float32) / 255.0
        sh, sw = sl.shape[:2]
        hw, hh = sw / 2, sh / 2
        dz = dense_z_pos[i] - CENTER_Z

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
        mask3 = np.stack([mask] * 3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, warped * alpha_val), canvas)

    result = np.clip(canvas * 255, 0, 255).astype(np.uint8)

    # Draw marker
    if marker_xyz is not None:
        mx, my, mz = marker_xyz
        dz = mz - CENTER_Z
        rx = cos_y * mx + sin_y * dz
        ry = my
        rz = -sin_y * mx + cos_y * dz
        ry2 = cos_x * ry - sin_x * rz
        sx, sy = int(rx + cx), int(ry2 + cy)
        if 10 < sx < W - 10 and 10 < sy < H - 10:
            # Cyan glow marker
            cv2.circle(result, (sx, sy), 20, (180, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(result, (sx, sy), 14, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(result, (sx, sy), 7, (255, 255, 0), -1, cv2.LINE_AA)

    return result


# ══════════════════════════════════════════════════════════════
# LOAD LANDMARK DATA + COMPUTE DISPLAY POSITIONS
# ══════════════════════════════════════════════════════════════
print("Loading landmark data...")
pkl = np.load(f'{BASE}/png_exports/registration_per_tile_pkl/row2_1/pkl_transform_row2_1.npz')
iv = pkl['pcd_invivo_jy306']   # (z, y, x)
ev = pkl['ev_nd2']             # (x, y, z)
M2d = pkl['M2d_jy306_to_nd2']

# Load stitch transform for row2_1
with open(f'{BASE}/registration_video/stitch_v5_params.json') as f:
    stitch_params = json.load(f)
M_cum = np.array(stitch_params['cumulative_iou']['row2_1'])  # 3x3

print("Computing landmark display positions in 3D volume...")
lm_display = {}
for idx in SELECTED:
    # ev_nd2 coords (x,y) in tile 4200x4200 space
    x_tile = ev[idx, 0]
    y_tile = ev[idx, 1]
    z_nd2 = ev[idx, 2]

    # Map tile coords to canvas coords using cumulative IOU transform
    pt = np.array([x_tile, y_tile, 1.0])
    canvas_pt = M_cum @ pt
    cx_canvas = canvas_pt[0]
    cy_canvas = canvas_pt[1]

    # Map canvas coords to display coords (subtract crop, scale)
    dx = (cx_canvas - c0) * scale_up - disp_w / 2
    dy = (cy_canvas - r0) * scale_up - disp_h / 2

    # Z: find closest z_display position
    closest_z_idx = int(np.argmin(np.abs(z_indices - z_nd2)))
    dz = z_display[closest_z_idx]

    lm_display[idx] = (dx, dy, dz)
    print(f"  LM#{idx}: canvas=({cx_canvas:.0f},{cy_canvas:.0f}) -> display=({dx:.0f},{dy:.0f}), z_idx={closest_z_idx}")


# ══════════════════════════════════════════════════════════════
# GENERATE 4 PANELS PER LANDMARK
# ══════════════════════════════════════════════════════════════
print("\nLoading JY306 + nd2 for panel crops...")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz, hy, wx = jy306.shape

nd2_files = sorted(glob.glob(f'{BASE}/png_exports/registration_video/row2_1/GFP_z*.png'))
nd2_stack = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in nd2_files])

CROP_JY = 60
CROP_ND2 = 160

panels = {}
for idx in SELECTED:
    z_lm = int(round(iv[idx, 0]))
    y_lm = int(round(iv[idx, 1]))
    x_lm = int(round(iv[idx, 2]))

    # In-vivo crop (exact z, not MIP)
    jy_slice = norm8(jy306[min(nz-1, max(0, z_lm))])
    y0 = max(0, y_lm - CROP_JY); y1 = min(hy, y_lm + CROP_JY)
    x0 = max(0, x_lm - CROP_JY); x1 = min(wx, x_lm + CROP_JY)
    crop_jy = jy_slice[y0:y1, x0:x1]
    crop_jy_r = cv2.resize(crop_jy, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
    iv_hot = make_hot(crop_jy_r)
    # Mark center
    cv2.circle(iv_hot, (PANEL_SZ//2, PANEL_SZ//2), 8, CYAN, 1, cv2.LINE_AA)

    # Ex-vivo crop
    x_nd2 = int(round(ev[idx, 0])); y_nd2 = int(round(ev[idx, 1]))
    best_z = min(len(nd2_stack)-1, max(0, int(round(ev[idx, 2]))))
    nd2_slice = norm8(nd2_stack[best_z])
    yn0 = max(0, y_nd2 - CROP_ND2); yn1 = min(4200, y_nd2 + CROP_ND2)
    xn0 = max(0, x_nd2 - CROP_ND2); xn1 = min(4200, x_nd2 + CROP_ND2)
    crop_nd2 = nd2_slice[yn0:yn1, xn0:xn1]
    ev_panel = np.zeros((PANEL_SZ, PANEL_SZ, 3), np.uint8)
    crop_nd2_r = cv2.resize(crop_nd2, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
    ev_panel[:, :, 1] = crop_nd2_r
    cv2.circle(ev_panel, (PANEL_SZ//2, PANEL_SZ//2), 8, CYAN, 1, cv2.LINE_AA)

    # Overlay (warp invivo into nd2 space)
    jy_full_u8 = norm8(jy306[min(nz-1, max(0, z_lm))])
    warped_jy = cv2.warpAffine(jy_full_u8, M2d, (4200, 4200), borderValue=0)
    crop_warped = warped_jy[yn0:yn1, xn0:xn1]
    ov = np.zeros((crop_nd2.shape[0], crop_nd2.shape[1], 3), np.uint8)
    ov[:, :, 1] = crop_nd2  # green = exvivo
    ov[:, :, 2] = norm8(crop_warped)  # red = invivo warped
    overlay = cv2.resize(ov, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)
    cv2.circle(overlay, (PANEL_SZ//2, PANEL_SZ//2), 8, CYAN, 1, cv2.LINE_AA)

    panels[idx] = (iv_hot, overlay, ev_panel)
    print(f"  LM#{idx}: z={z_lm}, nd2_z={best_z}")

del jy306, nd2_stack  # free memory

# Load calcium patches
print("Loading calcium patches...")
cal_patches = {}
for idx in SELECTED:
    vpath = f'{PATCH_DIR}/patch_{idx}.mp4'
    if not os.path.exists(vpath):
        print(f"  WARNING: {vpath} not found")
        continue
    cap = cv2.VideoCapture(vpath)
    frames = []
    while True:
        ret, frm = cap.read()
        if not ret: break
        frames.append(frm)
    cap.release()
    cal_patches[idx] = frames
    print(f"  LM#{idx}: {len(frames)} calcium frames")


# ══════════════════════════════════════════════════════════════
# RENDER FRAMES
# ══════════════════════════════════════════════════════════════
print("\nRendering cell identity cards...")
os.makedirs(FRAME_DIR, exist_ok=True)

# Panel layout
GAP_P = 24
total_pw = PANEL_SZ * 4 + GAP_P * 3
x_panel_start = (W - total_pw) // 2
y_panels = H - PANEL_SZ - 50

frame_idx = 0

def render_cell_frame(idx, fi, cal_offset=0):
    """Render one frame for a given cell landmark."""
    if idx not in panels or idx not in cal_patches:
        return np.zeros((H, W, 3), np.uint8)

    iv_panel, overlay, ev_panel = panels[idx]
    cal_frames = cal_patches[idx]
    mx, my, mz = lm_display[idx]

    # Gentle rotation (slow oscillation)
    rot_y = 0.15 * math.sin(fi / N_PER_CELL * math.pi * 2)
    rot_x = INIT_ROT_X

    frame = render_3d(rot_y, rot_x, marker_xyz=(mx, my, mz))

    # Calcium panel (advance through calcium video)
    cal_idx = (cal_offset + fi * 3) % len(cal_frames)
    cal_raw = cal_frames[cal_idx]
    cal_disp = cv2.resize(cal_raw, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_LANCZOS4)

    # Draw panels
    panel_list = [cal_disp, iv_panel, overlay, ev_panel]
    labels = ['CALCIUM ACTIVITY', 'IN-VIVO (HOT)', 'REGISTERED', 'EX-VIVO (GFP)']

    for pi, (panel, label) in enumerate(zip(panel_list, labels)):
        px = x_panel_start + pi * (PANEL_SZ + GAP_P)
        fy = y_panels
        frame[fy:fy + PANEL_SZ, px:px + PANEL_SZ] = panel
        # Border
        cv2.rectangle(frame, (px, fy), (px + PANEL_SZ - 1, fy + PANEL_SZ - 1),
                      (80, 80, 80), 1)
        # Label
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.35, 1)
        cv2.putText(frame, label, (px + (PANEL_SZ - tw) // 2, fy - 6),
                    FONT, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    # Title
    z_lm = int(round(iv[idx, 0]))
    title = f'CELL #{idx}  --  ROW2_1  z = {z_lm}'
    (tw, _), _ = cv2.getTextSize(title, FONT, 0.55, 1)
    cv2.putText(frame, title, ((W - tw) // 2, y_panels - 28),
                FONT, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    caption(frame, 'MATCHED  CELL  IDENTITY  CARDS')
    return frame


# Render all cells with crossfades
valid_cells = [idx for idx in SELECTED if idx in panels and idx in cal_patches]
print(f"  {len(valid_cells)} valid cells: {valid_cells}")

for ci, idx in enumerate(valid_cells):
    print(f"  Cell #{idx} ({ci+1}/{len(valid_cells)})")

    for fi in range(N_PER_CELL):
        frame = render_cell_frame(idx, fi)

        # Crossfade at boundaries
        if ci > 0 and fi < N_CROSSFADE:
            # Fade in from previous cell
            prev_idx = valid_cells[ci - 1]
            prev_fi = N_PER_CELL - N_CROSSFADE + fi
            prev_frame = render_cell_frame(prev_idx, prev_fi)
            t = ease(fi / (N_CROSSFADE - 1))
            frame = cv2.addWeighted(prev_frame, 1 - t, frame, t, 0)

        # Skip last N_CROSSFADE frames (they'll be covered by next cell's fade-in)
        if ci < len(valid_cells) - 1 and fi >= N_PER_CELL - N_CROSSFADE:
            continue

        frame_idx += 1
        cv2.imwrite(f'{FRAME_DIR}/frame_{frame_idx:05d}.png', frame)

    print(f"    -> {frame_idx} frames so far")

print(f"\nDone! {frame_idx} frames -> {FRAME_DIR}/")
print(f"Duration: {frame_idx/FPS:.1f}s @ {FPS}fps")
print(f"\nTo encode: ffmpeg -y -framerate {FPS} -i {FRAME_DIR}/frame_%05d.png -vcodec libx264 -pix_fmt yuv420p -crf 18 {OUT}")