"""
Scene 5b multi-tile 3D: After row5_1 3D rotation, zoom out and show 5 more tiles
as 3D volumes, all rotating together. Then merge/transition to stitched 3D volume.

Replaces frames after row5_1 hold (frame 6468) through stitched crossfade (frame 6516).
"""

import numpy as np, cv2, math, os, glob, shutil
from collections import Counter

BASE = '/Users/neurolab/neuroinformatics/margaret'
FRAMES_DIR = f'{BASE}/animation/frames_combined_edited'
FRAMES_NEW = f'{BASE}/animation/frames_multi_tile_3d'

W, H = 1920, 1080
FPS = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

Z_SPACING = 4
INTERP_PER_GAP = 2
INIT_ROT_X = -0.3

# All 19 tiles
TILES = ['row1_3',
         'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
         'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
         'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
         'row5_1']

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

def ncc(a, b):
    mask = (a > 5) & (b > 5)
    if mask.sum() < 100: return -1
    af = a[mask].astype(np.float32); af -= af.mean()
    bf = b[mask].astype(np.float32); bf -= bf.mean()
    return float(np.sum(af * bf) / (np.sqrt(np.sum(af**2) * np.sum(bf**2)) + 1e-8))

def caption(frame, text, alpha=1.0):
    if alpha < 0.01: return
    ts, th = 0.72, 1
    (tw, _), _ = cv2.getTextSize(text, FONT, ts, th)
    x = (W - tw) // 2; y = H - 42
    col = tuple(int(v * alpha) for v in WHITE)
    cv2.putText(frame, text, (x, y), FONT, ts, col, th, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────
# STEP 1: Build 3D data for all tiles
# ──────────────────────────────────────────────────────────────
import tifffile

print("Loading JY306 in-vivo volume...")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_jy = jy306.shape[0]

tile_data = {}  # tile -> dict with overlay_slices, dense, dense_z, etc.

for tile in TILES:
    print(f"\nBuilding 3D for {tile}...")

    nd2_files = sorted(glob.glob(f'{BASE}/png_exports/registration_video/{tile}/GFP_z*.png'))
    if not nd2_files:
        print(f"  SKIPPED: no GFP files")
        continue
    nd2_stack = np.array([cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in nd2_files])

    pkl_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile}/pkl_transform_{tile}.npz'
    if not os.path.exists(pkl_path):
        print(f"  SKIPPED: no pkl")
        continue
    pkl = np.load(pkl_path)
    M2d = pkl['M2d_jy306_to_nd2']
    iv = pkl['pcd_invivo_jy306']
    ev = pkl['ev_nd2']
    n_lm = len(iv)

    MODE_Z = int(round(np.median(iv[:, 0])))
    MODE_Z = max(0, min(nz_jy - 1, MODE_Z))

    # Crop region
    margin_nd2 = 350
    crop_x0 = max(0, int(ev[:, 0].min() - margin_nd2))
    crop_y0 = max(0, int(ev[:, 1].min() - margin_nd2))
    crop_x1 = min(4200, int(ev[:, 0].max() + margin_nd2))
    crop_y1 = min(4200, int(ev[:, 1].max() + margin_nd2))

    # Display size for grid cell (smaller for 19 tiles)
    CELL_H = 400
    scale_nd2 = CELL_H / (crop_y1 - crop_y0)
    cell_w = int((crop_x1 - crop_x0) * scale_nd2)
    cell_h = CELL_H

    # z range
    iv_z_min = max(0, int(iv[:, 0].min()) - 1)
    iv_z_max = min(nz_jy - 1, int(iv[:, 0].max()) + 1)
    z_range = list(range(iv_z_min, iv_z_max + 1))

    overlay_slices = []
    for z_iv in z_range:
        iv_u8 = norm8(jy306[z_iv])
        warped_iv = cv2.warpAffine(iv_u8, M2d, (4200, 4200), flags=cv2.INTER_LINEAR, borderValue=0)
        warped_crop = warped_iv[crop_y0:crop_y1, crop_x0:crop_x1]

        # NCC z-match
        best_ncc, best_z = -1, 0
        for zi in range(len(nd2_stack)):
            nd2_c = nd2_stack[zi].astype(np.uint8)[crop_y0:min(crop_y1, nd2_stack.shape[1]),
                                                    crop_x0:min(crop_x1, nd2_stack.shape[2])]
            wc = warped_crop[:nd2_c.shape[0], :nd2_c.shape[1]]
            score = ncc(norm8(wc), norm8(nd2_c))
            if score > best_ncc: best_ncc, best_z = score, zi

        nd2_best = nd2_stack[best_z].astype(np.uint8)
        nd2_c = nd2_best[crop_y0:min(crop_y1, nd2_best.shape[0]),
                         crop_x0:min(crop_x1, nd2_best.shape[1])]
        wc = warped_crop[:nd2_c.shape[0], :nd2_c.shape[1]]

        ov = np.zeros((nd2_c.shape[0], nd2_c.shape[1], 3), np.uint8)
        # Ex-vivo GCaMP = magenta (B + R channels in BGR)
        ev_u8 = norm8(nd2_c)
        ov[:, :, 0] = ev_u8  # B
        ov[:, :, 2] = ev_u8  # R
        # In-vivo GCaMP = green
        iv_u8 = norm8(wc)[:nd2_c.shape[0], :nd2_c.shape[1]]
        iv_green = np.zeros_like(ov)
        iv_green[:, :, 1] = iv_u8
        ov = cv2.addWeighted(ov, 0.5, iv_green, 0.5, 0)

        ov_small = cv2.resize(ov, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        overlay_slices.append(ov_small)

    n_slices = len(overlay_slices)
    mid_idx = z_range.index(MODE_Z) if MODE_Z in z_range else n_slices // 2
    print(f"  {n_slices} slices, {cell_w}x{cell_h}")

    # Dense interpolation
    dense = []
    dense_z = []
    dense_real = []
    for i in range(n_slices):
        dense.append(overlay_slices[i])
        dense_z.append(i * Z_SPACING)
        dense_real.append(i)
        if i < n_slices - 1:
            for sub in range(1, INTERP_PER_GAP + 1):
                t_sub = sub / (INTERP_PER_GAP + 1)
                interp = (overlay_slices[i].astype(np.float32) * (1 - t_sub) +
                          overlay_slices[i + 1].astype(np.float32) * t_sub)
                dense.append(interp.astype(np.uint8))
                dense_z.append(i * Z_SPACING + t_sub * Z_SPACING)
                dense_real.append(-1)

    dense = np.array(dense)
    dense_z = np.array(dense_z, dtype=np.float64)
    center_z = (dense_z[-1] + dense_z[0]) / 2.0

    tile_data[tile] = {
        'dense': dense,
        'dense_z': dense_z,
        'dense_real': dense_real,
        'center_z': center_z,
        'n_slices': n_slices,
        'cell_w': cell_w,
        'cell_h': cell_h,
        'name': tile.upper().replace('_', ' '),
    }
    print(f"  {len(dense)} dense slices, center_z={center_z:.1f}")

del jy306

# Normalize per-tile brightness so dim tiles (e.g. row1_3) are visible
print("\nNormalizing per-tile brightness...")
for tile_name, td in tile_data.items():
    dense = td['dense']
    max_val = dense.max()
    if max_val < 120:  # dim tile — boost brightness
        boost = min(255.0 / max(max_val, 1), 3.0)
        td['dense'] = np.clip(dense.astype(np.float32) * boost, 0, 255).astype(np.uint8)
        print(f"  {tile_name}: boosted {boost:.1f}x (max was {max_val})")
    else:
        print(f"  {tile_name}: ok (max={max_val})")

print(f"\nBuilt 3D data for {len(tile_data)} tiles: {list(tile_data.keys())}")


# ──────────────────────────────────────────────────────────────
# STEP 2: Render functions
# ──────────────────────────────────────────────────────────────

def render_tile_3d_at(td, rot_y, rot_x, cx, cy, scale=1.0, alpha_val=0.7, z_offset=0.0):
    """Render a single tile's 3D stack at given position and scale.
    z_offset shifts the entire tile in 3D z-space before rotation."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    dense = td['dense']
    dense_z = td['dense_z']
    center_z = td['center_z']

    z_depths = []
    for i in range(len(dense)):
        dz = dense_z[i] - center_z + z_offset
        rz = cos_y * dz
        rz2 = cos_x * rz
        z_depths.append((rz2, i))
    z_depths.sort(key=lambda x: x[0])

    for depth, i in z_depths:
        sl = dense[i].astype(np.float32) / 255.0
        sh, sw = sl.shape[:2]
        hw, hh = sw * scale / 2, sh * scale / 2
        dz = dense_z[i] - center_z + z_offset

        corners_3d = np.array([
            [-hw, -hh, dz * scale], [hw, -hh, dz * scale],
            [hw, hh, dz * scale], [-hw, hh, dz * scale]
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

    return canvas


def render_multi_tile(tile_positions, rot_y, rot_x, tile_scales=None, tile_z_offsets=None):
    """Render multiple tiles at given positions, all with same rotation.
    tile_z_offsets: dict of tile_name -> z_offset in 3D space."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)

    for tile_name, (cx, cy) in tile_positions.items():
        if tile_name not in tile_data:
            continue
        # Skip tiles far off screen
        if cx < -500 or cx > W + 500 or cy < -500 or cy > H + 500:
            continue
        td = tile_data[tile_name]
        scale = tile_scales.get(tile_name, 1.0) if tile_scales else 1.0
        z_off = tile_z_offsets.get(tile_name, 0.0) if tile_z_offsets else 0.0
        tile_canvas = render_tile_3d_at(td, rot_y, rot_x, cx, cy, scale=scale, z_offset=z_off)
        # Composite: maximum blend
        mask = np.max(tile_canvas, axis=2) > 0.01
        mask3 = np.stack([mask] * 3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, tile_canvas), canvas)

    return np.clip(canvas * 255, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────
# STEP 3: Define grid layout
# ──────────────────────────────────────────────────────────────

# 4 rows x 5 columns grid for 19 tiles (one empty spot)
GRID_COLS = 5
GRID_ROWS = 4
CELL_GAP_X = 20
CELL_GAP_Y = 15

# Average cell size
avg_cell_w = sum(td['cell_w'] for td in tile_data.values()) // len(tile_data)
avg_cell_h = sum(td['cell_h'] for td in tile_data.values()) // len(tile_data)

grid_w = GRID_COLS * avg_cell_w + (GRID_COLS - 1) * CELL_GAP_X
grid_h = GRID_ROWS * avg_cell_h + (GRID_ROWS - 1) * CELL_GAP_Y
grid_x0 = (W - grid_w) // 2
grid_y0 = (H - grid_h) // 2

# Grid cell centers
grid_positions = {}
tile_list = list(tile_data.keys())
for idx, tile_name in enumerate(tile_list):
    row = idx // GRID_COLS
    col = idx % GRID_COLS
    cx = grid_x0 + col * (avg_cell_w + CELL_GAP_X) + avg_cell_w // 2
    cy = grid_y0 + row * (avg_cell_h + CELL_GAP_Y) + avg_cell_h // 2
    grid_positions[tile_name] = (cx, cy)

# Row5_1 starts at screen center (full size), then moves to grid position
row5_1_center = (W // 2, H // 2)
row5_1_grid = grid_positions.get('row5_1', row5_1_center)

# Scale: row5_1 at full size fills ~760px, at grid fills ~avg_cell_w
# Current row5_1 display: cell_w x cell_h (built at CELL_H=400)
# But the original row5_1 in scene5b was at DISP_H=0.72*1080=778px tall
# Scale factor to go from full to grid
FULL_SCALE = 778 / tile_data['row5_1']['cell_h']  # scales row5_1 to full screen size
GRID_SCALE = 1.0

print(f"\nGrid layout: {grid_w}x{grid_h}")
print(f"Cell avg: {avg_cell_w}x{avg_cell_h}")
print(f"Full scale: {FULL_SCALE:.2f}, grid scale: {GRID_SCALE:.2f}")
for t, (cx, cy) in grid_positions.items():
    print(f"  {t}: ({cx}, {cy})")


# ──────────────────────────────────────────────────────────────
# STEP 4: Generate frames
# ──────────────────────────────────────────────────────────────
import shutil
if os.path.exists(FRAMES_NEW):
    shutil.rmtree(FRAMES_NEW)
os.makedirs(FRAMES_NEW)

frame_count = 0

def save_frame(frame):
    global frame_count
    frame_count += 1
    cv2.imwrite(f'{FRAMES_NEW}/frame_{frame_count:05d}.png', frame)

# Phase A: Row5_1 zooms out from center to grid position (36 frames = 1.5s)
print("\nPhase A: row5_1 zoom out (1.5s)...")
for fi in range(36):
    t = ease(fi / 30)

    # Interpolate position and scale
    cx = int(row5_1_center[0] * (1 - t) + row5_1_grid[0] * t)
    cy = int(row5_1_center[1] * (1 - t) + row5_1_grid[1] * t)
    scale = FULL_SCALE * (1 - t) + GRID_SCALE * t

    # Slow rotation continues
    rot_y = t * 0.15  # slight rotation
    rot_x = INIT_ROT_X

    tile_canvas = render_tile_3d_at(tile_data['row5_1'], rot_y, rot_x, cx, cy, scale=scale)
    frame = np.clip(tile_canvas * 255, 0, 255).astype(np.uint8)

    # Other tiles fade in during second half
    if t > 0.4:
        fade = ease((t - 0.4) / 0.6)
        for tile_name in tile_list[1:]:  # skip row5_1
            gx, gy = grid_positions[tile_name]
            tc = render_tile_3d_at(tile_data[tile_name], rot_y, rot_x, gx, gy, scale=GRID_SCALE)
            tc_u8 = np.clip(tc * 255 * fade, 0, 255).astype(np.uint8)
            mask = np.max(tc_u8, axis=2) > 5
            mask3 = np.stack([mask] * 3, axis=-1)
            frame = np.where(mask3, np.maximum(frame, tc_u8), frame)

    # Labels
    if t > 0.6:
        label_alpha = ease((t - 0.6) / 0.4)
        for tile_name, (gx, gy) in grid_positions.items():
            if tile_name == 'row5_1' or tile_name not in tile_data:
                continue
            td = tile_data[tile_name]
            lbl = td['name']
            (tw, _), _ = cv2.getTextSize(lbl, FONT, 0.32, 1)
            lx = gx - tw // 2
            ly = gy - td['cell_h'] // 2 - 10
            col = tuple(int(v * label_alpha) for v in WHITE)
            cv2.putText(frame, lbl, (lx, ly), FONT, 0.32, col, 1, cv2.LINE_AA)

    # Row5_1 label
    td5 = tile_data['row5_1']
    lbl5 = td5['name']
    (tw5, _), _ = cv2.getTextSize(lbl5, FONT, 0.32, 1)
    label_y = cy - int(td5['cell_h'] * scale / 2) - 10
    cv2.putText(frame, lbl5, (cx - tw5 // 2, label_y), FONT, 0.32, WHITE, 1, cv2.LINE_AA)

    caption(frame, f'PER-TILE  3D  ALIGNMENT  --  {len(tile_data)}  OF  19  TILES')
    save_frame(frame)
    if fi % 12 == 0:
        print(f"  {fi+1}/36")


# Phase B: Hold all tiles at fixed angle (24 frames = 1s)
STATIC_ROT_Y = 0.25   # fixed slight angle to show thickness
STATIC_ROT_X = INIT_ROT_X
print("Phase B: all tiles at fixed angle (1s)...")
for fi in range(24):
    t = fi / 23.0
    rot_y = 0.15 + t * (STATIC_ROT_Y - 0.15)  # ease to static angle
    rot_x = STATIC_ROT_X

    positions = {tile_name: grid_positions[tile_name] for tile_name in tile_list}
    scales = {tile_name: GRID_SCALE for tile_name in tile_list}

    frame = render_multi_tile(positions, rot_y, rot_x, scales)

    # Tile labels
    for tile_name, (gx, gy) in grid_positions.items():
        if tile_name not in tile_data: continue
        td = tile_data[tile_name]
        lbl = td['name']
        (tw, _), _ = cv2.getTextSize(lbl, FONT, 0.32, 1)
        lx = gx - tw // 2
        ly = gy - td['cell_h'] // 2 - 10
        cv2.putText(frame, lbl, (lx, ly), FONT, 0.32, WHITE, 1, cv2.LINE_AA)

    caption(frame, f'PER-TILE  3D  ALIGNMENT  --  {len(tile_data)}  OF  19  TILES')
    save_frame(frame)
    if fi % 8 == 0:
        print(f"  {fi+1}/24")

# Phase B2: Zoom into top-left, hold+rotate, pan right across each row, repeat, zoom out
print("Phase B2: row-by-row fly-by...")

# Camera layout
row_ys = sorted(set(cy for _, cy in grid_positions.values()))
n_rows_grid = len(row_ys)

# Per-row x extents (leftmost and rightmost tile in each row)
row_x_ranges = {}
for tile_name, (gx, gy) in grid_positions.items():
    for ri, ry in enumerate(row_ys):
        if abs(gy - ry) < 5:
            if ri not in row_x_ranges:
                row_x_ranges[ri] = [gx, gx]
            else:
                row_x_ranges[ri][0] = min(row_x_ranges[ri][0], gx)
                row_x_ranges[ri][1] = max(row_x_ranges[ri][1], gx)

all_xs = [cx for cx, _ in grid_positions.values()]
x_min_grid, x_max_grid = min(all_xs), max(all_xs)
x_center_grid = (x_min_grid + x_max_grid) / 2
y_center_grid = (row_ys[0] + row_ys[-1]) / 2

ZOOM_LEVEL = 2.5

# Timing
ZOOM_IN_FRAMES = 18    # zoom from full grid to top-left
HOLD_FRAMES = 48       # hold steady at left of row while tiles rotate (2s)
PAN_FRAMES = 24        # pan left to right across the row (1s)
TRANS_FRAMES = 10      # transition down to next row's left side
ZOOM_OUT_FRAMES = 18   # zoom back out to full grid
PER_ROW = HOLD_FRAMES + PAN_FRAMES + TRANS_FRAMES  # 64
FLYBY_FRAMES = ZOOM_IN_FRAMES + n_rows_grid * PER_ROW - TRANS_FRAMES + ZOOM_OUT_FRAMES
# (last row doesn't need TRANS_FRAMES)
print(f"  {FLYBY_FRAMES} frames total ({n_rows_grid} rows, {FLYBY_FRAMES/FPS:.1f}s)")

for fi in range(FLYBY_FRAMES):
    # Determine camera position and zoom
    if fi < ZOOM_IN_FRAMES:
        # Zoom in: from full grid center to top-left of row 0
        zt = ease(fi / max(1, ZOOM_IN_FRAMES - 1))
        zoom = 1.0 + (ZOOM_LEVEL - 1.0) * zt
        row0_xl = row_x_ranges[0][0]
        cam_x = x_center_grid * (1 - zt) + row0_xl * zt
        cam_y = y_center_grid * (1 - zt) + row_ys[0] * zt
    elif fi >= FLYBY_FRAMES - ZOOM_OUT_FRAMES:
        # Zoom out: from last row right back to full grid
        zoom_fi = fi - (FLYBY_FRAMES - ZOOM_OUT_FRAMES)
        zt = ease(zoom_fi / max(1, ZOOM_OUT_FRAMES - 1))
        zoom = ZOOM_LEVEL * (1 - zt) + 1.0 * zt
        last_xr = row_x_ranges[n_rows_grid - 1][1]
        cam_x = last_xr * (1 - zt) + x_center_grid * zt
        cam_y = row_ys[-1] * (1 - zt) + y_center_grid * zt
    else:
        zoom = ZOOM_LEVEL
        # Which row and sub-phase?
        row_fi = fi - ZOOM_IN_FRAMES
        row_idx = min(row_fi // PER_ROW, n_rows_grid - 1)
        sub_fi = row_fi - row_idx * PER_ROW

        xl = row_x_ranges[row_idx][0]  # leftmost tile x
        xr = row_x_ranges[row_idx][1]  # rightmost tile x

        if sub_fi < HOLD_FRAMES:
            # Hold at left side of row — steady, tiles rotate
            cam_x = xl
            cam_y = row_ys[row_idx]
        elif sub_fi < HOLD_FRAMES + PAN_FRAMES:
            # Pan left to right across the row
            pan_t = ease((sub_fi - HOLD_FRAMES) / max(1, PAN_FRAMES - 1))
            cam_x = xl + (xr - xl) * pan_t
            cam_y = row_ys[row_idx]
        else:
            # Transition down to next row's left side
            trans_t = ease((sub_fi - HOLD_FRAMES - PAN_FRAMES) / max(1, TRANS_FRAMES - 1))
            next_row = min(row_idx + 1, n_rows_grid - 1)
            next_xl = row_x_ranges[next_row][0]
            cam_x = xr * (1 - trans_t) + next_xl * trans_t
            cam_y = row_ys[row_idx] * (1 - trans_t) + row_ys[next_row] * trans_t

    # Render: transform tile positions relative to camera
    positions = {}
    scales = {}
    for tile_name in tile_list:
        gx, gy = grid_positions[tile_name]
        px = W / 2 + (gx - cam_x) * zoom
        py = H / 2 + (gy - cam_y) * zoom
        positions[tile_name] = (int(px), int(py))
        scales[tile_name] = GRID_SCALE * zoom

    # Fixed angle — no spinning, just static tilt to show thickness
    rot_y = STATIC_ROT_Y
    rot_x = STATIC_ROT_X

    frame = render_multi_tile(positions, rot_y, rot_x, scales)

    # Labels (scaled with zoom)
    font_scale = 0.32 * min(zoom, 2.0)
    for tile_name, (px, py) in positions.items():
        if tile_name not in tile_data: continue
        if px < -200 or px > W + 200 or py < -200 or py > H + 200:
            continue
        td = tile_data[tile_name]
        lbl = td['name']
        (tw, _), _ = cv2.getTextSize(lbl, FONT, font_scale, 1)
        lx = px - tw // 2
        ly = py - int(td['cell_h'] * zoom / 2) - 8
        cv2.putText(frame, lbl, (lx, ly), FONT, font_scale, WHITE, 1, cv2.LINE_AA)

    caption(frame, f'PER-TILE  3D  ALIGNMENT  --  {len(tile_data)}  OF  19  TILES')
    save_frame(frame)
    if fi % 20 == 0:
        print(f"  {fi+1}/{FLYBY_FRAMES}")

# Phase B3: Brief hold at full grid, static angle (12 frames = 0.5s)
print("Phase B3: hold at grid (0.5s)...")
for fi in range(12):
    positions = {tile_name: grid_positions[tile_name] for tile_name in tile_list}
    scales = {tile_name: GRID_SCALE for tile_name in tile_list}

    rot_y = STATIC_ROT_Y
    rot_x = STATIC_ROT_X

    frame = render_multi_tile(positions, rot_y, rot_x, scales)

    for tile_name, (gx, gy) in grid_positions.items():
        if tile_name not in tile_data: continue
        td = tile_data[tile_name]
        lbl = td['name']
        (tw, _), _ = cv2.getTextSize(lbl, FONT, 0.32, 1)
        lx = gx - tw // 2
        ly = gy - td['cell_h'] // 2 - 10
        cv2.putText(frame, lbl, (lx, ly), FONT, 0.32, WHITE, 1, cv2.LINE_AA)

    caption(frame, f'PER-TILE  3D  ALIGNMENT  --  {len(tile_data)}  OF  19  TILES')
    save_frame(frame)


# ──────────────────────────────────────────────────────────────
# Phase C: Real spatial merge — tiles move from flat mosaic to 3D stack
# ──────────────────────────────────────────────────────────────

# Compute each tile's target z-offset based on median in-vivo z
tile_median_z = {}
for tile_name in tile_list:
    pkl_path = f'{BASE}/png_exports/registration_per_tile_pkl/{tile_name}/pkl_transform_{tile_name}.npz'
    if os.path.exists(pkl_path):
        pkl = np.load(pkl_path)
        iv = pkl['pcd_invivo_jy306']
        tile_median_z[tile_name] = float(np.median(iv[:, 0]))
    else:
        tile_median_z[tile_name] = 8.0

z_vals = list(tile_median_z.values())
z_min, z_max = min(z_vals), max(z_vals)

# Z-offset in 3D render pixels: spread tiles across z based on median z
# Each unit of in-vivo z maps to Z_PX_PER_UNIT pixels of 3D depth
Z_PX_PER_UNIT = 15  # pixels per in-vivo z-unit in 3D space
tile_final_z = {}
for tile_name in tile_list:
    tile_final_z[tile_name] = (tile_median_z[tile_name] - (z_min + z_max) / 2) * Z_PX_PER_UNIT

print(f"\nPhase C: spatial merge — tiles stack in 3D (3s)...")
print(f"  Z range: {z_min:.0f}-{z_max:.0f}, spread: {(z_max-z_min)*Z_PX_PER_UNIT:.0f}px")

MERGE_FRAMES = 72
for fi in range(MERGE_FRAMES):
    t = ease(fi / max(1, MERGE_FRAMES - 6))

    # All tiles move toward center x,y and separate in z
    positions = {}
    scales = {}
    z_offsets = {}
    for tile_name in tile_list:
        gx, gy = grid_positions[tile_name]
        # x,y: from grid position to screen center
        cx = int(gx * (1 - t) + W // 2 * t)
        cy = int(gy * (1 - t) + H // 2 * t)
        positions[tile_name] = (cx, cy)
        scales[tile_name] = GRID_SCALE
        # z: from 0 (flat) to final z-offset (stacked)
        z_offsets[tile_name] = tile_final_z[tile_name] * t

    rot_y = STATIC_ROT_Y
    rot_x = STATIC_ROT_X

    frame = render_multi_tile(positions, rot_y, rot_x, scales, tile_z_offsets=z_offsets)

    # Fade labels
    label_alpha = max(0, 1 - t * 3)
    if label_alpha > 0.05:
        for tile_name, (px, py) in positions.items():
            if tile_name not in tile_data: continue
            td = tile_data[tile_name]
            lbl = td['name']
            (tw, _), _ = cv2.getTextSize(lbl, FONT, 0.32, 1)
            lx = px - tw // 2
            ly = py - td['cell_h'] // 2 - 10
            col = tuple(int(v * label_alpha) for v in WHITE)
            cv2.putText(frame, lbl, (lx, ly), FONT, 0.32, col, 1, cv2.LINE_AA)

    caption(frame, 'COMBINING  ALL  19  TILES  INTO  STITCHED  3D')
    save_frame(frame)
    if fi % 12 == 0:
        print(f"  {fi+1}/{MERGE_FRAMES}")

# Phase D: Hold stacked result (36 frames = 1.5s)
print("Phase D: hold stacked 3D (1.5s)...")
for fi in range(36):
    positions = {tile_name: (W // 2, H // 2) for tile_name in tile_list}
    scales = {tile_name: GRID_SCALE for tile_name in tile_list}
    z_offsets = {tile_name: tile_final_z[tile_name] for tile_name in tile_list}

    # Gentle rotation to show the stacked volume
    rot_y = STATIC_ROT_Y + fi * 0.008
    rot_x = STATIC_ROT_X

    frame = render_multi_tile(positions, rot_y, rot_x, scales, tile_z_offsets=z_offsets)
    caption(frame, 'FULL  STITCHED  3D  ALIGNMENT  --  ALL  19  TILES')
    save_frame(frame)
    if fi % 12 == 0:
        print(f"  {fi+1}/36")


print(f"\nDone! {frame_count} frames saved to {FRAMES_NEW}/ ({frame_count/FPS:.1f}s)")