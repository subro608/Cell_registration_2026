"""
Three stacks v2: Full flow from 3 split grids → real tile merge into combined
3D volume → separate into 3 channel-specific 3D volumes.

All rendering uses per-tile data — no stitched volume intermediate.

Phases:
  1. Split (36fr): tiles slide from merged center to 3 columns (IV|EV|MS)
  2. Rotate split (36fr): rotate 3 grids to show depth
  3. Merge grids (36fr): 3 columns slide back to merged center grid
  4. Tile merge (72fr): tiles slide from grid to center with z-depth (real alignment)
  5. Scale-up (36fr): tiles grow to volume scale + z correction
  6. Rotation blend (72fr): 360° rotation, blend per-tile combined → per-tile channels
  7. Split channels (48fr): per-tile channels → 3 side-by-side volumes
  8. Hold split (48fr): rotate 3 channel volumes
  9. Merge back (36fr): 3 channel volumes → single combined

Total: 420 frames = 17.5s @ 24fps

Outputs to frames_three_stacks_v2/
"""
import numpy as np, cv2, math, os, pickle, shutil, time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/animation/frames_three_stacks_v2'
W, H = 1920, 1080
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

TILES = ['row1_3',
         'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
         'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
         'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
         'row5_1']

STATIC_ROT_Y = 0.25
STATIC_ROT_X = -0.3
SPIN_AMP = 0.12
SPIN_PERIOD = 120
VOLUME_SCALE = 1.8  # match scene5b

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))


def draw_axes(frame, rot_y, rot_x, cx=120, cy=None, ax_len=70, alpha=1.0):
    """Draw 3D axis trident that rotates with the volume."""
    if alpha < 0.01:
        return
    if cy is None:
        cy = H - 120
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)
    axes_3d = [
        (1, 0, 0),     # ML — horizontal in slice
        (0, -1, 0),    # AP — vertical in slice (up)
        (0, 0, 1),     # DV — depth (z-stack, down/ventral)
    ]
    ax_colors = [(0, 0, 180), (40, 40, 40), (200, 80, 0)]
    ax_labels = ['ML', 'AP', 'DV']
    for ai, (ux, uy, uz) in enumerate(axes_3d):
        rx = cos_y * ux + sin_y * uz
        ry = uy
        rz = -sin_y * ux + cos_y * uz
        ry2 = cos_x * ry - sin_x * rz
        px, py = int(cx + rx * ax_len), int(cy + ry2 * ax_len)
        nx, ny = int(cx - rx * ax_len), int(cy - ry2 * ax_len)
        col = tuple(int(c * alpha) for c in ax_colors[ai])
        cv2.line(frame, (nx, ny), (cx, cy), col, 2, cv2.LINE_AA)
        cv2.arrowedLine(frame, (cx, cy), (px, py), col, 3, cv2.LINE_AA, tipLength=0.15)
        dx_tip = rx / max(abs(rx), abs(ry2), 0.01) * 18
        dy_tip = ry2 / max(abs(rx), abs(ry2), 0.01) * 18
        lx, ly = int(px + dx_tip), int(py + dy_tip)
        cv2.putText(frame, ax_labels[ai], (lx - 10, ly + 5), FONT, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, ax_labels[ai], (lx - 10, ly + 5), FONT, 0.55, col, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 4, tuple(int(200*alpha) for _ in range(3)), -1, cv2.LINE_AA)


# ── Load pre-built assets ──
print("Loading pre-built assets...")
import time as _time
_t0 = _time.time()
with open(f'{BASE}/animation/scene5b_three_stacks_assets.pkl', 'rb') as f:
    assets = pickle.load(f)
print(f"  Loaded in {_time.time()-_t0:.0f}s")

# Per-tile data (already has dense, dense_with_ms, invivo, exvivo, merscope)
tile_data = {}
tile_list = []
for tile in TILES:
    if tile not in assets:
        continue
    tile_data[tile] = assets[tile]
    tile_list.append(tile)
    n = len(assets[tile]['dense'])
    w = assets[tile]['cell_w']
    h = assets[tile]['cell_h']
    print(f"  {tile}: {n} slices, {w}x{h}")
print(f"Loaded {len(tile_list)} tiles")

# Stitched volumes (combined has MERSCOPE baked in)
stitch = assets['_stitched']
stitch_vol_with_ms = stitch['combined']   # IV+EV+MERSCOPE
stitch_invivo = stitch['invivo']          # green only
stitch_exvivo = stitch['exvivo']          # magenta only
stitch_z = stitch['z']
vol_w = stitch['width']
vol_h = stitch['height']
nz = len(stitch_vol_with_ms)
avg_um_per_dpx = stitch['avg_um_per_dpx']
z_px_per_slice = 2.0 / avg_um_per_dpx
stitch_center_z = (stitch_z[0] + stitch_z[-1]) / 2
print(f"Stitched: {stitch_vol_with_ms.shape}")

# ── Z offsets (same computation as scene5b) ──
ANIM_Z_SCALE = 0.8
REAL_Z_PX_PER_SLICE = z_px_per_slice

z_offsets_raw = {tn: tile_data[tn]['stitch_z_offset'] for tn in tile_list}
z_center = (min(z_offsets_raw.values()) + max(z_offsets_raw.values())) / 2

tile_final_z = {}  # exaggerated z for merge animation
tile_real_z = {}   # physically correct z for final view
for tn in tile_list:
    tile_final_z[tn] = (z_offsets_raw[tn] - z_center) * ANIM_Z_SCALE
    tile_real_z[tn] = (z_offsets_raw[tn] - z_center) * REAL_Z_PX_PER_SLICE

# XY offsets for stitched positioning
xy_scale = 1.0 / (avg_um_per_dpx / 0.65)
x_mean = np.mean([tile_data[t]['canvas_x'] for t in tile_list])
y_mean = np.mean([tile_data[t]['canvas_y'] for t in tile_list])
tile_real_xy = {}
for tn in tile_list:
    dx = (tile_data[tn]['canvas_x'] - x_mean) * xy_scale
    dy = (tile_data[tn]['canvas_y'] - y_mean) * xy_scale
    tile_real_xy[tn] = (dx, dy)

print(f"  Z scaling: anim={ANIM_Z_SCALE}/slice, real={REAL_Z_PX_PER_SLICE:.3f} dpx/slice")

# ── Grid layout ──
GRID_COLS = 5
GRID_ROWS = 4
avg_cell_w = sum(td['cell_w'] for td in tile_data.values()) // len(tile_data)
avg_cell_h = sum(td['cell_h'] for td in tile_data.values()) // len(tile_data)
CELL_GAP_X, CELL_GAP_Y = 20, 15

col_w = W // 3
grid_w = GRID_COLS * avg_cell_w + (GRID_COLS - 1) * CELL_GAP_X
grid_h = GRID_ROWS * avg_cell_h + (GRID_ROWS - 1) * CELL_GAP_Y

margin = 40
fit_scale = min((col_w - margin) / grid_w, (H - margin - 60) / grid_h)
GRID_SCALE = fit_scale

single_margin = 80
single_fit = min((W - single_margin) / grid_w, (H - single_margin) / grid_h)
MERGED_SCALE = single_fit

print(f"Grid scale: {GRID_SCALE:.3f}, Merged: {MERGED_SCALE:.3f}")

def compute_grid_positions(center_x, center_y, scale):
    fitted_w = grid_w * scale
    fitted_h = grid_h * scale
    x0 = center_x - fitted_w / 2
    y0 = center_y - fitted_h / 2
    positions = {}
    for idx, tn in enumerate(tile_list):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        cx = x0 + (col * (avg_cell_w + CELL_GAP_X) + avg_cell_w / 2) * scale
        cy = y0 + (row * (avg_cell_h + CELL_GAP_Y) + avg_cell_h / 2) * scale
        positions[tn] = (int(cx), int(cy))
    return positions

COL_CENTERS = [W // 6, W // 2, W * 5 // 6]
COL_Y = H // 2 - 15
MERGED_CX, MERGED_CY = W // 2, H // 2

split_positions = [compute_grid_positions(cx, COL_Y, GRID_SCALE) for cx in COL_CENTERS]
merged_positions = compute_grid_positions(MERGED_CX, MERGED_CY, MERGED_SCALE)

# Volume column centers for split channel view (per-tile rendering)
VOL_COL_CENTERS = [W // 6, W // 2, W * 5 // 6]
# Compute per-tile split scale: each column needs overlapping tiles to fit in W/3
max_tile_w = max(td['cell_w'] for td in tile_data.values())
max_tile_h = max(td['cell_h'] for td in tile_data.values())
TILE_SPLIT_SCALE = min((W / 3 - 80) / max_tile_w, (H - 140) / max_tile_h) * 0.85
print(f"  TILE_SPLIT_SCALE: {TILE_SPLIT_SCALE:.3f}")


# ── Rendering functions ──

def render_tile_3d(slices, rot_y, rot_x, cx, cy, scale, dense_z, center_z, alpha_val=0.7, z_offset=0.0):
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    z_depths = []
    for i in range(len(slices)):
        dz = dense_z[i] - center_z + z_offset
        rz2 = cos_x * (cos_y * dz)
        z_depths.append((rz2, i))
    z_depths.sort(key=lambda x: x[0])

    for depth, i in z_depths:
        sl = slices[i].astype(np.float32) / 255.0
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
        M = cv2.getPerspectiveTransform(src_corners, rot_corners)
        warped = cv2.warpPerspective(sl, M, (W, H))
        mask = np.max(warped, axis=2) > 0.01
        mask3 = np.stack([mask] * 3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, warped * alpha_val), canvas)
    return canvas


def render_merscope_tile_3d(ms_img, rot_y, rot_x, cx, cy, scale, dense_z, center_z, alpha_val=0.7, z_offset=0.0):
    """Render MERSCOPE dots as a single flat plane at center z + offset."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)
    ms = ms_img.astype(np.float32) / 255.0
    sh, sw = ms.shape[:2]
    hw, hh = sw * scale / 2, sh * scale / 2
    dz = z_offset  # single plane
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
    M = cv2.getPerspectiveTransform(src_corners, rot_corners)
    warped = cv2.warpPerspective(ms, M, (W, H))
    mask = np.max(warped, axis=2) > 0.01
    mask3 = np.stack([mask] * 3, axis=-1)
    canvas = np.where(mask3, np.maximum(canvas, warped * alpha_val), canvas)
    return canvas


def render_column(channel, positions, rot_y, rot_x, scale, z_offsets=None):
    """Render all tiles for one channel at given positions."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for tn, (cx, cy) in positions.items():
        if tn not in tile_data:
            continue
        td = tile_data[tn]
        z_off = z_offsets.get(tn, 0.0) if z_offsets else 0.0
        if channel == 'merscope':
            ms = td['merscope']
            if ms is None:
                continue
            tc = render_merscope_tile_3d(ms, rot_y, rot_x, cx, cy, scale,
                                         td['dense_z'], td['center_z'], z_offset=z_off)
        else:
            tc = render_tile_3d(td[channel], rot_y, rot_x, cx, cy, scale,
                                td['dense_z'], td['center_z'], z_offset=z_off)
        mask = np.max(tc, axis=2) > 0.01
        mask3 = np.stack([mask] * 3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, tc), canvas)
    return canvas


def render_multi_tile(positions, rot_y, rot_x, scales, z_offsets=None):
    """Render all tiles combined (IV+EV+MERSCOPE pre-baked) at given positions."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for tn, (cx, cy) in positions.items():
        if tn not in tile_data:
            continue
        td = tile_data[tn]
        scale = scales.get(tn, GRID_SCALE)
        if scale < 0.05:
            continue
        z_off = z_offsets.get(tn, 0.0) if z_offsets else 0.0
        tc = render_tile_3d(td['dense_with_ms'], rot_y, rot_x, cx, cy, scale,
                            td['dense_z'], td['center_z'], z_offset=z_off)
        mask = np.max(tc, axis=2) > 0.01
        mask3 = np.stack([mask] * 3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, tc), canvas)
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)


def render_stitched_volume(volume, rot_y, rot_x, cx, cy, scale, subsample=4, alpha_val=0.7):
    """Render a volume (nz, h, w, 3) as 3D stack at given position."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    n_slices = len(volume)
    indices = range(0, n_slices, subsample)
    sh, sw = volume.shape[1], volume.shape[2]
    src_corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)

    z_depths = []
    for i in indices:
        dz = (stitch_z[i] - stitch_center_z) * z_px_per_slice
        rz2 = cos_x * (cos_y * dz)
        z_depths.append((rz2, i, dz))
    z_depths.sort(key=lambda x: x[0])

    for depth, i, dz in z_depths:
        sl = volume[i].astype(np.float32) / 255.0
        hw, hh = sw * scale / 2, sh * scale / 2
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
        M = cv2.getPerspectiveTransform(src_corners, rot_corners)
        warped = cv2.warpPerspective(sl, M, (W, H))
        mask = np.max(warped, axis=2) > 0.01
        mask3 = np.stack([mask] * 3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, warped * alpha_val), canvas)

    return np.clip(canvas * 255, 0, 255).astype(np.uint8)


def render_merscope_stitched(rot_y, rot_x, cx, cy, scale, alpha_val=0.7):
    """Render MERSCOPE as 19 per-tile gene dot planes at correct stitch z-offsets."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    planes = []
    for tn in tile_list:
        td = tile_data[tn]
        ms = td.get('merscope')
        if ms is None:
            continue
        dz = (td['stitch_z_offset'] - z_center) * z_px_per_slice
        rz2 = cos_x * (cos_y * dz)
        dx, dy = tile_real_xy[tn]
        planes.append((rz2, ms, dz, dx, dy))

    planes.sort(key=lambda x: x[0])

    for depth, ms_img, dz, dx, dy in planes:
        ms = ms_img.astype(np.float32) / 255.0
        sh, sw = ms.shape[:2]
        hw, hh = sw * scale / 2, sh * scale / 2
        tile_cx = cx + dx * scale
        tile_cy = cy + dy * scale

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
            rot_corners.append([rx + tile_cx, ry2 + tile_cy])
        rot_corners = np.array(rot_corners, dtype=np.float32)
        src_corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_corners, rot_corners)
        warped = cv2.warpPerspective(ms, M, (W, H))
        mask = np.max(warped, axis=2) > 0.01
        mask3 = np.stack([mask] * 3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, warped * alpha_val), canvas)

    return np.clip(canvas * 255, 0, 255).astype(np.uint8)


# ── Frame rendering ──

col_labels_data = [
    ('IN VIVO', COL_CENTERS[0]),
    ('EX VIVO', COL_CENTERS[1]),
    ('MERSCOPE', COL_CENTERS[2]),
]


def render_frame(job):
    frame_idx, params = job
    phase = params['phase']
    rot_y = params['rot_y']
    rot_x = params['rot_x']
    t = params.get('t', 0)

    frame = np.zeros((H, W, 3), dtype=np.uint8)

    if phase in ('split', 'rotate', 'merge_grids'):
        # ── Tile-grid based phases (3 channel columns) ──
        if phase == 'split':
            for ci, channel in enumerate(['invivo', 'exvivo', 'merscope']):
                positions = {}
                for tn in tile_list:
                    mx, my = merged_positions[tn]
                    sx, sy = split_positions[ci][tn]
                    positions[tn] = (int(mx * (1 - t) + sx * t), int(my * (1 - t) + sy * t))
                scale = MERGED_SCALE * (1 - t) + GRID_SCALE * t
                col_canvas = render_column(channel, positions, rot_y, rot_x, scale)
                col_u8 = np.clip(col_canvas * 255, 0, 255).astype(np.uint8)
                mask = np.max(col_u8, axis=2) > 2
                frame[mask] = np.maximum(frame[mask], col_u8[mask])

        elif phase == 'rotate':
            for ci, channel in enumerate(['invivo', 'exvivo', 'merscope']):
                col_canvas = render_column(channel, split_positions[ci], rot_y, rot_x, GRID_SCALE)
                col_u8 = np.clip(col_canvas * 255, 0, 255).astype(np.uint8)
                mask = np.max(col_u8, axis=2) > 2
                frame[mask] = np.maximum(frame[mask], col_u8[mask])

        elif phase == 'merge_grids':
            for ci, channel in enumerate(['invivo', 'exvivo', 'merscope']):
                positions = {}
                for tn in tile_list:
                    sx, sy = split_positions[ci][tn]
                    mx, my = merged_positions[tn]
                    positions[tn] = (int(sx * (1 - t) + mx * t), int(sy * (1 - t) + my * t))
                scale = GRID_SCALE * (1 - t) + MERGED_SCALE * t
                col_canvas = render_column(channel, positions, rot_y, rot_x, scale)
                col_u8 = np.clip(col_canvas * 255, 0, 255).astype(np.uint8)
                mask = np.max(col_u8, axis=2) > 2
                frame[mask] = np.maximum(frame[mask], col_u8[mask])

    elif phase == 'tile_merge':
        # ── Real tile merge: tiles slide from grid to center with z-depth ──
        vol_s = 1.0 + (VOLUME_SCALE - 1.0) * t
        positions = {}
        scales = {}
        z_offsets = {}
        for tn in tile_list:
            gx, gy = merged_positions[tn]
            positions[tn] = (int(gx * (1 - t) + W // 2 * t), int(gy * (1 - t) + H // 2 * t))
            scales[tn] = MERGED_SCALE * vol_s
            z_offsets[tn] = tile_final_z[tn] * t * VOLUME_SCALE
        frame = render_multi_tile(positions, rot_y, rot_x, scales, z_offsets)

    elif phase == 'scale_up':
        # ── Scale-up: tiles grow + z transition from animated to physically correct ──
        c_end_scale = MERGED_SCALE * VOLUME_SCALE
        c_target_scale = VOLUME_SCALE
        cur_scale = c_end_scale + (c_target_scale - c_end_scale) * t
        positions = {}
        scales = {}
        z_offsets = {}
        for tn in tile_list:
            positions[tn] = (W // 2, H // 2)
            scales[tn] = cur_scale
            z_anim = tile_final_z[tn] * VOLUME_SCALE
            z_real = tile_real_z[tn] * VOLUME_SCALE
            z_offsets[tn] = z_anim + (z_real - z_anim) * t
        frame = render_multi_tile(positions, rot_y, rot_x, scales, z_offsets)

    elif phase == 'rotate_blend':
        # ── 360° rotation: blend per-tile combined → per-tile channels at back-facing ──
        blend_t = t
        if blend_t <= 0:
            positions = {tn: (W//2, H//2) for tn in tile_list}
            scales = {tn: VOLUME_SCALE for tn in tile_list}
            z_offsets = {tn: tile_real_z[tn] * VOLUME_SCALE for tn in tile_list}
            frame = render_multi_tile(positions, rot_y, rot_x, scales, z_offsets)
        elif blend_t >= 1:
            for channel in ['invivo', 'exvivo', 'merscope']:
                positions = {tn: (W//2, H//2) for tn in tile_list}
                z_offs = {tn: tile_real_z[tn] * VOLUME_SCALE for tn in tile_list}
                col_canvas = render_column(channel, positions, rot_y, rot_x, VOLUME_SCALE, z_offs)
                col_u8 = np.clip(col_canvas * 255, 0, 255).astype(np.uint8)
                mask = np.max(col_u8, axis=2) > 2
                frame[mask] = np.maximum(frame[mask], col_u8[mask])
        else:
            positions = {tn: (W//2, H//2) for tn in tile_list}
            scales = {tn: VOLUME_SCALE for tn in tile_list}
            z_offsets = {tn: tile_real_z[tn] * VOLUME_SCALE for tn in tile_list}
            f_tiles = render_multi_tile(positions, rot_y, rot_x, scales, z_offsets)
            f_channels = np.zeros((H, W, 3), dtype=np.uint8)
            for channel in ['invivo', 'exvivo', 'merscope']:
                z_offs = {tn: tile_real_z[tn] * VOLUME_SCALE for tn in tile_list}
                col_canvas = render_column(channel, positions, rot_y, rot_x, VOLUME_SCALE, z_offs)
                col_u8 = np.clip(col_canvas * 255, 0, 255).astype(np.uint8)
                mask = np.max(col_u8, axis=2) > 2
                f_channels[mask] = np.maximum(f_channels[mask], col_u8[mask])
            frame = cv2.addWeighted(f_tiles, 1 - blend_t, f_channels, blend_t, 0)

    elif phase == 'split_channels':
        # ── Per-tile channels split to 3 side-by-side columns ──
        for ci, channel in enumerate(['invivo', 'exvivo', 'merscope']):
            cx = int(W / 2 * (1 - t) + VOL_COL_CENTERS[ci] * t)
            scale = VOLUME_SCALE * (1 - t) + TILE_SPLIT_SCALE * t
            positions = {tn: (cx, H // 2) for tn in tile_list}
            z_offs = {tn: tile_real_z[tn] * scale for tn in tile_list}
            col_canvas = render_column(channel, positions, rot_y, rot_x, scale, z_offs)
            col_u8 = np.clip(col_canvas * 255, 0, 255).astype(np.uint8)
            mask = np.max(col_u8, axis=2) > 2
            frame[mask] = np.maximum(frame[mask], col_u8[mask])

    elif phase == 'hold_split_vols':
        # ── Rotate 3 separate per-tile channel volumes ──
        for ci, channel in enumerate(['invivo', 'exvivo', 'merscope']):
            positions = {tn: (VOL_COL_CENTERS[ci], H // 2) for tn in tile_list}
            z_offs = {tn: tile_real_z[tn] * TILE_SPLIT_SCALE for tn in tile_list}
            col_canvas = render_column(channel, positions, rot_y, rot_x, TILE_SPLIT_SCALE, z_offs)
            col_u8 = np.clip(col_canvas * 255, 0, 255).astype(np.uint8)
            mask = np.max(col_u8, axis=2) > 2
            frame[mask] = np.maximum(frame[mask], col_u8[mask])

    elif phase == 'merge_channels':
        # ── 3 per-tile channel volumes merge back to center ──
        for ci, channel in enumerate(['invivo', 'exvivo', 'merscope']):
            cx = int(VOL_COL_CENTERS[ci] * (1 - t) + W / 2 * t)
            scale = TILE_SPLIT_SCALE * (1 - t) + VOLUME_SCALE * t
            positions = {tn: (cx, H // 2) for tn in tile_list}
            z_offs = {tn: tile_real_z[tn] * scale for tn in tile_list}
            col_canvas = render_column(channel, positions, rot_y, rot_x, scale, z_offs)
            col_u8 = np.clip(col_canvas * 255, 0, 255).astype(np.uint8)
            mask = np.max(col_u8, axis=2) > 2
            frame[mask] = np.maximum(frame[mask], col_u8[mask])

    # Labels
    labels = params.get('labels', None)
    if labels:
        for text, lx, alpha in labels:
            if alpha < 0.05:
                continue
            ts = 0.55
            (tw, _), _ = cv2.getTextSize(text, FONT, ts, 1)
            col = tuple(int(v * alpha) for v in WHITE)
            cv2.putText(frame, text, (lx - tw // 2, H - 30), FONT, ts, col, 1, cv2.LINE_AA)

    # Draw 3D axis widget
    draw_axes(frame, rot_y, rot_x)

    cap = params.get('caption', '')
    if cap:
        ts = 0.72
        (tw, _), _ = cv2.getTextSize(cap, FONT, ts, 1)
        cv2.putText(frame, cap, ((W - tw) // 2, H - 55), FONT, ts, WHITE, 1, cv2.LINE_AA)

    cv2.imwrite(f'{OUT_DIR}/frame_{frame_idx:05d}.png', frame)
    return frame_idx


# ── Build jobs ──
print("\nBuilding frame jobs...")
jobs = []
fi = 0
spin_fi = 0

def spin_rot():
    return STATIC_ROT_Y + SPIN_AMP * math.sin(2 * math.pi * spin_fi / SPIN_PERIOD)

ROT_SWING = 0.5  # for rotate phases

# Phase 1: Split (36fr)
SPLIT_FR = 36
for i in range(SPLIT_FR):
    t = ease(i / max(1, SPLIT_FR - 1))
    rot_y = spin_rot()
    label_alpha = ease(max(0, (t - 0.5) / 0.5))
    labels = [(text, lx, label_alpha) for text, lx in col_labels_data]
    fi += 1; spin_fi += 1
    jobs.append((fi, {'phase': 'split', 't': t, 'rot_y': rot_y, 'rot_x': STATIC_ROT_X,
                       'labels': labels, 'caption': 'REGISTERED  MULTIMODAL  STACKS'}))

# Phase 2: Rotate split (36fr)
ROTATE_FR = 36
for i in range(ROTATE_FR):
    rot_y = STATIC_ROT_Y + ROT_SWING * math.sin(2 * math.pi * i / ROTATE_FR)
    labels = [(text, lx, 1.0) for text, lx in col_labels_data]
    fi += 1; spin_fi += 1
    jobs.append((fi, {'phase': 'rotate', 'rot_y': rot_y, 'rot_x': STATIC_ROT_X,
                       'labels': labels, 'caption': 'REGISTERED  MULTIMODAL  STACKS'}))

# Phase 3: Merge 3 grids to center (36fr)
MERGE_GRIDS_FR = 36
for i in range(MERGE_GRIDS_FR):
    t = ease(i / max(1, MERGE_GRIDS_FR - 1))
    rot_y = spin_rot()
    label_alpha = 1 - ease(min(1.0, t * 2))
    labels = [(text, lx, label_alpha) for text, lx in col_labels_data]
    fi += 1; spin_fi += 1
    jobs.append((fi, {'phase': 'merge_grids', 't': t, 'rot_y': rot_y, 'rot_x': STATIC_ROT_X,
                       'labels': labels, 'caption': 'REGISTERED  MULTIMODAL  STACKS'}))

# Phase 4: Real tile merge — tiles slide to center with z-depth (72fr)
TILE_MERGE_FR = 72
for i in range(TILE_MERGE_FR):
    t = ease(i / max(1, TILE_MERGE_FR - 6))
    rot_y = spin_rot()
    fi += 1; spin_fi += 1
    jobs.append((fi, {'phase': 'tile_merge', 't': t, 'rot_y': rot_y, 'rot_x': STATIC_ROT_X,
                       'caption': 'COMBINING  TILES  INTO  STITCHED  3D  VOLUME'}))

# Phase 5: Scale-up + z correction (36fr)
SCALE_UP_FR = 36
for i in range(SCALE_UP_FR):
    t = ease(i / max(1, SCALE_UP_FR - 1))
    rot_y = spin_rot()
    fi += 1; spin_fi += 1
    jobs.append((fi, {'phase': 'scale_up', 't': t, 'rot_y': rot_y, 'rot_x': STATIC_ROT_X,
                       'caption': 'COMBINING  TILES  INTO  STITCHED  3D  VOLUME'}))

# Phase 6: Rotation blend (72fr) — 360° rotation, blend per-tile combined → per-tile channels
ROTATE_BLEND_FR = 72
ROT_BLEND_START = STATIC_ROT_Y  # where scale_up ends
ROT_BLEND_SWEEP = 2 * math.pi   # full 360°
for i in range(ROTATE_BLEND_FR):
    t_rot = i / max(1, ROTATE_BLEND_FR - 1)
    rot_y = ROT_BLEND_START + ROT_BLEND_SWEEP * t_rot
    rot_x = STATIC_ROT_X
    blend_t = ease(max(0, min(1, (t_rot - 0.2) / 0.35)))  # blend at 20-55% (back-facing)
    fi += 1; spin_fi += 1
    jobs.append((fi, {'phase': 'rotate_blend', 't': blend_t, 'rot_y': rot_y, 'rot_x': rot_x,
                       'caption': 'REGISTERED  MULTIMODAL  VOLUME'}))

# Phase 7: Split per-tile channels into 3 column volumes (48fr)
SPLIT_CH_FR = 48
for i in range(SPLIT_CH_FR):
    t = ease(i / max(1, SPLIT_CH_FR - 1))
    rot_y = spin_rot()
    label_alpha = ease(max(0, (t - 0.5) / 0.5))
    labels = [(text, lx, label_alpha) for text, lx in col_labels_data]
    fi += 1; spin_fi += 1
    jobs.append((fi, {'phase': 'split_channels', 't': t, 'rot_y': rot_y, 'rot_x': STATIC_ROT_X,
                       'labels': labels, 'caption': 'IN VIVO  |  EX VIVO  |  MERSCOPE'}))

# Phase 8: Hold split channel volumes (48fr)
HOLD_SPLIT_FR = 48
for i in range(HOLD_SPLIT_FR):
    rot_y = STATIC_ROT_Y + ROT_SWING * math.sin(2 * math.pi * i / HOLD_SPLIT_FR)
    labels = [(text, lx, 1.0) for text, lx in col_labels_data]
    fi += 1; spin_fi += 1
    jobs.append((fi, {'phase': 'hold_split_vols', 'rot_y': rot_y, 'rot_x': STATIC_ROT_X,
                       'labels': labels, 'caption': 'IN VIVO  |  EX VIVO  |  MERSCOPE'}))

# Phase 9: Merge channel volumes back (36fr)
MERGE_CH_FR = 36
for i in range(MERGE_CH_FR):
    t = ease(i / max(1, MERGE_CH_FR - 1))
    rot_y = spin_rot()
    label_alpha = 1 - ease(min(1.0, t * 2))
    labels = [(text, lx, label_alpha) for text, lx in col_labels_data]
    fi += 1; spin_fi += 1
    jobs.append((fi, {'phase': 'merge_channels', 't': t, 'rot_y': rot_y, 'rot_x': STATIC_ROT_X,
                       'labels': labels, 'caption': 'REGISTERED  MULTIMODAL  VOLUME'}))

print(f"  {len(jobs)} frames to render")

# ── Render ──
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

N_WORKERS = min(os.cpu_count(), 8)
print(f"\nRendering with {N_WORKERS} threads...")
t0 = time.time()
done = 0

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(render_frame, job): job[0] for job in jobs}
    for future in as_completed(futures):
        idx = future.result()
        done += 1
        if done % 10 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = (len(jobs) - done) / rate
            print(f"  {done}/{len(jobs)} ({rate:.1f} fr/s, ~{remaining:.0f}s remaining)")

elapsed = time.time() - t0
print(f"\nDone! {len(jobs)} frames in {elapsed:.0f}s")
print(f"Saved to {OUT_DIR}/")

# Encode preview
print("\nEncoding preview...")
os.system(f'ffmpeg -y -framerate 24 -i {OUT_DIR}/frame_%05d.png '
          f'-c:v libx264 -pix_fmt yuv420p -crf 18 '
          f'{BASE}/animation/three_stacks_v2_preview.mp4 2>/dev/null')
print("Saved three_stacks_v2_preview.mp4")
