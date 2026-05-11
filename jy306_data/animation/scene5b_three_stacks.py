"""
Render the 3-stack split sequence: 19 tiles shown as 3 side-by-side groups.
  Left:   In-vivo (green channel only)
  Center: Ex-vivo (magenta channels only)
  Right:  MERSCOPE gene dots (rainbow)

Phases:
  1. Split (36fr): tiles slide from merged center to 3 columns
  2. Rotate (48fr): gentle rotation of all 3 groups
  3. Merge (36fr): slide back to center

Outputs to frames_three_stacks/ folder.
"""
import numpy as np, cv2, math, os, pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/animation/frames_three_stacks'
W, H = 1920, 1080
FPS = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

TILES = ['row1_3',
         'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
         'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
         'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
         'row5_1']

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

# ── Load assets ──
print("Loading assets...")
with open(f'{BASE}/animation/scene5b_assets_v3.pkl', 'rb') as f:
    assets = pickle.load(f)

# Build per-tile data with split channels
tile_data = {}
tile_list = []
for tile in TILES:
    if tile not in assets:
        continue
    a = assets[tile]
    dense = a['dense']  # (n, h, w, 3) uint8, BGR
    # Split channels
    # In OpenCV BGR: B=0, G=1, R=2
    # Magenta (ex-vivo) = R+B channels, Green (in-vivo) = G channel
    n, h, w, _ = dense.shape

    # In-vivo: green channel only → display as green
    invivo = np.zeros_like(dense)
    invivo[:, :, :, 1] = dense[:, :, :, 1]  # G channel

    # Ex-vivo: R+B channels → display as magenta
    exvivo = np.zeros_like(dense)
    exvivo[:, :, :, 0] = dense[:, :, :, 0]  # B
    exvivo[:, :, :, 2] = dense[:, :, :, 2]  # R

    # MERSCOPE dots
    ms = a.get('merscope')  # (h, w, 3) or None

    tile_data[tile] = {
        'dense': dense,
        'invivo': invivo,
        'exvivo': exvivo,
        'merscope': ms,
        'dense_z': a['dense_z'],
        'center_z': a['center_z'],
        'cell_w': a['cell_w'],
        'cell_h': a['cell_h'],
    }
    tile_list.append(tile)
    print(f"  {tile}: {n} slices, {w}x{h}")

print(f"Loaded {len(tile_list)} tiles")

# ── Grid layout (same as main script) ──
GRID_COLS = 5
GRID_ROWS = 4
avg_cell_w = sum(td['cell_w'] for td in tile_data.values()) // len(tile_data)
avg_cell_h = sum(td['cell_h'] for td in tile_data.values()) // len(tile_data)
CELL_GAP_X, CELL_GAP_Y = 20, 15

# For 3-column layout, each column gets W/3
# Grid within each column
col_w = W // 3
grid_w = GRID_COLS * avg_cell_w + (GRID_COLS - 1) * CELL_GAP_X
grid_h = GRID_ROWS * avg_cell_h + (GRID_ROWS - 1) * CELL_GAP_Y

# Scale to fit each column
margin = 40
fit_scale = min((col_w - margin) / grid_w, (H - margin - 60) / grid_h)  # 60 for labels
GRID_SCALE = fit_scale
print(f"Grid scale for 3-col: {GRID_SCALE:.3f}")

# Also compute merged (single) grid scale
single_margin = 80
single_fit = min((W - single_margin) / grid_w, (H - single_margin) / grid_h)
MERGED_SCALE = single_fit

# Grid positions relative to column center
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

# Column centers for split view
COL_CENTERS = [W // 6, W // 2, W * 5 // 6]
COL_Y = H // 2 - 15  # slightly up to leave room for labels

# Merged center
MERGED_CX, MERGED_CY = W // 2, H // 2

# Pre-compute positions
split_positions = [compute_grid_positions(cx, COL_Y, GRID_SCALE) for cx in COL_CENTERS]
merged_positions = compute_grid_positions(MERGED_CX, MERGED_CY, MERGED_SCALE)

SPIN_AMP = 0.12
SPIN_PERIOD = 120
STATIC_ROT_Y = 0.25
STATIC_ROT_X = -0.3


def render_tile_3d(slices, rot_y, rot_x, cx, cy, scale, dense_z, center_z, alpha_val=0.7):
    """Render a single tile's slices as 3D stack."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    z_depths = []
    for i in range(len(slices)):
        dz = dense_z[i] - center_z
        rz2 = cos_x * (cos_y * dz)
        z_depths.append((rz2, i))
    z_depths.sort(key=lambda x: x[0])

    for depth, i in z_depths:
        sl = slices[i].astype(np.float32) / 255.0
        sh, sw = sl.shape[:2]
        hw, hh = sw * scale / 2, sh * scale / 2
        dz = dense_z[i] - center_z

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


def render_merscope_tile_3d(ms_img, rot_y, rot_x, cx, cy, scale, dense_z, center_z, alpha_val=0.7):
    """Render MERSCOPE dots as 3D stack (same image on each z-slice)."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    # Use fewer slices for dots (subsample)
    n = len(dense_z)
    indices = list(range(0, n, max(1, n // 5)))  # ~5 slices
    ms = ms_img.astype(np.float32) / 255.0
    sh, sw = ms.shape[:2]

    z_depths = []
    for i in indices:
        dz = dense_z[i] - center_z
        rz2 = cos_x * (cos_y * dz)
        z_depths.append((rz2, i, dz))
    z_depths.sort(key=lambda x: x[0])

    for depth, i, dz in z_depths:
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
        src_corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_corners, rot_corners)
        warped = cv2.warpPerspective(ms, M, (W, H))
        mask = np.max(warped, axis=2) > 0.01
        mask3 = np.stack([mask] * 3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, warped * alpha_val), canvas)

    return canvas


def render_column(channel, positions, rot_y, rot_x, scale):
    """Render all tiles for one channel (invivo/exvivo/merscope)."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for tn, (cx, cy) in positions.items():
        if tn not in tile_data:
            continue
        td = tile_data[tn]
        if channel == 'merscope':
            ms = td['merscope']
            if ms is None:
                continue
            tc = render_merscope_tile_3d(ms, rot_y, rot_x, cx, cy, scale,
                                         td['dense_z'], td['center_z'])
        else:
            slices = td[channel]
            tc = render_tile_3d(slices, rot_y, rot_x, cx, cy, scale,
                                td['dense_z'], td['center_z'])
        mask = np.max(tc, axis=2) > 0.01
        mask3 = np.stack([mask] * 3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, tc), canvas)
    return canvas


def render_frame(job):
    frame_idx, params = job
    phase = params['phase']
    rot_y = params['rot_y']
    rot_x = params['rot_x']
    t = params.get('t', 0)

    frame = np.zeros((H, W, 3), dtype=np.uint8)

    if phase == 'split':
        # Interpolate from merged to split positions
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
        # Three separate columns
        for ci, channel in enumerate(['invivo', 'exvivo', 'merscope']):
            col_canvas = render_column(channel, split_positions[ci], rot_y, rot_x, GRID_SCALE)
            col_u8 = np.clip(col_canvas * 255, 0, 255).astype(np.uint8)
            mask = np.max(col_u8, axis=2) > 2
            frame[mask] = np.maximum(frame[mask], col_u8[mask])

    elif phase == 'merge':
        # Interpolate from split back to merged
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

    # Column labels
    labels = params.get('labels', None)
    if labels:
        for text, lx, alpha in labels:
            if alpha < 0.05:
                continue
            ts = 0.55
            (tw, _), _ = cv2.getTextSize(text, FONT, ts, 1)
            col = tuple(int(v * alpha) for v in WHITE)
            cv2.putText(frame, text, (lx - tw // 2, H - 30), FONT, ts, col, 1, cv2.LINE_AA)

    # Caption
    cap = params.get('caption', '')
    if cap:
        ts = 0.72
        (tw, _), _ = cv2.getTextSize(cap, FONT, ts, 1)
        cv2.putText(frame, cap, ((W - tw) // 2, H - 55), FONT, ts, WHITE, 1, cv2.LINE_AA)

    out_path = f'{OUT_DIR}/frame_{frame_idx:05d}.png'
    cv2.imwrite(out_path, frame)
    return frame_idx


# ── Build jobs ──
print("\nBuilding frame jobs...")
jobs = []
fi = 0
spin_fi = 0

col_labels = [
    ('IN-VIVO', COL_CENTERS[0]),
    ('EX-VIVO', COL_CENTERS[1]),
    ('MERSCOPE', COL_CENTERS[2]),
]

# Phase 1: Split (36 frames)
SPLIT_FRAMES = 36
for i in range(SPLIT_FRAMES):
    t = ease(i / max(1, SPLIT_FRAMES - 1))
    rot_y = STATIC_ROT_Y + SPIN_AMP * math.sin(2 * math.pi * spin_fi / SPIN_PERIOD)
    label_alpha = ease(max(0, (t - 0.5) / 0.5))
    labels = [(text, lx, label_alpha) for text, lx in col_labels]

    fi += 1
    jobs.append((fi, {
        'phase': 'split', 't': t,
        'rot_y': rot_y, 'rot_x': STATIC_ROT_X,
        'labels': labels,
        'caption': 'REGISTERED  MULTIMODAL  STACKS',
    }))
    spin_fi += 1

# Phase 2: Rotate (48 frames) — pronounced swing to show 3D depth
ROTATE_FRAMES = 48
ROT_SWING = 0.5  # ±0.5 rad (~29°) swing
for i in range(ROTATE_FRAMES):
    # Full sine swing: 0 → +0.5 → 0 → -0.5 → 0
    rot_y = STATIC_ROT_Y + ROT_SWING * math.sin(2 * math.pi * i / ROTATE_FRAMES)
    labels = [(text, lx, 1.0) for text, lx in col_labels]

    fi += 1
    jobs.append((fi, {
        'phase': 'rotate',
        'rot_y': rot_y, 'rot_x': STATIC_ROT_X,
        'labels': labels,
        'caption': 'REGISTERED  MULTIMODAL  STACKS',
    }))
    spin_fi += 1

# Phase 3: Merge back (36 frames)
MERGE_FRAMES = 36
for i in range(MERGE_FRAMES):
    t = ease(i / max(1, MERGE_FRAMES - 1))
    rot_y = STATIC_ROT_Y + SPIN_AMP * math.sin(2 * math.pi * spin_fi / SPIN_PERIOD)
    label_alpha = 1 - ease(min(1.0, t * 2))
    labels = [(text, lx, label_alpha) for text, lx in col_labels]

    fi += 1
    jobs.append((fi, {
        'phase': 'merge', 't': t,
        'rot_y': rot_y, 'rot_x': STATIC_ROT_X,
        'labels': labels,
        'caption': 'REGISTERED  MULTIMODAL  STACKS',
    }))
    spin_fi += 1

print(f"  {len(jobs)} frames to render")

# ── Render ──
import shutil, time
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
