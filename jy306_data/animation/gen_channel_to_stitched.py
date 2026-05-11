"""
Generate transition frames: per-tile channels at center → stitched 3D volume.
360° rotation with blend at back-facing, then hold stitched.

Outputs to frames_channel_to_stitched/
"""
import numpy as np, cv2, math, os, pickle, shutil, time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/animation/frames_channel_to_stitched_v2'
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
VOLUME_SCALE = 1.8

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))


def draw_axes(frame, rot_y, rot_x, cx=120, cy=None, ax_len=70):
    if cy is None:
        cy = H - 120
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)
    axes_3d = [(1, 0, 0), (0, -1, 0), (0, 0, 1)]
    ax_colors = [(0, 0, 180), (40, 40, 40), (200, 80, 0)]
    ax_labels = ['ML', 'AP', 'DV']
    for ai, (ux, uy, uz) in enumerate(axes_3d):
        rx = cos_y * ux + sin_y * uz
        ry = uy
        rz = -sin_y * ux + cos_y * uz
        ry2 = cos_x * ry - sin_x * rz
        px, py = int(cx + rx * ax_len), int(cy + ry2 * ax_len)
        nx, ny = int(cx - rx * ax_len), int(cy - ry2 * ax_len)
        cv2.line(frame, (nx, ny), (cx, cy), ax_colors[ai], 2, cv2.LINE_AA)
        cv2.arrowedLine(frame, (cx, cy), (px, py), ax_colors[ai], 3, cv2.LINE_AA, tipLength=0.15)
        dx_tip = rx / max(abs(rx), abs(ry2), 0.01) * 18
        dy_tip = ry2 / max(abs(rx), abs(ry2), 0.01) * 18
        lx, ly = int(px + dx_tip), int(py + dy_tip)
        cv2.putText(frame, ax_labels[ai], (lx - 10, ly + 5), FONT, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, ax_labels[ai], (lx - 10, ly + 5), FONT, 0.55, ax_colors[ai], 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 4, (200, 200, 200), -1, cv2.LINE_AA)


# ── Load assets ──
print("Loading assets...")
t0 = time.time()
with open(f'{BASE}/animation/scene5b_three_stacks_assets.pkl', 'rb') as f:
    assets = pickle.load(f)
print(f"  Loaded in {time.time()-t0:.0f}s")

tile_data = {}
tile_list = []
for tile in TILES:
    if tile not in assets:
        continue
    tile_data[tile] = assets[tile]
    tile_list.append(tile)

stitch = assets['_stitched']
stitch_vol_with_ms = stitch['combined']
stitch_z = stitch['z']
avg_um_per_dpx = stitch['avg_um_per_dpx']
z_px_per_slice = 2.0 / avg_um_per_dpx
stitch_center_z = (stitch_z[0] + stitch_z[-1]) / 2
vol_h, vol_w = stitch_vol_with_ms.shape[1], stitch_vol_with_ms.shape[2]

z_offsets_raw = {tn: tile_data[tn]['stitch_z_offset'] for tn in tile_list}
z_center = (min(z_offsets_raw.values()) + max(z_offsets_raw.values())) / 2
REAL_Z_PX = z_px_per_slice
tile_real_z = {}
for tn in tile_list:
    tile_real_z[tn] = (z_offsets_raw[tn] - z_center) * REAL_Z_PX

print(f"Loaded {len(tile_list)} tiles, stitched: {stitch_vol_with_ms.shape}")


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
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)
    ms = ms_img.astype(np.float32) / 255.0
    sh, sw = ms.shape[:2]
    hw, hh = sw * scale / 2, sh * scale / 2
    dz = z_offset
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


def render_multi_tile_center(rot_y, rot_x):
    """Render all tiles combined (dense_with_ms) overlapping at center."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for tn in tile_list:
        td = tile_data[tn]
        z_off = tile_real_z[tn] * VOLUME_SCALE
        tc = render_tile_3d(td['dense_with_ms'], rot_y, rot_x, W//2, H//2, VOLUME_SCALE,
                             td['dense_z'], td['center_z'], z_offset=z_off)
        mask = np.max(tc, axis=2) > 0.01
        mask3 = np.stack([mask]*3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, tc), canvas)
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)


def render_stitched_volume(rot_y, rot_x, subsample=4):
    """Render the full stitched combined volume."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)
    n_slices = len(stitch_vol_with_ms)
    indices = range(0, n_slices, subsample)
    sh, sw = stitch_vol_with_ms.shape[1], stitch_vol_with_ms.shape[2]
    src_corners = np.array([[0,0],[sw,0],[sw,sh],[0,sh]], dtype=np.float32)
    z_depths = []
    for i in indices:
        dz = (stitch_z[i] - stitch_center_z) * z_px_per_slice
        rz2 = cos_x * (cos_y * dz)
        z_depths.append((rz2, i, dz))
    z_depths.sort(key=lambda x: x[0])
    for depth, i, dz in z_depths:
        sl = stitch_vol_with_ms[i].astype(np.float32) / 255.0
        hw, hh = sw * VOLUME_SCALE / 2, sh * VOLUME_SCALE / 2
        corners_3d = np.array([
            [-hw, -hh, dz * VOLUME_SCALE], [hw, -hh, dz * VOLUME_SCALE],
            [hw, hh, dz * VOLUME_SCALE], [-hw, hh, dz * VOLUME_SCALE]
        ], dtype=np.float64)
        rot_corners = []
        for c in corners_3d:
            rx = cos_y * c[0] + sin_y * c[2]
            ry = c[1]
            rz = -sin_y * c[0] + cos_y * c[2]
            ry2 = cos_x * ry - sin_x * rz
            rot_corners.append([rx + W//2, ry2 + H//2])
        rot_corners = np.array(rot_corners, dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_corners, rot_corners)
        warped = cv2.warpPerspective(sl, M, (W, H))
        mask = np.max(warped, axis=2) > 0.01
        mask3 = np.stack([mask]*3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, warped * 0.7), canvas)
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)


def render_frame(job):
    fi, rot_y, rot_x, blend_t, is_hold = job

    if is_hold:
        frame = render_stitched_volume(rot_y, rot_x, subsample=3)
    elif blend_t <= 0:
        frame = render_multi_tile_center(rot_y, rot_x)
    elif blend_t >= 1:
        frame = render_stitched_volume(rot_y, rot_x)
    else:
        f_ch = render_multi_tile_center(rot_y, rot_x)
        f_st = render_stitched_volume(rot_y, rot_x)
        frame = cv2.addWeighted(f_ch, 1 - blend_t, f_st, blend_t, 0)

    draw_axes(frame, rot_y, rot_x)

    cap = 'STITCHED  MULTIMODAL  3D  VOLUME' if (blend_t > 0.5 or is_hold) else 'REGISTERED  MULTIMODAL  VOLUME'
    ts = 0.72
    (tw, _), _ = cv2.getTextSize(cap, FONT, ts, 1)
    cv2.putText(frame, cap, ((W - tw) // 2, H - 42), FONT, ts, WHITE, 1, cv2.LINE_AA)

    cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', frame)
    return fi


# ── Build jobs ──
jobs = []
fi = 0

# Phase 1: 360° rotation — blend per-tile channels → stitched at back-facing (72fr)
ROT_START = STATIC_ROT_Y
ROT_SWEEP = 2 * math.pi
for i in range(72):
    t = i / 71
    rot_y = ROT_START + ROT_SWEEP * t
    blend_t = ease(max(0, min(1, (t - 0.2) / 0.35)))  # blend at 20-55%
    fi += 1
    jobs.append((fi, rot_y, STATIC_ROT_X, blend_t, False))

# Phase 2: Hold stitched volume (24fr)
for i in range(24):
    rot_y = ROT_START + 0.08 * math.sin(2 * math.pi * i / 48)
    fi += 1
    jobs.append((fi, rot_y, STATIC_ROT_X, 1.0, True))

print(f"{len(jobs)} frames to render")

# ── Render ──
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

N_WORKERS = min(os.cpu_count(), 6)
print(f"Rendering with {N_WORKERS} threads...")
t0 = time.time()
done = 0

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(render_frame, job): job[0] for job in jobs}
    for future in as_completed(futures):
        idx = future.result()
        done += 1
        if done % 5 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = (len(jobs) - done) / rate
            print(f"  {done}/{len(jobs)} ({rate:.1f} fr/s, ~{remaining:.0f}s remaining)")

elapsed = time.time() - t0
print(f"\nDone! {len(jobs)} frames in {elapsed:.0f}s")
print(f"Saved to {OUT_DIR}/")
