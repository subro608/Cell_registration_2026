"""
3-layer 3D stack: in-vivo, ex-vivo, MERSCOPE as 3 planes stacked in z.
Uses the stitched volume split into channels.

Phases:
  1. Combined (36fr): single 3D stack with all 3 layers interleaved, gentle rotate
  2. Split (48fr): 3 layers separate in z, rotate to show separation
  3. Hold split (36fr): hold separated, rotate
  4. Merge (36fr): layers come back together
  5. Hold merged (24fr): final combined view

Outputs to frames_3layer_stack/
"""
import numpy as np, cv2, math, os, pickle, shutil, time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/animation/frames_3layer_stack'
W, H = 1920, 1080
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

# ── Load stitched volume ──
print("Loading assets...")
with open(f'{BASE}/animation/scene5b_assets_v3.pkl', 'rb') as f:
    assets = pickle.load(f)

stitch = assets['_stitched']
volume = stitch['volume']  # (nz, h, w, 3) uint8 BGR
z_vals = stitch['z']
print(f"Stitched volume: {volume.shape}, z=[{z_vals[0]:.0f},{z_vals[-1]:.0f}]")

nz, vol_h, vol_w, _ = volume.shape

# Split into 3 channel volumes
# BGR: B=0, G=1, R=2. Magenta=R+B, Green=G
vol_invivo = np.zeros_like(volume)
vol_invivo[:, :, :, 1] = volume[:, :, :, 1]  # green channel only

vol_exvivo = np.zeros_like(volume)
vol_exvivo[:, :, :, 0] = volume[:, :, :, 0]  # B
vol_exvivo[:, :, :, 2] = volume[:, :, :, 2]  # R

# MERSCOPE: build from per-tile dots, or use a representative slice
# Since dots are 2D per tile, we'll build a combined MERSCOPE image from all tiles
print("Building MERSCOPE composite...")
tiles = [t for t in assets.keys() if not t.startswith('_')]
tiles = sorted(tiles, key=lambda t: assets[t]['stitch_z_offset'])
max_w = stitch['width']

# Build a single MERSCOPE composite (padded like stitched volume)
ms_composite = np.zeros((vol_h, max_w, 3), dtype=np.uint8)
for t in tiles:
    a = assets[t]
    ms = a.get('merscope')
    if ms is None:
        continue
    cw = a['cell_w']
    pad_l = (max_w - cw) // 2
    # Resize ms to match cell dimensions
    if ms.shape[0] != vol_h or ms.shape[1] != cw:
        ms = cv2.resize(ms, (cw, vol_h), interpolation=cv2.INTER_AREA)
    # Paste (max blend)
    region = ms_composite[:, pad_l:pad_l + cw]
    mask = np.max(ms, axis=2) > np.max(region, axis=2)
    region[mask] = ms[mask]

print(f"MERSCOPE composite: {ms_composite.shape}")

# Subsample volumes for speed
SUBSAMPLE = 4  # render every 4th slice
indices = list(range(0, nz, SUBSAMPLE))
vol_iv_sub = vol_invivo[indices]
vol_ev_sub = vol_exvivo[indices]
z_sub = z_vals[indices]
n_sub = len(indices)
print(f"Subsampled to {n_sub} slices")

# Scale
SCALE = 1.5
center_z = (z_sub[0] + z_sub[-1]) / 2

# Compute z-pixel scale
avg_um_per_dpx = np.mean([(assets[t]['crop_h_nd2'] * 0.645) / assets[t]['cell_h']
                           for t in tiles if t in assets])
z_px_per_slice = 2.0 / avg_um_per_dpx


def render_3layer(rot_y, rot_x, separation=0.0, labels_alpha=0.0):
    """
    Render 3 layers: in-vivo (top), ex-vivo (middle), MERSCOPE (bottom).
    separation: z-distance between layers (0 = merged, positive = separated)
    """
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cos_y, sin_y = math.cos(rot_y), math.sin(rot_y)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)
    cx, cy = W // 2, H // 2

    # Three groups: invivo (top), exvivo (mid), merscope (bottom)
    # Each group has its own z-offset for separation
    layers = [
        ('invivo', vol_iv_sub, z_sub, separation),      # pushed up in z
        ('exvivo', vol_ev_sub, z_sub, 0.0),              # center
        ('merscope', None, None, -separation),            # pushed down in z
    ]

    # Collect all renderable slices with depth
    all_slices = []
    for layer_name, vol, zs, z_off in layers:
        if layer_name == 'merscope':
            # Single MERSCOPE plane at center z + offset
            dz = z_off * z_px_per_slice
            rz2 = cos_x * (cos_y * dz)
            all_slices.append((rz2, ms_composite, dz, vol_w, vol_h))
        else:
            for i in range(n_sub):
                dz = (zs[i] - center_z) * z_px_per_slice + z_off * z_px_per_slice
                rz2 = cos_x * (cos_y * dz)
                all_slices.append((rz2, vol[i], dz, vol_w, vol_h))

    all_slices.sort(key=lambda x: x[0])

    src_corners = np.array([[0, 0], [vol_w, 0], [vol_w, vol_h], [0, vol_h]], dtype=np.float32)

    for depth, sl_data, dz, sw, sh in all_slices:
        sl = sl_data.astype(np.float32) / 255.0
        if sl.shape[0] != sh or sl.shape[1] != sw:
            sl = cv2.resize(sl, (sw, sh)).astype(np.float32)
            if sl.max() > 1:
                sl /= 255.0
        hw, hh = sw * SCALE / 2, sh * SCALE / 2

        corners_3d = np.array([
            [-hw, -hh, dz * SCALE], [hw, -hh, dz * SCALE],
            [hw, hh, dz * SCALE], [-hw, hh, dz * SCALE]
        ], dtype=np.float64)

        rot_corners = []
        for c in corners_3d:
            rx = cos_y * c[0] + sin_y * c[2]
            ry = c[1]
            rz = -sin_y * c[0] + cos_y * c[2]
            ry2 = cos_x * ry - sin_x * rz
            rot_corners.append([rx + cx, ry2 + cy])

        rot_corners_f = np.array(rot_corners, dtype=np.float32)
        sc = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(sc, rot_corners_f)
        warped = cv2.warpPerspective(sl, M, (W, H))
        mask = np.max(warped, axis=2) > 0.01
        mask3 = np.stack([mask] * 3, axis=-1)
        canvas = np.where(mask3, np.maximum(canvas, warped * 0.7), canvas)

    frame = np.clip(canvas * 255, 0, 255).astype(np.uint8)

    # Layer labels when separated
    if labels_alpha > 0.05:
        col = tuple(int(v * labels_alpha) for v in WHITE)
        label_data = [
            ('IN-VIVO', separation),
            ('EX-VIVO', 0.0),
            ('MERSCOPE', -separation),
        ]
        for text, z_off in label_data:
            dz = z_off * z_px_per_slice
            # Project label position
            ry2 = cos_x * (-sh * SCALE / 2 - 15) - sin_x * (dz * SCALE)
            ly = int(cy + ry2)
            ts = 0.5
            (tw, _), _ = cv2.getTextSize(text, FONT, ts, 1)
            cv2.putText(frame, text, ((W - tw) // 2, ly - 5), FONT, ts, col, 1, cv2.LINE_AA)

    return frame


def render_frame(job):
    fi, params = job
    p = {k: v for k, v in params.items() if k != 'caption'}
    frame = render_3layer(**p)

    cap = params.get('caption', '')
    if cap:
        ts = 0.72
        (tw, _), _ = cv2.getTextSize(cap, FONT, ts, 1)
        cv2.putText(frame, cap, ((W - tw) // 2, H - 42), FONT, ts, WHITE, 1, cv2.LINE_AA)

    cv2.imwrite(f'{OUT_DIR}/frame_{fi:05d}.png', frame)
    return fi


# ── Build jobs ──
print("\nBuilding frame jobs...")
jobs = []
fi = 0
ROT_BASE = 0.25
ROT_X = -0.3
MAX_SEP = 40  # max z-separation between layers (in native slices)

# Phase 1: Combined, gentle rotate (36fr)
for i in range(36):
    rot_y = ROT_BASE + 0.3 * math.sin(2 * math.pi * i / 72)
    fi += 1
    jobs.append((fi, {'rot_y': rot_y, 'rot_x': ROT_X, 'separation': 0.0,
                       'labels_alpha': 0.0, 'caption': 'REGISTERED  MULTIMODAL  VOLUME'}))

# Phase 2: Split apart (48fr)
for i in range(48):
    t = ease(i / 47)
    sep = MAX_SEP * t
    rot_y = ROT_BASE + 0.4 * math.sin(2 * math.pi * (36 + i) / 72)
    la = ease(max(0, (t - 0.3) / 0.7))
    fi += 1
    jobs.append((fi, {'rot_y': rot_y, 'rot_x': ROT_X, 'separation': sep,
                       'labels_alpha': la, 'caption': 'IN-VIVO  |  EX-VIVO  |  MERSCOPE'}))

# Phase 3: Hold split, rotate (36fr)
for i in range(36):
    rot_y = ROT_BASE + 0.5 * math.sin(2 * math.pi * i / 36)
    fi += 1
    jobs.append((fi, {'rot_y': rot_y, 'rot_x': ROT_X, 'separation': MAX_SEP,
                       'labels_alpha': 1.0, 'caption': 'IN-VIVO  |  EX-VIVO  |  MERSCOPE'}))

# Phase 4: Merge back (36fr)
for i in range(36):
    t = ease(i / 35)
    sep = MAX_SEP * (1 - t)
    rot_y = ROT_BASE + 0.3 * math.sin(2 * math.pi * (120 + i) / 72)
    la = 1 - ease(min(1.0, t * 2))
    fi += 1
    jobs.append((fi, {'rot_y': rot_y, 'rot_x': ROT_X, 'separation': sep,
                       'labels_alpha': la, 'caption': 'REGISTERED  MULTIMODAL  VOLUME'}))

# Phase 5: Hold merged (24fr)
for i in range(24):
    rot_y = ROT_BASE + 0.2 * math.sin(2 * math.pi * (156 + i) / 72)
    fi += 1
    jobs.append((fi, {'rot_y': rot_y, 'rot_x': ROT_X, 'separation': 0.0,
                       'labels_alpha': 0.0, 'caption': 'REGISTERED  MULTIMODAL  VOLUME'}))

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
