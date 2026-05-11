"""
Fix scene5b intro transition.
Replace frames 1-72 (A0 hold + A0 mask + start of Phase A) with smooth intermediate frames:
  Step 1 (18fr): Fade out yellow circles, scale bar, caption from scene5 last frame
  Step 2 (30fr): Slide tile from right-half position to center
  Step 3 (12fr): Hold at center (flat), then scene5b Phase A continues from frame 73+

Reads scene5's last frame, generates clean intermediate assets, writes directly to frames_multi_tile_3d.
"""
import numpy as np, cv2, math, os, pickle

BASE = '/Users/neurolab/neuroinformatics/margaret'
FRAMES_DIR = f'{BASE}/animation/frames_multi_tile_3d'
W, H = 1920, 1080

def ease(t):
    return float(0.5 - 0.5 * math.cos(math.pi * max(0., min(1., t))))

# Load scene5 last frame
import subprocess
tmp_img = '/tmp/_s5b_last_frame.png'
subprocess.run(['ffmpeg', '-y', '-sseof', '-0.1', '-i',
                f'{BASE}/animation/scene5_all_tiles_v2_h264.mp4',
                '-frames:v', '1', tmp_img], capture_output=True)
s5_last = cv2.imread(tmp_img)
print(f"Scene5 last frame: {s5_last.shape}")

# Also get 2nd-to-last frame (no yellow circles, just the overlay)
subprocess.run(['ffmpeg', '-y', '-sseof', '-2.5', '-i',
                f'{BASE}/animation/scene5_all_tiles_v2_h264.mp4',
                '-frames:v', '1', '/tmp/_s5b_clean.png'], capture_output=True)
s5_clean = cv2.imread('/tmp/_s5b_clean.png')
print(f"Scene5 clean frame (no circles): {s5_clean.shape}")

# Load scene5b assets to render row5_1 tile
with open(f'{BASE}/animation/scene5b_assets_v3.pkl', 'rb') as f:
    assets = pickle.load(f)

a = assets['row5_1']
dense = a['dense']
dense_z = a['dense_z']
center_z = a['center_z']
cell_w, cell_h = a['cell_w'], a['cell_h']
FULL_SCALE = 778 / cell_h  # from main script

def render_tile_flat(cx, cy, scale):
    """Render row5_1 as flat (rot=0) single mid-slice at given position/scale."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    # Use middle slice
    mid = len(dense) // 2
    sl = dense[mid]
    sh, sw = sl.shape[:2]
    new_w = int(sw * scale)
    new_h = int(sh * scale)
    resized = cv2.resize(sl, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    x0 = cx - new_w // 2
    y0 = cy - new_h // 2
    # Clip to canvas
    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(W, x0 + new_w)
    dst_y1 = min(H, y0 + new_h)
    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = resized[src_y0:src_y0+(dst_y1-dst_y0),
                                                          src_x0:src_x0+(dst_x1-dst_x0)]
    return canvas

def render_tile_3d(cx, cy, scale, rot_x):
    """Render row5_1 as 3D stack with x-tilt at given position."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    cos_x, sin_x = math.cos(rot_x), math.sin(rot_x)

    z_depths = []
    for i in range(len(dense)):
        dz = dense_z[i] - center_z
        rz2 = cos_x * dz
        z_depths.append((rz2, i, dz))
    z_depths.sort(key=lambda x: x[0])

    for depth, i, dz in z_depths:
        sl = dense[i].astype(np.float32) / 255.0
        sh, sw = sl.shape[:2]
        hw, hh = sw * scale / 2, sh * scale / 2

        corners_3d = np.array([
            [-hw, -hh, dz * scale], [hw, -hh, dz * scale],
            [hw, hh, dz * scale], [-hw, hh, dz * scale]
        ], dtype=np.float64)

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
        canvas = np.where(mask3, np.maximum(canvas, warped * 0.7), canvas)

    return np.clip(canvas * 255, 0, 255).astype(np.uint8)

# Scene5 tile center position (right half)
S5_CX, S5_CY = 1200, 430

# ── Step 1 (18 frames): Fade out annotations ──
# Blend from s5_last (with circles/scalebar) to s5_clean (without)
print("Step 1: Fade out annotations...")
for fi in range(18):
    t = ease(fi / 17)
    frame = cv2.addWeighted(s5_last, 1 - t, s5_clean, t, 0)
    path = f'{FRAMES_DIR}/frame_{fi+1:05d}.png'
    cv2.imwrite(path, frame)

# ── Step 2 (30 frames): Slide from right to center + introduce 3D tilt ──
# Blend from scene5's clean frame to scene5b's rendered tile while moving
print("Step 2: Slide to center + 3D tilt...")
for fi in range(30):
    t = ease(fi / 29)
    cx = int(S5_CX * (1 - t) + (W // 2) * t)
    cy = int(S5_CY * (1 - t) + (H // 2) * t)
    rot_x = 0.0 * (1 - t) + (-0.3) * t

    # Render scene5b tile at current position
    if abs(rot_x) < 0.02:
        tile_frame = render_tile_flat(cx, cy, FULL_SCALE)
    else:
        tile_frame = render_tile_3d(cx, cy, FULL_SCALE, rot_x)

    # For early frames, blend from scene5 clean to scene5b rendering
    blend_t = ease(min(1.0, fi / 15))  # fast blend in first half
    frame = cv2.addWeighted(s5_clean, 1 - blend_t, tile_frame, blend_t, 0)

    idx = 18 + fi + 1
    path = f'{FRAMES_DIR}/frame_{idx:05d}.png'
    cv2.imwrite(path, frame)

# ── Step 3 (12 frames): Hold at center with tilt (matches Phase A start) ──
print("Step 3: Hold at center...")
tile_centered = render_tile_3d(W // 2, H // 2, FULL_SCALE, -0.3)
for fi in range(12):
    idx = 48 + fi + 1
    path = f'{FRAMES_DIR}/frame_{idx:05d}.png'
    cv2.imwrite(path, tile_centered)

print(f"\nReplaced frames 1-60 (A0 transition)")
print("Phase A (zoom out to grid) starts at frame 61+")
