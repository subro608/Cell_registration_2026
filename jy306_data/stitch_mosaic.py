"""
Stitch all 21 nd2 tiles into a mosaic using saved tile_positions.npz.
Generates MIP mosaic for visual verification, then full 3D stitched volume.
Finally resamples to isotropic voxels (0.6455 x 0.6455 x 2.0 um).
"""

import numpy as np
import nd2
import cv2
import os

# ============================================================
# Load tile positions
# ============================================================
d = np.load('registration_video/tile_positions.npz', allow_pickle=True)
gp = d['global_pos'].item()
grid = d['grid'].item()

# Shift so all positions are >= 0
min_x = min(p[0] for p in gp.values())
min_y = min(p[1] for p in gp.values())
positions = {k: (x - min_x, y - min_y) for k, (x, y) in gp.items()}

canvas_w = max(x for x, y in positions.values()) + 4200
canvas_h = max(y for x, y in positions.values()) + 4200
print(f"Canvas size: {canvas_w} x {canvas_h}")

# ============================================================
# Step 1: MIP mosaic for visual verification
# ============================================================
print("\n=== Building GFP MIP mosaic ===")
canvas_mip = np.zeros((canvas_h, canvas_w), dtype=np.float64)
weight = np.zeros((canvas_h, canvas_w), dtype=np.float64)

for row_name, tiles in sorted(grid.items()):
    for tile_num in tiles:
        key = f"{row_name}_{tile_num}"
        nd2_path = f"registration_video/{row_name}/{tile_num}.nd2"
        if not os.path.exists(nd2_path):
            print(f"  MISSING: {nd2_path}")
            continue

        print(f"  Loading {key}...", end=" ", flush=True)
        with nd2.ND2File(nd2_path) as f:
            data = f.asarray()  # (12, 2, 4200, 4200)
        gfp = data[:, 1].astype(np.float32)  # GFP channel
        mip = np.max(gfp, axis=0)  # (4200, 4200)

        x0, y0 = positions[key]
        # Linear blend weight: ramp from edges
        h, w = mip.shape
        wy = np.minimum(np.arange(h), np.arange(h-1, -1, -1)).astype(np.float32)
        wx = np.minimum(np.arange(w), np.arange(w-1, -1, -1)).astype(np.float32)
        blend_w = np.outer(wy, wx).clip(1)  # avoid zero weight

        canvas_mip[y0:y0+h, x0:x0+w] += mip * blend_w
        weight[y0:y0+h, x0:x0+w] += blend_w
        print(f"placed at ({x0}, {y0})")

# Normalize
mask = weight > 0
canvas_mip[mask] /= weight[mask]

# Save MIP
out_dir = 'png_exports/overlays'
os.makedirs(out_dir, exist_ok=True)

# Normalize to 8-bit
p2, p98 = np.percentile(canvas_mip[canvas_mip > 0], [2, 99.5])
mip_8 = np.clip((canvas_mip - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
cv2.imwrite(f'{out_dir}/tile_mosaic_stitched_mip.png', mip_8)
print(f"\nSaved MIP mosaic: {out_dir}/tile_mosaic_stitched_mip.png")
print(f"  Size: {mip_8.shape[1]} x {mip_8.shape[0]}")

# Also save a downscaled version for quick viewing
scale = 0.25
small = cv2.resize(mip_8, (int(mip_8.shape[1]*scale), int(mip_8.shape[0]*scale)))
cv2.imwrite(f'{out_dir}/tile_mosaic_stitched_mip_quarter.png', small)
print(f"  Quarter-size: {small.shape[1]} x {small.shape[0]}")

print("\n=== MIP mosaic done. Check the output before proceeding to full 3D stitching. ===")
