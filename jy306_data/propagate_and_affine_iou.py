"""
Propagate all landmarks to IOU-only stitched space, then compute 3D affine.

1. Load all landmark files (landmarks_nd2_native_*.npz)
2. Map ex-vivo nd2 coords through cumulative IOU → stitched 1µm iso (µm)
3. Convert in-vivo JY306 s80 coords to µm
4. Fit 3D affine (least-squares): in-vivo µm → stitched ex-vivo µm
5. Save all results

Usage:
    python3 propagate_and_affine_iou.py
"""

import numpy as np
import cv2
import os
import json
import glob

BASE = '/Users/neurolab/neuroinformatics/margaret'

NATIVE_XY_UM = 0.645
NATIVE_Z_UM = 2.0

# In-vivo JY306 s80 pixel sizes
IV_XY_UM = 0.6835
IV_Z_UM = 3.0

# ============================================================
# Load stitch params
# ============================================================
print("Loading IOU-only stitch params...")
with open(f'{BASE}/registration_video/stitch_iou_only_params.json') as f:
    params = json.load(f)

TILE_ORDER = params['tile_order']
tile_z_offsets = {k: int(v) for k, v in params['tile_z_offsets'].items()}
cum_iou = {}
for k, v in params['cumulative_iou'].items():
    cum_iou[k] = np.array(v)

# ============================================================
# Discover landmark files
# ============================================================
lm_files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz'))
legacy = f'{BASE}/registration_video/landmarks_27_nd2_native.npz'
if os.path.exists(legacy):
    lm_files.append(legacy)

tile_to_idx = {k: i for i, k in enumerate(TILE_ORDER)}
print(f"Found {len(lm_files)} landmark files")

# ============================================================
# Propagate all landmarks
# ============================================================
all_ev_stitched = []   # ex-vivo in stitched 1µm iso (µm)
all_iv_um = []         # in-vivo in µm
all_tiles = []
all_cell_ids = []

for lm_file in lm_files:
    bn = os.path.basename(lm_file)
    if 'landmarks_27_nd2_native' in bn:
        tile = 'row2_1'
    else:
        tile = bn.replace('landmarks_nd2_native_', '').replace('.npz', '')

    if tile not in tile_to_idx:
        print(f"  SKIP {tile}: not in tile order")
        continue

    d = np.load(lm_file)
    ev_nd2 = d['ev_nd2']           # (N, 3) = (col, row, z_merc) in nd2 pixels
    pcd_iv = d['pcd_invivo_jy306']  # (N, 3) = (z, y, x) in JY306 pixels
    N = ev_nd2.shape[0]

    # Find best nd2 z per cell
    img_dir = f'{BASE}/png_exports/registration_video/{tile}'
    nd2_slices = []
    for zi in range(12):
        img = cv2.imread(f'{img_dir}/GFP_z{zi:03d}.png', cv2.IMREAD_UNCHANGED)
        if img is None:
            nd2_slices.append(np.zeros((4200, 4200), dtype=np.float32))
        else:
            nd2_slices.append(img.astype(np.float32))
    nd2_slices = np.array(nd2_slices)
    H_nd2 = nd2_slices.shape[1]

    for i in range(N):
        x_nd2 = ev_nd2[i, 0]
        y_nd2 = ev_nd2[i, 1]

        # Best z in nd2
        c = int(round(np.clip(x_nd2, 10, H_nd2 - 11)))
        r = int(round(np.clip(y_nd2, 10, H_nd2 - 11)))
        intensities = [nd2_slices[z][r-10:r+10, c-10:c+10].mean() for z in range(12)]
        z_nd2 = np.argmax(intensities)

        # Propagate through cumulative IOU → stitched 1µm iso
        M = cum_iou[tile]
        p = M @ np.array([x_nd2, y_nd2, 1.0])
        x_canvas, y_canvas = p[0], p[1]

        x_iso = x_canvas * NATIVE_XY_UM  # µm
        y_iso = y_canvas * NATIVE_XY_UM
        z_iso = (tile_z_offsets[tile] + z_nd2) * NATIVE_Z_UM

        all_ev_stitched.append([x_iso, y_iso, z_iso])

        # In-vivo: JY306 s80 pixel → µm
        iv_z, iv_y, iv_x = pcd_iv[i]
        all_iv_um.append([iv_x * IV_XY_UM, iv_y * IV_XY_UM, iv_z * IV_Z_UM])

        all_tiles.append(tile)
        all_cell_ids.append(i)

    print(f"  {tile}: {N} cells")
    del nd2_slices

all_ev_stitched = np.array(all_ev_stitched)  # (N_total, 3) x,y,z in µm
all_iv_um = np.array(all_iv_um)              # (N_total, 3) x,y,z in µm
N_total = len(all_ev_stitched)

print(f"\nTotal landmarks: {N_total}")
print(f"  Ex-vivo stitched range: x=[{all_ev_stitched[:,0].min():.0f}, {all_ev_stitched[:,0].max():.0f}] "
      f"y=[{all_ev_stitched[:,1].min():.0f}, {all_ev_stitched[:,1].max():.0f}] "
      f"z=[{all_ev_stitched[:,2].min():.0f}, {all_ev_stitched[:,2].max():.0f}] µm")
print(f"  In-vivo range: x=[{all_iv_um[:,0].min():.0f}, {all_iv_um[:,0].max():.0f}] "
      f"y=[{all_iv_um[:,1].min():.0f}, {all_iv_um[:,1].max():.0f}] "
      f"z=[{all_iv_um[:,2].min():.0f}, {all_iv_um[:,2].max():.0f}] µm")

# ============================================================
# Compute 3D affine: in-vivo → stitched ex-vivo
# ============================================================
# Least squares: A @ [x, y, z, 1]^T = [x', y', z']^T
# A is 3x4, source = in-vivo, target = stitched ex-vivo
print("\nFitting 3D affine (in-vivo → stitched ex-vivo)...")

src = all_iv_um        # (N, 3)
tgt = all_ev_stitched  # (N, 3)

# Build design matrix: [x, y, z, 1]
ones = np.ones((N_total, 1))
src_h = np.hstack([src, ones])  # (N, 4)

# Solve: tgt = src_h @ A^T  →  A^T = pinv(src_h) @ tgt
A_T, residuals, rank, sv = np.linalg.lstsq(src_h, tgt, rcond=None)
A = A_T.T  # (3, 4)

print(f"  Affine matrix (3x4):")
for row in A:
    print(f"    [{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} | {row[3]:10.2f}]")

# Evaluate fit
predicted = src_h @ A_T
errors = np.sqrt(np.sum((predicted - tgt) ** 2, axis=1))
print(f"\n  Reprojection error (µm):")
print(f"    Mean: {errors.mean():.2f}")
print(f"    Median: {np.median(errors):.2f}")
print(f"    Max: {errors.max():.2f}")
print(f"    Std: {errors.std():.2f}")

# Decompose affine into scale, rotation, translation
R = A[:, :3]
t = A[:, 3]
U, S, Vt = np.linalg.svd(R)
print(f"\n  Scale factors (SVD): {S[0]:.4f}, {S[1]:.4f}, {S[2]:.4f}")
print(f"  Translation: ({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) µm")
det = np.linalg.det(R)
print(f"  det(R) = {det:.4f} ({'proper' if det > 0 else 'improper (reflection)'})")

# ============================================================
# Save results
# ============================================================
out_path = f'{BASE}/registration_video/affine_3d_iou_results.npz'
np.savez(out_path,
         ev_stitched_um=all_ev_stitched,
         iv_um=all_iv_um,
         affine_3x4=A,
         predicted_ev=predicted,
         errors=errors,
         tiles=np.array(all_tiles),
         cell_ids=np.array(all_cell_ids))
print(f"\nSaved: {out_path}")
print("Done!")
