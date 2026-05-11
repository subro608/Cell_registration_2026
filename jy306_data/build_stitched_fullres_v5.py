"""
Build full-resolution stitched volume using IOU rigid + full-res elastix (v5).

Pipeline per tile:
  1. Load 12 GFP z-slices (4200x4200)
  2. Apply PAIR IOU rigid warp (align to previous tile, 4200x4200)
  3. Apply PAIR elastix B-spline (align to previous tile, 4200x4200, via transformix)
  4. Apply CUMULATIVE transform (previous tile's accumulated transform) to place on canvas
  5. Stack in z with row1_4 gap
  6. Resample to 1µm isotropic

Uses /usr/bin/python3 for SimpleITK.

Usage:
    /usr/bin/python3 build_stitched_fullres_v5.py
"""

import numpy as np
import cv2
import os
import json
import time
import tifffile
import SimpleITK as sitk

BASE = '/Users/neurolab/neuroinformatics/margaret'
ELASTIX_DIR = f'{BASE}/png_exports/z_stitch_elastix_fullres_v5'
OUT_DIR = f'{BASE}/registration_video/stitched'

TILE_ORDER = [
    'row1_1', 'row1_2', 'row1_3',
    'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
    'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
    'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
    'row5_1',
]

ROW1_4_INSERT_AFTER = 'row1_3'
ROW1_4_SLICES = 6

NATIVE_XY_UM = 0.645
NATIVE_Z_UM = 2.0
TARGET_UM = 1.0

def affine_2x3_to_3x3(M):
    return np.vstack([M, [0, 0, 1]])

def affine_3x3_to_2x3(M):
    return M[:2, :]

def normalize_u8(img):
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    p2, p995 = np.percentile(vals, [2, 99.5])
    return np.clip((img - p2) / max(p995 - p2, 1) * 255, 0, 255).astype(np.uint8)

# ============================================================
# Load transforms
# ============================================================
print("Loading IOU transforms...")
with open(f'{BASE}/registration_video/auto_align_transforms_iou_v4.json') as f:
    iou_transforms = json.load(f)

masks_data = np.load(f'{BASE}/registration_video/via_masks_v4.npz')
masks = {k: masks_data[k] for k in masks_data.files}

# ============================================================
# Build cumulative transforms
# The cumulative transform for tile K maps from tile K's
# "pair-aligned" space (after IOU+elastix) to the final canvas.
#
# Tile 0: identity (anchor)
# Tile 1: pair warp (IOU+elastix) aligns it to tile 0, then identity places on canvas
# Tile 2: pair warp aligns to tile 1, then tile 1's cumulative places on canvas
#
# Since elastix is a deformation field (not an affine), we can't compose it
# into a single matrix. Instead:
#   Step 1: Apply pair IOU rigid in 4200x4200 space
#   Step 2: Apply pair elastix in 4200x4200 space (via transformix)
#   Step 3: Apply cumulative affine (from previous pairs' IOU rigids) to canvas
#
# The cumulative affine only accumulates the IOU rigid parts.
# Each tile's elastix is applied independently in 4200x4200.
# ============================================================
print("Computing cumulative IOU transforms...")
cum_iou = {TILE_ORDER[0]: np.eye(3)}
for i in range(len(TILE_ORDER) - 1):
    a, b = TILE_ORDER[i], TILE_ORDER[i+1]
    pair_key = f'{a}_to_{b}'
    warp = np.array(iou_transforms[pair_key]['warp_matrix'], dtype=np.float64)
    M = affine_2x3_to_3x3(warp)
    cum_iou[b] = cum_iou[a] @ M

# Compute canvas bounds
corners = np.array([[0,0],[4200,0],[4200,4200],[0,4200]], dtype=np.float64)
all_corners = []
for k in TILE_ORDER:
    for c in corners:
        p = cum_iou[k] @ [c[0], c[1], 1]
        all_corners.append(p[:2])
all_corners = np.array(all_corners)

x_min, y_min = all_corners.min(axis=0)
x_max, y_max = all_corners.max(axis=0)
offset_x, offset_y = -x_min, -y_min
canvas_w = int(np.ceil(x_max - x_min))
canvas_h = int(np.ceil(y_max - y_min))

# Add offset
for k in TILE_ORDER:
    cum_iou[k][0, 2] += offset_x
    cum_iou[k][1, 2] += offset_y

print(f"  Canvas: {canvas_w} x {canvas_h}")

# ============================================================
# Build stitched volume
# ============================================================
n_tiles = len(TILE_ORDER)
total_z = n_tiles * 12 + ROW1_4_SLICES  # 258
mem_gb = total_z * canvas_h * canvas_w * 2 / 1e9
print(f"\nVolume: ({total_z}, {canvas_h}, {canvas_w}) = {mem_gb:.1f} GB")

volume = np.zeros((total_z, canvas_h, canvas_w), dtype=np.uint16)

z_offset = 0
tile_z_offsets = {}

for tile_idx, key in enumerate(TILE_ORDER):
    t0 = time.time()
    print(f"\n  [{tile_idx+1}/{n_tiles}] {key} (z={z_offset})...", flush=True)
    tile_z_offsets[key] = z_offset

    # Load 12 z-slices
    img_dir = f'{BASE}/png_exports/registration_video/{key}'
    slices = []
    for zi in range(12):
        img = cv2.imread(f'{img_dir}/GFP_z{zi:03d}.png', cv2.IMREAD_UNCHANGED)
        if img is None:
            slices.append(np.zeros((4200, 4200), dtype=np.uint16))
        else:
            slices.append(img.astype(np.uint16))

    mask = masks.get(key, np.ones((4200, 4200), dtype=np.uint8))

    for zi in range(12):
        sl = slices[zi].astype(np.float32) * mask.astype(np.float32)

        if tile_idx == 0:
            # First tile: just place with cumulative (identity + offset)
            M_cum = affine_3x3_to_2x3(cum_iou[key])
            warped = cv2.warpAffine(sl, M_cum, (canvas_w, canvas_h),
                                    flags=cv2.INTER_LINEAR, borderValue=0)
        else:
            # Step 1: Apply PAIR IOU rigid in 4200x4200
            prev_key = TILE_ORDER[tile_idx - 1]
            pair_key = f'{prev_key}_to_{key}'
            pair_warp = np.array(iou_transforms[pair_key]['warp_matrix'], dtype=np.float32)
            sl_rigid = cv2.warpAffine(sl, pair_warp, (4200, 4200),
                                      flags=cv2.INTER_LINEAR, borderValue=0)

            # Step 2: Apply PAIR elastix in 4200x4200 (via transformix)
            tfm_file = f'{ELASTIX_DIR}/{pair_key}/TransformParameters.0.txt'
            if os.path.exists(tfm_file):
                # Normalize for transformix (same as what elastix saw)
                sl_n = normalize_u8(sl_rigid).astype(np.float32)
                sl_itk = sitk.GetImageFromArray(sl_n)
                tfm = sitk.ReadParameterFile(tfm_file)
                transformix = sitk.TransformixImageFilter()
                transformix.SetTransformParameterMap(tfm)
                transformix.SetMovingImage(sl_itk)
                transformix.LogToConsoleOff()
                try:
                    transformix.Execute()
                    sl_elx = sitk.GetArrayFromImage(transformix.GetResultImage()).astype(np.float32)
                    # Recover original intensity scale
                    # sl_elx is in normalized [0, 255] space, map back
                    # Actually just use the elastix-warped normalized image
                    # and rescale to original range
                    vals = sl_rigid[sl_rigid > 0]
                    if len(vals) > 0:
                        p2, p995 = np.percentile(vals, [2, 99.5])
                        sl_deformed = sl_elx / 255.0 * (p995 - p2) + p2
                    else:
                        sl_deformed = sl_elx
                except Exception as e:
                    sl_deformed = sl_rigid
            else:
                sl_deformed = sl_rigid

            # Step 3: Apply PREVIOUS tile's cumulative to place on canvas
            # The pair warp aligned this tile to prev's native space.
            # prev's cumulative maps prev's native to canvas.
            M_prev = affine_3x3_to_2x3(cum_iou[prev_key])
            warped = cv2.warpAffine(sl_deformed, M_prev, (canvas_w, canvas_h),
                                    flags=cv2.INTER_LINEAR, borderValue=0)

        volume[z_offset + zi] = np.clip(warped, 0, 65535).astype(np.uint16)

    dt = time.time() - t0
    print(f"    z={z_offset}-{z_offset+11}, {dt:.1f}s")
    z_offset += 12

    if key == ROW1_4_INSERT_AFTER:
        print(f"    Inserting {ROW1_4_SLICES}-slice gap for row1_4")
        z_offset += ROW1_4_SLICES

print(f"\nTotal z: {z_offset}")
volume = volume[:z_offset]

# Save native
native_path = f'{OUT_DIR}/stitched_gfp_fullres_v5.tif'
print(f"\nSaving native volume ({volume.shape})...")
tifffile.imwrite(native_path, volume, bigtiff=True)
print(f"  Saved: {native_path} ({volume.nbytes/1e9:.1f} GB)")

# ============================================================
# Resample to 1µm isotropic (chunked)
# ============================================================
from scipy.ndimage import zoom

z_factor = NATIVE_Z_UM / TARGET_UM
xy_factor = NATIVE_XY_UM / TARGET_UM
nz_actual = z_offset

iso_path = f'{OUT_DIR}/stitched_gfp_fullres_v5_1um_isotropic.tif'
CHUNK_Z = 24
n_chunks = int(np.ceil(nz_actual / CHUNK_Z))
print(f"\nResampling to 1µm isotropic (z_factor={z_factor}, xy_factor={xy_factor})...")

with tifffile.TiffWriter(iso_path, bigtiff=True) as tif:
    for ci in range(n_chunks):
        z0 = ci * CHUNK_Z
        z1 = min(nz_actual, z0 + CHUNK_Z)
        print(f"  Chunk {ci+1}/{n_chunks}: z={z0}-{z1-1}...", end=" ", flush=True)
        chunk = volume[z0:z1].astype(np.float32)
        chunk_iso = zoom(chunk, (z_factor, xy_factor, xy_factor), order=1)
        chunk_iso = np.clip(chunk_iso, 0, 65535).astype(np.uint16)
        for z in range(chunk_iso.shape[0]):
            tif.write(chunk_iso[z])
        print(f"{chunk_iso.shape[0]} slices")

print(f"  Saved: {iso_path}")

del volume

# ============================================================
# Save stitch parameters for landmark propagation
# ============================================================
params = {
    'tile_order': TILE_ORDER,
    'tile_z_offsets': tile_z_offsets,
    'canvas_w': canvas_w,
    'canvas_h': canvas_h,
    'offset_x': float(offset_x),
    'offset_y': float(offset_y),
    'native_xy_um': NATIVE_XY_UM,
    'native_z_um': NATIVE_Z_UM,
    'elastix_dir': ELASTIX_DIR,
}
# Save cumulative transforms
cum_save = {}
for k in TILE_ORDER:
    cum_save[k] = cum_iou[k].tolist()
params['cumulative_iou'] = cum_save

with open(f'{BASE}/registration_video/stitch_v5_params.json', 'w') as f:
    json.dump(params, f, indent=2)
print(f"\nStitch params saved to registration_video/stitch_v5_params.json")

print("\nDone!")
