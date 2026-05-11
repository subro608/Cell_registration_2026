"""
Run SimpleElastix B-spline at FULL RESOLUTION (DS=1, 4200x4200) for all 20 pairs.
Saves TransformParameters for point propagation.

Uses /usr/bin/python3 for SimpleITK.

Usage:
    /usr/bin/python3 elastix_fullres_v5.py
"""

import numpy as np
import cv2
import os
import json
import time
import SimpleITK as sitk

BASE = '/Users/neurolab/neuroinformatics/margaret'

TILE_ORDER = [
    'row1_1', 'row1_2', 'row1_3',
    'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
    'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
    'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
    'row5_1',
]

# Elastix params — full resolution
GRID_SPACING = 64       # larger grid at full res (equivalent to 32 at DS=2)
N_RESOLUTIONS = 4
MAX_ITER = 1000
METRIC = 'AdvancedNormalizedCorrelation'

OUT_DIR = f'{BASE}/png_exports/z_stitch_elastix_fullres_v5'
os.makedirs(OUT_DIR, exist_ok=True)

def normalize_float(img):
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.float32)
    p2, p995 = np.percentile(vals, [2, 99.5])
    return np.clip((img - p2) / max(p995 - p2, 1), 0, 1).astype(np.float32)

def compute_ncc(a, b, mask):
    a_m = a[mask].astype(np.float64)
    b_m = b[mask].astype(np.float64)
    if len(a_m) == 0: return 0.0
    a_m -= a_m.mean(); b_m -= b_m.mean()
    denom = np.sqrt(np.sum(a_m**2) * np.sum(b_m**2))
    return float(np.sum(a_m * b_m) / denom) if denom > 0 else 0.0

# ============================================================
print("Loading masks and IOU transforms...")
masks_data = np.load(f'{BASE}/registration_video/via_masks_v4.npz')
masks = {k: masks_data[k] for k in masks_data.files}

with open(f'{BASE}/registration_video/auto_align_transforms_iou_v4.json', 'r') as f:
    iou_transforms = json.load(f)

all_pairs = [(TILE_ORDER[i], TILE_ORDER[i+1]) for i in range(len(TILE_ORDER)-1)]

# Check which pairs are already done
pairs_todo = []
for a, b in all_pairs:
    pair_key = f'{a}_to_{b}'
    pair_dir = f'{OUT_DIR}/{pair_key}'
    if os.path.exists(f'{pair_dir}/TransformParameters.0.txt'):
        print(f"  SKIP {pair_key} (already done)")
    else:
        pairs_todo.append((a, b))

print(f"\n{len(pairs_todo)} pairs to process, {20 - len(pairs_todo)} already done\n")

# Load boundary slices
print("Loading boundary slices...")
needed_keys = set(k for a, b in pairs_todo for k in (a, b))
first_slices, last_slices = {}, {}
for key in TILE_ORDER:
    if key not in needed_keys:
        continue
    img_dir = f'{BASE}/png_exports/registration_video/{key}'
    first_frame = cv2.imread(f'{img_dir}/GFP_z000.png', cv2.IMREAD_GRAYSCALE)
    last_frame  = cv2.imread(f'{img_dir}/GFP_z011.png', cv2.IMREAD_GRAYSCALE)
    if first_frame is None or last_frame is None:
        print(f"  WARNING: missing PNGs for {key}")
        continue
    mask = masks.get(key, np.ones((4200, 4200), dtype=np.uint8))
    first_slices[key] = first_frame.astype(np.float32) * mask.astype(np.float32)
    last_slices[key]  = last_frame.astype(np.float32) * mask.astype(np.float32)
    print(f"  {key} loaded")

results = {}

for idx, (key_a, key_b) in enumerate(pairs_todo):
    pair_key = f'{key_a}_to_{key_b}'
    is_cross = key_a.split('_')[0] != key_b.split('_')[0]
    tag = ' [CROSS]' if is_cross else ''
    print(f"\n=== [{idx+1}/{len(pairs_todo)}] {pair_key}{tag} ===")

    ref = last_slices[key_a]
    mov = first_slices[key_b]
    h, w = ref.shape

    mask_a = masks.get(key_a, np.ones((h, w), dtype=np.uint8))
    mask_b = masks.get(key_b, np.ones((h, w), dtype=np.uint8))

    # Apply IOU rigid at full resolution
    tfm = iou_transforms[pair_key]
    warp = np.array(tfm['warp_matrix'], dtype=np.float32)
    mov_rigid = cv2.warpAffine(mov, warp, (w, h), flags=cv2.INTER_LINEAR)
    mask_mov_w = cv2.warpAffine(mask_b, warp, (w, h), flags=cv2.INTER_NEAREST)
    mask_both = (mask_a > 0) & (mask_mov_w > 0)

    ref_n = normalize_float(ref)
    mov_rigid_n = normalize_float(mov_rigid)
    ncc_rigid = compute_ncc(ref_n, mov_rigid_n, mask_both)
    print(f"  Rigid NCC: {ncc_rigid:.4f}")

    # Normalize to uint8 for elastix (full resolution, no DS)
    ref_u8 = (np.clip(ref_n, 0, 1) * 255).astype(np.uint8).astype(np.float32)
    mov_u8 = (np.clip(mov_rigid_n, 0, 1) * 255).astype(np.uint8).astype(np.float32)

    # Run elastix at full resolution
    ref_itk = sitk.GetImageFromArray(ref_u8)
    mov_itk = sitk.GetImageFromArray(mov_u8)

    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(ref_itk)
    elastix.SetMovingImage(mov_itk)

    pm = sitk.GetDefaultParameterMap('bspline')
    pm['NumberOfResolutions'] = [str(N_RESOLUTIONS)]
    pm['MaximumNumberOfIterations'] = [str(MAX_ITER)]
    pm['Metric'] = [METRIC]
    pm['FinalGridSpacingInPhysicalUnits'] = [str(GRID_SPACING)]
    pm['BSplineInterpolationOrder'] = ['3']
    pm['FinalBSplineInterpolationOrder'] = ['3']
    pm['WriteResultImage'] = ['false']
    pm['NumberOfSpatialSamples'] = ['8192']  # more samples at full res
    pm['NewSamplesEveryIteration'] = ['true']
    elastix.SetParameterMap(pm)
    elastix.LogToConsoleOff()

    t0 = time.time()
    print(f"  Running elastix (fullres {w}x{h}, grid={GRID_SPACING}, iter={MAX_ITER})...", flush=True)
    try:
        elastix.Execute()
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s")

        # Save transform
        pair_dir = f'{OUT_DIR}/{pair_key}'
        os.makedirs(pair_dir, exist_ok=True)
        tmap = elastix.GetTransformParameterMap()
        for i, t in enumerate(tmap):
            sitk.WriteParameterFile(t, f'{pair_dir}/TransformParameters.{i}.txt')

        # Compute NCC of result
        result = sitk.GetArrayFromImage(elastix.GetResultImage()).astype(np.float32)
        result_n = normalize_float(result)
        ncc_elx = compute_ncc(ref_n, result_n, mask_both)
        print(f"  Elastix NCC: {ncc_elx:.4f} (rigid was {ncc_rigid:.4f}, delta={ncc_elx-ncc_rigid:+.4f})")

        results[pair_key] = {
            'ncc_rigid': float(ncc_rigid),
            'ncc_elastix': float(ncc_elx),
            'time_s': float(dt),
        }

        # Save overlay
        ov = np.zeros((h, w, 3), dtype=np.uint8)
        ov[:, :, 1] = (ref_n * mask_both * 255).astype(np.uint8)
        ov[:, :, 0] = ov[:, :, 2] = (result_n * mask_both * 255).astype(np.uint8)
        ov_small = cv2.resize(ov, (w // 4, h // 4))
        cv2.imwrite(f'{pair_dir}/overlay.png', ov_small)

    except Exception as e:
        print(f"  FAILED: {e}")
        results[pair_key] = {'ncc_rigid': float(ncc_rigid), 'error': str(e)}

    # Save results after each pair
    with open(f'{OUT_DIR}/results_fullres_v5.json', 'w') as f:
        json.dump(results, f, indent=2)

print(f"\nDone! Results saved to {OUT_DIR}/results_fullres_v5.json")
