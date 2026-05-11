"""
Compare SimpleElastix deformable registration vs Farneback on row4_5 -> row4_6.
Uses B-spline deformable registration with mutual information metric.
"""

import numpy as np
import nd2
import cv2
import os
import json
import SimpleITK as sitk

BASE = '/Users/neurolab/neuroinformatics/margaret'

TILE_ORDER = [
    'row1_1', 'row1_2', 'row1_3',
    'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
    'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
    'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
    'row5_1',
]

PAIR_IDX = 18  # row4_5 -> row4_6
key_a = TILE_ORDER[PAIR_IDX]
key_b = TILE_ORDER[PAIR_IDX + 1]
pair_key = f'{key_a}_to_{key_b}'

def nd2_path(key):
    row, tile = key.split('_')
    if row == 'row5':
        return f'{BASE}/registration_video/row5/Row5/{tile}.nd2'
    return f'{BASE}/registration_video/{row}/{tile}.nd2'

def normalize_float(img):
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.float32)
    p2, p995 = np.percentile(vals, [2, 99.5])
    return np.clip((img - p2) / max(p995 - p2, 1), 0, 1).astype(np.float32)

def normalize_8bit(img):
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    p2, p995 = np.percentile(vals, [2, 99.5])
    return np.clip((img - p2) / max(p995 - p2, 1) * 255, 0, 255).astype(np.uint8)

def compute_ncc(a, b, mask):
    a_m = a[mask].astype(np.float64)
    b_m = b[mask].astype(np.float64)
    if len(a_m) == 0:
        return 0.0
    a_m -= a_m.mean(); b_m -= b_m.mean()
    denom = np.sqrt(np.sum(a_m**2) * np.sum(b_m**2))
    return float(np.sum(a_m * b_m) / denom) if denom > 0 else 0.0

def compute_mse(a, b, mask):
    a_m = a[mask].astype(np.float64)
    b_m = b[mask].astype(np.float64)
    return float(np.mean((a_m - b_m)**2)) if len(a_m) > 0 else 0.0

def make_overlay(ref_n, mov_n, mask):
    h, w = ref_n.shape
    r = ref_n * mask.astype(np.float32)
    m = mov_n * mask.astype(np.float32)
    ov = np.zeros((h, w, 3), dtype=np.float32)
    ov[:, :, 0] = m; ov[:, :, 1] = r; ov[:, :, 2] = m
    return (ov * 255).astype(np.uint8)

# ============================================================
print(f"=== SimpleElastix deformable: {pair_key} ===\n")

print("Loading masks and transforms...")
masks_data = np.load(f'{BASE}/registration_video/via_masks_v2.npz')
masks = {k: masks_data[k] for k in masks_data.files}

with open(f'{BASE}/registration_video/auto_align_transforms_iou.json', 'r') as f:
    iou_transforms = json.load(f)

print(f"Loading {key_a}...")
with nd2.ND2File(nd2_path(key_a)) as f:
    data_a = f.asarray()
mask_a = masks.get(key_a, np.ones((4200, 4200), dtype=np.uint8))
ref = data_a[-1, 1].astype(np.float32) * mask_a.astype(np.float32)

print(f"Loading {key_b}...")
with nd2.ND2File(nd2_path(key_b)) as f:
    data_b = f.asarray()
mask_b = masks.get(key_b, np.ones((4200, 4200), dtype=np.uint8))
mov = data_b[0, 1].astype(np.float32) * mask_b.astype(np.float32)

h, w = ref.shape

# Apply IOU rigid
tfm = iou_transforms[pair_key]
warp = np.array(tfm['warp_matrix'], dtype=np.float32)
mov_rigid = cv2.warpAffine(mov, warp, (w, h), flags=cv2.INTER_LINEAR)
mask_mov_w = cv2.warpAffine(mask_b, warp, (w, h), flags=cv2.INTER_NEAREST)
mask_both = (mask_a > 0) & (mask_mov_w > 0)

ref_n = normalize_float(ref)
mov_rigid_n = normalize_float(mov_rigid)

ncc_rigid = compute_ncc(ref_n, mov_rigid_n, mask_both)
mse_rigid = compute_mse(ref_n, mov_rigid_n, mask_both)
print(f"Rigid baseline: NCC={ncc_rigid:.4f}, MSE={mse_rigid:.6f}")

out_dir = f'{BASE}/png_exports/z_stitch_tuning/{pair_key}'
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# Downsample for Elastix (same as Farneback: DS=2)
DS = 2
print(f"\nDownsampling by {DS}x for Elastix (same as Farneback)...")
ref_8 = normalize_8bit(ref)
mov_rigid_8 = normalize_8bit(mov_rigid)
sh, sw = h // DS, w // DS
ref_ds = cv2.resize(ref_8, (sw, sh))
mov_ds = cv2.resize(mov_rigid_8, (sw, sh))
mask_ds = cv2.resize(mask_both.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)

# Convert to SimpleITK
ref_itk = sitk.GetImageFromArray(ref_ds.astype(np.float32))
mov_itk = sitk.GetImageFromArray(mov_ds.astype(np.float32))

# ============================================================
# Run Elastix B-spline deformable registration
print("Running SimpleElastix B-spline registration...")

elastix = sitk.ElastixImageFilter()
elastix.SetFixedImage(ref_itk)
elastix.SetMovingImage(mov_itk)

# Parameter map: B-spline deformable
pm = sitk.GetDefaultParameterMap('bspline')
pm['NumberOfResolutions'] = ['4']
pm['MaximumNumberOfIterations'] = ['500']
pm['Metric'] = ['AdvancedNormalizedCorrelation']
pm['FinalGridSpacingInPhysicalUnits'] = ['32']   # coarse = 64, fine = 16
pm['BSplineInterpolationOrder'] = ['3']
pm['FinalBSplineInterpolationOrder'] = ['3']
pm['WriteResultImage'] = ['false']
pm['NumberOfSpatialSamples'] = ['4096']
pm['NewSamplesEveryIteration'] = ['true']

elastix.SetParameterMap(pm)
elastix.LogToConsoleOff()

elastix.Execute()

result_itk = elastix.GetResultImage()
mov_elx = sitk.GetArrayFromImage(result_itk).astype(np.float32)

# Upscale result back to full resolution
mov_elx_full = cv2.resize(mov_elx, (w, h), interpolation=cv2.INTER_LINEAR)

# Normalize and compute metrics
mov_elx_n = normalize_float(mov_elx_full)
ncc_elx = compute_ncc(ref_n, mov_elx_n, mask_both)
mse_elx = compute_mse(ref_n, mov_elx_n, mask_both)
print(f"Elastix B-spline: NCC={ncc_elx:.4f}, MSE={mse_elx:.6f}")
print(f"NCC improvement: {ncc_elx - ncc_rigid:+.4f}")

# ============================================================
# Load Farneback best for comparison (win=76, fsk=65, blur=13, iter=10)
print("\nRecomputing Farneback best (win=76, fsk=65, blur=13, iter=10)...")
ref_8_f = normalize_8bit(ref)
mov_rigid_8_f = normalize_8bit(mov_rigid)
sh_f, sw_f = h // 2, w // 2
ref_s = cv2.GaussianBlur(cv2.resize(ref_8_f, (sw_f, sh_f)), (13, 13), 0)
mov_s = cv2.GaussianBlur(cv2.resize(mov_rigid_8_f, (sw_f, sh_f)), (13, 13), 0)

flow_small = cv2.calcOpticalFlowFarneback(
    ref_s, mov_s, flow=None, pyr_scale=0.5, levels=5,
    winsize=76, iterations=10, poly_n=7, poly_sigma=1.5,
    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

flow = np.zeros((h, w, 2), dtype=np.float32)
flow[:, :, 0] = cv2.resize(flow_small[:, :, 0], (w, h)) * 2
flow[:, :, 1] = cv2.resize(flow_small[:, :, 1], (w, h)) * 2
flow[:, :, 0] = cv2.GaussianBlur(flow[:, :, 0], (65, 65), 0)
flow[:, :, 1] = cv2.GaussianBlur(flow[:, :, 1], (65, 65), 0)

ys, xs = np.mgrid[:h, :w].astype(np.float32)
mov_far = cv2.remap(mov_rigid, xs + flow[:, :, 0], ys + flow[:, :, 1], cv2.INTER_LINEAR)
mov_far_n = normalize_float(mov_far)
ncc_far = compute_ncc(ref_n, mov_far_n, mask_both)
mse_far = compute_mse(ref_n, mov_far_n, mask_both)
print(f"Farneback best: NCC={ncc_far:.4f}, MSE={mse_far:.6f}")

# ============================================================
# Side-by-side comparison: rigid | farneback | elastix
print("\nGenerating comparison...")
ov_rigid = make_overlay(ref_n, mov_rigid_n, mask_both)
ov_far   = make_overlay(ref_n, mov_far_n, mask_both)
ov_elx   = make_overlay(ref_n, mov_elx_n, mask_both)

gap = 20
combined = np.zeros((h, w * 3 + gap * 2, 3), dtype=np.uint8)
combined[:, :w] = ov_rigid
combined[:, w + gap:w * 2 + gap] = ov_far
combined[:, w * 2 + gap * 2:] = ov_elx

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(combined, f'IOU RIGID (NCC={ncc_rigid:.4f})', (20, 60), font, 1.8, (0, 255, 0), 3, cv2.LINE_AA)
cv2.putText(combined, f'FARNEBACK win=76 (NCC={ncc_far:.4f})', (w + gap + 20, 60), font, 1.8, (0, 255, 255), 3, cv2.LINE_AA)
cv2.putText(combined, f'ELASTIX B-spline (NCC={ncc_elx:.4f})', (w * 2 + gap * 2 + 20, 60), font, 1.8, (255, 200, 0), 3, cv2.LINE_AA)

small = cv2.resize(combined, (combined.shape[1] // 4, combined.shape[0] // 4))
out_path = f'{out_dir}/elastix_vs_farneback.png'
cv2.imwrite(out_path, cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
print(f"Saved: {out_path}")

# Save standalone Elastix overlay
elx_small = cv2.resize(ov_elx, (w // 4, h // 4))
elx_path = f'{out_dir}/elastix_overlay.png'
cv2.imwrite(elx_path, cv2.cvtColor(elx_small, cv2.COLOR_RGB2BGR))
print(f"Saved: {elx_path}")

# Save standalone Farneback overlay
far_small = cv2.resize(ov_far, (w // 4, h // 4))
far_path = f'{out_dir}/farneback_overlay.png'
cv2.imwrite(far_path, cv2.cvtColor(far_small, cv2.COLOR_RGB2BGR))
print(f"Saved: {far_path}")

print(f"\n=== SUMMARY ===")
print(f"  Rigid:      NCC={ncc_rigid:.4f}")
print(f"  Farneback:  NCC={ncc_far:.4f}  (+{ncc_far - ncc_rigid:.4f})")
print(f"  Elastix:    NCC={ncc_elx:.4f}  (+{ncc_elx - ncc_rigid:.4f})")
