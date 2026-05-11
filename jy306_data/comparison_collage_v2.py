"""
Row-wise comparison collage: IOU Rigid | Farneback v3 | Elastix B-spline
Grouped by tile row. 500x500 thumbnails, green/magenta overlay.
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

ROW_GROUPS = {
    'Row 1':          [('row1_1','row1_2'), ('row1_2','row1_3')],
    'Row 1to2 cross': [('row1_3','row2_1')],
    'Row 2':          [('row2_1','row2_2'), ('row2_2','row2_3'), ('row2_3','row2_4'), ('row2_4','row2_5')],
    'Row 2to3 cross': [('row2_5','row3_1')],
    'Row 3':          [('row3_1','row3_2'), ('row3_2','row3_3'), ('row3_3','row3_4'), ('row3_4','row3_5'), ('row3_5','row3_6')],
    'Row 3to4 cross': [('row3_6','row4_1')],
    'Row 4':          [('row4_1','row4_2'), ('row4_2','row4_3'), ('row4_3','row4_4'), ('row4_4','row4_5'), ('row4_5','row4_6')],
    'Row 4to5 cross': [('row4_6','row5_1')],
}

METHODS = ['IOU Rigid', 'Farneback v3', 'Elastix B-spline']
CELL_W, CELL_H = 500, 500
LABEL_H, GAP = 50, 8

# Farneback v3 params (moderate, used across all pairs)
FB_DS = 2
FB_WINSIZE = 76
FB_BLUR_K = 13
FB_FSK = 65
FB_LEVELS = 5
FB_ITER = 15

# Elastix params
ELX_DS = 2
ELX_GRID = 32
ELX_RES = 4
ELX_ITER = 500

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

def apply_flow(img, flow):
    h, w = img.shape[:2]
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    return cv2.remap(img, xs + flow[:,:,0], ys + flow[:,:,1], cv2.INTER_LINEAR)

def make_overlay(ref_n, mov_n, mask):
    h, w = ref_n.shape
    r = ref_n * mask.astype(np.float32)
    m = mov_n * mask.astype(np.float32)
    ov = np.zeros((h, w, 3), dtype=np.float32)
    ov[:,:,0] = m; ov[:,:,1] = r; ov[:,:,2] = m
    return (ov * 255).astype(np.uint8)

def compute_ncc(a, b, mask):
    a_m = a[mask].astype(np.float64); b_m = b[mask].astype(np.float64)
    if len(a_m) == 0: return 0.0
    a_m -= a_m.mean(); b_m -= b_m.mean()
    denom = np.sqrt(np.sum(a_m**2) * np.sum(b_m**2))
    return float(np.sum(a_m * b_m) / denom) if denom > 0 else 0.0

def run_farneback(ref, mov_rigid, h, w):
    ref_8 = normalize_8bit(ref)
    mov_8 = normalize_8bit(mov_rigid)
    sh, sw = h // FB_DS, w // FB_DS
    ref_s = cv2.GaussianBlur(cv2.resize(ref_8, (sw, sh)), (FB_BLUR_K, FB_BLUR_K), 0)
    mov_s = cv2.GaussianBlur(cv2.resize(mov_8, (sw, sh)), (FB_BLUR_K, FB_BLUR_K), 0)
    flow_small = cv2.calcOpticalFlowFarneback(
        ref_s, mov_s, flow=None, pyr_scale=0.5, levels=FB_LEVELS,
        winsize=FB_WINSIZE, iterations=FB_ITER, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[:,:,0] = cv2.resize(flow_small[:,:,0], (w, h)) * FB_DS
    flow[:,:,1] = cv2.resize(flow_small[:,:,1], (w, h)) * FB_DS
    flow[:,:,0] = cv2.GaussianBlur(flow[:,:,0], (FB_FSK, FB_FSK), 0)
    flow[:,:,1] = cv2.GaussianBlur(flow[:,:,1], (FB_FSK, FB_FSK), 0)
    return apply_flow(mov_rigid, flow)

def run_elastix(ref, mov_rigid, h, w):
    sh, sw = h // ELX_DS, w // ELX_DS
    ref_ds = cv2.resize(normalize_8bit(ref), (sw, sh)).astype(np.float32)
    mov_ds = cv2.resize(normalize_8bit(mov_rigid), (sw, sh)).astype(np.float32)
    ref_itk = sitk.GetImageFromArray(ref_ds)
    mov_itk = sitk.GetImageFromArray(mov_ds)
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(ref_itk)
    elastix.SetMovingImage(mov_itk)
    pm = sitk.GetDefaultParameterMap('bspline')
    pm['NumberOfResolutions'] = [str(ELX_RES)]
    pm['MaximumNumberOfIterations'] = [str(ELX_ITER)]
    pm['Metric'] = ['AdvancedNormalizedCorrelation']
    pm['FinalGridSpacingInPhysicalUnits'] = [str(ELX_GRID)]
    pm['BSplineInterpolationOrder'] = ['3']
    pm['FinalBSplineInterpolationOrder'] = ['3']
    pm['WriteResultImage'] = ['false']
    pm['NumberOfSpatialSamples'] = ['4096']
    pm['NewSamplesEveryIteration'] = ['true']
    elastix.SetParameterMap(pm)
    elastix.LogToConsoleOff()
    elastix.Execute()
    result = sitk.GetArrayFromImage(elastix.GetResultImage()).astype(np.float32)
    return cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

# ============================================================
print("Loading data...")
masks_data = np.load(f'{BASE}/registration_video/via_masks_v2.npz')
masks = {k: masks_data[k] for k in masks_data.files}

with open(f'{BASE}/registration_video/auto_align_transforms_iou.json', 'r') as f:
    iou_transforms = json.load(f)

print("Loading boundary slices...")
first_slices, last_slices = {}, {}
for key in TILE_ORDER:
    print(f"  {key}...", end=" ", flush=True)
    with nd2.ND2File(nd2_path(key)) as f:
        data = f.asarray()
    mask = masks.get(key, np.ones((4200,4200), dtype=np.uint8))
    first_slices[key] = data[0,1].astype(np.float32) * mask.astype(np.float32)
    last_slices[key]  = data[-1,1].astype(np.float32) * mask.astype(np.float32)
    print("done")

out_dir = f'{BASE}/png_exports/z_stitch_comparison_v2'
os.makedirs(out_dir, exist_ok=True)

for group_name, group_pairs in ROW_GROUPS.items():
    print(f"\n=== {group_name} ({len(group_pairs)} pairs) ===")
    n_methods = len(METHODS)
    n_pairs = len(group_pairs)
    sheet_w = n_methods * (CELL_W + GAP) + GAP
    sheet_h = n_pairs * (CELL_H + LABEL_H + GAP) + GAP + 60
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    # Header
    colors = [(200,255,200), (200,255,255), (255,230,150)]
    for mi, (mname, col) in enumerate(zip(METHODS, colors)):
        x0 = GAP + mi * (CELL_W + GAP)
        cv2.putText(sheet, mname, (x0+10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)

    for pi, (key_a, key_b) in enumerate(group_pairs):
        pair_key = f'{key_a}_to_{key_b}'
        print(f"  {pair_key}...", end=" ", flush=True)

        ref = last_slices[key_a]
        mov = first_slices[key_b]
        h, w = ref.shape

        mask_a = masks.get(key_a, np.ones((h,w), dtype=np.uint8))
        mask_b = masks.get(key_b, np.ones((h,w), dtype=np.uint8))

        # IOU rigid
        tfm = iou_transforms[pair_key]
        warp = np.array(tfm['warp_matrix'], dtype=np.float32)
        mov_rigid = cv2.warpAffine(mov, warp, (w,h), flags=cv2.INTER_LINEAR)
        mask_mov_w = cv2.warpAffine(mask_b, warp, (w,h), flags=cv2.INTER_NEAREST)
        mask_both = (mask_a > 0) & (mask_mov_w > 0)

        ref_n = normalize_float(ref)
        mov_rigid_n = normalize_float(mov_rigid)
        ncc_rigid = compute_ncc(ref_n, mov_rigid_n, mask_both)

        # Farneback
        mov_fb = run_farneback(ref, mov_rigid, h, w)
        mov_fb_n = normalize_float(mov_fb)
        ncc_fb = compute_ncc(ref_n, mov_fb_n, mask_both)

        # Elastix
        mov_elx = run_elastix(ref, mov_rigid, h, w)
        mov_elx_n = normalize_float(mov_elx)
        ncc_elx = compute_ncc(ref_n, mov_elx_n, mask_both)

        overlays = [
            make_overlay(ref_n, mov_rigid_n, mask_both),
            make_overlay(ref_n, mov_fb_n, mask_both),
            make_overlay(ref_n, mov_elx_n, mask_both),
        ]
        nccs = [ncc_rigid, ncc_fb, ncc_elx]

        y0 = 60 + GAP + pi * (CELL_H + LABEL_H + GAP)
        for mi, (ov, ncc) in enumerate(zip(overlays, nccs)):
            x0 = GAP + mi * (CELL_W + GAP)
            thumb = cv2.resize(ov, (CELL_W, CELL_H))
            sheet[y0:y0+CELL_H, x0:x0+CELL_W] = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
            cv2.putText(sheet, f'NCC={ncc:.4f}', (x0+5, y0+CELL_H-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,100), 1, cv2.LINE_AA)

        # Pair label
        cv2.putText(sheet, pair_key, (GAP+5, y0+CELL_H+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2, cv2.LINE_AA)
        print(f"rigid={ncc_rigid:.3f} fb={ncc_fb:.3f} elx={ncc_elx:.3f}")

    safe = group_name.replace(' ', '_')
    path = f'{out_dir}/{safe}.png'
    cv2.imwrite(path, sheet)
    print(f"  Saved: {path}")

print(f"\nAll collages saved to: {out_dir}/")
