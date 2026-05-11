"""
5-column comparison collage TEST (one pair).
Col 1: Raw overlap + annotation contours (no masking)
Col 2: IOU Rigid
Col 3: Farneback v3
Col 4: Elastix 500 iter
Col 5: Elastix 2000 iter
"""
import numpy as np
import cv2
import os
import json
import SimpleITK as sitk

BASE = '/Users/neurolab/neuroinformatics/margaret'
PNG_DIR = f'{BASE}/png_exports/registration_video'
OUT_DIR = f'{BASE}/png_exports/z_stitch_comparison_v5'
os.makedirs(OUT_DIR, exist_ok=True)

# Test pair
TEST_PAIR = ('row3_1', 'row3_2')

# Farneback v3 params
FB_DS = 2; FB_WINSIZE = 76; FB_BLUR_K = 13; FB_FSK = 65; FB_LEVELS = 5; FB_ITER = 15

METHODS = ['Raw + Annotation', 'IOU Rigid', 'Farneback v3', 'Elastix 500iter', 'Elastix 2000iter']
CELL_W, CELL_H = 600, 600
LABEL_H, GAP = 50, 8
HDR_H = 60

def normalize_float(img):
    vals = img[img > 0]
    if len(vals) == 0: return np.zeros_like(img, dtype=np.float32)
    p2, p995 = np.percentile(vals, [2, 99.5])
    return np.clip((img - p2) / max(p995 - p2, 1), 0, 1).astype(np.float32)

def normalize_8bit(img):
    vals = img[img > 0]
    if len(vals) == 0: return np.zeros_like(img, dtype=np.uint8)
    p2, p995 = np.percentile(vals, [2, 99.5])
    return np.clip((img - p2) / max(p995 - p2, 1) * 255, 0, 255).astype(np.uint8)

def compute_ncc(a, b, mask):
    a_m = a[mask].astype(np.float64); b_m = b[mask].astype(np.float64)
    if len(a_m) == 0: return 0.0
    a_m -= a_m.mean(); b_m -= b_m.mean()
    denom = np.sqrt(np.sum(a_m**2) * np.sum(b_m**2))
    return float(np.sum(a_m * b_m) / denom) if denom > 0 else 0.0

def apply_flow(img, flow):
    h, w = img.shape[:2]
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    return cv2.remap(img, xs + flow[:,:,0], ys + flow[:,:,1], cv2.INTER_LINEAR)

MAGENTA_BOOST = 1.0  # increase magenta brightness

def make_overlay_masked(ref_n, mov_n, mask):
    h, w = ref_n.shape
    ov = np.zeros((h, w, 3), dtype=np.float32)
    r = ref_n * mask.astype(np.float32)
    m = np.clip(mov_n * mask.astype(np.float32) * MAGENTA_BOOST, 0, 1)
    ov[:,:,1] = r; ov[:,:,0] = m; ov[:,:,2] = m
    return (ov * 255).astype(np.uint8)

def make_overlay_raw(ref_raw, mov_raw):
    """Raw overlay without masking, full image."""
    ref_n = normalize_float(ref_raw)
    mov_n = normalize_float(mov_raw)
    h, w = ref_n.shape
    ov = np.zeros((h, w, 3), dtype=np.float32)
    ov[:,:,1] = ref_n
    m = np.clip(mov_n * MAGENTA_BOOST, 0, 1)
    ov[:,:,0] = m; ov[:,:,2] = m
    return (ov * 255).astype(np.uint8)

def draw_mask_contour(img, mask, color, thickness=12):
    cnts, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, thickness)
    return img

def run_farneback(ref, mov_rigid):
    h, w = ref.shape
    ref_8 = normalize_8bit(ref); mov_8 = normalize_8bit(mov_rigid)
    sh, sw = h // FB_DS, w // FB_DS
    ref_s = cv2.GaussianBlur(cv2.resize(ref_8, (sw, sh)), (FB_BLUR_K, FB_BLUR_K), 0)
    mov_s = cv2.GaussianBlur(cv2.resize(mov_8, (sw, sh)), (FB_BLUR_K, FB_BLUR_K), 0)
    flow_s = cv2.calcOpticalFlowFarneback(ref_s, mov_s, None, 0.5, FB_LEVELS, FB_WINSIZE, FB_ITER, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[:,:,0] = cv2.GaussianBlur(cv2.resize(flow_s[:,:,0], (w, h)) * FB_DS, (FB_FSK, FB_FSK), 0)
    flow[:,:,1] = cv2.GaussianBlur(cv2.resize(flow_s[:,:,1], (w, h)) * FB_DS, (FB_FSK, FB_FSK), 0)
    return apply_flow(mov_rigid, flow)

def run_elastix(ref, mov_rigid, n_iter):
    h, w = ref.shape
    sh, sw = h // 2, w // 2
    ref_ds = cv2.resize(normalize_8bit(ref), (sw, sh)).astype(np.float32)
    mov_ds = cv2.resize(normalize_8bit(mov_rigid), (sw, sh)).astype(np.float32)
    ref_itk = sitk.GetImageFromArray(ref_ds)
    mov_itk = sitk.GetImageFromArray(mov_ds)
    el = sitk.ElastixImageFilter()
    el.SetFixedImage(ref_itk); el.SetMovingImage(mov_itk)
    pm = sitk.GetDefaultParameterMap('bspline')
    pm['NumberOfResolutions'] = ['4']
    pm['MaximumNumberOfIterations'] = [str(n_iter)]
    pm['Metric'] = ['AdvancedNormalizedCorrelation']
    pm['FinalGridSpacingInPhysicalUnits'] = ['32']
    pm['BSplineInterpolationOrder'] = ['3']
    pm['FinalBSplineInterpolationOrder'] = ['3']
    pm['WriteResultImage'] = ['false']
    pm['NumberOfSpatialSamples'] = ['4096']
    pm['NewSamplesEveryIteration'] = ['true']
    el.SetParameterMap(pm); el.LogToConsoleOff(); el.Execute()
    result = sitk.GetArrayFromImage(el.GetResultImage()).astype(np.float32)
    return cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

# ---- Load data ----
print("Loading masks and transforms...")
masks_data = np.load(f'{BASE}/registration_video/via_masks_v4.npz')
masks = {k: masks_data[k] for k in masks_data.files}
with open(f'{BASE}/registration_video/auto_align_transforms_iou_v4.json') as f:
    iou_transforms = json.load(f)

key_a, key_b = TEST_PAIR
pair_key = f'{key_a}_to_{key_b}'
print(f"\nProcessing {pair_key}...")

ref_raw = cv2.imread(f'{PNG_DIR}/{key_a}/GFP_z011.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
mov_raw = cv2.imread(f'{PNG_DIR}/{key_b}/GFP_z000.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
h, w = ref_raw.shape

mask_a = masks.get(key_a, np.ones((h,w), dtype=np.uint8))
mask_b = masks.get(key_b, np.ones((h,w), dtype=np.uint8))

# IOU rigid
tfm = iou_transforms[pair_key]
warp = np.array(tfm['warp_matrix'], dtype=np.float32)
mov_rigid = cv2.warpAffine(mov_raw, warp, (w,h), flags=cv2.INTER_LINEAR)
mask_b_w  = cv2.warpAffine(mask_b, warp, (w,h), flags=cv2.INTER_NEAREST)
mask_both = (mask_a > 0) & (mask_b_w > 0)

ref_n       = normalize_float(ref_raw * mask_a)
mov_rigid_n = normalize_float(mov_rigid * mask_b_w)
ncc_rigid   = compute_ncc(ref_n, mov_rigid_n, mask_both)
print(f"  IOU rigid NCC={ncc_rigid:.4f}")

# Farneback
print("  Running Farneback...")
mov_fb   = run_farneback(ref_raw * mask_a, mov_rigid * mask_b_w)
mov_fb_n = normalize_float(mov_fb)
ncc_fb   = compute_ncc(ref_n, mov_fb_n, mask_both)
print(f"  Farneback NCC={ncc_fb:.4f}")

# Elastix 500
print("  Running Elastix 500 iter...")
mov_e500   = run_elastix(ref_raw * mask_a, mov_rigid * mask_b_w, 500)
mov_e500_n = normalize_float(mov_e500)
ncc_e500   = compute_ncc(ref_n, mov_e500_n, mask_both)
print(f"  Elastix 500 NCC={ncc_e500:.4f}")

# Elastix 2000
print("  Running Elastix 2000 iter...")
mov_e2000   = run_elastix(ref_raw * mask_a, mov_rigid * mask_b_w, 2000)
mov_e2000_n = normalize_float(mov_e2000)
ncc_e2000   = compute_ncc(ref_n, mov_e2000_n, mask_both)
print(f"  Elastix 2000 NCC={ncc_e2000:.4f}")

# ---- Build collage ----
n_methods = len(METHODS)
sheet_w = n_methods * (CELL_W + GAP) + GAP
sheet_h = HDR_H + GAP + CELL_H + LABEL_H + GAP
sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

# Header
colors = [(220,220,220), (200,255,200), (200,255,255), (255,200,150), (255,230,100)]
for mi, (mname, col) in enumerate(zip(METHODS, colors)):
    x0 = GAP + mi * (CELL_W + GAP)
    cv2.putText(sheet, mname, (x0+8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)

# Panel 1: Raw + annotation contours (centroid shift only for rough alignment)
cx_a = np.where(mask_a > 0)[1].mean(); cy_a = np.where(mask_a > 0)[0].mean()
cx_b = np.where(mask_b > 0)[1].mean(); cy_b = np.where(mask_b > 0)[0].mean()
tx, ty = cx_a - cx_b, cy_a - cy_b
M_cent = np.float32([[1,0,tx],[0,1,ty]])
mov_cent = cv2.warpAffine(mov_raw, M_cent, (w,h))
mask_b_cent = cv2.warpAffine(mask_b, M_cent, (w,h), flags=cv2.INTER_NEAREST)
ov_raw = make_overlay_raw(ref_raw, mov_cent)
draw_mask_contour(ov_raw, mask_a,      (0, 255, 0),   12)  # green = ref
draw_mask_contour(ov_raw, mask_b_cent, (255, 0, 255), 12)  # magenta = mov

panels = [
    (ov_raw,                                                        None),
    (make_overlay_masked(ref_n, mov_rigid_n, mask_both),            ncc_rigid),
    (make_overlay_masked(ref_n, mov_fb_n,    mask_both),            ncc_fb),
    (make_overlay_masked(ref_n, mov_e500_n,  mask_both),            ncc_e500),
    (make_overlay_masked(ref_n, mov_e2000_n, mask_both),            ncc_e2000),
]

y0 = HDR_H + GAP
for mi, (ov, ncc) in enumerate(panels):
    x0 = GAP + mi * (CELL_W + GAP)
    thumb = cv2.resize(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), (CELL_W, CELL_H))
    sheet[y0:y0+CELL_H, x0:x0+CELL_W] = thumb
    if ncc is not None:
        cv2.putText(sheet, f'NCC={ncc:.4f}', (x0+5, y0+CELL_H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,100), 1, cv2.LINE_AA)

cv2.putText(sheet, pair_key, (GAP+5, y0+CELL_H+35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

out_path = f'{OUT_DIR}/test_{pair_key}.png'
cv2.imwrite(out_path, sheet)
print(f"\nSaved: {out_path}")
