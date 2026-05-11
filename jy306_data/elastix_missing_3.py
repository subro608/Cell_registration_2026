"""
Regenerate 3 missing pairs with transform saving:
row1_1->row1_2, row1_2->row1_3, row4_6->row5_1
Runs for both 500iter and 2000iter output folders.
"""
import numpy as np
import cv2
import os
import json
import SimpleITK as sitk

BASE = '/Users/neurolab/neuroinformatics/margaret'
PNG_DIR = f'{BASE}/png_exports/registration_video'

MISSING_PAIRS = [
    ('row1_1', 'row1_2'),
    ('row1_2', 'row1_3'),
    ('row4_6', 'row5_1'),
]

CONFIGS = [
    {'iter': 500,  'out_dir': f'{BASE}/png_exports/z_stitch_elastix_v4_500iter'},
    {'iter': 2000, 'out_dir': f'{BASE}/png_exports/z_stitch_elastix_v4'},
]

DS = 2; GRID_SPACING = 32; N_RESOLUTIONS = 4
METRIC = 'AdvancedNormalizedCorrelation'

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

def make_overlay(ref_n, mov_n, mask):
    h, w = ref_n.shape
    r = ref_n * mask.astype(np.float32)
    m = mov_n * mask.astype(np.float32)
    ov = np.zeros((h, w, 3), dtype=np.float32)
    ov[:,:,0] = m; ov[:,:,1] = r; ov[:,:,2] = m
    return (ov * 255).astype(np.uint8)

def run_elastix_save(ref_ds, mov_ds, n_iter, pair_dir):
    ref_itk = sitk.GetImageFromArray(ref_ds)
    mov_itk = sitk.GetImageFromArray(mov_ds)
    el = sitk.ElastixImageFilter()
    el.SetFixedImage(ref_itk); el.SetMovingImage(mov_itk)
    pm = sitk.GetDefaultParameterMap('bspline')
    pm['NumberOfResolutions'] = [str(N_RESOLUTIONS)]
    pm['MaximumNumberOfIterations'] = [str(n_iter)]
    pm['Metric'] = [METRIC]
    pm['FinalGridSpacingInPhysicalUnits'] = [str(GRID_SPACING)]
    pm['BSplineInterpolationOrder'] = ['3']
    pm['FinalBSplineInterpolationOrder'] = ['3']
    pm['WriteResultImage'] = ['false']
    pm['NumberOfSpatialSamples'] = ['4096']
    pm['NewSamplesEveryIteration'] = ['true']
    el.SetParameterMap(pm); el.LogToConsoleOff(); el.Execute()
    # Save transform parameters
    tmap = el.GetTransformParameterMap()
    for i, t in enumerate(tmap):
        sitk.WriteParameterFile(t, f'{pair_dir}/TransformParameters.{i}.txt')
    return sitk.GetArrayFromImage(el.GetResultImage()).astype(np.float32)

# Load masks and IOU transforms
print("Loading masks and transforms...")
masks_data = np.load(f'{BASE}/registration_video/via_masks_v4.npz')
masks = {k: masks_data[k] for k in masks_data.files}
with open(f'{BASE}/registration_video/auto_align_transforms_iou_v4.json') as f:
    iou_transforms = json.load(f)

for cfg in CONFIGS:
    n_iter = cfg['iter']
    out_dir = cfg['out_dir']
    print(f"\n=== Running {n_iter} iter -> {out_dir} ===")

    for key_a, key_b in MISSING_PAIRS:
        pair_key = f'{key_a}_to_{key_b}'
        print(f"  {pair_key}...", end=' ', flush=True)

        ref = cv2.imread(f'{PNG_DIR}/{key_a}/GFP_z011.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mov = cv2.imread(f'{PNG_DIR}/{key_b}/GFP_z000.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
        h, w = ref.shape

        mask_a = masks.get(key_a, np.ones((h,w), dtype=np.uint8))
        mask_b = masks.get(key_b, np.ones((h,w), dtype=np.uint8))

        tfm = iou_transforms[pair_key]
        warp = np.array(tfm['warp_matrix'], dtype=np.float32)
        mov_rigid  = cv2.warpAffine(mov, warp, (w,h), flags=cv2.INTER_LINEAR)
        mask_b_w   = cv2.warpAffine(mask_b, warp, (w,h), flags=cv2.INTER_NEAREST)
        mask_both  = (mask_a > 0) & (mask_b_w > 0)

        ref_n      = normalize_float(ref * mask_a)
        mov_rigid_n = normalize_float(mov_rigid * mask_b_w)
        ncc_rigid  = compute_ncc(ref_n, mov_rigid_n, mask_both)

        sh, sw = h // DS, w // DS
        ref_ds = cv2.resize(normalize_8bit(ref * mask_a), (sw, sh)).astype(np.float32)
        mov_ds = cv2.resize(normalize_8bit(mov_rigid * mask_b_w), (sw, sh)).astype(np.float32)

        pair_dir = f'{out_dir}/{pair_key}'
        os.makedirs(pair_dir, exist_ok=True)

        result_ds  = run_elastix_save(ref_ds, mov_ds, n_iter, pair_dir)
        result_full = cv2.resize(result_ds, (w, h), interpolation=cv2.INTER_LINEAR)
        result_n   = normalize_float(result_full)
        ncc_elx    = compute_ncc(ref_n, result_n, mask_both)

        # Save overlay
        ov = make_overlay(ref_n, result_n, mask_both)
        combined = np.zeros((h, w*2+20, 3), dtype=np.uint8)
        combined[:,:w] = cv2.cvtColor(make_overlay(ref_n, mov_rigid_n, mask_both), cv2.COLOR_RGB2BGR)
        combined[:,w+20:] = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, f'IOU RIGID (NCC={ncc_rigid:.4f})', (20,60), font, 2.0, (0,255,0), 3, cv2.LINE_AA)
        cv2.putText(combined, f'ELASTIX {n_iter}iter (NCC={ncc_elx:.4f})', (w+40,60), font, 2.0, (255,200,0), 3, cv2.LINE_AA)
        cv2.imwrite(f'{pair_dir}/overlay.png', combined)

        print(f"rigid={ncc_rigid:.4f} elx={ncc_elx:.4f} — transform saved")

print("\nDone.")
