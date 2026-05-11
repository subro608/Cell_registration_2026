"""
Row-wise comparison collage: for each pair, show all methods side by side.
Columns: IOU rigid | ECC | Deformable v1 | Deformable v2 | Deformable v3
Rows grouped by tile row (row1, row2, row3, row4, row5).
Only green/magenta overlay (no checkerboard).
"""

import numpy as np
import nd2
import cv2
import os
import json

BASE = '/Users/neurolab/neuroinformatics/margaret'

TILE_ORDER = [
    'row1_1', 'row1_2', 'row1_3',
    'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
    'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
    'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
    'row5_1',
]

ROW_GROUPS = {
    'Row 1': [('row1_1', 'row1_2'), ('row1_2', 'row1_3')],
    'Row 1→2 (cross)': [('row1_3', 'row2_1')],
    'Row 2': [('row2_1', 'row2_2'), ('row2_2', 'row2_3'), ('row2_3', 'row2_4'), ('row2_4', 'row2_5')],
    'Row 2→3 (cross)': [('row2_5', 'row3_1')],
    'Row 3': [('row3_1', 'row3_2'), ('row3_2', 'row3_3'), ('row3_3', 'row3_4'), ('row3_4', 'row3_5'), ('row3_5', 'row3_6')],
    'Row 3→4 (cross)': [('row3_6', 'row4_1')],
    'Row 4': [('row4_1', 'row4_2'), ('row4_2', 'row4_3'), ('row4_3', 'row4_4'), ('row4_4', 'row4_5'), ('row4_5', 'row4_6')],
    'Row 4→5 (cross)': [('row4_6', 'row5_1')],
}

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
    map_x = xs + flow[:, :, 0]
    map_y = ys + flow[:, :, 1]
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

def compute_deformable(ref_8, mov_rigid_8, ds, winsize, blur_k, flow_smooth_k, levels, iterations):
    h, w = ref_8.shape
    sh, sw = h // ds, w // ds
    ref_s = cv2.GaussianBlur(cv2.resize(ref_8, (sw, sh)), (blur_k, blur_k), 0)
    mov_s = cv2.GaussianBlur(cv2.resize(mov_rigid_8, (sw, sh)), (blur_k, blur_k), 0)
    flow_small = cv2.calcOpticalFlowFarneback(
        ref_s, mov_s, flow=None, pyr_scale=0.5, levels=levels,
        winsize=winsize, iterations=iterations, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[:, :, 0] = cv2.resize(flow_small[:, :, 0], (w, h)) * ds
    flow[:, :, 1] = cv2.resize(flow_small[:, :, 1], (w, h)) * ds
    flow[:, :, 0] = cv2.GaussianBlur(flow[:, :, 0], (flow_smooth_k, flow_smooth_k), 0)
    flow[:, :, 1] = cv2.GaussianBlur(flow[:, :, 1], (flow_smooth_k, flow_smooth_k), 0)
    return flow

def make_overlay(ref_n, mov_n, mask_both):
    h, w = ref_n.shape
    r = ref_n * mask_both.astype(np.float32)
    m = mov_n * mask_both.astype(np.float32)
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[:, :, 0] = m
    overlay[:, :, 1] = r
    overlay[:, :, 2] = m
    return (overlay * 255).astype(np.uint8)

# ============================================================
print("Loading data...")
masks_data = np.load(f'{BASE}/registration_video/via_masks_v2.npz')
masks = {k: masks_data[k] for k in masks_data.files}

with open(f'{BASE}/registration_video/auto_align_transforms_iou.json', 'r') as f:
    iou_transforms = json.load(f)

# Load ECC transforms
ecc_path = f'{BASE}/registration_video/auto_align_transforms.json'
if os.path.exists(ecc_path):
    with open(ecc_path, 'r') as f:
        ecc_transforms = json.load(f)
else:
    ecc_transforms = {}

pairs = [(TILE_ORDER[i], TILE_ORDER[i+1]) for i in range(len(TILE_ORDER) - 1)]

# Load boundary slices
print("Loading boundary slices...")
first_slices = {}
last_slices = {}
for key in TILE_ORDER:
    path = nd2_path(key)
    print(f"  {key}...", end=" ", flush=True)
    with nd2.ND2File(path) as f:
        data = f.asarray()
    mask = masks.get(key, np.ones((4200, 4200), dtype=np.uint8))
    first_slices[key] = data[0, 1].astype(np.float32) * mask.astype(np.float32)
    last_slices[key] = data[-1, 1].astype(np.float32) * mask.astype(np.float32)
    print("done")

print()

out_dir = f'{BASE}/png_exports/z_stitch_comparison'
os.makedirs(out_dir, exist_ok=True)

# Methods: name, label
METHODS = ['RAW', 'IOU Rigid', 'ECC', 'Deform v1', 'Deform v3', 'Deform v2']

# Thumbnail size for each cell
CELL_W = 500
CELL_H = 500
LABEL_H = 40
GAP = 8

for group_name, group_pairs in ROW_GROUPS.items():
    print(f"\n=== {group_name} ({len(group_pairs)} pairs) ===")

    n_methods = len(METHODS)
    n_pairs = len(group_pairs)

    # Sheet: cols = methods, rows = pairs
    sheet_w = n_methods * (CELL_W + GAP) + GAP
    sheet_h = n_pairs * (CELL_H + LABEL_H + GAP) + GAP + 60  # 60 for header
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    # Header row with method names
    for mi, method_name in enumerate(METHODS):
        x0 = GAP + mi * (CELL_W + GAP)
        cv2.putText(sheet, method_name, (x0 + 10, 40),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    for pi, (key_a, key_b) in enumerate(group_pairs):
        pair_key = f'{key_a}_to_{key_b}'
        print(f"  {pair_key}...", end=" ", flush=True)

        ref = last_slices[key_a]
        mov = first_slices[key_b]
        h, w = ref.shape

        mask_ref = masks.get(key_a, np.ones((h, w), dtype=np.uint8))
        mask_mov = masks.get(key_b, np.ones((h, w), dtype=np.uint8))

        ref_n = normalize_float(ref)

        # IOU rigid warp
        iou_tfm = iou_transforms.get(pair_key)
        iou_warp = np.array(iou_tfm['warp_matrix'], dtype=np.float32) if iou_tfm else np.eye(2, 3, dtype=np.float32)
        mov_iou = cv2.warpAffine(mov, iou_warp, (w, h), flags=cv2.INTER_LINEAR)
        mask_mov_iou = cv2.warpAffine(mask_mov, iou_warp, (w, h), flags=cv2.INTER_NEAREST)
        mask_both_iou = (mask_ref > 0) & (mask_mov_iou > 0)

        # ECC warp
        ecc_tfm = ecc_transforms.get(pair_key)
        if ecc_tfm:
            ecc_warp = np.array(ecc_tfm['warp_matrix'], dtype=np.float32)
            mov_ecc = cv2.warpAffine(mov, ecc_warp, (w, h), flags=cv2.INTER_LINEAR)
            mask_mov_ecc = cv2.warpAffine(mask_mov, ecc_warp, (w, h), flags=cv2.INTER_NEAREST)
            mask_both_ecc = (mask_ref > 0) & (mask_mov_ecc > 0)
        else:
            mov_ecc = mov_iou
            mask_both_ecc = mask_both_iou

        # Raw (no alignment)
        mask_both_raw = (mask_ref > 0) & (mask_mov > 0)

        # Deformable versions from IOU rigid base
        ref_8 = normalize_8bit(ref)
        mov_iou_8 = normalize_8bit(mov_iou)

        # v1: smooth
        flow_v1 = compute_deformable(ref_8, mov_iou_8, ds=2, winsize=128, blur_k=21, flow_smooth_k=51, levels=5, iterations=10)
        mov_def_v1 = apply_flow(mov_iou, flow_v1)

        # v3: moderate
        flow_v3 = compute_deformable(ref_8, mov_iou_8, ds=2, winsize=64, blur_k=15, flow_smooth_k=35, levels=5, iterations=12)
        mov_def_v3 = apply_flow(mov_iou, flow_v3)

        # v2: aggressive
        flow_v2 = compute_deformable(ref_8, mov_iou_8, ds=2, winsize=32, blur_k=9, flow_smooth_k=21, levels=7, iterations=15)
        mov_def_v2 = apply_flow(mov_iou, flow_v2)

        # Generate overlays for each method
        overlays = []

        # RAW
        overlays.append(make_overlay(ref_n, normalize_float(mov), mask_both_raw))
        # IOU Rigid
        overlays.append(make_overlay(ref_n, normalize_float(mov_iou), mask_both_iou))
        # ECC
        overlays.append(make_overlay(ref_n, normalize_float(mov_ecc), mask_both_ecc))
        # Deform v1
        overlays.append(make_overlay(ref_n, normalize_float(mov_def_v1), mask_both_iou))
        # Deform v3
        overlays.append(make_overlay(ref_n, normalize_float(mov_def_v3), mask_both_iou))
        # Deform v2
        overlays.append(make_overlay(ref_n, normalize_float(mov_def_v2), mask_both_iou))

        # Place in sheet
        y0 = 60 + GAP + pi * (CELL_H + LABEL_H + GAP)
        for mi, ov in enumerate(overlays):
            x0 = GAP + mi * (CELL_W + GAP)
            thumb = cv2.resize(ov, (CELL_W, CELL_H))
            # Convert RGB to BGR for sheet
            sheet[y0:y0+CELL_H, x0:x0+CELL_W] = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)

        # Label
        label_y = y0 + CELL_H + 25
        cv2.putText(sheet, pair_key, (GAP + 5, label_y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        print("done")

    # Save this row group
    safe_name = group_name.replace(' ', '_').replace('→', 'to').replace('(', '').replace(')', '')
    sheet_path = f'{out_dir}/{safe_name}.png'
    cv2.imwrite(sheet_path, sheet)
    print(f"  Saved: {sheet_path}")

print(f"\nAll collages saved to: {out_dir}/")
