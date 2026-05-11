"""
Deformable registration: start from IOU rigid alignment, then apply smooth
optical-flow-based deformable registration (Farneback).

For each pair:
1. Load IOU rigid transform, apply it
2. Run smooth deformable registration (large window Farneback optical flow)
3. Generate green/magenta overlay QC
4. Save displacement fields and contact sheet
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
    """Warp image using dense optical flow field."""
    h, w = img.shape[:2]
    # Build remap coordinates
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    map_x = xs + flow[:, :, 0]
    map_y = ys + flow[:, :, 1]
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

# ============================================================
print("Loading masks and IOU transforms...")
masks_data = np.load(f'{BASE}/registration_video/via_masks_v2.npz')
masks = {k: masks_data[k] for k in masks_data.files}

with open(f'{BASE}/registration_video/auto_align_transforms_iou.json', 'r') as f:
    iou_transforms = json.load(f)

pairs = [(TILE_ORDER[i], TILE_ORDER[i+1]) for i in range(len(TILE_ORDER) - 1)]
print(f"{len(pairs)} pairs\n")

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

qc_dir = f'{BASE}/png_exports/z_stitch_qc_deformable_v3'
os.makedirs(qc_dir, exist_ok=True)

results = {}

for idx, (key_a, key_b) in enumerate(pairs):
    is_cross = key_a.split('_')[0] != key_b.split('_')[0]
    tag = ' [CROSS-ROW]' if is_cross else ''
    pair_key = f'{key_a}_to_{key_b}'
    print(f"=== Pair {idx}: {pair_key}{tag} ===")

    ref = last_slices[key_a]
    mov = first_slices[key_b]
    h, w = ref.shape

    # Step 1: Apply IOU rigid transform
    tfm = iou_transforms[pair_key]
    warp = np.array(tfm['warp_matrix'], dtype=np.float32)
    mov_rigid = cv2.warpAffine(mov, warp, (w, h), flags=cv2.INTER_LINEAR)

    # Also warp the mask
    mask_mov = masks.get(key_b, np.ones((h, w), dtype=np.uint8))
    mask_ref = masks.get(key_a, np.ones((h, w), dtype=np.uint8))
    mask_mov_w = cv2.warpAffine(mask_mov, warp, (w, h), flags=cv2.INTER_NEAREST)
    mask_both = (mask_ref > 0) & (mask_mov_w > 0)

    print(f"  Rigid: a={tfm['angle_deg']:.1f}° tx={tfm['translation'][0]:.0f} ty={tfm['translation'][1]:.0f}")

    # Step 2: Smooth deformable registration (Farneback optical flow)
    # Work at half resolution for speed, large window for smoothness
    ds = 2
    sh, sw = h // ds, w // ds

    ref_8 = normalize_8bit(ref)
    mov_rigid_8 = normalize_8bit(mov_rigid)

    ref_s = cv2.resize(ref_8, (sw, sh))
    mov_s = cv2.resize(mov_rigid_8, (sw, sh))

    # Moderate blur
    ref_s = cv2.GaussianBlur(ref_s, (15, 15), 0)
    mov_s = cv2.GaussianBlur(mov_s, (15, 15), 0)

    print("  Computing optical flow (Farneback, moderate)...")
    # Middle ground: winsize=64, moderate smoothing
    flow_small = cv2.calcOpticalFlowFarneback(
        ref_s, mov_s,
        flow=None,
        pyr_scale=0.5,
        levels=5,
        winsize=64,         # between 32 (aggressive) and 128 (smooth)
        iterations=12,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    # Upscale flow to full resolution
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[:, :, 0] = cv2.resize(flow_small[:, :, 0], (w, h)) * ds
    flow[:, :, 1] = cv2.resize(flow_small[:, :, 1], (w, h)) * ds

    # Moderate smoothing of the flow field
    flow[:, :, 0] = cv2.GaussianBlur(flow[:, :, 0], (35, 35), 0)
    flow[:, :, 1] = cv2.GaussianBlur(flow[:, :, 1], (35, 35), 0)

    # Apply deformable warp
    mov_deformed = apply_flow(mov_rigid, flow)

    # Stats
    mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    mag_masked = mag[mask_both]
    mean_disp = np.mean(mag_masked) if len(mag_masked) > 0 else 0
    max_disp = np.max(mag_masked) if len(mag_masked) > 0 else 0
    print(f"  Deformation: mean={mean_disp:.1f}px, max={max_disp:.1f}px")

    # Step 3: QC — green/magenta overlay (rigid vs deformable side by side)
    ref_n = normalize_float(ref)
    mov_rigid_n = normalize_float(mov_rigid)
    mov_deform_n = normalize_float(mov_deformed)

    # Apply mask intersection
    ref_n_m = ref_n * mask_both.astype(np.float32)
    mov_rigid_m = mov_rigid_n * mask_both.astype(np.float32)
    mov_deform_m = mov_deform_n * mask_both.astype(np.float32)

    # Rigid overlay (left)
    overlay_rigid = np.zeros((h, w, 3), dtype=np.float32)
    overlay_rigid[:, :, 0] = mov_rigid_m
    overlay_rigid[:, :, 1] = ref_n_m
    overlay_rigid[:, :, 2] = mov_rigid_m

    # Deformable overlay (right)
    overlay_deform = np.zeros((h, w, 3), dtype=np.float32)
    overlay_deform[:, :, 0] = mov_deform_m
    overlay_deform[:, :, 1] = ref_n_m
    overlay_deform[:, :, 2] = mov_deform_m

    gap = 20
    combined = np.zeros((h, w * 2 + gap, 3), dtype=np.float32)
    combined[:, :w] = overlay_rigid
    combined[:, w + gap:] = overlay_deform
    combined = (combined * 255).astype(np.uint8)

    # Labels
    cv2.putText(combined, 'IOU RIGID', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                 (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, 'DEFORMABLE', (w + gap + 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                 (0, 255, 255), 3, cv2.LINE_AA)
    info_str = f'{pair_key}{tag}  mean_disp={mean_disp:.0f}px  max={max_disp:.0f}px'
    cv2.putText(combined, info_str, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                 (200, 200, 200), 2, cv2.LINE_AA)

    qc_small = cv2.resize(combined, (combined.shape[1] // 4, combined.shape[0] // 4))
    out_path = f'{qc_dir}/{pair_key}.png'
    cv2.imwrite(out_path, cv2.cvtColor(qc_small, cv2.COLOR_RGB2BGR))
    print(f"  QC: {out_path}")

    results[pair_key] = {
        'iou_angle': tfm['angle_deg'],
        'iou_tx': tfm['translation'][0],
        'iou_ty': tfm['translation'][1],
        'iou_val': tfm['iou'],
        'mean_displacement_px': float(mean_disp),
        'max_displacement_px': float(max_disp),
        'is_cross_row': is_cross,
    }
    print()

# Save results
save_path = f'{BASE}/registration_video/deformable_results_v3.json'
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {save_path}")

# Contact sheet — overlay only (no checkerboard)
print("\nBuilding contact sheet...")
cols = 4
rows_grid = (len(pairs) + cols - 1) // cols

sample = cv2.imread(f'{qc_dir}/{pairs[0][0]}_to_{pairs[0][1]}.png')
thumb_h, thumb_w = sample.shape[:2]
gap_s = 20
label_h = 70
sheet_w = cols * (thumb_w + gap_s) + gap_s
sheet_h = rows_grid * (thumb_h + label_h + gap_s) + gap_s
sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

for idx, (key_a, key_b) in enumerate(pairs):
    pair_key = f'{key_a}_to_{key_b}'
    is_cross = key_a.split('_')[0] != key_b.split('_')[0]
    tag = ' [CROSS]' if is_cross else ''

    qc_path = f'{qc_dir}/{pair_key}.png'
    if not os.path.exists(qc_path):
        continue
    img = cv2.imread(qc_path)

    r, c = divmod(idx, cols)
    x0 = gap_s + c * (thumb_w + gap_s)
    y0 = gap_s + r * (thumb_h + label_h + gap_s)
    sheet[y0:y0+thumb_h, x0:x0+thumb_w] = img

    info = results[pair_key]
    cv2.putText(sheet, f'{pair_key}{tag}', (x0, y0 + thumb_h + 22),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(sheet, f'IOU={info["iou_val"]:.3f} disp={info["mean_displacement_px"]:.0f}/{info["max_displacement_px"]:.0f}px',
                 (x0, y0 + thumb_h + 44),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

sheet_path = f'{qc_dir}/contact_sheet_deformable.png'
cv2.imwrite(sheet_path, sheet)
print(f"Contact sheet: {sheet_path}")

# High-res version too
print("Building high-res contact sheet...")
sample_hr = cv2.imread(f'{qc_dir}/{pairs[0][0]}_to_{pairs[0][1]}.png')
th_h, th_w = sample_hr.shape[:2]
gap_hr = 30
label_hr = 80
sheet_w_hr = cols * (th_w + gap_hr) + gap_hr
sheet_h_hr = rows_grid * (th_h + label_hr + gap_hr) + gap_hr
sheet_hr = np.zeros((sheet_h_hr, sheet_w_hr, 3), dtype=np.uint8)

for idx, (key_a, key_b) in enumerate(pairs):
    pair_key = f'{key_a}_to_{key_b}'
    is_cross = key_a.split('_')[0] != key_b.split('_')[0]
    tag = ' [CROSS]' if is_cross else ''
    qc_path = f'{qc_dir}/{pair_key}.png'
    if not os.path.exists(qc_path):
        continue
    img = cv2.imread(qc_path)
    r, c = divmod(idx, cols)
    x0 = gap_hr + c * (th_w + gap_hr)
    y0 = gap_hr + r * (th_h + label_hr + gap_hr)
    sheet_hr[y0:y0+th_h, x0:x0+th_w] = img
    info = results[pair_key]
    cv2.putText(sheet_hr, f'{pair_key}{tag}', (x0, y0 + th_h + 30),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(sheet_hr, f'IOU={info["iou_val"]:.3f} mean_disp={info["mean_displacement_px"]:.0f}px max={info["max_displacement_px"]:.0f}px',
                 (x0, y0 + th_h + 60),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

sheet_hr_path = f'{qc_dir}/contact_sheet_deformable_hires.png'
cv2.imwrite(sheet_hr_path, sheet_hr)
print(f"High-res: {sheet_hr_path}")

print("\n=== SUMMARY ===")
for pk, info in results.items():
    tag = ' [CROSS]' if info['is_cross_row'] else ''
    print(f"  {pk}{tag}: IOU={info['iou_val']:.3f} mean_disp={info['mean_displacement_px']:.0f}px max={info['max_displacement_px']:.0f}px")
