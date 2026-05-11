"""
Automated slice alignment: IOU-only (no ECC).
Coarse IOU search -> Fine IOU search -> done.
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

def mask_centroid(mask):
    ys, xs = np.where(mask > 0)
    return float(np.mean(xs)), float(np.mean(ys))

def compute_iou(mask_a, mask_b_warped):
    a = mask_a > 0
    b = mask_b_warped > 0
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    if union == 0:
        return 0.0
    return float(intersection) / float(union)

def build_warp(angle_deg, tx, ty, cx, cy):
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    return np.array([
        [cos_a, -sin_a, cx * (1 - cos_a) + cy * sin_a + tx],
        [sin_a,  cos_a, cy * (1 - cos_a) - cx * sin_a + ty]
    ], dtype=np.float32)

def iou_search(mask_ref, mask_mov, cx_mov, cy_mov,
               angle_range, angle_step, tx_range, ty_range, t_step, downsample):
    h, w = mask_ref.shape
    sh, sw = h // downsample, w // downsample
    mask_ref_s = cv2.resize(mask_ref, (sw, sh), interpolation=cv2.INTER_NEAREST)
    mask_mov_small = cv2.resize(mask_mov, (sw, sh), interpolation=cv2.INTER_NEAREST)
    cx_s, cy_s = cx_mov / downsample, cy_mov / downsample

    best_iou = -1
    best_params = (0, 0, 0)

    for angle in np.arange(-angle_range, angle_range + angle_step, angle_step):
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        warp_rot = np.array([
            [cos_a, -sin_a, cx_s * (1 - cos_a) + cy_s * sin_a],
            [sin_a,  cos_a, cy_s * (1 - cos_a) - cx_s * sin_a]
        ], dtype=np.float32)
        mask_rot_s = cv2.warpAffine(mask_mov_small, warp_rot, (sw, sh),
                                     flags=cv2.INTER_NEAREST)

        for tx in np.arange(-tx_range, tx_range + t_step, t_step):
            for ty in np.arange(-ty_range, ty_range + t_step, t_step):
                tx_s = tx / downsample
                ty_s = ty / downsample
                M_t = np.array([[1, 0, tx_s], [0, 1, ty_s]], dtype=np.float32)
                shifted = cv2.warpAffine(mask_rot_s, M_t, (sw, sh),
                                          flags=cv2.INTER_NEAREST)
                iou = compute_iou(mask_ref_s, shifted)
                if iou > best_iou:
                    best_iou = iou
                    best_params = (angle, tx, ty)

    return best_params[0], best_params[1], best_params[2], best_iou

def align_pair(mask_ref, mask_mov):
    cx_mov, cy_mov = mask_centroid(mask_mov)
    cx_ref, cy_ref = mask_centroid(mask_ref)
    print(f"  Centroid ref=({cx_ref:.0f},{cy_ref:.0f}), mov=({cx_mov:.0f},{cy_mov:.0f})")

    # Coarse search
    print("  IOU coarse (±20°/0.5°, ±400px/20px, ds=4)...")
    a_c, tx_c, ty_c, iou_c = iou_search(
        mask_ref, mask_mov, cx_mov, cy_mov,
        angle_range=20, angle_step=0.5,
        tx_range=400, ty_range=400, t_step=20, downsample=4)
    print(f"  Coarse: a={a_c:.1f}° tx={tx_c:.0f} ty={ty_c:.0f} IOU={iou_c:.4f}")

    # Medium search: narrow window CENTERED on coarse result
    print(f"  IOU medium (centered ±3°/0.25°, ±40px/4px, ds=2)...")
    h, w = mask_ref.shape
    sh2, sw2 = h // 2, w // 2
    mask_ref_s2 = cv2.resize(mask_ref, (sw2, sh2), interpolation=cv2.INTER_NEAREST)
    mask_mov_s2 = cv2.resize(mask_mov, (sw2, sh2), interpolation=cv2.INTER_NEAREST)
    cx_s2, cy_s2 = cx_mov / 2, cy_mov / 2

    best_iou_m = iou_c
    best_m = (a_c, tx_c, ty_c)

    for angle in np.arange(a_c - 3, a_c + 3.25, 0.25):
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        warp_rot = np.array([
            [cos_a, -sin_a, cx_s2 * (1 - cos_a) + cy_s2 * sin_a],
            [sin_a,  cos_a, cy_s2 * (1 - cos_a) - cx_s2 * sin_a]
        ], dtype=np.float32)
        mask_rot = cv2.warpAffine(mask_mov_s2, warp_rot, (sw2, sh2),
                                   flags=cv2.INTER_NEAREST)
        for tx in np.arange(tx_c - 40, tx_c + 44, 4):
            for ty in np.arange(ty_c - 40, ty_c + 44, 4):
                M_t = np.array([[1, 0, tx / 2], [0, 1, ty / 2]], dtype=np.float32)
                shifted = cv2.warpAffine(mask_rot, M_t, (sw2, sh2),
                                          flags=cv2.INTER_NEAREST)
                iou = compute_iou(mask_ref_s2, shifted)
                if iou > best_iou_m:
                    best_iou_m = iou
                    best_m = (angle, tx, ty)

    a_m, tx_m, ty_m, iou_m = best_m[0], best_m[1], best_m[2], best_iou_m
    print(f"  Medium: a={a_m:.2f}° tx={tx_m:.0f} ty={ty_m:.0f} IOU={iou_m:.4f}")

    # Fine search: narrow window CENTERED on medium result
    print("  IOU fine (centered ±1°/0.1°, ±16px/2px, ds=1)...")
    h, w = mask_ref.shape
    sh, sw = h, w  # ds=1
    mask_ref_full = mask_ref
    mask_mov_full = mask_mov
    cx_s, cy_s = cx_mov, cy_mov

    best_iou_f = iou_m
    best_f = (a_m, tx_m, ty_m)

    for angle in np.arange(a_m - 1, a_m + 1.1, 0.1):
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        warp_rot = np.array([
            [cos_a, -sin_a, cx_s * (1 - cos_a) + cy_s * sin_a],
            [sin_a,  cos_a, cy_s * (1 - cos_a) - cx_s * sin_a]
        ], dtype=np.float32)
        mask_rot = cv2.warpAffine(mask_mov_full, warp_rot, (sw, sh),
                                   flags=cv2.INTER_NEAREST)
        for tx in np.arange(tx_m - 16, tx_m + 18, 2):
            for ty in np.arange(ty_m - 16, ty_m + 18, 2):
                M_t = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
                shifted = cv2.warpAffine(mask_rot, M_t, (sw, sh),
                                          flags=cv2.INTER_NEAREST)
                iou = compute_iou(mask_ref_full, shifted)
                if iou > best_iou_f:
                    best_iou_f = iou
                    best_f = (angle, tx, ty)

    a_f, tx_f, ty_f, iou_f = best_f[0], best_f[1], best_f[2], best_iou_f
    print(f"  Fine: a={a_f:.2f}° tx={tx_f:.0f} ty={ty_f:.0f} IOU={iou_f:.4f}")

    warp = build_warp(a_f, tx_f, ty_f, cx_mov, cy_mov)
    return warp, iou_f, a_f

def make_qc(ref, mov, warp, mask_ref, mask_mov, pair_key, info_str):
    h, w = ref.shape
    mov_w = cv2.warpAffine(mov, warp, (w, h), flags=cv2.INTER_LINEAR)
    ref_n = normalize_float(ref)
    mov_n = normalize_float(mov_w)

    mask_mov_w = cv2.warpAffine(mask_mov.astype(np.uint8), warp, (w, h),
                                 flags=cv2.INTER_NEAREST)
    mask_both = (mask_ref > 0) & (mask_mov_w > 0)
    ref_n *= mask_both.astype(np.float32)
    mov_n *= mask_both.astype(np.float32)

    block = 80
    yy, xx = np.mgrid[:h, :w]
    checker = ((yy // block) + (xx // block)) % 2 == 0
    checker_img = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        checker_img[:, :, c] = np.where(checker, ref_n, mov_n)

    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[:, :, 0] = mov_n
    overlay[:, :, 1] = ref_n
    overlay[:, :, 2] = mov_n

    gap = 20
    combined = np.zeros((h, w * 2 + gap, 3), dtype=np.float32)
    combined[:, :w] = checker_img
    combined[:, w + gap:] = overlay
    combined = (combined * 255).astype(np.uint8)

    cv2.putText(combined, info_str, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                 (0, 255, 0), 2, cv2.LINE_AA)

    qc_small = cv2.resize(combined, (combined.shape[1] // 4, combined.shape[0] // 4))
    out_path = f'{BASE}/png_exports/z_stitch_qc_iou/{pair_key}.png'
    cv2.imwrite(out_path, cv2.cvtColor(qc_small, cv2.COLOR_RGB2BGR))
    return out_path

# ============================================================
print("Loading masks...")
masks_data = np.load(f'{BASE}/registration_video/via_masks_v2.npz')
masks = {k: masks_data[k] for k in masks_data.files}

pairs = [(TILE_ORDER[i], TILE_ORDER[i+1]) for i in range(len(TILE_ORDER) - 1)]
print(f"{len(pairs)} pairs\n")

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

qc_dir = f'{BASE}/png_exports/z_stitch_qc_iou'
os.makedirs(qc_dir, exist_ok=True)

transforms = {}
for idx, (key_a, key_b) in enumerate(pairs):
    is_cross = key_a.split('_')[0] != key_b.split('_')[0]
    tag = ' [CROSS-ROW]' if is_cross else ''
    pair_key = f'{key_a}_to_{key_b}'
    print(f"=== Pair {idx}: {pair_key}{tag} ===")

    ref = last_slices[key_a]
    mov = first_slices[key_b]
    mask_ref = masks.get(key_a, np.ones((4200, 4200), dtype=np.uint8))
    mask_mov = masks.get(key_b, np.ones((4200, 4200), dtype=np.uint8))

    warp, iou_val, angle_f = align_pair(mask_ref, mask_mov)
    tx_f, ty_f = warp[0, 2], warp[1, 2]

    transforms[pair_key] = {
        'warp_matrix': warp.tolist(),
        'angle_deg': float(angle_f),
        'translation': [float(tx_f), float(ty_f)],
        'iou': float(iou_val),
        'is_cross_row': is_cross,
    }

    info_str = f'{pair_key}{tag}  a={angle_f:.1f} tx={tx_f:.0f} ty={ty_f:.0f} IOU={iou_val:.3f}'
    qc_path = make_qc(ref, mov, warp, mask_ref, mask_mov, pair_key, info_str)
    print(f"  QC: {qc_path}\n")

save_path = f'{BASE}/registration_video/auto_align_transforms_iou.json'
with open(save_path, 'w') as f:
    json.dump(transforms, f, indent=2)
print(f"Transforms saved: {save_path}")

# Contact sheet
print("\nBuilding contact sheet...")
cols = 4
rows_grid = (len(pairs) + cols - 1) // cols
thumb_w, thumb_h = 600, 300
gap = 10
sheet_w = cols * (thumb_w + gap) + gap
sheet_h = rows_grid * (thumb_h + 50 + gap) + gap
sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

for idx, (key_a, key_b) in enumerate(pairs):
    pair_key = f'{key_a}_to_{key_b}'
    is_cross = key_a.split('_')[0] != key_b.split('_')[0]
    tag = ' [CROSS]' if is_cross else ''
    qc_path = f'{qc_dir}/{pair_key}.png'
    if not os.path.exists(qc_path):
        continue
    img = cv2.imread(qc_path)
    img = cv2.resize(img, (thumb_w, thumb_h))
    r, c = divmod(idx, cols)
    x0 = gap + c * (thumb_w + gap)
    y0 = gap + r * (thumb_h + 50 + gap)
    sheet[y0:y0+thumb_h, x0:x0+thumb_w] = img
    info = transforms[pair_key]
    cv2.putText(sheet, f'{pair_key}{tag}', (x0, y0 + thumb_h + 18),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(sheet, f'IOU={info["iou"]:.3f} a={info["angle_deg"]:.1f} tx={info["translation"][0]:.0f} ty={info["translation"][1]:.0f}',
                 (x0, y0 + thumb_h + 36),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

sheet_path = f'{qc_dir}/contact_sheet_iou.png'
cv2.imwrite(sheet_path, sheet)
print(f"Contact sheet: {sheet_path}")

print("\n=== SUMMARY ===")
for pk, info in transforms.items():
    tag = ' [CROSS]' if info['is_cross_row'] else ''
    print(f"  {pk}{tag}: a={info['angle_deg']:.2f}° tx={info['translation'][0]:.0f} ty={info['translation'][1]:.0f} IOU={info['iou']:.3f}")
