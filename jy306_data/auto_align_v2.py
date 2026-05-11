"""
Automated slice alignment v2: centroid-based initialization + ECC refinement.

Strategy:
1. Compute mask centroid for each tile
2. Translate moving image so centroids match (robust initialization)
3. Coarse rotation search around centroid-aligned position
4. Multi-scale ECC refinement
5. Generate checkerboard + green/magenta QC overlays
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

def normalize_8bit(img):
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    p2, p995 = np.percentile(vals, [2, 99.5])
    return np.clip((img - p2) / max(p995 - p2, 1) * 255, 0, 255).astype(np.uint8)

def normalize_float(img):
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.float32)
    p2, p995 = np.percentile(vals, [2, 99.5])
    return np.clip((img - p2) / max(p995 - p2, 1), 0, 1).astype(np.float32)

def mask_centroid(mask):
    """Compute centroid of binary mask."""
    ys, xs = np.where(mask > 0)
    return float(np.mean(xs)), float(np.mean(ys))

def align_pair(ref, mov, mask_ref, mask_mov):
    """Align mov to ref using centroid init + rotation search + ECC."""
    h, w = ref.shape

    # Step 1: Centroid-based translation
    cx_ref, cy_ref = mask_centroid(mask_ref)
    cx_mov, cy_mov = mask_centroid(mask_mov)
    tx_init = cx_ref - cx_mov
    ty_init = cy_ref - cy_mov
    print(f"  Centroid ref=({cx_ref:.0f},{cy_ref:.0f}), mov=({cx_mov:.0f},{cy_mov:.0f})")
    print(f"  Centroid shift: tx={tx_init:.0f}, ty={ty_init:.0f}")

    ref_8 = normalize_8bit(ref)
    mov_8 = normalize_8bit(mov)

    # Step 2: Coarse rotation search around centroid-aligned position
    # Rotate around the MOVING image's centroid, then translate
    print("  Coarse rotation search (±20°, step=0.25)...")
    best_score = -1
    best_angle = 0
    ds = 4
    sh, sw = h // ds, w // ds
    ref_s = cv2.GaussianBlur(cv2.resize(ref_8, (sw, sh)), (15, 15), 0).astype(np.float32)

    for angle in np.arange(-20, 20.25, 0.25):
        # Build warp: rotate around mov centroid, then translate to align centroids
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        # Rotation around mov centroid + centroid translation
        warp_test = np.array([
            [cos_a, -sin_a, cx_mov*(1-cos_a) + cy_mov*sin_a + tx_init],
            [sin_a,  cos_a, cy_mov*(1-cos_a) - cx_mov*sin_a + ty_init]
        ], dtype=np.float32)

        mov_rot = cv2.warpAffine(mov_8, warp_test, (w, h), flags=cv2.INTER_LINEAR)
        mov_s = cv2.GaussianBlur(cv2.resize(mov_rot, (sw, sh)), (15, 15), 0).astype(np.float32)

        # NCC on masked region
        shift, response = cv2.phaseCorrelate(ref_s, mov_s)
        if response > best_score:
            best_score = response
            best_angle = angle

    print(f"  Best angle: {best_angle:.2f}° (score={best_score:.4f})")

    # Step 3: Build initial warp from best angle + centroid shift
    rad = np.radians(best_angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    init_warp = np.array([
        [cos_a, -sin_a, cx_mov*(1-cos_a) + cy_mov*sin_a + tx_init],
        [sin_a,  cos_a, cy_mov*(1-cos_a) - cx_mov*sin_a + ty_init]
    ], dtype=np.float32)

    # Step 4: Multi-scale ECC refinement
    cc = 0.0
    warp = init_warp.copy()
    for ds_ecc in [4, 2]:
        sh2, sw2 = h // ds_ecc, w // ds_ecc
        ref_e = cv2.GaussianBlur(cv2.resize(ref_8, (sw2, sh2)), (11, 11), 0)
        mov_e = cv2.GaussianBlur(cv2.resize(mov_8, (sw2, sh2)), (11, 11), 0)

        warp_scaled = warp.copy()
        warp_scaled[0, 2] /= ds_ecc
        warp_scaled[1, 2] /= ds_ecc

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
        try:
            cc, warp_out = cv2.findTransformECC(ref_e, mov_e, warp_scaled,
                                                 cv2.MOTION_EUCLIDEAN, criteria,
                                                 inputMask=None, gaussFiltSize=5)
            warp_out[0, 2] *= ds_ecc
            warp_out[1, 2] *= ds_ecc
            warp = warp_out
            angle_f = np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))
            print(f"  ECC {ds_ecc}x: angle={angle_f:.2f}°, tx={warp[0,2]:.1f}, ty={warp[1,2]:.1f}, ECC={cc:.4f}")
        except cv2.error as e:
            print(f"  ECC {ds_ecc}x failed: {e}")

    angle_f = np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))
    return warp, cc, angle_f

def make_qc(ref, mov, warp, mask_ref, mask_mov, pair_key, info_str):
    """Generate checkerboard + green/magenta QC image."""
    h, w = ref.shape

    mov_w = cv2.warpAffine(mov, warp, (w, h), flags=cv2.INTER_LINEAR)
    ref_n = normalize_float(ref)
    mov_n = normalize_float(mov_w)

    # Mask intersection
    mask_mov_w = cv2.warpAffine(mask_mov.astype(np.uint8), warp, (w, h), flags=cv2.INTER_NEAREST)
    mask_both = (mask_ref > 0) & (mask_mov_w > 0)
    ref_n *= mask_both.astype(np.float32)
    mov_n *= mask_both.astype(np.float32)

    # Checkerboard
    block = 80
    yy, xx = np.mgrid[:h, :w]
    checker = ((yy // block) + (xx // block)) % 2 == 0
    checker_img = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        checker_img[:, :, c] = np.where(checker, ref_n, mov_n)

    # Green/magenta
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[:, :, 0] = mov_n
    overlay[:, :, 1] = ref_n
    overlay[:, :, 2] = mov_n

    gap = 20
    combined = np.zeros((h, w * 2 + gap, 3), dtype=np.float32)
    combined[:, :w] = checker_img
    combined[:, w + gap:] = overlay
    combined = (combined * 255).astype(np.uint8)

    # Label
    cv2.putText(combined, info_str, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                 (0, 255, 0), 2, cv2.LINE_AA)

    qc_small = cv2.resize(combined, (combined.shape[1] // 4, combined.shape[0] // 4))
    out_path = f'{BASE}/png_exports/z_stitch_qc_aligned/{pair_key}.png'
    cv2.imwrite(out_path, cv2.cvtColor(qc_small, cv2.COLOR_RGB2BGR))
    return out_path

# ============================================================
# Main
# ============================================================
print("Loading masks...")
masks_data = np.load(f'{BASE}/registration_video/via_masks_v2.npz')
masks = {k: masks_data[k] for k in masks_data.files}

pairs = [(TILE_ORDER[i], TILE_ORDER[i+1]) for i in range(len(TILE_ORDER) - 1)]
print(f"{len(pairs)} pairs\n")

# Pre-load boundary slices
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

# Align all pairs
transforms = {}
qc_dir = f'{BASE}/png_exports/z_stitch_qc_aligned'
os.makedirs(qc_dir, exist_ok=True)

for idx, (key_a, key_b) in enumerate(pairs):
    is_cross = key_a.split('_')[0] != key_b.split('_')[0]
    tag = ' [CROSS-ROW]' if is_cross else ''
    pair_key = f'{key_a}_to_{key_b}'
    print(f"=== Pair {idx}: {pair_key}{tag} ===")

    ref = last_slices[key_a]
    mov = first_slices[key_b]
    mask_ref = masks.get(key_a, np.ones((4200, 4200), dtype=np.uint8))
    mask_mov = masks.get(key_b, np.ones((4200, 4200), dtype=np.uint8))

    warp, ecc_val, angle_f = align_pair(ref, mov, mask_ref, mask_mov)
    tx_f, ty_f = warp[0, 2], warp[1, 2]

    transforms[pair_key] = {
        'warp_matrix': warp.tolist(),
        'angle_deg': float(angle_f),
        'translation': [float(tx_f), float(ty_f)],
        'ecc': float(ecc_val),
        'is_cross_row': is_cross,
    }

    info_str = f'{pair_key}{tag}  a={angle_f:.1f} tx={tx_f:.0f} ty={ty_f:.0f} ECC={ecc_val:.3f}'
    qc_path = make_qc(ref, mov, warp, mask_ref, mask_mov, pair_key, info_str)
    print(f"  QC: {qc_path}\n")

# Save transforms
save_path = f'{BASE}/registration_video/auto_align_transforms_v2.json'
with open(save_path, 'w') as f:
    json.dump(transforms, f, indent=2)
print(f"Transforms saved: {save_path}")

# Contact sheet
print("\nBuilding contact sheet...")
n_pairs = len(pairs)
cols = 4
rows = (n_pairs + cols - 1) // cols
thumb_w, thumb_h = 600, 300
gap = 10
sheet_w = cols * (thumb_w + gap) + gap
sheet_h = rows * (thumb_h + 50 + gap) + gap

sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

for idx, (key_a, key_b) in enumerate(pairs):
    pair_key = f'{key_a}_to_{key_b}'
    is_cross = key_a.split('_')[0] != key_b.split('_')[0]
    tag = ' [CROSS-ROW]' if is_cross else ''

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
    label1 = f'{pair_key}{tag}'
    label2 = f'ECC={info["ecc"]:.3f} a={info["angle_deg"]:.1f} tx={info["translation"][0]:.0f} ty={info["translation"][1]:.0f}'
    cv2.putText(sheet, label1, (x0, y0 + thumb_h + 18),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(sheet, label2, (x0, y0 + thumb_h + 36),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

sheet_path = f'{qc_dir}/contact_sheet_v2.png'
cv2.imwrite(sheet_path, sheet)
print(f"Contact sheet: {sheet_path}")

print("\n=== SUMMARY ===")
for pair_key, info in transforms.items():
    tag = ' [CROSS]' if info['is_cross_row'] else ''
    print(f"  {pair_key}{tag}: a={info['angle_deg']:.2f}° tx={info['translation'][0]:.0f} ty={info['translation'][1]:.0f} ECC={info['ecc']:.3f}")
