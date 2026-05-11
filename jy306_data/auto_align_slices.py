"""
Automated slice-to-slice alignment using ECC (Enhanced Correlation Coefficient).

For each adjacent tile pair:
1. Load last z-slice of tile A, first z-slice of tile B (GFP, masked)
2. Within-row: use phase correlation + ECC refinement (small shifts expected)
3. Cross-row: coarse rotation search (±30°) + ECC refinement
4. Save transforms and regenerate QC contact sheet

Usage:
    python auto_align_slices.py
"""

import numpy as np
import nd2
import cv2
import os
import json
import sys

# ============================================================
# Config
# ============================================================
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

# ============================================================
# Load slices
# ============================================================
def load_boundary_slices(key, masks):
    """Load first and last z-slice GFP from tile, masked."""
    path = nd2_path(key)
    with nd2.ND2File(path) as f:
        data = f.asarray()  # (12, 2, 4200, 4200)
    first_gfp = data[0, 1].astype(np.float32)
    last_gfp = data[-1, 1].astype(np.float32)
    mask = masks.get(key, np.ones((4200, 4200), dtype=np.uint8))
    first_gfp *= mask.astype(np.float32)
    last_gfp *= mask.astype(np.float32)
    return first_gfp, last_gfp

def normalize_8bit(img):
    """Normalize to uint8 for ECC/correlation."""
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    p2, p995 = np.percentile(vals, [2, 99.5])
    out = np.clip((img - p2) / max(p995 - p2, 1) * 255, 0, 255).astype(np.uint8)
    return out

def normalize_float(img):
    """Normalize to 0-1 float."""
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.float32)
    p2, p995 = np.percentile(vals, [2, 99.5])
    return np.clip((img - p2) / max(p995 - p2, 1), 0, 1).astype(np.float32)

# ============================================================
# Alignment methods
# ============================================================
def phase_correlation_align(ref, mov):
    """Phase correlation for translation-only estimation."""
    # Work at reduced resolution for speed
    scale = 0.5
    h, w = ref.shape
    sh, sw = int(h * scale), int(w * scale)
    ref_s = cv2.resize(ref, (sw, sh))
    mov_s = cv2.resize(mov, (sw, sh))

    # Phase correlation
    ref_f = np.float32(ref_s)
    mov_f = np.float32(mov_s)
    shift, response = cv2.phaseCorrelate(ref_f, mov_f)

    tx, ty = shift[0] / scale, shift[1] / scale
    return tx, ty, response

def ecc_refine(ref, mov, init_warp=None, mode='euclidean', downsample=2):
    """Refine alignment using ECC optimization.

    Args:
        ref: reference image (uint8)
        mov: moving image (uint8)
        init_warp: initial 2x3 warp matrix (None = identity)
        mode: 'euclidean' (rotation+translation) or 'affine'
        downsample: factor to downsample for speed

    Returns:
        warp_matrix (2x3), ecc_value
    """
    h, w = ref.shape
    sh, sw = h // downsample, w // downsample

    ref_s = cv2.resize(ref, (sw, sh))
    mov_s = cv2.resize(mov, (sw, sh))

    # Apply Gaussian blur to help convergence
    ref_s = cv2.GaussianBlur(ref_s, (15, 15), 0)
    mov_s = cv2.GaussianBlur(mov_s, (15, 15), 0)

    if mode == 'euclidean':
        motion = cv2.MOTION_EUCLIDEAN
    elif mode == 'affine':
        motion = cv2.MOTION_AFFINE
    else:
        motion = cv2.MOTION_TRANSLATION

    if init_warp is not None:
        warp = init_warp.copy().astype(np.float32)
        # Scale translation for downsampled coords
        warp[0, 2] /= downsample
        warp[1, 2] /= downsample
    else:
        warp = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)

    try:
        cc, warp_out = cv2.findTransformECC(ref_s, mov_s, warp, motion, criteria,
                                             inputMask=None, gaussFiltSize=5)
        # Scale translation back to full resolution
        warp_out[0, 2] *= downsample
        warp_out[1, 2] *= downsample
        return warp_out, cc
    except cv2.error as e:
        print(f"    ECC failed: {e}")
        if init_warp is not None:
            return init_warp, 0.0
        return np.eye(2, 3, dtype=np.float32), 0.0

def coarse_rotation_search(ref, mov, angle_range=30, angle_step=1.0, downsample=4):
    """Brute-force rotation search at coarse resolution.

    For each angle, rotate mov, then compute phase correlation for translation.
    Returns best (angle, tx, ty, score).
    """
    h, w = ref.shape
    sh, sw = h // downsample, w // downsample
    cx, cy = w / 2, h / 2

    ref_s = cv2.resize(ref, (sw, sh))
    ref_blur = cv2.GaussianBlur(ref_s, (21, 21), 0).astype(np.float32)

    best_score = -1
    best_params = (0, 0, 0)

    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)

    for angle in angles:
        # Rotate mov around center
        M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        mov_rot = cv2.warpAffine(mov, M_rot, (w, h), flags=cv2.INTER_LINEAR)
        mov_s = cv2.resize(mov_rot, (sw, sh))
        mov_blur = cv2.GaussianBlur(mov_s, (21, 21), 0).astype(np.float32)

        # Phase correlation for translation
        shift, response = cv2.phaseCorrelate(ref_blur, mov_blur)

        if response > best_score:
            best_score = response
            best_params = (angle, shift[0] * downsample, shift[1] * downsample)

    return best_params[0], best_params[1], best_params[2], best_score

def build_warp_from_angle_translation(angle_deg, tx, ty, cx, cy):
    """Build 2x3 euclidean warp matrix from angle and translation."""
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    # Rotation around center + translation
    warp = np.array([
        [cos_a, -sin_a, tx + cx * (1 - cos_a) + cy * sin_a],
        [sin_a,  cos_a, ty + cy * (1 - cos_a) - cx * sin_a]
    ], dtype=np.float32)
    return warp

def warp_to_angle_translation(warp):
    """Extract angle and translation from 2x3 warp matrix."""
    angle_rad = np.arctan2(warp[1, 0], warp[0, 0])
    angle_deg = np.degrees(angle_rad)
    tx = warp[0, 2]
    ty = warp[1, 2]
    return angle_deg, tx, ty

# ============================================================
# QC visualization
# ============================================================
def make_qc_overlay(ref, mov, warp, mask_ref=None, mask_mov=None):
    """Generate checkerboard + green/magenta overlay."""
    h, w = ref.shape

    # Warp moving image
    mov_w = cv2.warpAffine(mov, warp, (w, h), flags=cv2.INTER_LINEAR)

    # Normalize
    ref_n = normalize_float(ref)
    mov_n = normalize_float(mov_w)

    # Mask intersection if available
    if mask_ref is not None and mask_mov is not None:
        mask_mov_w = cv2.warpAffine(mask_mov.astype(np.uint8), warp, (w, h),
                                     flags=cv2.INTER_NEAREST)
        mask_both = (mask_ref > 0) & (mask_mov_w > 0)
        ref_n *= mask_both.astype(np.float32)
        mov_n *= mask_both.astype(np.float32)

    # Checkerboard
    block = 100
    checker = np.zeros((h, w), dtype=bool)
    for cy in range(0, h, block):
        for cx in range(0, w, block):
            if ((cy // block) + (cx // block)) % 2 == 0:
                checker[cy:cy+block, cx:cx+block] = True

    checker_img = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        checker_img[:, :, c] = np.where(checker, ref_n, mov_n)

    # Green/magenta overlay
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[:, :, 0] = mov_n   # R = moving (magenta)
    overlay[:, :, 1] = ref_n   # G = reference (green)
    overlay[:, :, 2] = mov_n   # B = moving (magenta)

    # Combine side by side
    gap = 20
    combined = np.zeros((h, w * 2 + gap, 3), dtype=np.float32)
    combined[:, :w] = checker_img
    combined[:, w + gap:] = overlay

    return (combined * 255).astype(np.uint8)

# ============================================================
# Main
# ============================================================
def main():
    print("Loading masks...")
    masks_data = np.load(f'{BASE}/registration_video/via_masks_v2.npz')
    masks = {k: masks_data[k] for k in masks_data.files}
    print(f"  {len(masks)} masks loaded")

    # Build pairs
    pairs = [(TILE_ORDER[i], TILE_ORDER[i+1]) for i in range(len(TILE_ORDER) - 1)]
    print(f"  {len(pairs)} pairs to align\n")

    # Pre-load all boundary slices
    print("Loading boundary slices...")
    first_slices = {}  # key -> first z-slice
    last_slices = {}   # key -> last z-slice

    for key in TILE_ORDER:
        print(f"  {key}...", end=" ", flush=True)
        first_s, last_s = load_boundary_slices(key, masks)
        first_slices[key] = first_s
        last_slices[key] = last_s
        print("done")

    # Align each pair
    transforms = {}
    qc_dir = f'{BASE}/png_exports/z_stitch_qc_aligned'
    os.makedirs(qc_dir, exist_ok=True)

    for idx, (key_a, key_b) in enumerate(pairs):
        is_cross = key_a.split('_')[0] != key_b.split('_')[0]
        tag = ' [CROSS-ROW]' if is_cross else ''
        pair_key = f'{key_a}_to_{key_b}'
        print(f"\n--- Pair {idx}: {pair_key}{tag} ---")

        ref = last_slices[key_a]     # last z of tile A = reference
        mov = first_slices[key_b]    # first z of tile B = moving

        ref_8 = normalize_8bit(ref)
        mov_8 = normalize_8bit(mov)

        h, w = ref.shape
        cx, cy = w / 2, h / 2

        if is_cross:
            # Cross-row: coarse rotation search + ECC refinement
            print("  Coarse rotation search (±30°)...")
            angle, tx, ty, score = coarse_rotation_search(ref_8, mov_8,
                                                           angle_range=30, angle_step=0.5,
                                                           downsample=4)
            print(f"  Coarse: angle={angle:.1f}°, tx={tx:.0f}, ty={ty:.0f}, score={score:.4f}")

            # Build initial warp for ECC
            init_warp = build_warp_from_angle_translation(angle, tx, ty, cx, cy)

            # ECC refinement at 2x downsample
            print("  ECC refinement (2x)...")
            warp, ecc_val = ecc_refine(ref_8, mov_8, init_warp=init_warp,
                                        mode='euclidean', downsample=2)
            angle_f, tx_f, ty_f = warp_to_angle_translation(warp)
            print(f"  Refined: angle={angle_f:.2f}°, tx={tx_f:.1f}, ty={ty_f:.1f}, ECC={ecc_val:.4f}")

        else:
            # Within-row: phase correlation for initial translation, then ECC
            print("  Phase correlation...")
            tx, ty, pc_score = phase_correlation_align(ref_8, mov_8)
            print(f"  Phase corr: tx={tx:.1f}, ty={ty:.1f}, score={pc_score:.4f}")

            # Build initial warp (translation only)
            init_warp = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

            # ECC refinement
            print("  ECC refinement (2x)...")
            warp, ecc_val = ecc_refine(ref_8, mov_8, init_warp=init_warp,
                                        mode='euclidean', downsample=2)
            angle_f, tx_f, ty_f = warp_to_angle_translation(warp)
            print(f"  Refined: angle={angle_f:.2f}°, tx={tx_f:.1f}, ty={ty_f:.1f}, ECC={ecc_val:.4f}")

        # Store transform
        transforms[pair_key] = {
            'warp_matrix': warp.tolist(),
            'angle_deg': float(angle_f),
            'translation': [float(tx_f), float(ty_f)],
            'ecc': float(ecc_val),
            'is_cross_row': is_cross,
        }

        # Generate QC overlay
        mask_a = masks.get(key_a)
        mask_b = masks.get(key_b)
        qc = make_qc_overlay(ref, mov, warp, mask_a, mask_b)

        # Add label
        label = f'{pair_key}{tag}  angle={angle_f:.2f} tx={tx_f:.0f} ty={ty_f:.0f} ECC={ecc_val:.3f}'
        cv2.putText(qc, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                     (0, 255, 0), 2, cv2.LINE_AA)

        out_path = f'{qc_dir}/{pair_key}.png'
        # Downscale for reasonable file size
        qc_small = cv2.resize(qc, (qc.shape[1] // 4, qc.shape[0] // 4))
        cv2.imwrite(out_path, cv2.cvtColor(qc_small, cv2.COLOR_RGB2BGR))
        print(f"  QC: {out_path}")

    # Save transforms
    save_path = f'{BASE}/registration_video/auto_align_transforms.json'
    with open(save_path, 'w') as f:
        json.dump(transforms, f, indent=2)
    print(f"\nTransforms saved: {save_path}")

    # ============================================================
    # Build contact sheet
    # ============================================================
    print("\nBuilding contact sheet...")
    n_pairs = len(pairs)
    cols = 4
    rows = (n_pairs + cols - 1) // cols
    thumb_w, thumb_h = 600, 300
    gap = 10
    sheet_w = cols * (thumb_w + gap) + gap
    sheet_h = rows * (thumb_h + 50 + gap) + gap  # 50px for label

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

        # Label
        info = transforms[pair_key]
        label1 = f'{pair_key}{tag}'
        label2 = f'ECC={info["ecc"]:.3f} angle={info["angle_deg"]:.1f}'
        cv2.putText(sheet, label1, (x0, y0 + thumb_h + 18),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(sheet, label2, (x0, y0 + thumb_h + 36),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    sheet_path = f'{qc_dir}/contact_sheet_aligned.png'
    cv2.imwrite(sheet_path, sheet)
    print(f"Contact sheet: {sheet_path}")

    # Summary
    print("\n=== SUMMARY ===")
    for pair_key, info in transforms.items():
        tag = ' [CROSS-ROW]' if info['is_cross_row'] else ''
        print(f"  {pair_key}{tag}: angle={info['angle_deg']:.2f}°, "
              f"tx={info['translation'][0]:.1f}, ty={info['translation'][1]:.1f}, "
              f"ECC={info['ecc']:.3f}")

if __name__ == '__main__':
    main()
