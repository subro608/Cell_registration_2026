"""
Tune deformable registration per pair.
Base: v1 params (winsize=128, blur_k=21, ds=2, levels=5).
User specifies flow_smooth_k and iterations per pair.
Output: NCC + overlay QC.
"""

import numpy as np
import nd2
import cv2
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/Users/neurolab/neuroinformatics/margaret'

TILE_ORDER = [
    'row1_1', 'row1_2', 'row1_3',
    'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
    'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
    'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
    'row5_1',
]

# --- Which pair to tune ---
PAIR_IDX = 15  # 15 = row4_2 -> row4_3
key_a = TILE_ORDER[PAIR_IDX]
key_b = TILE_ORDER[PAIR_IDX + 1]
pair_key = f'{key_a}_to_{key_b}'

# v1 base params (fixed)
DS = 2
WINSIZE = 76
BLUR_K = 13
LEVELS = 5
POLY_N = 7
POLY_SIGMA = 1.5

# User-specified params
FLOW_SMOOTH_K = 65
ITERATIONS = 15

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

def compute_ncc(a, b, mask):
    a_m = a[mask].astype(np.float64)
    b_m = b[mask].astype(np.float64)
    if len(a_m) == 0:
        return 0.0
    a_m = a_m - a_m.mean()
    b_m = b_m - b_m.mean()
    denom = np.sqrt(np.sum(a_m**2) * np.sum(b_m**2))
    if denom == 0:
        return 0.0
    return float(np.sum(a_m * b_m) / denom)

def compute_mse(a, b, mask):
    a_m = a[mask].astype(np.float64)
    b_m = b[mask].astype(np.float64)
    if len(a_m) == 0:
        return 0.0
    return float(np.mean((a_m - b_m)**2))

def make_overlay(ref_n, mov_n, mask):
    h, w = ref_n.shape
    r = ref_n * mask.astype(np.float32)
    m = mov_n * mask.astype(np.float32)
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[:, :, 0] = m
    overlay[:, :, 1] = r
    overlay[:, :, 2] = m
    return (overlay * 255).astype(np.uint8)

# ============================================================
print(f"=== Tuning {pair_key} ===")
print(f"Base: winsize={WINSIZE}, blur_k={BLUR_K}, ds={DS}, levels={LEVELS}")
print(f"User params: flow_smooth_k={FLOW_SMOOTH_K}, iterations={ITERATIONS}")
print()

# Load data
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

print(f"IOU rigid: angle={tfm['angle_deg']:.2f}° tx={tfm['translation'][0]:.0f} ty={tfm['translation'][1]:.0f}")

# Prepare downsampled images for flow computation
ref_8 = normalize_8bit(ref)
mov_rigid_8 = normalize_8bit(mov_rigid)
sh, sw = h // DS, w // DS
ref_s = cv2.GaussianBlur(cv2.resize(ref_8, (sw, sh)), (BLUR_K, BLUR_K), 0)
mov_s = cv2.GaussianBlur(cv2.resize(mov_rigid_8, (sw, sh)), (BLUR_K, BLUR_K), 0)

# Baseline metrics (rigid only)
ref_n = normalize_float(ref)
mov_rigid_n = normalize_float(mov_rigid)
ncc_rigid = compute_ncc(ref_n, mov_rigid_n, mask_both)
mse_rigid = compute_mse(ref_n, mov_rigid_n, mask_both)
print(f"Rigid baseline: NCC={ncc_rigid:.4f}, MSE={mse_rigid:.6f}")

# Output dir
out_dir = f'{BASE}/png_exports/z_stitch_tuning/{pair_key}'
os.makedirs(out_dir, exist_ok=True)

# === TWO-PASS MODE ===
# Pass 1: winsize=76, current params, iter=10
# Pass 2: winsize=128, fsk=65, gentle cleanup, iter=5
TWO_PASS = False

iter_nccs = []
iter_mses = []
iter_mean_disps = []
iter_max_disps = []
iter_labels = []

if TWO_PASS:
    PASS1_ITER = 10
    PASS2_WINSIZE = 128
    PASS2_BLUR_K = 21
    PASS2_FSK = 65
    PASS2_ITER = 5

    print(f"\n=== PASS 1: winsize={WINSIZE}, blur_k={BLUR_K}, fsk={FLOW_SMOOTH_K}, iter 1-{PASS1_ITER} ===")
    for itr in range(1, PASS1_ITER + 1):
        print(f"  P1 iter={itr}/{PASS1_ITER}...", end=" ", flush=True)

        flow_small = cv2.calcOpticalFlowFarneback(
            ref_s, mov_s, flow=None, pyr_scale=0.5, levels=LEVELS,
            winsize=WINSIZE, iterations=itr, poly_n=POLY_N, poly_sigma=POLY_SIGMA,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow = np.zeros((h, w, 2), dtype=np.float32)
        flow[:, :, 0] = cv2.resize(flow_small[:, :, 0], (w, h)) * DS
        flow[:, :, 1] = cv2.resize(flow_small[:, :, 1], (w, h)) * DS
        flow[:, :, 0] = cv2.GaussianBlur(flow[:, :, 0], (FLOW_SMOOTH_K, FLOW_SMOOTH_K), 0)
        flow[:, :, 1] = cv2.GaussianBlur(flow[:, :, 1], (FLOW_SMOOTH_K, FLOW_SMOOTH_K), 0)

        mov_def = apply_flow(mov_rigid, flow)
        mov_def_n = normalize_float(mov_def)

        ncc = compute_ncc(ref_n, mov_def_n, mask_both)
        mse = compute_mse(ref_n, mov_def_n, mask_both)
        mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        mag_masked = mag[mask_both]
        mean_disp = float(np.mean(mag_masked)) if len(mag_masked) > 0 else 0
        max_disp = float(np.max(mag_masked)) if len(mag_masked) > 0 else 0

        iter_nccs.append(ncc)
        iter_mses.append(mse)
        iter_mean_disps.append(mean_disp)
        iter_max_disps.append(max_disp)
        iter_labels.append(f"P1:{itr}")
        print(f"NCC={ncc:.4f}  MSE={mse:.6f}  disp={mean_disp:.1f}/{max_disp:.1f}px")

    # Use best pass 1 flow (iter=PASS1_ITER)
    # mov_def and flow are already from the last iteration
    p1_flow = flow.copy()
    mov_after_p1 = mov_def.copy()

    print(f"\n=== PASS 2: winsize={PASS2_WINSIZE}, blur_k={PASS2_BLUR_K}, fsk={PASS2_FSK}, iter 1-{PASS2_ITER} ===")
    # Prepare pass 2 inputs: ref vs warped moving
    mov_p1_8 = normalize_8bit(mov_after_p1)
    ref_s2 = cv2.GaussianBlur(cv2.resize(ref_8, (sw, sh)), (PASS2_BLUR_K, PASS2_BLUR_K), 0)
    mov_s2 = cv2.GaussianBlur(cv2.resize(mov_p1_8, (sw, sh)), (PASS2_BLUR_K, PASS2_BLUR_K), 0)

    for itr in range(1, PASS2_ITER + 1):
        print(f"  P2 iter={itr}/{PASS2_ITER}...", end=" ", flush=True)

        flow2_small = cv2.calcOpticalFlowFarneback(
            ref_s2, mov_s2, flow=None, pyr_scale=0.5, levels=LEVELS,
            winsize=PASS2_WINSIZE, iterations=itr, poly_n=POLY_N, poly_sigma=POLY_SIGMA,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow2 = np.zeros((h, w, 2), dtype=np.float32)
        flow2[:, :, 0] = cv2.resize(flow2_small[:, :, 0], (w, h)) * DS
        flow2[:, :, 1] = cv2.resize(flow2_small[:, :, 1], (w, h)) * DS
        flow2[:, :, 0] = cv2.GaussianBlur(flow2[:, :, 0], (PASS2_FSK, PASS2_FSK), 0)
        flow2[:, :, 1] = cv2.GaussianBlur(flow2[:, :, 1], (PASS2_FSK, PASS2_FSK), 0)

        # Combined flow: apply pass2 on top of pass1 result
        mov_def2 = apply_flow(mov_after_p1, flow2)
        mov_def2_n = normalize_float(mov_def2)

        # Total displacement = pass1 + pass2
        total_flow = p1_flow + flow2
        ncc = compute_ncc(ref_n, mov_def2_n, mask_both)
        mse = compute_mse(ref_n, mov_def2_n, mask_both)
        mag = np.sqrt(total_flow[:, :, 0]**2 + total_flow[:, :, 1]**2)
        mag_masked = mag[mask_both]
        mean_disp = float(np.mean(mag_masked)) if len(mag_masked) > 0 else 0
        max_disp = float(np.max(mag_masked)) if len(mag_masked) > 0 else 0

        iter_nccs.append(ncc)
        iter_mses.append(mse)
        iter_mean_disps.append(mean_disp)
        iter_max_disps.append(max_disp)
        iter_labels.append(f"P2:{itr}")
        print(f"NCC={ncc:.4f}  MSE={mse:.6f}  disp={mean_disp:.1f}/{max_disp:.1f}px")

    mov_def = mov_def2
    mov_def_n = mov_def2_n

else:
    # Single pass mode
    print(f"\nRunning iterations 1 to {ITERATIONS}, recording metrics at each...")
    for itr in range(1, ITERATIONS + 1):
        print(f"  iter={itr}/{ITERATIONS}...", end=" ", flush=True)

        flow_small = cv2.calcOpticalFlowFarneback(
            ref_s, mov_s, flow=None, pyr_scale=0.5, levels=LEVELS,
            winsize=WINSIZE, iterations=itr, poly_n=POLY_N, poly_sigma=POLY_SIGMA,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        flow = np.zeros((h, w, 2), dtype=np.float32)
        flow[:, :, 0] = cv2.resize(flow_small[:, :, 0], (w, h)) * DS
        flow[:, :, 1] = cv2.resize(flow_small[:, :, 1], (w, h)) * DS
        flow[:, :, 0] = cv2.GaussianBlur(flow[:, :, 0], (FLOW_SMOOTH_K, FLOW_SMOOTH_K), 0)
        flow[:, :, 1] = cv2.GaussianBlur(flow[:, :, 1], (FLOW_SMOOTH_K, FLOW_SMOOTH_K), 0)

        mov_def = apply_flow(mov_rigid, flow)
        mov_def_n = normalize_float(mov_def)

        ncc = compute_ncc(ref_n, mov_def_n, mask_both)
        mse = compute_mse(ref_n, mov_def_n, mask_both)
        mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        mag_masked = mag[mask_both]
        mean_disp = float(np.mean(mag_masked)) if len(mag_masked) > 0 else 0
        max_disp = float(np.max(mag_masked)) if len(mag_masked) > 0 else 0

        iter_nccs.append(ncc)
        iter_mses.append(mse)
        iter_mean_disps.append(mean_disp)
        iter_max_disps.append(max_disp)
        iter_labels.append(str(itr))
        print(f"NCC={ncc:.4f}  MSE={mse:.6f}  disp={mean_disp:.1f}/{max_disp:.1f}px")

# Final values
ncc_def = iter_nccs[-1]
mse_def = iter_mses[-1]
mean_disp_final = iter_mean_disps[-1]
max_disp_final = iter_max_disps[-1]
print(f"\nFinal: NCC={ncc_def:.4f} (rigid={ncc_rigid:.4f}, delta={ncc_def - ncc_rigid:+.4f})")

# Save per-iteration results
result = {
    'pair': pair_key,
    'base_params': {'winsize': WINSIZE, 'blur_k': BLUR_K, 'ds': DS, 'levels': LEVELS},
    'user_params': {'flow_smooth_k': FLOW_SMOOTH_K, 'iterations': ITERATIONS},
    'rigid': {'ncc': ncc_rigid, 'mse': mse_rigid},
    'per_iteration': {
        'labels': iter_labels,
        'ncc': iter_nccs, 'mse': iter_mses,
        'mean_disp': iter_mean_disps, 'max_disp': iter_max_disps,
    },
}
with open(f'{out_dir}/tuning_result_win{WINSIZE}_fsk{FLOW_SMOOTH_K}_blur{BLUR_K}_iter{ITERATIONS}.json', 'w') as f:
    json.dump(result, f, indent=2)

# ============================================================
# Plot: 4-panel metric curves vs iteration
print("\nGenerating plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
title = f'{pair_key} — win={WINSIZE}, fsk={FLOW_SMOOTH_K}, blur={BLUR_K}'
if TWO_PASS:
    title += f' | 2-pass (P2: win={PASS2_WINSIZE})'
fig.suptitle(title, fontsize=14)

xs = range(len(iter_nccs))

# NCC
ax = axes[0, 0]
ax.plot(xs, iter_nccs, 'o-', color='tab:blue', markersize=5, linewidth=2)
ax.axhline(y=ncc_rigid, color='red', linestyle='--', linewidth=1.5, label=f'Rigid ({ncc_rigid:.4f})')
if TWO_PASS:
    ax.axvline(x=PASS1_ITER - 0.5, color='gray', linestyle=':', linewidth=1.5, label='Pass 1|2 boundary')
ax.set_xticks(xs)
ax.set_xticklabels(iter_labels, rotation=45, fontsize=7)
ax.set_xlabel('Step')
ax.set_ylabel('NCC')
ax.set_title('NCC vs Step (higher = better)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# MSE
ax = axes[0, 1]
ax.plot(xs, iter_mses, 'o-', color='tab:orange', markersize=5, linewidth=2)
ax.axhline(y=mse_rigid, color='red', linestyle='--', linewidth=1.5, label=f'Rigid ({mse_rigid:.6f})')
if TWO_PASS:
    ax.axvline(x=PASS1_ITER - 0.5, color='gray', linestyle=':', linewidth=1.5)
ax.set_xticks(xs)
ax.set_xticklabels(iter_labels, rotation=45, fontsize=7)
ax.set_xlabel('Step')
ax.set_ylabel('MSE')
ax.set_title('MSE vs Step (lower = better)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Mean displacement
ax = axes[1, 0]
ax.plot(xs, iter_mean_disps, 'o-', color='tab:green', markersize=5, linewidth=2)
if TWO_PASS:
    ax.axvline(x=PASS1_ITER - 0.5, color='gray', linestyle=':', linewidth=1.5)
ax.set_xticks(xs)
ax.set_xticklabels(iter_labels, rotation=45, fontsize=7)
ax.set_xlabel('Step')
ax.set_ylabel('Mean Displacement (px)')
ax.set_title('Mean Displacement vs Step')
ax.grid(True, alpha=0.3)

# Max displacement
ax = axes[1, 1]
ax.plot(xs, iter_max_disps, 'o-', color='tab:red', markersize=5, linewidth=2)
if TWO_PASS:
    ax.axvline(x=PASS1_ITER - 0.5, color='gray', linestyle=':', linewidth=1.5)
ax.set_xticks(xs)
ax.set_xticklabels(iter_labels, rotation=45, fontsize=7)
ax.set_xlabel('Step')
ax.set_ylabel('Max Displacement (px)')
ax.set_title('Max Displacement vs Step')
ax.grid(True, alpha=0.3)

plt.tight_layout()
tag = '_2pass' if TWO_PASS else ''
plot_path = f'{out_dir}/metrics_win{WINSIZE}_fsk{FLOW_SMOOTH_K}_blur{BLUR_K}_iter{ITERATIONS}{tag}.png'
plt.savefig(plot_path, dpi=150)
print(f"Saved: {plot_path}")

# ============================================================
# Side by side overlay: rigid | final deformable
# Recompute final flow (already have it from last iteration)
ov_rigid = make_overlay(ref_n, mov_rigid_n, mask_both)
ov_def = make_overlay(ref_n, mov_def_n, mask_both)

gap = 20
combined = np.zeros((h, w * 2 + gap, 3), dtype=np.uint8)
combined[:, :w] = ov_rigid
combined[:, w + gap:] = ov_def

cv2.putText(combined, f'IOU RIGID (NCC={ncc_rigid:.4f})', (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3, cv2.LINE_AA)
cv2.putText(combined, f'DEFORM fsk={FLOW_SMOOTH_K} iter={ITERATIONS} (NCC={ncc_def:.4f})',
            (w + gap + 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
info = f'disp: mean={mean_disp_final:.0f}px max={max_disp_final:.0f}px  NCC delta={ncc_def - ncc_rigid:+.4f}'
cv2.putText(combined, info, (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2, cv2.LINE_AA)

small = cv2.resize(combined, (combined.shape[1] // 4, combined.shape[0] // 4))
out_path = f'{out_dir}/overlay_win{WINSIZE}_fsk{FLOW_SMOOTH_K}_blur{BLUR_K}_iter{ITERATIONS}{tag}.png'
cv2.imwrite(out_path, cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
print(f"Saved: {out_path}")
print(f"\nAll outputs in: {out_dir}/")
