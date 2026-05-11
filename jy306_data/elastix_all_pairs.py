"""
Run SimpleElastix B-spline deformable registration for all 20 pairs.
Saves per-pair overlay, per-iteration metric plot, and final contact sheet.
"""

import numpy as np
import nd2
import cv2
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import SimpleITK as sitk

BASE = '/Users/neurolab/neuroinformatics/margaret'

TILE_ORDER = [
    'row1_1', 'row1_2', 'row1_3',
    'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
    'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
    'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
    'row5_1',
]

# Elastix params
DS = 2
GRID_SPACING = 32       # B-spline control point spacing (pixels at DS resolution)
N_RESOLUTIONS = 4
MAX_ITER = 500
ITER_CHECKPOINTS = [10, 25, 50, 100, 150, 200, 300, 400, 500]
METRIC = 'AdvancedNormalizedCorrelation'

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
    if len(a_m) == 0: return 0.0
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

def run_elastix_at_iter(ref_ds, mov_ds, n_iter):
    """Run Elastix for n_iter iterations and return warped DS image."""
    ref_itk = sitk.GetImageFromArray(ref_ds.astype(np.float32))
    mov_itk = sitk.GetImageFromArray(mov_ds.astype(np.float32))
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(ref_itk)
    elastix.SetMovingImage(mov_itk)
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
    elastix.SetParameterMap(pm)
    elastix.LogToConsoleOff()
    elastix.Execute()
    return sitk.GetArrayFromImage(elastix.GetResultImage()).astype(np.float32)

# ============================================================
print("Loading masks and IOU transforms...")
masks_data = np.load(f'{BASE}/registration_video/via_masks_v2.npz')
masks = {k: masks_data[k] for k in masks_data.files}

with open(f'{BASE}/registration_video/auto_align_transforms_iou.json', 'r') as f:
    iou_transforms = json.load(f)

pairs = [(TILE_ORDER[i], TILE_ORDER[i+1]) for i in range(len(TILE_ORDER)-1)]

# Load all boundary slices
print("Loading boundary slices...")
first_slices, last_slices = {}, {}
for key in TILE_ORDER:
    print(f"  {key}...", end=" ", flush=True)
    with nd2.ND2File(nd2_path(key)) as f:
        data = f.asarray()
    mask = masks.get(key, np.ones((4200, 4200), dtype=np.uint8))
    first_slices[key] = data[0, 1].astype(np.float32) * mask.astype(np.float32)
    last_slices[key] = data[-1, 1].astype(np.float32) * mask.astype(np.float32)
    print("done")

out_dir = f'{BASE}/png_exports/z_stitch_elastix'
os.makedirs(out_dir, exist_ok=True)

all_results = {}
overlay_thumbs = []

params_str = f'DS={DS}, grid={GRID_SPACING}px, res={N_RESOLUTIONS}, iter={MAX_ITER}, metric={METRIC}'
print(f"\nParams: {params_str}\n")

for idx, (key_a, key_b) in enumerate(pairs):
    is_cross = key_a.split('_')[0] != key_b.split('_')[0]
    tag = ' [CROSS]' if is_cross else ''
    pair_key = f'{key_a}_to_{key_b}'
    print(f"=== [{idx+1}/20] {pair_key}{tag} ===")

    ref = last_slices[key_a]
    mov = first_slices[key_b]
    h, w = ref.shape

    mask_a = masks.get(key_a, np.ones((h, w), dtype=np.uint8))
    mask_b = masks.get(key_b, np.ones((h, w), dtype=np.uint8))

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
    print(f"  Rigid: NCC={ncc_rigid:.4f}  MSE={mse_rigid:.6f}")

    # Downsample
    sh, sw = h // DS, w // DS
    ref_ds = cv2.resize(normalize_8bit(ref), (sw, sh)).astype(np.float32)
    mov_ds = cv2.resize(normalize_8bit(mov_rigid), (sw, sh)).astype(np.float32)

    # Run Elastix at each checkpoint, compute NCC directly from result
    pair_dir = f'{out_dir}/{pair_key}'
    os.makedirs(pair_dir, exist_ok=True)

    print(f"  Running Elastix at checkpoints: {ITER_CHECKPOINTS}")
    iter_nccs, iter_mses = [], []
    mov_warped_n = mov_rigid_n  # fallback
    ncc_elx, mse_elx = ncc_rigid, mse_rigid

    for cp in ITER_CHECKPOINTS:
        print(f"    iter={cp}...", end=" ", flush=True)
        try:
            result_ds = run_elastix_at_iter(ref_ds, mov_ds, cp)
            result_full = cv2.resize(result_ds, (w, h), interpolation=cv2.INTER_LINEAR)
            result_n = normalize_float(result_full)
            ncc = compute_ncc(ref_n, result_n, mask_both)
            mse = compute_mse(ref_n, result_n, mask_both)
            iter_nccs.append(ncc)
            iter_mses.append(mse)
            print(f"NCC={ncc:.4f}")
            if cp == ITER_CHECKPOINTS[-1]:
                mov_warped_n = result_n
                ncc_elx = ncc
                mse_elx = mse
        except Exception as e:
            print(f"FAILED: {e}")
            iter_nccs.append(ncc_rigid)
            iter_mses.append(mse_rigid)

    # Save per-pair metric plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{pair_key}{tag}\n{params_str}', fontsize=10)

    ax = axes[0]
    ax.plot(ITER_CHECKPOINTS, iter_nccs, 'o-', color='tab:blue', linewidth=2, markersize=6)
    ax.axhline(y=ncc_rigid, color='red', linestyle='--', linewidth=1.5, label=f'Rigid ({ncc_rigid:.4f})')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('NCC')
    ax.set_title('NCC vs Iterations (directly measured)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ITER_CHECKPOINTS, iter_mses, 'o-', color='tab:orange', linewidth=2, markersize=6)
    ax.axhline(y=mse_rigid, color='red', linestyle='--', linewidth=1.5, label=f'Rigid ({mse_rigid:.6f})')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('MSE')
    ax.set_title(f'MSE vs Iterations  |  Final NCC={ncc_elx:.4f}  Δ={ncc_elx-ncc_rigid:+.4f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{pair_dir}/metrics.png', dpi=120)
    plt.close()

    # Save overlay: rigid | elastix side by side
    ov_rigid = make_overlay(ref_n, mov_rigid_n, mask_both)
    ov_elx = make_overlay(ref_n, mov_warped_n, mask_both)

    gap = 20
    combined = np.zeros((h, w * 2 + gap, 3), dtype=np.uint8)
    combined[:, :w] = ov_rigid
    combined[:, w + gap:] = ov_elx

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, f'IOU RIGID (NCC={ncc_rigid:.4f})', (20, 60), font, 2.0, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, f'ELASTIX B-spline (NCC={ncc_elx:.4f})', (w + gap + 20, 60), font, 2.0, (255, 200, 0), 3, cv2.LINE_AA)
    cv2.putText(combined, params_str, (20, h - 30), font, 0.9, (180, 180, 180), 2, cv2.LINE_AA)

    small = cv2.resize(combined, (combined.shape[1] // 4, combined.shape[0] // 4))
    cv2.imwrite(f'{pair_dir}/overlay.png', cv2.cvtColor(small, cv2.COLOR_RGB2BGR))

    overlay_thumbs.append((pair_key, tag, small, ncc_rigid, ncc_elx))

    all_results[pair_key] = {
        'ncc_rigid': ncc_rigid, 'mse_rigid': mse_rigid,
        'ncc_elastix': ncc_elx, 'mse_elastix': mse_elx,
        'ncc_improvement': ncc_elx - ncc_rigid,
        'is_cross': is_cross,
        'params': {'ds': DS, 'grid_spacing': GRID_SPACING, 'n_resolutions': N_RESOLUTIONS,
                   'max_iter': MAX_ITER, 'metric': METRIC},
    }
    print()

# Save results JSON
with open(f'{out_dir}/elastix_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"Results saved: {out_dir}/elastix_results.json")

# ============================================================
# Contact sheet
print("Building contact sheet...")
cols = 4
rows_grid = (len(pairs) + cols - 1) // cols
sample = overlay_thumbs[0][2]
th, tw = sample.shape[:2]
gap_s, label_h = 10, 60
sheet_w = cols * (tw + gap_s) + gap_s
sheet_h = rows_grid * (th + label_h + gap_s) + gap_s
sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

for i, (pk, tag, thumb, ncc_r, ncc_e) in enumerate(overlay_thumbs):
    r, c = divmod(i, cols)
    x0 = gap_s + c * (tw + gap_s)
    y0 = gap_s + r * (th + label_h + gap_s)
    sheet[y0:y0+th, x0:x0+tw] = thumb
    cv2.putText(sheet, f'{pk}{tag}', (x0, y0+th+18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)
    cv2.putText(sheet, f'R={ncc_r:.3f} E={ncc_e:.3f} d={ncc_e-ncc_r:+.3f}',
                (x0, y0+th+38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1, cv2.LINE_AA)

cv2.imwrite(f'{out_dir}/contact_sheet.png', sheet)
print(f"Contact sheet: {out_dir}/contact_sheet.png")

# Summary bar chart
print("Building summary chart...")
fig, ax = plt.subplots(figsize=(16, 5))
pks = list(all_results.keys())
nccs_r = [all_results[pk]['ncc_rigid'] for pk in pks]
nccs_e = [all_results[pk]['ncc_elastix'] for pk in pks]
x = np.arange(len(pks))
w_bar = 0.35
ax.bar(x - w_bar/2, nccs_r, w_bar, label='Rigid', color='#e74c3c', alpha=0.8)
ax.bar(x + w_bar/2, nccs_e, w_bar, label='Elastix', color='#2ecc71', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([pk.replace('_to_', '→') for pk in pks], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('NCC')
ax.set_title(f'All Pairs: Rigid vs Elastix B-spline\n{params_str}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{out_dir}/summary_ncc.png', dpi=150)
print(f"Summary: {out_dir}/summary_ncc.png")

print("\n=== DONE ===")
for pk, info in all_results.items():
    tag = ' [X]' if info['is_cross'] else ''
    print(f"  {pk}{tag}: rigid={info['ncc_rigid']:.4f} elastix={info['ncc_elastix']:.4f} Δ={info['ncc_improvement']:+.4f}")
