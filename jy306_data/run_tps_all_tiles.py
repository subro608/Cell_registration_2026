#!/usr/bin/env python3
"""TPS + IRLS registration for all 19 tiles.
Generates per-tile plots: IRLS iterations vs LOO error, NCC comparison.
Saves transform data and summary."""
import numpy as np
import cv2
import os
import json
import glob
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from collections import defaultdict

BASE = '/Users/neurolab/neuroinformatics/margaret'
IV_XY_UM = 0.6835
IV_Z_UM = 3.0
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0

OUT_BASE = f'{BASE}/png_exports/registration_per_tile_tps'
os.makedirs(OUT_BASE, exist_ok=True)

TPS_LAM = 20000
N_IRLS = 10


# ============================================================
# TPS functions
# ============================================================
def tps_coeffs_weighted(src_pts, dst_pts, weights, lam=0.0):
    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    n = len(src)

    dx = src[:, 0:1] - src[:, 0:1].T
    dy = src[:, 1:2] - src[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    r = np.maximum(dists, 1e-10)
    K = r**2 * np.log(r)
    np.fill_diagonal(K, 0)

    W_diag = np.diag(w)
    P = np.hstack([np.ones((n, 1)), src])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = W_diag @ K + lam * np.eye(n)
    L[:n, n:] = W_diag @ P
    L[n:, :n] = P.T

    rhs = np.zeros((n + 3, 2))
    rhs[:n] = W_diag @ dst

    try:
        params = np.linalg.solve(L, rhs)
    except np.linalg.LinAlgError:
        params = np.linalg.lstsq(L, rhs, rcond=None)[0]

    return params[:n], params[n:], src


def tps_transform(W, a, src, query_pt):
    qx, qy = query_pt
    result = a[0] + a[1] * qx + a[2] * qy
    for i in range(len(src)):
        r = np.sqrt((qx - src[i, 0])**2 + (qy - src[i, 1])**2)
        if r > 1e-10:
            result = result + W[i] * r**2 * np.log(r)
    return (float(result[0]), float(result[1]))


def tps_transform_grid(W, a, src, grid_x, grid_y):
    """Transform a grid of points through TPS. Returns (map_x, map_y)."""
    h, w = grid_x.shape
    map_x = np.zeros_like(grid_x, dtype=np.float32)
    map_y = np.zeros_like(grid_y, dtype=np.float32)

    # Precompute for vectorized kernel eval
    src_arr = np.array(src, dtype=np.float64)
    n = len(src_arr)

    for yi in range(h):
        for xi in range(w):
            qx, qy = float(grid_x[yi, xi]), float(grid_y[yi, xi])
            rx = qx - src_arr[:, 0]
            ry = qy - src_arr[:, 1]
            r = np.sqrt(rx**2 + ry**2)
            r = np.maximum(r, 1e-10)
            U = r**2 * np.log(r)
            val = a[0] + a[1] * qx + a[2] * qy + (W * U[:, None]).sum(axis=0)
            map_x[yi, xi] = float(val[0])
            map_y[yi, xi] = float(val[1])

    return map_x, map_y


def fit_tps_irls(pred_pts, actual_pts, lam, n_irls, track_iters=False):
    """Fit TPS with IRLS. Returns (W, a, src, weights, iter_history)."""
    n = len(pred_pts)
    weights = np.ones(n)
    history = []

    for it in range(n_irls):
        W, a, src = tps_coeffs_weighted(pred_pts, actual_pts, weights, lam=lam)

        # Training errors
        train_errs = []
        for k in range(n):
            pt = tps_transform(W, a, src, pred_pts[k])
            err = np.sqrt((pt[0] - actual_pts[k][0])**2 + (pt[1] - actual_pts[k][1])**2)
            train_errs.append(max(err, 0.1))

        if track_iters:
            # LOO error for this iteration
            loo_errs = []
            for j in range(n):
                src_loo = [pred_pts[k] for k in range(n) if k != j]
                dst_loo = [actual_pts[k] for k in range(n) if k != j]
                w_loo = np.array([weights[k] for k in range(n) if k != j])
                W_l, a_l, s_l = tps_coeffs_weighted(src_loo, dst_loo, w_loo, lam=lam)
                pt = tps_transform(W_l, a_l, s_l, pred_pts[j])
                err = np.sqrt((pt[0] - actual_pts[j][0])**2 + (pt[1] - actual_pts[j][1])**2)
                loo_errs.append(err)
            history.append({
                'iter': it,
                'train_mean': np.mean(train_errs),
                'train_median': np.median(train_errs),
                'loo_mean': np.mean(loo_errs),
                'loo_median': np.median(loo_errs),
                'n_better': sum(1 for le, ae in zip(loo_errs,
                    [np.sqrt((p[0]-a2[0])**2+(p[1]-a2[1])**2) for p, a2 in zip(pred_pts, actual_pts)])
                    if le < ae),
            })

        # Reweight
        med_err = max(np.median(train_errs), 0.1)
        for k in range(n):
            weights[k] = 1.0 if train_errs[k] <= med_err else med_err / train_errs[k]

    return W, a, src, weights, history


def ncc(a, b):
    mask = (a > 0) & (b > 0)
    if mask.sum() < 100:
        return -1.0
    am = a[mask].astype(np.float64)
    bm = b[mask].astype(np.float64)
    am -= am.mean(); bm -= bm.mean()
    d2 = np.sqrt((am**2).sum() * (bm**2).sum())
    return float((am * bm).sum() / d2) if d2 > 1e-10 else -1.0


def norm8(img):
    vals = img[img > 0]
    if len(vals) < 100:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


# ============================================================
# Load shared data
# ============================================================
print("Loading stitch params...")
with open(f'{BASE}/registration_video/stitch_iou_only_params.json') as f:
    params = json.load(f)
TILE_ORDER = params['tile_order']

print("Loading JY306 in-vivo...")
iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv = iv_vol_raw.shape[0]

print("Finding landmark files...")
lm_files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_nd2_native_*.npz'))
legacy = f'{BASE}/registration_video/landmarks_27_nd2_native.npz'
if os.path.exists(legacy):
    lm_files.append(legacy)

tile_lm_files = {}
for lm_file in lm_files:
    bn = os.path.basename(lm_file)
    if 'landmarks_27_nd2_native' in bn:
        tile = 'row2_1'
    else:
        tile = bn.replace('landmarks_nd2_native_', '').replace('.npz', '')
    if tile in TILE_ORDER:
        tile_lm_files[tile] = lm_file
print(f"  {len(tile_lm_files)} tiles with landmarks")


# ============================================================
# Process each tile
# ============================================================
summary = []

for tile in sorted(tile_lm_files.keys()):
    print(f"\n{'='*60}")
    print(f"  {tile}")
    print(f"{'='*60}")

    out_dir = f'{OUT_BASE}/{tile}'
    os.makedirs(out_dir, exist_ok=True)

    # Load transform data from elastix pipeline (has affine + landmarks)
    tfm_path = f'{BASE}/png_exports/registration_per_tile_elastix/{tile}/transform_{tile}.npz'
    if not os.path.exists(tfm_path):
        print(f"  SKIP: no transform data")
        continue

    d = np.load(tfm_path, allow_pickle=True)
    pcd_iv = d['pcd_iv']
    ev_nd2 = d['ev_nd2']
    predicted = d['predicted']
    errors = d['errors']
    M_inv = d['M_inv']
    offset_inv = d['offset_inv']
    nd2_z_gauss = d['nd2_z_gauss']
    N_LM = len(pcd_iv)

    if N_LM < 4:
        print(f"  SKIP: only {N_LM} landmarks")
        continue

    # Build correspondences
    pred_pts = [(float(predicted[i, 0] / ND2_XY_UM), float(predicted[i, 1] / ND2_XY_UM)) for i in range(N_LM)]
    actual_pts = [(float(ev_nd2[i, 0]), float(ev_nd2[i, 1])) for i in range(N_LM)]

    aff_errs = [np.sqrt((p[0]-a[0])**2 + (p[1]-a[1])**2) for p, a in zip(pred_pts, actual_pts)]
    aff_mean = np.mean(aff_errs)

    # Fit TPS + IRLS with iteration tracking
    print(f"  {N_LM} lm | affine: {aff_mean:.1f}px ({aff_mean*ND2_XY_UM:.1f}µm)")
    print(f"  Fitting TPS (lam={TPS_LAM}, IRLS={N_IRLS}) with LOO tracking...", flush=True)

    W, a, src, weights, history = fit_tps_irls(
        pred_pts, actual_pts, TPS_LAM, N_IRLS, track_iters=True)

    # Final LOO evaluation
    loo_errs = []
    for j in range(N_LM):
        src_loo = [pred_pts[k] for k in range(N_LM) if k != j]
        dst_loo = [actual_pts[k] for k in range(N_LM) if k != j]
        w_loo = np.array([weights[k] for k in range(N_LM) if k != j])
        W_l, a_l, s_l = tps_coeffs_weighted(src_loo, dst_loo, w_loo, lam=TPS_LAM)
        pt = tps_transform(W_l, a_l, s_l, pred_pts[j])
        err = np.sqrt((pt[0] - actual_pts[j][0])**2 + (pt[1] - actual_pts[j][1])**2)
        loo_errs.append(err)

    tps_mean = np.mean(loo_errs)
    n_better = sum(1 for ae, te in zip(aff_errs, loo_errs) if te < ae)
    print(f"  TPS: {tps_mean:.1f}px ({tps_mean*ND2_XY_UM:.1f}µm) | better={n_better}/{N_LM}")

    # ---- NCC comparison per z-pair ----
    # Load nd2 slices
    img_dir = f'{BASE}/png_exports/registration_video/{tile}'
    nd2_slices = []
    for zi in range(12):
        img = cv2.imread(f'{img_dir}/GFP_z{zi:03d}.png', cv2.IMREAD_UNCHANGED)
        if img is None:
            nd2_slices.append(np.zeros((4200, 4200), dtype=np.float32))
        else:
            nd2_slices.append(img.astype(np.float32))
    nd2_slices = np.array(nd2_slices)
    nd2_h, nd2_w = nd2_slices.shape[1], nd2_slices.shape[2]

    # Group by z-pairs
    z_pair_to_lm = defaultdict(list)
    for i in range(N_LM):
        z_iv = int(round(np.clip(pcd_iv[i, 0], 0, nz_iv - 1)))
        z_nd2 = int(round(np.clip(nd2_z_gauss[i], 0, 11)))
        z_pair_to_lm[(z_iv, z_nd2)].append(i)

    z_iv_best_nd2 = {}
    for (z_iv, z_nd2), lm_list in z_pair_to_lm.items():
        if z_iv not in z_iv_best_nd2 or len(lm_list) > len(z_pair_to_lm.get((z_iv, z_iv_best_nd2[z_iv]), [])):
            z_iv_best_nd2[z_iv] = z_nd2

    # Fit inverse TPS (actual → predicted) for image warping
    # This maps fixed-space coords → moving-space coords for cv2.remap
    W_inv, a_inv, src_inv, _, _ = fit_tps_irls(
        actual_pts, pred_pts, TPS_LAM, N_IRLS, track_iters=False)

    # Compute NCC per z_iv (using best z_nd2)
    DS = max(1, nd2_w // 600)
    ds_w, ds_h = nd2_w // DS, nd2_h // DS

    ncc_affine_list = []
    ncc_tps_list = []
    z_iv_labels = []

    for z_iv in sorted(z_iv_best_nd2.keys()):
        z_nd2 = z_iv_best_nd2[z_iv]
        nd2_sl = nd2_slices[z_nd2]

        # Affine warp
        M2d = np.array([
            [M_inv[2, 2], M_inv[2, 1], M_inv[2, 0] * z_nd2 + offset_inv[2]],
            [M_inv[1, 2], M_inv[1, 1], M_inv[1, 0] * z_nd2 + offset_inv[1]],
        ], dtype=np.float64)
        iv_affine = cv2.warpAffine(iv_vol_raw[z_iv], M2d, (nd2_w, nd2_h),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)

        ncc_aff = ncc(nd2_sl, iv_affine)

        # TPS warp: build remap at downscaled resolution
        grid_x_ds, grid_y_ds = np.meshgrid(
            np.linspace(0, nd2_w - 1, ds_w),
            np.linspace(0, nd2_h - 1, ds_h))

        # Inverse TPS: for each output pixel, find input pixel
        map_x_ds = np.zeros((ds_h, ds_w), dtype=np.float32)
        map_y_ds = np.zeros((ds_h, ds_w), dtype=np.float32)

        src_inv_arr = np.array(src_inv, dtype=np.float64)
        n_src = len(src_inv_arr)
        for yi in range(ds_h):
            for xi in range(ds_w):
                qx, qy = float(grid_x_ds[yi, xi]), float(grid_y_ds[yi, xi])
                rx = qx - src_inv_arr[:, 0]
                ry = qy - src_inv_arr[:, 1]
                r = np.sqrt(rx**2 + ry**2)
                r = np.maximum(r, 1e-10)
                U = r**2 * np.log(r)
                val = a_inv[0] + a_inv[1] * qx + a_inv[2] * qy + (W_inv * U[:, None]).sum(axis=0)
                map_x_ds[yi, xi] = float(val[0])
                map_y_ds[yi, xi] = float(val[1])

        # Upscale maps to full res
        map_x = cv2.resize(map_x_ds, (nd2_w, nd2_h), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(map_y_ds, (nd2_w, nd2_h), interpolation=cv2.INTER_LINEAR)

        # Apply TPS warp to affine result
        iv_tps = cv2.remap(iv_affine, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
        ncc_tps = ncc(nd2_sl, iv_tps)

        ncc_affine_list.append(ncc_aff)
        ncc_tps_list.append(ncc_tps)
        z_iv_labels.append(f"z={z_iv}")
        print(f"    z_iv={z_iv:2d}→z_nd2={z_nd2:2d}: NCC aff={ncc_aff:.3f} tps={ncc_tps:.3f}", flush=True)

    # ---- Plot 1: IRLS iterations vs LOO error ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    iters = [h['iter'] for h in history]
    loo_means = [h['loo_mean'] for h in history]
    loo_meds = [h['loo_median'] for h in history]
    train_means = [h['train_mean'] for h in history]
    ax.plot(iters, loo_means, 'b-o', label='LOO mean', linewidth=2)
    ax.plot(iters, loo_meds, 'b--s', label='LOO median', linewidth=1.5)
    ax.plot(iters, train_means, 'r-^', label='Train mean', linewidth=1.5)
    ax.axhline(aff_mean, color='gray', linestyle=':', label=f'Affine mean ({aff_mean:.1f}px)')
    ax.set_xlabel('IRLS iteration')
    ax.set_ylabel('Error (px)')
    ax.set_title(f'{tile}: IRLS convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: NCC comparison
    ax = axes[1]
    x = np.arange(len(z_iv_labels))
    w_bar = 0.35
    ax.bar(x - w_bar/2, ncc_affine_list, w_bar, label='Affine', color='steelblue', alpha=0.8)
    ax.bar(x + w_bar/2, ncc_tps_list, w_bar, label='TPS+IRLS', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(z_iv_labels, fontsize=8)
    ax.set_ylabel('NCC')
    ax.set_title(f'{tile}: NCC per z-slice')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Per-landmark error comparison
    ax = axes[2]
    ax.scatter(aff_errs, loo_errs, c=['green' if t < a else 'red' for a, t in zip(aff_errs, loo_errs)],
               s=30, alpha=0.7, edgecolors='k', linewidths=0.5)
    mx = max(max(aff_errs), max(loo_errs)) * 1.1
    ax.plot([0, mx], [0, mx], 'k--', alpha=0.5, label='equal')
    ax.set_xlabel('Affine error (px)')
    ax.set_ylabel('TPS+IRLS LOO error (px)')
    ax.set_title(f'{tile}: per-landmark ({n_better}/{N_LM} improved)')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{tile}: TPS+IRLS (λ={TPS_LAM}, {N_IRLS} iter) — '
                 f'aff={aff_mean:.1f}px→tps={tps_mean:.1f}px ({(1-tps_mean/aff_mean)*100:.0f}%)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plot_path = f'{out_dir}/tps_metrics_{tile}.png'
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    # Save transform
    np.savez(f'{out_dir}/tps_transform_{tile}.npz',
             W=W, a_coeff=a, src_pts=np.array(src), weights=weights,
             W_inv=W_inv, a_inv=a_inv, src_inv=np.array(src_inv),
             pred_pts=np.array(pred_pts), actual_pts=np.array(actual_pts),
             aff_errs=np.array(aff_errs), loo_errs=np.array(loo_errs),
             ncc_affine=np.array(ncc_affine_list), ncc_tps=np.array(ncc_tps_list),
             z_iv_labels=np.array(z_iv_labels),
             lam=TPS_LAM, n_irls=N_IRLS)
    print(f"  Saved transform: {out_dir}/tps_transform_{tile}.npz")

    mean_ncc_aff = np.mean(ncc_affine_list)
    mean_ncc_tps = np.mean(ncc_tps_list)
    summary.append((tile, N_LM, aff_mean, tps_mean, n_better, mean_ncc_aff, mean_ncc_tps))

    del nd2_slices

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*90}")
print(f"{'Tile':>10} {'N_lm':>5} {'Aff_px':>7} {'TPS_px':>7} {'Reduc':>6} {'Better':>8} {'NCC_aff':>8} {'NCC_tps':>8}")
print(f"{'='*90}")
for tile, n, aff, tps, nb, ncc_a, ncc_t in summary:
    pct = f"{(1-tps/aff)*100:.0f}%"
    print(f"{tile:>10} {n:5d} {aff:7.1f} {tps:7.1f} {pct:>6} {nb:4d}/{n:<3d} {ncc_a:8.3f} {ncc_t:8.3f}")

print(f"\nAll saved to {OUT_BASE}/")

# Summary plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
tiles = [s[0] for s in summary]
aff_means = [s[2] for s in summary]
tps_means = [s[3] for s in summary]
ncc_affs = [s[5] for s in summary]
ncc_tpss = [s[6] for s in summary]

ax = axes[0]
x = np.arange(len(tiles))
ax.bar(x - 0.2, [a * ND2_XY_UM for a in aff_means], 0.4, label='Affine', color='steelblue')
ax.bar(x + 0.2, [t * ND2_XY_UM for t in tps_means], 0.4, label='TPS+IRLS', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(tiles, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean error (µm)')
ax.set_title('Landmark error: Affine vs TPS+IRLS')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
ax.bar(x - 0.2, ncc_affs, 0.4, label='Affine', color='steelblue')
ax.bar(x + 0.2, ncc_tpss, 0.4, label='TPS+IRLS', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(tiles, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean NCC')
ax.set_title('NCC: Affine vs TPS+IRLS')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'All tiles: TPS+IRLS (λ={TPS_LAM}, IRLS={N_IRLS})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_BASE}/summary_all_tiles.png', dpi=150)
plt.close(fig)
print(f"Saved summary plot: {OUT_BASE}/summary_all_tiles.png")
