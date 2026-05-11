#!/usr/bin/env python3
"""Test different RBF kernels for landmark correction: TPS, Gaussian, Multiquadric.
Global pooling + lambda sweep + LOO cross-validation on row3_6 and row2_1."""
import numpy as np
import cv2
import os
import json
import tifffile
from scipy.ndimage import median_filter
from collections import defaultdict

BASE = '/Users/neurolab/neuroinformatics/margaret'
IV_XY_UM = 0.6835
ND2_XY_UM = 0.645

TILES = ['row2_1', 'row3_6']


def rbf_coeffs(src_pts, dst_pts, kernel='tps', lam=0.0, sigma=100.0):
    """Compute RBF interpolation coefficients mapping src → dst.
    Kernels: 'tps' (r²log r), 'gaussian' (exp(-r²/2σ²)), 'multiquadric' (√(r²+σ²))
    Returns (W, a, src_array)."""
    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)
    n = len(src)

    # Distance matrix
    dx = src[:, 0:1] - src[:, 0:1].T
    dy = src[:, 1:2] - src[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)

    if kernel == 'tps':
        r = np.maximum(dists, 1e-10)
        K = r**2 * np.log(r)
        np.fill_diagonal(K, 0)
    elif kernel == 'gaussian':
        K = np.exp(-dists**2 / (2 * sigma**2))
    elif kernel == 'multiquadric':
        K = np.sqrt(dists**2 + sigma**2)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Build system: [K + lam*I, P; P^T, 0] [W; a] = [dst; 0]
    P = np.hstack([np.ones((n, 1)), src])  # (n, 3)
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K + lam * np.eye(n)
    L[:n, n:] = P
    L[n:, :n] = P.T

    rhs = np.zeros((n + 3, 2))
    rhs[:n] = dst

    try:
        params = np.linalg.solve(L, rhs)
    except np.linalg.LinAlgError:
        params = np.linalg.lstsq(L, rhs, rcond=None)[0]

    W = params[:n]
    a = params[n:]
    return W, a, src


def rbf_transform_point(W, a, src, query_pt, kernel='tps', sigma=100.0):
    """Transform a single point through RBF."""
    qx, qy = query_pt
    n = len(src)

    result = a[0] + a[1] * qx + a[2] * qy

    for i in range(n):
        r = np.sqrt((qx - src[i, 0])**2 + (qy - src[i, 1])**2)
        if kernel == 'tps':
            if r > 1e-10:
                u = r**2 * np.log(r)
            else:
                u = 0.0
        elif kernel == 'gaussian':
            u = np.exp(-r**2 / (2 * sigma**2))
        elif kernel == 'multiquadric':
            u = np.sqrt(r**2 + sigma**2)
        result = result + W[i] * u

    return (float(result[0]), float(result[1]))


def loo_eval(pred_pts, actual_pts, kernel, lam, sigma=100.0):
    """Leave-one-out cross-validation. Returns list of (aff_err, rbf_err) per landmark."""
    n = len(pred_pts)
    results = []
    for j in range(n):
        src_loo = [pred_pts[k] for k in range(n) if k != j]
        dst_loo = [actual_pts[k] for k in range(n) if k != j]
        W, a, src_arr = rbf_coeffs(src_loo, dst_loo, kernel=kernel, lam=lam, sigma=sigma)
        rbf_pt = rbf_transform_point(W, a, src_arr, pred_pts[j], kernel=kernel, sigma=sigma)
        rbf_err = np.sqrt((rbf_pt[0] - actual_pts[j][0])**2 +
                          (rbf_pt[1] - actual_pts[j][1])**2)
        aff_err = np.sqrt((pred_pts[j][0] - actual_pts[j][0])**2 +
                          (pred_pts[j][1] - actual_pts[j][1])**2)
        results.append((aff_err, rbf_err))
    return results


# Load shared data
print("Loading in-vivo...")
iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv = iv_vol_raw.shape[0]

for tile in TILES:
    print(f"\n{'='*70}")
    print(f"  {tile}")
    print(f"{'='*70}")

    d = np.load(f'{BASE}/png_exports/registration_per_tile_elastix/{tile}/transform_{tile}.npz', allow_pickle=True)
    pcd_iv = d['pcd_iv']
    ev_nd2 = d['ev_nd2']
    predicted = d['predicted']
    errors = d['errors']
    nd2_z_gauss = d['nd2_z_gauss']
    N_LM = len(pcd_iv)

    # Build correspondences
    pred_pts = []
    actual_pts = []
    for i in range(N_LM):
        pr_x = float(predicted[i, 0] / ND2_XY_UM)
        pr_y = float(predicted[i, 1] / ND2_XY_UM)
        ex_x = float(ev_nd2[i, 0])
        ex_y = float(ev_nd2[i, 1])
        pred_pts.append((pr_x, pr_y))
        actual_pts.append((ex_x, ex_y))

    aff_mean = np.mean([np.sqrt((p[0]-a[0])**2 + (p[1]-a[1])**2) for p, a in zip(pred_pts, actual_pts)])
    print(f"  {N_LM} landmarks | affine mean: {aff_mean:.1f}px ({aff_mean*ND2_XY_UM:.1f}µm)\n")

    # Kernel + lambda/sigma sweep
    configs = []

    # TPS with lambda sweep
    for lam in [100, 1000, 5000, 10000, 20000, 50000, 100000]:
        configs.append(('tps', lam, 0))

    # Gaussian with sigma sweep (sigma in pixels)
    for sigma in [50, 100, 200, 400, 800]:
        for lam in [0, 1, 10, 100]:
            configs.append(('gaussian', lam, sigma))

    # Multiquadric with sigma sweep
    for sigma in [50, 100, 200, 400, 800]:
        for lam in [0, 1, 10, 100]:
            configs.append(('multiquadric', lam, sigma))

    print(f"  {'kernel':>12} {'lam':>8} {'sigma':>6} | {'mean_loo':>8} {'med_loo':>8} {'better':>8} {'worst':>8}")
    print(f"  {'-'*70}")

    best_mean = 999
    best_cfg = None
    best_results = None

    for kernel, lam, sigma in configs:
        res = loo_eval(pred_pts, actual_pts, kernel, lam, sigma)
        aff_errs = [r[0] for r in res]
        rbf_errs = [r[1] for r in res]
        mean_rbf = np.mean(rbf_errs)
        med_rbf = np.median(rbf_errs)
        n_better = sum(1 for a, r in res if r < a)
        worst = max(rbf_errs)

        if mean_rbf < best_mean:
            best_mean = mean_rbf
            best_cfg = (kernel, lam, sigma)
            best_results = res

        # Only print interesting ones
        if mean_rbf < aff_mean * 0.9:
            sig_str = f"{sigma:6d}" if sigma > 0 else f"{'n/a':>6}"
            print(f"  {kernel:>12} {lam:8d} {sig_str} | {mean_rbf:8.1f} {med_rbf:8.1f} {n_better:5d}/{N_LM} {worst:8.1f}")

    print(f"\n  BEST: {best_cfg[0]} lam={best_cfg[1]} sigma={best_cfg[2]}")
    print(f"  Mean: aff={aff_mean:.1f}px → rbf={best_mean:.1f}px ({(1-best_mean/aff_mean)*100:.0f}% reduction)")
    print(f"  In µm: aff={aff_mean*ND2_XY_UM:.1f}µm → rbf={best_mean*ND2_XY_UM:.1f}µm")

    # Detailed best results
    rbf_errs = [r[1] for r in best_results]
    aff_errs = [r[0] for r in best_results]
    n_better = sum(1 for a, r in best_results if r < a)
    print(f"  Better: {n_better}/{N_LM} ({100*n_better/N_LM:.0f}%)")
    print(f"  Median: aff={np.median(aff_errs):.1f}px → rbf={np.median(rbf_errs):.1f}px")
    print(f"  Worst:  aff={max(aff_errs):.1f}px → rbf={max(rbf_errs):.1f}px")
