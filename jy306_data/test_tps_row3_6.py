#!/usr/bin/env python3
"""Test Thin-Plate Spline (TPS) registration on row3_6 — compare to affine-only.
TPS is purely landmark-driven: no image intensity matching."""
import numpy as np
import cv2
import os
import json
import tifffile
from scipy.ndimage import median_filter
from collections import defaultdict

BASE = '/Users/neurolab/neuroinformatics/margaret'
IV_XY_UM = 0.6835
IV_Z_UM = 3.0
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0
TILE = 'row3_6'

print("Loading stitch params...")
with open(f'{BASE}/registration_video/stitch_iou_only_params.json') as f:
    params = json.load(f)

print("Loading JY306 in-vivo...")
iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv = iv_vol_raw.shape[0]

print("Median filter (for registration)...")
iv_vol_filt = np.zeros_like(iv_vol_raw)
for z in range(nz_iv):
    bg = median_filter(iv_vol_raw[z], size=15)
    iv_vol_filt[z] = np.clip(iv_vol_raw[z] - bg, 0, None)

print(f"Loading nd2 slices for {TILE}...")
img_dir = f'{BASE}/png_exports/registration_video/{TILE}'
nd2_slices = []
for zi in range(12):
    img = cv2.imread(f'{img_dir}/GFP_z{zi:03d}.png', cv2.IMREAD_UNCHANGED)
    if img is None:
        nd2_slices.append(np.zeros((4200, 4200), dtype=np.float32))
    else:
        nd2_slices.append(img.astype(np.float32))
nd2_slices = np.array(nd2_slices)
nd2_h, nd2_w = nd2_slices.shape[1], nd2_slices.shape[2]

print("Loading transform data...")
d = np.load(f'{BASE}/png_exports/registration_per_tile_elastix/{TILE}/transform_{TILE}.npz', allow_pickle=True)
pcd_iv = d['pcd_iv']
ev_nd2 = d['ev_nd2']
predicted = d['predicted']
errors = d['errors']
M_inv = d['M_inv']
offset_inv = d['offset_inv']
nd2_z_gauss = d['nd2_z_gauss']
N_LM = len(pcd_iv)

# Group by z-pairs
z_pair_to_lm = defaultdict(list)
for i in range(N_LM):
    z_iv = int(round(np.clip(pcd_iv[i, 0], 0, nz_iv - 1)))
    z_nd2 = int(round(np.clip(nd2_z_gauss[i], 0, 11)))
    z_pair_to_lm[(z_iv, z_nd2)].append(i)

# Group by z_iv for TPS (pool all nd2_z for same z_iv)
z_iv_to_lm = defaultdict(list)
for i in range(N_LM):
    z_iv = int(round(np.clip(pcd_iv[i, 0], 0, nz_iv - 1)))
    z_iv_to_lm[z_iv].append(i)


def ncc(a, b):
    mask = (a > 0) & (b > 0)
    if mask.sum() < 100:
        return -1.0
    am = a[mask].astype(np.float64)
    bm = b[mask].astype(np.float64)
    am -= am.mean(); bm -= bm.mean()
    d2 = np.sqrt((am**2).sum() * (bm**2).sum())
    return float((am * bm).sum() / d2) if d2 > 1e-10 else -1.0


def tps_coeffs(src_pts, dst_pts, lam=0.0):
    """Compute TPS coefficients mapping src → dst.
    src_pts, dst_pts: list of (x, y) tuples.
    lam: regularization (0 = exact interpolation).
    Returns (W, a) where W is (N,2) weights and a is (3,2) affine part."""
    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)
    n = len(src)

    # TPS kernel: U(r) = r^2 * log(r)
    def U(r):
        r = np.maximum(r, 1e-10)
        return r**2 * np.log(r)

    # Distance matrix
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = np.sqrt((src[i, 0] - src[j, 0])**2 + (src[i, 1] - src[j, 1])**2)
    K = U(dists)

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

    W = params[:n]  # (n, 2)
    a = params[n:]  # (3, 2) — [a0; ax; ay]
    return W, a, src


def tps_transform_point(W, a, src, query_pt):
    """Transform a single point through TPS."""
    qx, qy = query_pt
    n = len(src)

    # Affine part
    result = a[0] + a[1] * qx + a[2] * qy

    # Non-linear part
    for i in range(n):
        r = np.sqrt((qx - src[i, 0])**2 + (qy - src[i, 1])**2)
        if r > 1e-10:
            u = r**2 * np.log(r)
            result = result + W[i] * u

    return (float(result[0]), float(result[1]))


# ============================================================
# GLOBAL TPS: pool ALL landmarks across all z_iv/z_nd2
# TPS is point-to-point — no images needed.
# The affine error has a smooth spatial pattern; all 47 landmarks
# sample this pattern regardless of z.
# ============================================================

# Build global predicted→actual correspondences
all_pred = []
all_actual = []
all_info = []  # (lm_index, z_iv, z_nd2)
for i in range(N_LM):
    pr_x = float(predicted[i, 0] / ND2_XY_UM)
    pr_y = float(predicted[i, 1] / ND2_XY_UM)
    ex_x = float(ev_nd2[i, 0])
    ex_y = float(ev_nd2[i, 1])
    z_iv = int(round(np.clip(pcd_iv[i, 0], 0, nz_iv - 1)))
    z_nd2 = int(round(np.clip(nd2_z_gauss[i], 0, 11)))
    all_pred.append((pr_x, pr_y))
    all_actual.append((ex_x, ex_y))
    all_info.append((i, z_iv, z_nd2))

# Sweep lambda values
lambdas = [0, 1, 10, 100, 1000, 10000, 100000]

print(f"\n{'='*70}")
print(f"  {TILE}: Global TPS with {N_LM} landmarks — lambda sweep (LOO)")
print(f"{'='*70}\n")

best_lam = 0
best_mean = 999

for lam in lambdas:
    loo_errs = []
    for j in range(N_LM):
        src_loo = [all_pred[k] for k in range(N_LM) if k != j]
        dst_loo = [all_actual[k] for k in range(N_LM) if k != j]
        W, a, src_arr = tps_coeffs(src_loo, dst_loo, lam=lam)
        tps_pt = tps_transform_point(W, a, src_arr, all_pred[j])
        err = np.sqrt((tps_pt[0] - all_actual[j][0])**2 +
                      (tps_pt[1] - all_actual[j][1])**2)
        loo_errs.append(err)

    aff_errs = [np.sqrt((all_pred[j][0] - all_actual[j][0])**2 +
                         (all_pred[j][1] - all_actual[j][1])**2) for j in range(N_LM)]

    n_better = sum(1 for j in range(N_LM) if loo_errs[j] < aff_errs[j])
    mean_loo = np.mean(loo_errs)
    mean_aff = np.mean(aff_errs)

    if mean_loo < best_mean:
        best_mean = mean_loo
        best_lam = lam

    print(f"  lambda={lam:>7}: TPS_LOO={mean_loo:.1f}px | aff={mean_aff:.1f}px | better={n_better}/{N_LM} ({100*n_better/N_LM:.0f}%) | median_loo={np.median(loo_errs):.1f}px")

print(f"\n  Best lambda: {best_lam} (mean LOO = {best_mean:.1f}px)")

# Detailed results with best lambda
print(f"\n{'='*70}")
print(f"  Detailed LOO results with lambda={best_lam}")
print(f"{'='*70}")
print(f"{'LM':>5} {'z_iv':>5} {'z_nd2':>5} {'aff_px':>8} {'tps_px':>8} {'better':>7}")
print(f"{'-'*50}")

results = []
for j in range(N_LM):
    src_loo = [all_pred[k] for k in range(N_LM) if k != j]
    dst_loo = [all_actual[k] for k in range(N_LM) if k != j]
    W, a, src_arr = tps_coeffs(src_loo, dst_loo, lam=best_lam)
    tps_pt = tps_transform_point(W, a, src_arr, all_pred[j])

    tps_err = np.sqrt((tps_pt[0] - all_actual[j][0])**2 +
                      (tps_pt[1] - all_actual[j][1])**2)
    aff_err = np.sqrt((all_pred[j][0] - all_actual[j][0])**2 +
                       (all_pred[j][1] - all_actual[j][1])**2)

    i, z_iv, z_nd2 = all_info[j]
    better = tps_err < aff_err
    results.append({'i': i, 'z_iv': z_iv, 'z_nd2': z_nd2,
                    'aff_err_px': aff_err, 'tps_err_px': tps_err})
    print(f"{i:5d} {z_iv:5d} {z_nd2:5d} {aff_err:8.1f} {tps_err:8.1f} {'YES' if better else 'no':>7}")

n_better = sum(1 for r in results if r['tps_err_px'] < r['aff_err_px'])
aff_mean = np.mean([r['aff_err_px'] for r in results])
tps_mean = np.mean([r['tps_err_px'] for r in results])
print(f"\nTPS better: {n_better}/{len(results)} ({100*n_better/len(results):.0f}%)")
print(f"Mean affine: {aff_mean:.1f}px ({aff_mean*ND2_XY_UM:.1f}µm) | Mean TPS: {tps_mean:.1f}px ({tps_mean*ND2_XY_UM:.1f}µm)")
