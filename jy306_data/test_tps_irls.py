#!/usr/bin/env python3
"""TPS + IRLS: iteratively reweighted least squares for robust landmark correction.
Tests on row2_1 and row3_6, sweeps lambda + IRLS iterations."""
import numpy as np
import json
import tifffile

BASE = '/Users/neurolab/neuroinformatics/margaret'
ND2_XY_UM = 0.645
TILES = ['row2_1', 'row3_6']


def tps_coeffs_weighted(src_pts, dst_pts, weights, lam=0.0):
    """TPS with per-landmark weights. Higher weight = more influence."""
    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    n = len(src)

    # TPS kernel
    dx = src[:, 0:1] - src[:, 0:1].T
    dy = src[:, 1:2] - src[:, 1:2].T
    dists = np.sqrt(dx**2 + dy**2)
    r = np.maximum(dists, 1e-10)
    K = r**2 * np.log(r)
    np.fill_diagonal(K, 0)

    # Weight matrix
    W_diag = np.diag(w)

    # Build system: [W*K + lam*I, W*P; P^T, 0] [c; a] = [W*dst; 0]
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
    """Transform a point through TPS."""
    qx, qy = query_pt
    result = a[0] + a[1] * qx + a[2] * qy
    for i in range(len(src)):
        r = np.sqrt((qx - src[i, 0])**2 + (qy - src[i, 1])**2)
        if r > 1e-10:
            result = result + W[i] * r**2 * np.log(r)
    return (float(result[0]), float(result[1]))


def loo_irls(pred_pts, actual_pts, lam, n_irls=5):
    """LOO cross-validation with IRLS.
    For each left-out point, run IRLS on the remaining N-1 landmarks."""
    n = len(pred_pts)
    results = []

    for j in range(n):
        src_loo = [pred_pts[k] for k in range(n) if k != j]
        dst_loo = [actual_pts[k] for k in range(n) if k != j]
        n_loo = len(src_loo)

        # IRLS iterations
        weights = np.ones(n_loo)
        for it in range(n_irls):
            W, a, src_arr = tps_coeffs_weighted(src_loo, dst_loo, weights, lam=lam)

            # Compute training errors
            train_errs = []
            for k in range(n_loo):
                pt = tps_transform(W, a, src_arr, src_loo[k])
                err = np.sqrt((pt[0] - dst_loo[k][0])**2 + (pt[1] - dst_loo[k][1])**2)
                train_errs.append(max(err, 0.1))  # avoid div by zero

            # Reweight: Huber-like — normal weight if err < median, downweight if high
            med_err = max(np.median(train_errs), 0.1)
            for k in range(n_loo):
                if train_errs[k] <= med_err:
                    weights[k] = 1.0
                else:
                    weights[k] = med_err / train_errs[k]

        # Final model — transform the left-out point
        tps_pt = tps_transform(W, a, src_arr, pred_pts[j])
        tps_err = np.sqrt((tps_pt[0] - actual_pts[j][0])**2 +
                          (tps_pt[1] - actual_pts[j][1])**2)
        aff_err = np.sqrt((pred_pts[j][0] - actual_pts[j][0])**2 +
                          (pred_pts[j][1] - actual_pts[j][1])**2)
        results.append((aff_err, tps_err))

    return results


def loo_plain(pred_pts, actual_pts, lam):
    """LOO without IRLS (baseline)."""
    n = len(pred_pts)
    results = []
    for j in range(n):
        src_loo = [pred_pts[k] for k in range(n) if k != j]
        dst_loo = [actual_pts[k] for k in range(n) if k != j]
        weights = np.ones(len(src_loo))
        W, a, src_arr = tps_coeffs_weighted(src_loo, dst_loo, weights, lam=lam)
        pt = tps_transform(W, a, src_arr, pred_pts[j])
        tps_err = np.sqrt((pt[0] - actual_pts[j][0])**2 + (pt[1] - actual_pts[j][1])**2)
        aff_err = np.sqrt((pred_pts[j][0] - actual_pts[j][0])**2 +
                          (pred_pts[j][1] - actual_pts[j][1])**2)
        results.append((aff_err, tps_err))
    return results


# Load data
print("Loading in-vivo (for landmark data)...")

for tile in TILES:
    print(f"\n{'='*70}")
    print(f"  {tile}")
    print(f"{'='*70}")

    d = np.load(f'{BASE}/png_exports/registration_per_tile_elastix/{tile}/transform_{tile}.npz', allow_pickle=True)
    ev_nd2 = d['ev_nd2']
    predicted = d['predicted']
    N_LM = len(ev_nd2)

    pred_pts = []
    actual_pts = []
    for i in range(N_LM):
        pred_pts.append((float(predicted[i, 0] / ND2_XY_UM), float(predicted[i, 1] / ND2_XY_UM)))
        actual_pts.append((float(ev_nd2[i, 0]), float(ev_nd2[i, 1])))

    aff_errs_all = [np.sqrt((p[0]-a[0])**2 + (p[1]-a[1])**2) for p, a in zip(pred_pts, actual_pts)]
    aff_mean = np.mean(aff_errs_all)
    print(f"  {N_LM} landmarks | affine: mean={aff_mean:.1f}px ({aff_mean*ND2_XY_UM:.1f}µm)\n")

    # Test configs
    print(f"  {'config':>30} | {'mean':>6} {'median':>7} {'better':>8} {'worst':>7} | {'vs_aff':>7}")
    print(f"  {'-'*80}")

    best_mean = 999
    best_label = ""

    for lam in [1000, 5000, 10000, 20000, 50000, 100000]:
        # Plain TPS (no IRLS)
        res = loo_plain(pred_pts, actual_pts, lam)
        errs = [r[1] for r in res]
        m = np.mean(errs)
        md = np.median(errs)
        nb = sum(1 for a, r in res if r < a)
        w = max(errs)
        label = f"TPS lam={lam}"
        pct = f"{(1-m/aff_mean)*100:+.0f}%"
        print(f"  {label:>30} | {m:6.1f} {md:7.1f} {nb:5d}/{N_LM} {w:7.1f} | {pct:>7}")
        if m < best_mean:
            best_mean = m
            best_label = label
            best_res = res

        # TPS + IRLS
        for n_irls in [3, 5, 10]:
            res = loo_irls(pred_pts, actual_pts, lam, n_irls=n_irls)
            errs = [r[1] for r in res]
            m = np.mean(errs)
            md = np.median(errs)
            nb = sum(1 for a, r in res if r < a)
            w = max(errs)
            label = f"TPS lam={lam} IRLS={n_irls}"
            pct = f"{(1-m/aff_mean)*100:+.0f}%"
            print(f"  {label:>30} | {m:6.1f} {md:7.1f} {nb:5d}/{N_LM} {w:7.1f} | {pct:>7}")
            if m < best_mean:
                best_mean = m
                best_label = label
                best_res = res

    print(f"\n  BEST: {best_label}")
    print(f"  Mean: {aff_mean:.1f}px → {best_mean:.1f}px ({(1-best_mean/aff_mean)*100:.0f}% reduction)")
    print(f"  In µm: {aff_mean*ND2_XY_UM:.1f}µm → {best_mean*ND2_XY_UM:.1f}µm")
    nb = sum(1 for a, r in best_res if r < a)
    print(f"  Better: {nb}/{N_LM} ({100*nb/N_LM:.0f}%)")
    print(f"  Median: {np.median([r[0] for r in best_res]):.1f}px → {np.median([r[1] for r in best_res]):.1f}px")
