#!/usr/bin/env python3
"""Test ANTs SyN registration on row3_6 — compare to elastix and affine-only.
Downscales to ~1000px for speed, measures landmark displacement at full res."""
import numpy as np
import cv2
import os
import json
import tifffile
import ants
from scipy.ndimage import median_filter
from collections import defaultdict

BASE = '/Users/neurolab/neuroinformatics/margaret'
IV_XY_UM = 0.6835
IV_Z_UM = 3.0
ND2_XY_UM = 0.645
ND2_Z_UM = 2.0
TILE = 'row3_6'
DS = 4  # downscale factor for registration

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

z_pair_to_lm = defaultdict(list)
for i in range(N_LM):
    z_iv = int(round(np.clip(pcd_iv[i, 0], 0, nz_iv - 1)))
    z_nd2 = int(round(np.clip(nd2_z_gauss[i], 0, 11)))
    z_pair_to_lm[(z_iv, z_nd2)].append(i)


def norm_f(img):
    vals = img[img > 0]
    if len(vals) < 100:
        return img.copy()
    lo, hi = np.percentile(vals, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1), 0, 1).astype(np.float32)


def ncc(a, b):
    mask = (a > 0) & (b > 0)
    if mask.sum() < 100:
        return -1.0
    am = a[mask].astype(np.float64)
    bm = b[mask].astype(np.float64)
    am -= am.mean(); bm -= bm.mean()
    d2 = np.sqrt((am**2).sum() * (bm**2).sum())
    return float((am * bm).sum() / d2) if d2 > 1e-10 else -1.0


print(f"\n{'='*70}")
print(f"  {TILE}: ANTs SyN CC registration ({len(z_pair_to_lm)} z-pairs, DS={DS})")
print(f"{'='*70}")

results = []
for (z_iv, z_nd2), lm_indices in sorted(z_pair_to_lm.items()):
    nd2_sl = nd2_slices[z_nd2]

    # 2D affine warp at full res
    M2d = np.array([
        [M_inv[2, 2], M_inv[2, 1], M_inv[2, 0] * z_nd2 + offset_inv[2]],
        [M_inv[1, 2], M_inv[1, 1], M_inv[1, 0] * z_nd2 + offset_inv[1]],
    ], dtype=np.float64)
    iv_affine_filt = cv2.warpAffine(iv_vol_filt[z_iv], M2d, (nd2_w, nd2_h),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
    iv_affine_raw = cv2.warpAffine(iv_vol_raw[z_iv], M2d, (nd2_w, nd2_h),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)

    # Downscale for registration
    ds_h, ds_w = nd2_h // DS, nd2_w // DS
    fixed_ds = cv2.resize(nd2_sl, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
    moving_ds = cv2.resize(iv_affine_filt, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

    fixed_norm = norm_f(fixed_ds)
    moving_norm = norm_f(moving_ds)

    fixed_ants = ants.from_numpy(fixed_norm)
    moving_ants = ants.from_numpy(moving_norm)

    print(f"  ANTs iv_z={z_iv:2d} → nd2_z={z_nd2:2d} ({len(lm_indices)} lm) ...", end="", flush=True)

    try:
        reg = ants.registration(
            fixed=fixed_ants,
            moving=moving_ants,
            type_of_transform='SyNOnly',
            syn_metric='CC',
            syn_sampling=4,
            reg_iterations=(100, 70, 50),
            verbose=False
        )

        # Transform landmark points (in downscaled coords)
        for i in lm_indices:
            ex_x, ex_y = float(ev_nd2[i, 0]), float(ev_nd2[i, 1])
            pr_x = float(predicted[i, 0] / ND2_XY_UM)
            pr_y = float(predicted[i, 1] / ND2_XY_UM)

            # Downscaled coordinates
            pr_x_ds, pr_y_ds = pr_x / DS, pr_y / DS
            ex_x_ds, ex_y_ds = ex_x / DS, ex_y / DS

            # Binary warp: Gaussian blob at predicted position, warp, find centroid
            blob_ds = np.zeros((ds_h, ds_w), dtype=np.float32)
            bx_ds, by_ds = int(round(pr_x_ds)), int(round(pr_y_ds))
            r = 8
            y1, y2 = max(0, by_ds - r), min(ds_h, by_ds + r + 1)
            x1, x2 = max(0, bx_ds - r), min(ds_w, bx_ds + r + 1)
            for yy in range(y1, y2):
                for xx in range(x1, x2):
                    d2 = (xx - bx_ds)**2 + (yy - by_ds)**2
                    blob_ds[yy, xx] = 255.0 * np.exp(-d2 / (2 * 3**2))

            blob_ants = ants.from_numpy(blob_ds)
            warped_blob = ants.apply_transforms(
                fixed=fixed_ants, moving=blob_ants,
                transformlist=reg['fwdtransforms']
            ).numpy()

            thresh = warped_blob.max() * 0.3
            mask_b = warped_blob > thresh
            ys_b, xs_b = np.where(mask_b)
            if len(xs_b) > 0:
                weights_b = warped_blob[mask_b]
                ants_x_ds = float(np.average(xs_b, weights=weights_b))
                ants_y_ds = float(np.average(ys_b, weights=weights_b))
            else:
                ants_x_ds, ants_y_ds = pr_x_ds, pr_y_ds  # fallback

            # Scale back to full res
            ants_x = ants_x_ds * DS
            ants_y = ants_y_ds * DS

            ants_err = np.sqrt((ants_x - ex_x)**2 + (ants_y - ex_y)**2)
            aff_err_px = np.sqrt((pr_x - ex_x)**2 + (pr_y - ex_y)**2)

            better = "YES" if ants_err < aff_err_px else "no"
            print(f" LM#{i}:aff={aff_err_px:.1f} ants={ants_err:.1f}({better})", end="")

            results.append({
                'i': i, 'z_iv': z_iv, 'z_nd2': z_nd2,
                'aff_err_um': float(errors[i]),
                'aff_err_px': float(aff_err_px),
                'ants_err_px': float(ants_err),
            })

        print("", flush=True)

    except Exception as e:
        print(f"  FAILED: {e}")
        for i in lm_indices:
            pr_x = float(predicted[i, 0] / ND2_XY_UM)
            pr_y = float(predicted[i, 1] / ND2_XY_UM)
            ex_x, ex_y = float(ev_nd2[i, 0]), float(ev_nd2[i, 1])
            aff_err_px = np.sqrt((pr_x - ex_x)**2 + (pr_y - ex_y)**2)
            results.append({
                'i': i, 'z_iv': z_iv, 'z_nd2': z_nd2,
                'aff_err_um': float(errors[i]),
                'aff_err_px': float(aff_err_px),
                'ants_err_px': float(aff_err_px),  # fallback
            })

# Summary
print(f"\n{'='*70}")
print(f"{'LM':>5} {'z_iv':>5} {'z_nd2':>5} {'aff_px':>8} {'ants_px':>8} {'better':>7}")
print(f"{'='*70}")
n_better = 0
for r in results:
    better = r['ants_err_px'] < r['aff_err_px']
    if better:
        n_better += 1
    print(f"{r['i']:5d} {r['z_iv']:5d} {r['z_nd2']:5d} {r['aff_err_px']:8.1f} {r['ants_err_px']:8.1f} {'YES' if better else 'no':>7}")

print(f"\nANTs better: {n_better}/{len(results)} ({100*n_better/max(1,len(results)):.0f}%)")
aff_mean = np.mean([r['aff_err_px'] for r in results])
ants_mean = np.mean([r['ants_err_px'] for r in results])
print(f"Mean affine: {aff_mean:.1f}px | Mean ANTs: {ants_mean:.1f}px")
