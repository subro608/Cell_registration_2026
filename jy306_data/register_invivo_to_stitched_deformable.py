#!/usr/bin/env python3
"""
Register in-vivo (JY306 s80) to stitched ex-vivo (1µm iso) using 878 matched cell landmarks.
Step 2: Deformable registration using affine + RBF correction.

Strategy:
  1. Apply affine transform (from Step 1) as global initialization
  2. Fit RBF (thin-plate spline) to residuals: correction = RBF(affine_predicted) → target
  3. Evaluate with leave-one-out cross-validation to check overfitting
"""
import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import KFold

BASE = "/Users/neurolab/neuroinformatics/margaret"

# ============================================================
# 1. Load Step 1 results
# ============================================================
print("Loading affine results...")
data = np.load(f"{BASE}/registration_video/affine_3d_invivo_to_stitched.npz")
M = data['M_affine']       # (3, 4)
iv_um = data['iv_um']      # (878, 3) in-vivo in µm
ev_um = data['ev_um']      # (878, 3) ex-vivo in µm (target)

N = len(iv_um)
print(f"Loaded {N} landmarks")

# Apply affine to get initial prediction
A = np.hstack([iv_um, np.ones((N, 1))])  # (N, 4)
ev_affine = (M @ A.T).T  # (N, 3) affine-predicted ex-vivo coords

res_affine = np.linalg.norm(ev_affine - ev_um, axis=1)
print(f"Affine residuals: mean={res_affine.mean():.2f}, median={np.median(res_affine):.2f} µm")

# ============================================================
# 2. Fit RBF correction on top of affine
# ============================================================
# RBF maps affine-predicted coords → true ex-vivo coords
# This corrects for local non-linear tissue deformation
print("\n--- Fitting RBF (thin-plate spline) on affine residuals ---")

# Residuals to correct
residuals = ev_um - ev_affine  # (N, 3) what we need to add to affine prediction

# Fit one RBF per axis: f(affine_pred) → correction
# Using thin_plate_spline kernel with smoothing to avoid overfitting
for smoothing in [0, 1, 10, 50, 100, 500, 1000]:
    rbf_z = RBFInterpolator(ev_affine, residuals[:, 0], kernel='thin_plate_spline', smoothing=smoothing)
    rbf_y = RBFInterpolator(ev_affine, residuals[:, 1], kernel='thin_plate_spline', smoothing=smoothing)
    rbf_x = RBFInterpolator(ev_affine, residuals[:, 2], kernel='thin_plate_spline', smoothing=smoothing)

    # Training error
    corr_z = rbf_z(ev_affine)
    corr_y = rbf_y(ev_affine)
    corr_x = rbf_x(ev_affine)
    ev_corrected = ev_affine + np.column_stack([corr_z, corr_y, corr_x])
    res_train = np.linalg.norm(ev_corrected - ev_um, axis=1)
    print(f"  smoothing={smoothing:6.0f}: train mean={res_train.mean():.2f}, "
          f"median={np.median(res_train):.2f}, max={res_train.max():.2f} µm")

# ============================================================
# 3. Cross-validation to pick best smoothing
# ============================================================
print("\n--- 10-fold cross-validation ---")
kf = KFold(n_splits=10, shuffle=True, random_state=42)

best_smooth = None
best_cv_mean = np.inf

for smoothing in [0, 1, 10, 50, 100, 500, 1000, 2000, 5000, 10000]:
    cv_errors = []
    for train_idx, test_idx in kf.split(iv_um):
        # Affine predictions for all
        ev_aff_train = ev_affine[train_idx]
        ev_aff_test = ev_affine[test_idx]
        res_train = residuals[train_idx]

        # Fit RBF on training set
        rbf_z = RBFInterpolator(ev_aff_train, res_train[:, 0],
                                kernel='thin_plate_spline', smoothing=smoothing)
        rbf_y = RBFInterpolator(ev_aff_train, res_train[:, 1],
                                kernel='thin_plate_spline', smoothing=smoothing)
        rbf_x = RBFInterpolator(ev_aff_train, res_train[:, 2],
                                kernel='thin_plate_spline', smoothing=smoothing)

        # Predict on test set
        corr = np.column_stack([rbf_z(ev_aff_test), rbf_y(ev_aff_test), rbf_x(ev_aff_test)])
        ev_pred_test = ev_aff_test + corr
        err = np.linalg.norm(ev_pred_test - ev_um[test_idx], axis=1)
        cv_errors.extend(err.tolist())

    cv_errors = np.array(cv_errors)
    cv_mean = cv_errors.mean()
    cv_med = np.median(cv_errors)
    print(f"  smoothing={smoothing:6.0f}: CV mean={cv_mean:.2f}, median={cv_med:.2f}, "
          f"<10µm={100*(cv_errors<10).mean():.1f}%, <20µm={100*(cv_errors<20).mean():.1f}%, "
          f"<50µm={100*(cv_errors<50).mean():.1f}%")

    if cv_mean < best_cv_mean:
        best_cv_mean = cv_mean
        best_smooth = smoothing

print(f"\nBest smoothing: {best_smooth} (CV mean={best_cv_mean:.2f} µm)")

# ============================================================
# 4. Fit final model with best smoothing
# ============================================================
print(f"\n--- Fitting final RBF (smoothing={best_smooth}) ---")
rbf_z = RBFInterpolator(ev_affine, residuals[:, 0],
                        kernel='thin_plate_spline', smoothing=best_smooth)
rbf_y = RBFInterpolator(ev_affine, residuals[:, 1],
                        kernel='thin_plate_spline', smoothing=best_smooth)
rbf_x = RBFInterpolator(ev_affine, residuals[:, 2],
                        kernel='thin_plate_spline', smoothing=best_smooth)

# Final training residuals
corr_z = rbf_z(ev_affine)
corr_y = rbf_y(ev_affine)
corr_x = rbf_x(ev_affine)
ev_final = ev_affine + np.column_stack([corr_z, corr_y, corr_x])
res_final = np.linalg.norm(ev_final - ev_um, axis=1)

print(f"\n--- Final deformable residuals (µm) ---")
print(f"  Mean:   {res_final.mean():.2f} µm  (was {res_affine.mean():.2f} affine)")
print(f"  Median: {np.median(res_final):.2f} µm  (was {np.median(res_affine):.2f} affine)")
print(f"  Std:    {res_final.std():.2f} µm")
print(f"  Max:    {res_final.max():.2f} µm  (was {res_affine.max():.2f} affine)")
print(f"  Min:    {res_final.min():.2f} µm")
print(f"  <5µm:   {(res_final < 5).sum()}/{N} ({(res_final < 5).mean()*100:.1f}%)")
print(f"  <10µm:  {(res_final < 10).sum()}/{N} ({(res_final < 10).mean()*100:.1f}%)")
print(f"  <20µm:  {(res_final < 20).sum()}/{N} ({(res_final < 20).mean()*100:.1f}%)")
print(f"  <50µm:  {(res_final < 50).sum()}/{N} ({(res_final < 50).mean()*100:.1f}%)")

# Per-axis
axis_err = ev_final - ev_um
for ax, name in enumerate(['Z', 'Y', 'X']):
    print(f"  {name}-axis: mean={axis_err[:,ax].mean():.2f}, std={axis_err[:,ax].std():.2f}, "
          f"range=[{axis_err[:,ax].min():.2f}, {axis_err[:,ax].max():.2f}] µm")

# ============================================================
# 5. Save deformable transform
# ============================================================
# Save the RBF control points and weights so we can reconstruct later
# RBFInterpolator stores: _y (data values), _d (kernel matrix solution), etc.
# Easier to save the inputs and re-fit, or use pickle

import pickle

OUT_NPZ = f"{BASE}/registration_video/deformable_invivo_to_stitched.npz"
np.savez(OUT_NPZ,
    M_affine=M,
    best_smoothing=best_smooth,
    rbf_centers=ev_affine,          # (878, 3) RBF centers = affine-predicted coords
    rbf_residuals=residuals,        # (878, 3) RBF target = correction vectors
    iv_um=iv_um,
    ev_um=ev_um,
    ev_affine=ev_affine,
    ev_final=ev_final,
    residuals_affine=res_affine,
    residuals_final=res_final,
    cv_best_mean=best_cv_mean,
)
print(f"\nSaved: {OUT_NPZ}")

# Also pickle the fitted RBF objects for direct reuse
OUT_PKL = f"{BASE}/registration_video/deformable_rbf_models.pkl"
with open(OUT_PKL, 'wb') as f:
    pickle.dump({
        'rbf_z': rbf_z,
        'rbf_y': rbf_y,
        'rbf_x': rbf_x,
        'M_affine': M,
        'best_smoothing': best_smooth,
    }, f)
print(f"Saved: {OUT_PKL}")

# ============================================================
# 6. Demo: transform function
# ============================================================
print("\n--- Transform function ---")
print("To transform new in-vivo points (µm) to ex-vivo (µm):")
print("  1. ev_affine = M @ [iv_z, iv_y, iv_x, 1]")
print("  2. correction = [rbf_z(ev_affine), rbf_y(ev_affine), rbf_x(ev_affine)]")
print("  3. ev_final = ev_affine + correction")
