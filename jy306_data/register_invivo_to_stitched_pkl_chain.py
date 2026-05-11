#!/usr/bin/env python3
"""
Register in-vivo → stitched ex-vivo using per-tile pkl deformation fields + RBF correction.

Chain: JY306 → pkl inverse → merscope → inv(SIFT) → nd2 → cumulative_IOU → stitched 1µm iso
Then fit RBF to residuals of this chain.
"""
import numpy as np
import pickle
import json
import glob
import os
from scipy.ndimage import map_coordinates
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import KFold

BASE = "/Users/neurolab/neuroinformatics/margaret"

# ============================================================
# 1. Tile → pkl file mapping
# ============================================================
TILE_TO_MERSCOPE = {
    'row1_1': '1_1_merscope25', 'row1_2': '1_2_merscope24',
    'row1_3': '1_3_merscope23', 'row1_4': '1_4_merscope22',
    'row2_1': '2_1_merscope17', 'row2_2': '2_2_merscope18',
    'row2_3': '2_3_merscope19', 'row2_4': '2_4_merscope20',
    'row2_5': '2_5_merscope21', 'row3_1': '3_1_merscope16',
    'row3_2': '3_2_merscope15', 'row3_3': '3_3_merscope14',
    'row3_4': '3_4_merscope13', 'row3_5': '3_5_merscope12',
    'row3_6': '3_6_merscope11', 'row4_1': '4_1_merscope5',
    'row4_2': '4_2_merscope6',  'row4_3': '4_3_merscope7',
    'row4_4': '4_4_merscope8',  'row4_5': '4_5_merscope9',
    'row4_6': '4_6_merscope10', 'row5_1': '5_1_merscope4',
}

def find_pkl(tile):
    """Find the transformed pkl file for a tile."""
    merscope = TILE_TO_MERSCOPE[tile]
    pattern = f"{BASE}/transformation/{merscope}transformed*.pkl"
    files = glob.glob(pattern)
    if files:
        return files[0]
    return None

# ============================================================
# 2. Load stitching params
# ============================================================
params = json.load(open(f"{BASE}/registration_video/stitch_v5_params.json"))

# ============================================================
# 3. Helper functions
# ============================================================
def interp_vecfield(vf, pts):
    out = np.zeros_like(pts)
    for c in range(3):
        out[:, c] = map_coordinates(vf[..., c], pts.T, order=1, mode='nearest')
    return out

def point_inverse(pts, transforms):
    p = pts.copy().astype(np.float64)
    for t in reversed(transforms):
        key = list(t.keys())[0]
        val = t[key]
        if key == 'scale':
            p = p / val
        elif key == 'bhat':
            R, tv = val[:3].astype(np.float64), val[3].astype(np.float64)
            R_inv = np.linalg.inv(R)
            p = (p - tv) @ R_inv
        elif key == 'vec_field_total':
            disp = interp_vecfield(val, p)
            p = p - disp
    return p

def get_sift_affine(tile):
    """Get SIFT nd2→merscope affine for a tile. Falls back to global."""
    per_tile = f"{BASE}/registration_video/affine_nd2_to_merscope_sift_{tile}.npy"
    if os.path.exists(per_tile):
        return np.load(per_tile)
    return np.load(f"{BASE}/registration_video/affine_nd2_to_merscope_sift.npy")

def merscope_to_nd2(merscope_yx, sift_affine):
    """Convert merscope (dim2, dim1) → nd2 (col0, col1) using inverse of SIFT affine.

    SIFT maps: [nd2_col0, nd2_col1, 1] → [merscope_dim2, merscope_dim1]
    Inverse:   [merscope_dim2, merscope_dim1] → [nd2_col0, nd2_col1]
    """
    A_2x2 = sift_affine[:, :2]
    t = sift_affine[:, 2]
    A_inv = np.linalg.inv(A_2x2)
    nd2 = (A_inv @ (merscope_yx - t).T).T
    return nd2

def nd2_to_stitched(nd2_col0col1, tile, cell_z_nd2):
    """Convert nd2 (col0, col1) → stitched 1µm iso (z, y, x).

    cumulative_iou: H @ [col0, col1, 1] → [canvas_x_like, canvas_y_like]
    stitched_x = canvas_dim0 * 0.645
    stitched_y = canvas_dim1 * 0.645
    """
    H = np.array(params['cumulative_iou'][tile])
    z_off = params['tile_z_offsets'][tile]

    pts_h = np.column_stack([nd2_col0col1[:, 0], nd2_col0col1[:, 1],
                              np.ones(len(nd2_col0col1))])
    canvas = (H @ pts_h.T).T  # [dim0, dim1, 1]

    st_x = canvas[:, 0] * 0.645
    st_y = canvas[:, 1] * 0.645
    st_z = (z_off + cell_z_nd2) * 2.0

    return np.column_stack([st_z, st_y, st_x])

# ============================================================
# 4. Process all landmark tiles through pkl chain
# ============================================================
print("Processing landmarks through pkl chain...")
files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_stitched_v5_*.npz'))

all_iv_um = []      # in-vivo in µm
all_st_true = []    # true stitched coords (1µm iso)
all_st_pred = []    # pkl-chain predicted stitched coords
all_tiles = []      # tile name for each landmark

IV_XY_UM = 0.6835
IV_Z_UM = 3.0

pkl_cache = {}

for f in files:
    tile = 'row' + f.split('_row')[-1].replace('.npz', '')
    d = np.load(f)
    iv_jy = d['pcd_invivo_jy306']   # (N, 3) JY306 native pixels (z, y, x)
    st_true = d['stitched_coords']   # (N, 3) stitched 1µm iso (z, y, x)
    ev_nd2_known = d['ev_nd2']       # (N, 3) nd2 coords (col0, col1, z_frac)
    cell_z = d['cell_nd2_z']         # (N,) z-slice in nd2

    # Convert in-vivo to µm
    iv_um = iv_jy.copy().astype(np.float64)
    iv_um[:, 0] *= IV_Z_UM
    iv_um[:, 1] *= IV_XY_UM
    iv_um[:, 2] *= IV_XY_UM

    # Find pkl
    pkl_path = find_pkl(tile)
    if pkl_path is None:
        print(f"  {tile}: NO PKL FOUND — skipping")
        continue

    # Load pkl (cache to avoid re-reading)
    if pkl_path not in pkl_cache:
        with open(pkl_path, 'rb') as pf:
            pkl_cache[pkl_path] = pickle.load(pf)
    pkl_data = pkl_cache[pkl_path]
    transforms = pkl_data['transformations']

    # Step 1: pkl inverse — JY306 → merscope input space
    merscope_pts = point_inverse(iv_jy, transforms)  # (N, 3) = (z, dim1, dim2)

    # Step 2: inverse SIFT — merscope (dim2, dim1) → nd2 (col0, col1)
    sift = get_sift_affine(tile)
    # SIFT: [col0, col1, 1] → [dim2, dim1]
    merscope_for_inv = np.column_stack([merscope_pts[:, 2], merscope_pts[:, 1]])  # (dim2, dim1)
    nd2_pred = merscope_to_nd2(merscope_for_inv, sift)  # (N, 2) = (col0, col1)

    # Step 3: nd2 → stitched
    st_pred = nd2_to_stitched(nd2_pred, tile, cell_z)

    # Check nd2 prediction accuracy
    nd2_err = np.sqrt((nd2_pred[:, 0] - ev_nd2_known[:, 0])**2 +
                       (nd2_pred[:, 1] - ev_nd2_known[:, 1])**2)

    # Check stitched prediction accuracy
    st_err = np.linalg.norm(st_pred - st_true, axis=1)

    print(f"  {tile}: {len(iv_jy)} cells | "
          f"nd2 err={nd2_err.mean():.1f}px ({nd2_err.mean()*0.645:.1f}µm) | "
          f"stitched err={st_err.mean():.1f}µm")

    all_iv_um.append(iv_um)
    all_st_true.append(st_true)
    all_st_pred.append(st_pred)
    all_tiles.extend([tile] * len(iv_jy))

iv_um = np.vstack(all_iv_um)
st_true = np.vstack(all_st_true)
st_pred = np.vstack(all_st_pred)
N = len(iv_um)

print(f"\nTotal: {N} landmarks with pkl chain predictions")

# Overall pkl-chain accuracy
res_chain = np.linalg.norm(st_pred - st_true, axis=1)
print(f"\n--- Pkl chain residuals (µm) ---")
print(f"  Mean:   {res_chain.mean():.2f} µm")
print(f"  Median: {np.median(res_chain):.2f} µm")
print(f"  Std:    {res_chain.std():.2f} µm")
print(f"  Max:    {res_chain.max():.2f} µm")
print(f"  <10µm:  {(res_chain < 10).sum()}/{N} ({(res_chain < 10).mean()*100:.1f}%)")
print(f"  <20µm:  {(res_chain < 20).sum()}/{N} ({(res_chain < 20).mean()*100:.1f}%)")
print(f"  <50µm:  {(res_chain < 50).sum()}/{N} ({(res_chain < 50).mean()*100:.1f}%)")

# Compare with affine
aff_data = np.load(f"{BASE}/registration_video/affine_3d_invivo_to_stitched.npz")
M = aff_data['M_affine']
A_aff = np.hstack([iv_um, np.ones((N, 1))])
ev_affine = (M @ A_aff.T).T
res_affine = np.linalg.norm(ev_affine - st_true, axis=1)
print(f"\n--- Affine residuals (same {N} landmarks) ---")
print(f"  Mean:   {res_affine.mean():.2f} µm")
print(f"  Median: {np.median(res_affine):.2f} µm")

# ============================================================
# 5. Fit RBF on pkl-chain residuals
# ============================================================
residuals = st_true - st_pred  # (N, 3) what needs to be added to chain prediction

print(f"\n--- Cross-validating RBF on pkl-chain residuals ---")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
best_smooth = None
best_cv = np.inf

for smoothing in [0, 10, 100, 1000, 5000, 10000]:
    cv_errors = []
    for train_idx, test_idx in kf.split(iv_um):
        rbfs = []
        for ax in range(3):
            rbf = RBFInterpolator(st_pred[train_idx], residuals[train_idx, ax],
                                  kernel='thin_plate_spline', smoothing=smoothing)
            rbfs.append(rbf)
        corr = np.column_stack([r(st_pred[test_idx]) for r in rbfs])
        pred = st_pred[test_idx] + corr
        err = np.linalg.norm(pred - st_true[test_idx], axis=1)
        cv_errors.extend(err.tolist())

    cv_errors = np.array(cv_errors)
    cv_mean = cv_errors.mean()
    print(f"  s={smoothing:6}: CV mean={cv_mean:.2f}, median={np.median(cv_errors):.2f}, "
          f"<10µm={100*(cv_errors<10).mean():.1f}%, <20µm={100*(cv_errors<20).mean():.1f}%")

    if cv_mean < best_cv:
        best_cv = cv_mean
        best_smooth = smoothing

print(f"\nBest smoothing: {best_smooth} (CV mean={best_cv:.2f} µm)")

# ============================================================
# 6. Fit final model
# ============================================================
print(f"\n--- Fitting final RBF (smoothing={best_smooth}) ---")
rbfs_final = []
for ax in range(3):
    rbf = RBFInterpolator(st_pred, residuals[:, ax],
                          kernel='thin_plate_spline', smoothing=best_smooth)
    rbfs_final.append(rbf)

corr_final = np.column_stack([r(st_pred) for r in rbfs_final])
st_final = st_pred + corr_final
res_final = np.linalg.norm(st_final - st_true, axis=1)

print(f"\n--- Final pkl+RBF residuals (µm) ---")
print(f"  Mean:   {res_final.mean():.2f} µm  (chain: {res_chain.mean():.2f}, affine: {res_affine.mean():.2f})")
print(f"  Median: {np.median(res_final):.2f} µm")
print(f"  Std:    {res_final.std():.2f} µm")
print(f"  Max:    {res_final.max():.2f} µm")
print(f"  <5µm:   {(res_final < 5).sum()}/{N} ({(res_final < 5).mean()*100:.1f}%)")
print(f"  <10µm:  {(res_final < 10).sum()}/{N} ({(res_final < 10).mean()*100:.1f}%)")
print(f"  <20µm:  {(res_final < 20).sum()}/{N} ({(res_final < 20).mean()*100:.1f}%)")
print(f"  <50µm:  {(res_final < 50).sum()}/{N} ({(res_final < 50).mean()*100:.1f}%)")

# Per-axis
axis_err = st_final - st_true
for ax, name in enumerate(['Z', 'Y', 'X']):
    print(f"  {name}-axis: mean={axis_err[:,ax].mean():.2f}, std={axis_err[:,ax].std():.2f} µm")

# ============================================================
# 7. Save
# ============================================================
OUT = f"{BASE}/registration_video/pkl_chain_registration.npz"
np.savez(OUT,
    iv_um=iv_um,
    st_true=st_true,
    st_pred_chain=st_pred,
    st_pred_final=st_final,
    residuals_chain=res_chain,
    residuals_final=res_final,
    best_smoothing=best_smooth,
    cv_best_mean=best_cv,
)
print(f"\nSaved: {OUT}")

OUT_PKL = f"{BASE}/registration_video/pkl_chain_rbf_models.pkl"
with open(OUT_PKL, 'wb') as pf:
    pickle.dump({
        'rbfs': rbfs_final,
        'best_smoothing': best_smooth,
    }, pf)
print(f"Saved: {OUT_PKL}")
