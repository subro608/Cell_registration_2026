#!/usr/bin/env python3
"""
Register in-vivo (JY306 s80) to stitched ex-vivo (1µm iso) using 878 matched cell landmarks.
Step 1: Compute 3D affine transform via least-squares.
"""
import numpy as np
import glob

BASE = "/Users/neurolab/neuroinformatics/margaret"

# ============================================================
# 1. Load all 878 matched landmarks
# ============================================================
print("Loading landmarks...")
files = sorted(glob.glob(f'{BASE}/registration_video/landmarks_stitched_v5_*.npz'))

all_iv = []   # in-vivo: native s80 pixels (z, y, x) @ 0.6835µm XY, 3µm Z
all_ev = []   # ex-vivo: stitched 1µm iso (z, y, x)

for f in files:
    d = np.load(f)
    iv = d['pcd_invivo_jy306']   # (N, 3) in native s80 pixels (z, y, x)
    ev = d['stitched_coords']     # (N, 3) in stitched 1µm iso (z, y, x)
    all_iv.append(iv)
    all_ev.append(ev)
    tile = f.split('_')[-1].replace('.npz', '')
    print(f"  {tile}: {len(iv)} cells")

iv_pts = np.vstack(all_iv)  # (878, 3) native s80 pixels
ev_pts = np.vstack(all_ev)  # (878, 3) stitched 1µm iso

print(f"\nTotal: {len(iv_pts)} matched cells")
print(f"  In-vivo range:  z[{iv_pts[:,0].min():.1f}-{iv_pts[:,0].max():.1f}] "
      f"y[{iv_pts[:,1].min():.1f}-{iv_pts[:,1].max():.1f}] "
      f"x[{iv_pts[:,2].min():.1f}-{iv_pts[:,2].max():.1f}]")
print(f"  Ex-vivo range:  z[{ev_pts[:,0].min():.1f}-{ev_pts[:,0].max():.1f}] "
      f"y[{ev_pts[:,1].min():.1f}-{ev_pts[:,1].max():.1f}] "
      f"x[{ev_pts[:,2].min():.1f}-{ev_pts[:,2].max():.1f}]")

# ============================================================
# 2. Convert in-vivo from native pixels to µm
# ============================================================
IV_XY_UM = 0.6835
IV_Z_UM = 3.0

iv_um = iv_pts.copy().astype(np.float64)
iv_um[:, 0] *= IV_Z_UM    # z: native slices → µm
iv_um[:, 1] *= IV_XY_UM   # y: native pixels → µm
iv_um[:, 2] *= IV_XY_UM   # x: native pixels → µm

ev_um = ev_pts.copy().astype(np.float64)  # already 1µm iso, so coords = µm

print(f"\n  In-vivo (µm):   z[{iv_um[:,0].min():.1f}-{iv_um[:,0].max():.1f}] "
      f"y[{iv_um[:,1].min():.1f}-{iv_um[:,1].max():.1f}] "
      f"x[{iv_um[:,2].min():.1f}-{iv_um[:,2].max():.1f}]")
print(f"  Ex-vivo (µm):   z[{ev_um[:,0].min():.1f}-{ev_um[:,0].max():.1f}] "
      f"y[{ev_um[:,1].min():.1f}-{ev_um[:,1].max():.1f}] "
      f"x[{ev_um[:,2].min():.1f}-{ev_um[:,2].max():.1f}]")

# ============================================================
# 3. Compute 3D affine: in-vivo (µm) → ex-vivo (µm)
#    ev = M @ [iv; 1]  where M is (3, 4)
# ============================================================
print("\n--- Computing 3D affine (least-squares) ---")

N = len(iv_um)
# Build design matrix: (N, 4) = [z, y, x, 1]
A = np.hstack([iv_um, np.ones((N, 1))])  # (878, 4)

# Solve for M (3, 4): ev = A @ M.T  →  M.T = lstsq(A, ev)
M_T, residuals, rank, sv = np.linalg.lstsq(A, ev_um, rcond=None)
M = M_T.T  # (3, 4)

print(f"  Rank: {rank}")
print(f"  Singular values: {sv}")
print(f"\n  Affine matrix M (3×4):")
print(f"    {M[0]}")
print(f"    {M[1]}")
print(f"    {M[2]}")

# ============================================================
# 4. Evaluate: transform in-vivo → ex-vivo, compute residuals
# ============================================================
ev_pred = (M @ A.T).T  # (878, 3) predicted ex-vivo coords

residuals = np.linalg.norm(ev_pred - ev_um, axis=1)  # per-cell error in µm
print(f"\n--- Registration residuals (µm) ---")
print(f"  Mean:   {residuals.mean():.2f} µm")
print(f"  Median: {np.median(residuals):.2f} µm")
print(f"  Std:    {residuals.std():.2f} µm")
print(f"  Max:    {residuals.max():.2f} µm")
print(f"  Min:    {residuals.min():.2f} µm")
print(f"  <10µm:  {(residuals < 10).sum()}/{N} ({(residuals < 10).mean()*100:.1f}%)")
print(f"  <20µm:  {(residuals < 20).sum()}/{N} ({(residuals < 20).mean()*100:.1f}%)")
print(f"  <50µm:  {(residuals < 50).sum()}/{N} ({(residuals < 50).mean()*100:.1f}%)")

# Per-axis residuals
axis_err = ev_pred - ev_um
for ax, name in enumerate(['Z', 'Y', 'X']):
    print(f"  {name}-axis: mean={axis_err[:,ax].mean():.2f}, std={axis_err[:,ax].std():.2f}, "
          f"range=[{axis_err[:,ax].min():.2f}, {axis_err[:,ax].max():.2f}] µm")

# ============================================================
# 5. Also try similarity (rigid + uniform scale) and rigid
# ============================================================
from scipy.spatial.transform import Rotation

def fit_similarity(src, tgt):
    """Fit similarity transform: tgt = s * R @ src + t"""
    src_c = src.mean(axis=0)
    tgt_c = tgt.mean(axis=0)
    src_centered = src - src_c
    tgt_centered = tgt - tgt_c

    # Scale
    s = np.sqrt((tgt_centered**2).sum() / (src_centered**2).sum())

    # Rotation via SVD of cross-covariance
    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ D @ U.T

    t = tgt_c - s * R @ src_c
    return s, R, t

def fit_rigid(src, tgt):
    """Fit rigid transform: tgt = R @ src + t"""
    src_c = src.mean(axis=0)
    tgt_c = tgt.mean(axis=0)
    src_centered = src - src_c
    tgt_centered = tgt - tgt_c

    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ D @ U.T

    t = tgt_c - R @ src_c
    return R, t

# Similarity
s, R_sim, t_sim = fit_similarity(iv_um, ev_um)
ev_sim = s * (iv_um @ R_sim.T) + t_sim
res_sim = np.linalg.norm(ev_sim - ev_um, axis=1)
print(f"\n--- Similarity (rigid + scale) ---")
print(f"  Scale: {s:.4f}")
print(f"  Rotation angles: {Rotation.from_matrix(R_sim).as_euler('xyz', degrees=True)}")
print(f"  Translation: {t_sim}")
print(f"  Mean residual: {res_sim.mean():.2f} µm")
print(f"  Median: {np.median(res_sim):.2f} µm")
print(f"  <10µm: {(res_sim < 10).sum()}/{N} ({(res_sim < 10).mean()*100:.1f}%)")
print(f"  <20µm: {(res_sim < 20).sum()}/{N} ({(res_sim < 20).mean()*100:.1f}%)")

# Rigid
R_rig, t_rig = fit_rigid(iv_um, ev_um)
ev_rig = (iv_um @ R_rig.T) + t_rig
res_rig = np.linalg.norm(ev_rig - ev_um, axis=1)
print(f"\n--- Rigid (rotation + translation only) ---")
print(f"  Rotation angles: {Rotation.from_matrix(R_rig).as_euler('xyz', degrees=True)}")
print(f"  Translation: {t_rig}")
print(f"  Mean residual: {res_rig.mean():.2f} µm")
print(f"  Median: {np.median(res_rig):.2f} µm")
print(f"  <10µm: {(res_rig < 10).sum()}/{N} ({(res_rig < 10).mean()*100:.1f}%)")
print(f"  <20µm: {(res_rig < 20).sum()}/{N} ({(res_rig < 20).mean()*100:.1f}%)")

# ============================================================
# 6. Save best transform
# ============================================================
OUT = f"{BASE}/registration_video/affine_3d_invivo_to_stitched.npz"
np.savez(OUT,
    M_affine=M,                    # (3, 4) affine: ev = M @ [iv_um; 1]
    s_similarity=s,                # scalar scale
    R_similarity=R_sim,            # (3, 3) rotation
    t_similarity=t_sim,            # (3,) translation
    R_rigid=R_rig,                 # (3, 3) rotation
    t_rigid=t_rig,                 # (3,) translation
    iv_um=iv_um,                   # (878, 3) source points in µm
    ev_um=ev_um,                   # (878, 3) target points in µm
    residuals_affine=residuals,    # (878,) per-cell residuals
    residuals_similarity=res_sim,
    residuals_rigid=res_rig,
)
print(f"\nSaved: {OUT}")
print(f"\nBest transform: {'affine' if residuals.mean() < res_sim.mean() else 'similarity'}")
