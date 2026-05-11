"""
Build assets for scene 5 deformation animation (1 tile test: row2_1).

Runs the PKL inverse stage by stage on JY306, saving intermediate snapshots.
This shows the progressive registration from in-vivo → ex-vivo space.

Pipeline (inverse = stages applied in reverse order):
  Forward: scale→bhat→scale→bhat→bhat→scale→bhat→vecfield→vecfield→vecfield→bhat→vecfield
  Inverse: vecfield_inv→bhat_inv→vecfield_inv→vecfield_inv→vecfield_inv→bhat_inv→scale_inv→...

Saves: animation/scene5_deform_assets_row2_1.npz
"""

import pickle
import numpy as np
import cv2
import tifffile
import os
import glob
from scipy.ndimage import map_coordinates

BASE = '/Users/neurolab/neuroinformatics/margaret'
TILE = 'row2_1'
PKL_PATH = f'{BASE}/transformation/2_1_merscope17transformed_20250424104024.pkl'
OUT_NPZ = f'{BASE}/animation/scene5_deform_assets_{TILE}.npz'
OUT_DIR = f'{BASE}/animation/scene5_deform_test'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load PKL ──
print(f"Loading pkl for {TILE} ({PKL_PATH})...")
with open(PKL_PATH, 'rb') as f:
    d = pickle.load(f)

transforms = d['transformations']
pcd_invivo = d['pcd_invivo']    # (27,3) z,y,x in ~600px space
pcd_exvivo = d['pcd_exvivo']    # (27,3) z,y,x in ~600px space
transformed = d['transformed']  # (3, 16, 1704, 1704) exvivo in output space

n_stages = len(transforms)
CANVAS = transformed.shape[-1]  # 1704
NZ_PKL = transformed.shape[1]   # 16

print(f"  {n_stages} stages, canvas={CANVAS}, nz={NZ_PKL}")
print(f"  pcd_invivo: {pcd_invivo.shape}")
for i, t in enumerate(transforms):
    k = list(t.keys())[0]
    v = t[k]
    if isinstance(v, np.ndarray):
        print(f"  [{i:2d}] {k}: {v.shape}")
    else:
        print(f"  [{i:2d}] {k}: {v}")

# ── Load JY306 ──
print("\nLoading JY306 z-stack...")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_jy, hy_jy, wx_jy = jy306.shape
print(f"  JY306: {jy306.shape}")

# Mode z for row2_1 (from landmark data: z ranges 2-4, mode=3)
MODE_Z = 3

# ── Build JY306 volume in PKL canvas space ──
# Resize JY306 to fill (NZ_PKL, CANVAS, CANVAS) — same as inverse_pkl_transform.py
print(f"\nUpsampling JY306 to pkl canvas ({NZ_PKL}, {CANVAS}, {CANVAS})...")
vol = np.zeros((NZ_PKL, CANVAS, CANVAS), dtype=np.float32)
for z in range(min(nz_jy, NZ_PKL)):
    vol[z] = cv2.resize(jy306[z], (CANVAS, CANVAS), interpolation=cv2.INTER_LINEAR)
print(f"  vol: {vol.shape}, range=[{vol.min():.1f}, {vol.max():.1f}]")

# ── Inverse transform functions ──
def inv_scale(vol, scale):
    """Inverse of scale: sample at q*scale"""
    Z, Y, X = vol.shape
    z, y, x = np.mgrid[:Z, :Y, :X].astype(np.float32)
    return map_coordinates(vol, [z * scale, y * scale, x * scale],
                           order=1, mode='constant', cval=0).astype(np.float32)

def inv_affine(vol, bhat):
    """Inverse of affine: sample at q@R + t"""
    R = bhat[:3].astype(np.float64)
    tv = bhat[3].astype(np.float64)
    Z, Y, X = vol.shape
    z, y, x = np.mgrid[:Z, :Y, :X]
    pts = np.stack([z.ravel(), y.ravel(), x.ravel()], axis=1).astype(np.float64)
    pts_in = pts @ R + tv
    return map_coordinates(vol, pts_in.T.reshape(3, Z, Y, X),
                           order=1, mode='constant', cval=0).astype(np.float32)

def inv_vecfield(vol, vf):
    """Inverse of vec_field: sample at q + vf[q]"""
    Z, Y, X = vol.shape
    vf_z, vf_y, vf_x = vf.shape[:3]
    z, y, x = np.mgrid[:Z, :Y, :X].astype(np.float32)
    # Resize vf if needed
    if (vf_z, vf_y, vf_x) != (Z, Y, X):
        vf_resized = np.zeros((Z, Y, X, 3), dtype=np.float32)
        for c in range(3):
            for zi in range(Z):
                src_z = min(zi, vf_z - 1)
                vf_resized[zi, :, :, c] = cv2.resize(
                    vf[src_z, :, :, c].astype(np.float32), (X, Y),
                    interpolation=cv2.INTER_LINEAR)
        vf = vf_resized
    else:
        vf = vf.astype(np.float32)
    return map_coordinates(vol,
        [z + vf[..., 0], y + vf[..., 1], x + vf[..., 2]],
        order=1, mode='constant', cval=0).astype(np.float32)

# ── Apply inverse stages and save snapshots ──
print(f"\nApplying inverse stages ({n_stages-1} → 0)...")
stage_snapshots = []   # mode-z slice after each stage
stage_types = []       # 'scale', 'bhat', or 'vec_field_total'
stage_indices = []     # original stage index

# Save initial state (JY306 in canvas, before any inverse)
stage_snapshots.append(vol[MODE_Z].copy())
stage_types.append('initial')
stage_indices.append(-1)

result = vol.copy()
del vol  # free memory

for inv_i, orig_i in enumerate(range(n_stages - 1, -1, -1)):
    t = transforms[orig_i]
    key = list(t.keys())[0]
    val = t[key]
    print(f"  Inv stage {inv_i} (orig [{orig_i}]) {key}...", end=' ', flush=True)

    if key == 'scale':
        result = inv_scale(result, val)
    elif key == 'bhat':
        result = inv_affine(result, val)
    elif key == 'vec_field_total':
        result = inv_vecfield(result, val)

    nz_count = np.count_nonzero(result[MODE_Z])
    print(f"done  nonzero={nz_count}")

    stage_snapshots.append(result[MODE_Z].copy())
    stage_types.append(key)
    stage_indices.append(orig_i)

# ── Also load nd2 ex-vivo for overlay ──
print("\nLoading nd2 ex-vivo for row2_1...")
nd2_files = sorted(glob.glob(f'{BASE}/png_exports/registration_video/{TILE}/GFP_z*.png'))
nd2_stack = []
for f in nd2_files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    nd2_stack.append(img)
nd2_stack = np.array(nd2_stack)
print(f"  nd2: {nd2_stack.shape}")  # (12, 4200, 4200)

# Downsample nd2 to ~600px to match pkl input space
ND2_DS = 7
nd2_small_w = nd2_stack.shape[2] // ND2_DS  # 600
nd2_small_h = nd2_stack.shape[1] // ND2_DS
nd2_ds = np.array([cv2.resize(nd2_stack[z], (nd2_small_w, nd2_small_h),
                               interpolation=cv2.INTER_AREA)
                    for z in range(len(nd2_stack))])
print(f"  nd2 downsampled: {nd2_ds.shape}")  # (12, 600, 600)

# ── Also compute M2d affine warp for comparison ──
print("\nComputing M2d affine warp (for affine-only display)...")
npz = np.load(f'{BASE}/png_exports/registration_per_tile_pkl/{TILE}/pkl_transform_{TILE}.npz')
M2d = npz['M2d_jy306_to_nd2']
ev_nd2 = npz['ev_nd2']       # (x,y,z) in nd2 4200px
iv_jy306 = npz['pcd_invivo_jy306']  # (z,y,x) in JY306

# M2d warps JY306 → nd2 4200px space
jy_mode = jy306[MODE_Z]
iv_warped_m2d = cv2.warpAffine(jy_mode, M2d, (4200, 4200),
                                flags=cv2.INTER_LINEAR, borderValue=0)
print(f"  M2d warped: {iv_warped_m2d.shape}")

# ── Save test images ──
def norm8(img, lo=1, hi=99.5):
    v = img.ravel(); v = v[v > 0]
    if len(v) < 50: return np.zeros(img.shape, np.uint8)
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((img - p1) / max(p2 - p1, 1) * 255, 0, 255).astype(np.uint8)

# Save snapshots as test images
print(f"\nSaving {len(stage_snapshots)} stage snapshot images...")
for si, (snap, stype, sidx) in enumerate(zip(stage_snapshots, stage_types, stage_indices)):
    label = f"s{si:02d}_{stype}_orig{sidx}"
    cv2.imwrite(f'{OUT_DIR}/{label}.png', norm8(snap))

# Final inverse result overlaid with nd2 (both at ~600px in 1704 canvas)
final_inv = stage_snapshots[-1]  # final mode-z inverse result in 1704 canvas
# Find the nonzero bounding box of the final result
nz_y, nz_x = np.nonzero(final_inv > 0)
if len(nz_y) > 0:
    print(f"\nFinal inverse nonzero region: y={nz_y.min()}-{nz_y.max()}, x={nz_x.min()}-{nz_x.max()}")

    # Overlay: final inverse (red) + nd2 z=0 at 600px (green)
    # Place nd2 in 1704 canvas at top-left (it's ~600px)
    nd2_z0 = nd2_ds[0]  # best z for mode_z=3 landmarks — ev_nd2 z ≈ 0-2
    ov = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    ov[:nd2_small_h, :nd2_small_w, 1] = norm8(nd2_z0)  # green = exvivo
    ov[:, :, 2] = norm8(final_inv)  # red = warped invivo
    cv2.imwrite(f'{OUT_DIR}/overlay_final_inv_vs_nd2.png', ov)
    print(f"  Saved overlay_final_inv_vs_nd2.png")

# ── Save assets ──
print(f"\nSaving assets to {OUT_NPZ}...")
stage_snapshots_arr = np.array(stage_snapshots)  # (n_stages+1, 1704, 1704)
print(f"  stage_snapshots: {stage_snapshots_arr.shape}")

np.savez_compressed(OUT_NPZ,
    # Stage-by-stage inverse snapshots (mode-z slice at 1704 canvas)
    stage_snapshots=stage_snapshots_arr,
    stage_types=np.array(stage_types),
    stage_indices=np.array(stage_indices),
    canvas_size=CANVAS,
    mode_z=MODE_Z,
    # Landmark data
    pcd_invivo=pcd_invivo,   # (27,3) z,y,x in ~600px space
    pcd_exvivo=pcd_exvivo,   # (27,3) z,y,x in ~600px space
    ev_nd2=ev_nd2,           # (N,3) x,y,z in nd2 4200px
    iv_jy306=iv_jy306,       # (N,3) z,y,x in JY306 native
    M2d=M2d,                 # (2,3) JY306→nd2 affine
    # Ex-vivo reference
    nd2_ds=nd2_ds,           # (12, 600, 600) nd2 downsampled
    nd2_shape=np.array(nd2_stack.shape),  # (12, 4200, 4200)
    # M2d affine warp result
    iv_warped_m2d=iv_warped_m2d,  # (4200, 4200) — M2d warp of JY306 z=3
    # JY306 native for display
    jy306_mode_z=jy306[MODE_Z],  # (658, 629)
)

fsize = os.path.getsize(OUT_NPZ) / 1e6
print(f"Done! {OUT_NPZ} ({fsize:.1f} MB)")
print(f"  {len(stage_snapshots)} snapshots ({stage_snapshots_arr.shape})")
print(f"  Stage types: {list(zip(stage_indices, stage_types))}")