"""
Deformable registration: nd2 row2_1 MIP → exvivo_merscope_combined MIP.

Pipeline:
1. Start from ECC-refined affine (affine_nd2_to_merscope_ecc.npy)
2. Warp nd2 to MERSCOPE space using ECC affine
3. Compute Farneback optical flow between warped nd2 and MERSCOPE target
4. Combine: affine + flow field = full deformable mapping
5. Verify on 4 ground-truth landmarks

Output:
  registration_video/flow_nd2_to_merscope.npy — dense displacement field (2, H, W)
  png_exports/hot_pixel_tracking/deformable/ — QC images
"""

import numpy as np
import cv2
import tifffile
import os
import pickle

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT = os.path.join(BASE, 'png_exports/hot_pixel_tracking/deformable')
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Load ECC affine: nd2 (4200) → MERSCOPE (1627)
# ============================================================
M_ecc = np.load(os.path.join(BASE, 'registration_video/affine_nd2_to_merscope_ecc.npy'))
print(f"ECC affine:\n{M_ecc}")

# ============================================================
# Load images
# ============================================================
nd2_dir = os.path.join(BASE, 'png_exports/registration_video/row2_1')
nd2_slices = []
for i in range(12):
    img = cv2.imread(os.path.join(nd2_dir, f'GFP_z{i:03d}.png'), cv2.IMREAD_GRAYSCALE)
    nd2_slices.append(img.astype(np.float32))
nd2_mip = np.max(nd2_slices, axis=0)  # (4200, 4200)

merc_vol = tifffile.imread(os.path.join(BASE, 'exvivo_merscope_combined/2_1_merscope17.tif'))
merc_mip = np.max(merc_vol[:, :, :, 0], axis=0).astype(np.float32)
H_merc, W_merc = merc_mip.shape
print(f"nd2 MIP: {nd2_mip.shape}, MERSCOPE MIP: {merc_mip.shape}")

# ============================================================
# Warp nd2 to MERSCOPE space using ECC affine
# ============================================================
nd2_warped = cv2.warpAffine(nd2_mip, M_ecc.astype(np.float32), (W_merc, H_merc),
                             flags=cv2.INTER_LINEAR)

def norm8(img, p_lo=1, p_hi=99.5):
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

nd2_w8 = norm8(nd2_warped)
merc8 = norm8(merc_mip)

# ============================================================
# Try multiple Farneback parameter sets
# ============================================================
configs = {
    'farneback_small': dict(pyr_scale=0.5, levels=5, winsize=32, iterations=10, poly_n=7, poly_sigma=1.5),
    'farneback_large': dict(pyr_scale=0.5, levels=5, winsize=128, iterations=15, poly_n=7, poly_sigma=1.5),
    'farneback_xlarge': dict(pyr_scale=0.5, levels=7, winsize=256, iterations=20, poly_n=7, poly_sigma=1.5),
}

# Also try with Gaussian-blurred inputs (reduce noise)
nd2_blur = cv2.GaussianBlur(nd2_w8, (15, 15), 3)
merc_blur = cv2.GaussianBlur(merc8, (15, 15), 3)

flows = {}
for name, params in configs.items():
    print(f"\nComputing {name}...")
    flow = cv2.calcOpticalFlowFarneback(nd2_w8, merc8, flow=None, **params, flags=0)
    flows[name] = flow
    print(f"  dx range [{flow[:,:,0].min():.1f}, {flow[:,:,0].max():.1f}], "
          f"dy range [{flow[:,:,1].min():.1f}, {flow[:,:,1].max():.1f}]")
    mag = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    print(f"  magnitude: mean={mag.mean():.2f}, max={mag.max():.2f}, p99={np.percentile(mag,99):.2f}")

# Blurred version
print(f"\nComputing farneback_blurred...")
flow_blur = cv2.calcOpticalFlowFarneback(nd2_blur, merc_blur, flow=None,
                                          pyr_scale=0.5, levels=5, winsize=64,
                                          iterations=15, poly_n=7, poly_sigma=1.5, flags=0)
flows['farneback_blurred'] = flow_blur
mag = np.sqrt(flow_blur[:,:,0]**2 + flow_blur[:,:,1]**2)
print(f"  magnitude: mean={mag.mean():.2f}, max={mag.max():.2f}, p99={np.percentile(mag,99):.2f}")

# ============================================================
# Load pkl and build composite affine for point inversion
# ============================================================
pkl_path = os.path.join(BASE, 'transformation/2_1_merscope17transformed_20250424104024-001.pkl')
with open(pkl_path, 'rb') as f:
    pkl_data = pickle.load(f)
transforms = pkl_data['transformations']

def to_4x4(bhat):
    M = np.eye(4)
    M[:4, :3] = bhat
    return M

def scale_4x4(s):
    return np.diag([float(s), float(s), float(s), 1.0])

# Composite of all affine stages (bhat + scale, ignoring vec_fields)
composite = np.eye(4)
for t in transforms:
    if 'bhat' in t:
        composite = composite @ to_4x4(t['bhat'])
    elif 'scale' in t:
        composite = composite @ scale_4x4(t['scale'])

# 2D affine: native MERSCOPE (row,col) → 1704 canvas (row,col) at z_mean
z_m = 1.5
c = composite
M_nat_to_canvas = np.array([
    [c[2, 2], c[1, 2], z_m * c[0, 2] + c[3, 2]],  # col'
    [c[2, 1], c[1, 1], z_m * c[0, 1] + c[3, 1]]    # row'
], dtype=np.float64)

M3_nat_canvas = np.vstack([M_nat_to_canvas, [0, 0, 1]])
M3_canvas_nat = np.linalg.inv(M3_nat_canvas)

# canvas_to_jy306 scale
canvas_scale = 1704.0 / 578.0

def jy306_to_merscope_native(pt_xy):
    """Convert JY306 (col,row) → MERSCOPE native (col,row) via pkl composite affine inverse."""
    # JY306 ~578 space → 1704 canvas
    canvas_xy = np.array(pt_xy) * canvas_scale
    # 1704 canvas → MERSCOPE native
    h = np.array([canvas_xy[0], canvas_xy[1], 1.0])
    native_xy = M3_canvas_nat @ h
    return native_xy[:2]

# ============================================================
# Load landmarks and verify
# ============================================================
lm = np.load(os.path.join(BASE, 'registration_video/landmarks.npz'))
src_pts = lm['src_points'][:, :2]  # nd2 (col, row)
tgt_pts = lm['tgt_points'][:, :2]  # JY306 (col, row)

# Also check the landmarks_row21_jy306.npz for cross-reference
lm2_path = os.path.join(BASE, 'registration_video/landmarks_row21_jy306.npz')
if os.path.exists(lm2_path):
    lm2 = np.load(lm2_path, allow_pickle=True)
    print(f"\nRow21 JY306 landmarks: src={lm2['src_points'].shape}, tgt={lm2['tgt_points'].shape}")

print(f"\n{'='*70}")
print(f"{'Cell':>4}  {'ECC-only err':>12}  ", end="")
for name in flows:
    print(f"  {name:>16}", end="")
print()
print("-" * (20 + 18 * len(flows)))

best_flow_name = None
best_mean_err = 999

for flow_name, flow in flows.items():
    errs = []
    for i in range(len(src_pts)):
        # GT: nd2 → MERSCOPE via ECC affine
        nd2_pos = src_pts[i]
        gt_merc = M_ecc @ np.array([nd2_pos[0], nd2_pos[1], 1.0])

        # pkl-inverted: JY306 → MERSCOPE native (via composite affine only)
        jy_pos = tgt_pts[i]
        pkl_merc = jy306_to_merscope_native(jy_pos)

        # Apply flow correction: flow maps nd2_warped pixel → merc pixel
        # So corrected gt = gt_merc + flow[gt_merc]
        gx = int(round(np.clip(gt_merc[0], 0, W_merc - 1)))
        gy = int(round(np.clip(gt_merc[1], 0, H_merc - 1)))
        gt_def = gt_merc + np.array([flow[gy, gx, 0], flow[gy, gx, 1]])

        err = np.linalg.norm(pkl_merc - gt_def)
        errs.append(err)

    mean_err = np.mean(errs)
    if mean_err < best_mean_err:
        best_mean_err = mean_err
        best_flow_name = flow_name

# Print results table
for i in range(len(src_pts)):
    nd2_pos = src_pts[i]
    gt_merc = M_ecc @ np.array([nd2_pos[0], nd2_pos[1], 1.0])
    jy_pos = tgt_pts[i]
    pkl_merc = jy306_to_merscope_native(jy_pos)

    err_ecc = np.linalg.norm(pkl_merc - gt_merc)

    print(f"  L{i+1}  {err_ecc:12.1f}  ", end="")
    for flow_name, flow in flows.items():
        gx = int(round(np.clip(gt_merc[0], 0, W_merc - 1)))
        gy = int(round(np.clip(gt_merc[1], 0, H_merc - 1)))
        gt_def = gt_merc + np.array([flow[gy, gx, 0], flow[gy, gx, 1]])
        err = np.linalg.norm(pkl_merc - gt_def)
        print(f"  {err:16.1f}", end="")
    print()

# Print means
print(f"\n  Mean errors:")
for flow_name, flow in flows.items():
    errs = []
    for i in range(len(src_pts)):
        nd2_pos = src_pts[i]
        gt_merc = M_ecc @ np.array([nd2_pos[0], nd2_pos[1], 1.0])
        jy_pos = tgt_pts[i]
        pkl_merc = jy306_to_merscope_native(jy_pos)
        gx = int(round(np.clip(gt_merc[0], 0, W_merc - 1)))
        gy = int(round(np.clip(gt_merc[1], 0, H_merc - 1)))
        gt_def = gt_merc + np.array([flow[gy, gx, 0], flow[gy, gx, 1]])
        errs.append(np.linalg.norm(pkl_merc - gt_def))
    print(f"    {flow_name}: {np.mean(errs):.1f} px")

print(f"\nBest: {best_flow_name} ({best_mean_err:.1f} px)")

# ============================================================
# Save best flow
# ============================================================
best_flow = flows[best_flow_name]
np.save(os.path.join(BASE, 'registration_video/flow_nd2_to_merscope.npy'), best_flow)
print(f"Saved best flow field ({best_flow_name})")

# ============================================================
# Visualize: 3-panel comparison
# ============================================================
# Apply best flow
h, w = nd2_warped.shape
ys, xs = np.mgrid[:h, :w].astype(np.float32)
map_x = xs + best_flow[:, :, 0]
map_y = ys + best_flow[:, :, 1]
nd2_deformed = cv2.remap(nd2_warped, map_x, map_y, cv2.INTER_LINEAR)
nd2_d8 = norm8(nd2_deformed)

# Panel 1: ECC affine
ov1 = np.zeros((H_merc, W_merc, 3), dtype=np.uint8)
ov1[:, :, 1] = nd2_w8
ov1[:, :, 0] = merc8
ov1[:, :, 2] = merc8

# Panel 2: Deformable
ov2 = np.zeros((H_merc, W_merc, 3), dtype=np.uint8)
ov2[:, :, 1] = nd2_d8
ov2[:, :, 0] = merc8
ov2[:, :, 2] = merc8

# Panel 3: Flow magnitude
mag = np.sqrt(best_flow[:, :, 0]**2 + best_flow[:, :, 1]**2)
mag_vis = np.clip(mag / max(mag.max(), 1) * 255, 0, 255).astype(np.uint8)
mag_color = cv2.applyColorMap(mag_vis, cv2.COLORMAP_JET)

# Add landmark circles
DISP = 800
scale_d = DISP / H_merc

panels = [ov1, ov2, mag_color]
titles = ["ECC affine only", f"ECC + {best_flow_name}", f"Flow magnitude (max={mag.max():.1f}px)"]
resized = []
for p, title in zip(panels, titles):
    r = cv2.resize(p, (DISP, DISP))
    cv2.putText(r, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Draw landmarks
    for i in range(len(src_pts)):
        nd2_pos = src_pts[i]
        gt_merc = M_ecc @ np.array([nd2_pos[0], nd2_pos[1], 1.0])
        jy_pos = tgt_pts[i]
        pkl_merc = jy306_to_merscope_native(jy_pos)

        # Green = GT, Cyan = pkl-inverted
        cv2.circle(r, (int(gt_merc[0] * scale_d), int(gt_merc[1] * scale_d)), 8, (0, 255, 0), 2)
        cv2.circle(r, (int(pkl_merc[0] * scale_d), int(pkl_merc[1] * scale_d)), 8, (0, 255, 255), 2)
        cv2.putText(r, f"L{i+1}", (int(pkl_merc[0]*scale_d)+10, int(pkl_merc[1]*scale_d)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    resized.append(r)

panel_img = np.hstack(resized)
cv2.imwrite(os.path.join(OUT, 'affine_vs_deformable_3panel.png'), panel_img)
print(f"\nSaved {OUT}/affine_vs_deformable_3panel.png")

# ============================================================
# Also try SimpleElastix if available
# ============================================================
try:
    import SimpleITK as sitk
    print("\n\nSimpleITK available — trying B-spline registration...")

    fixed = sitk.GetImageFromArray(merc8)
    moving = sitk.GetImageFromArray(nd2_w8)

    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed)
    elastix.SetMovingImage(moving)

    # B-spline parameter map
    pm = sitk.GetDefaultParameterMap('bspline')
    pm['MaximumNumberOfIterations'] = ['2000']
    pm['FinalGridSpacingInPhysicalUnits'] = ['32']
    pm['NumberOfResolutions'] = ['4']
    pm['Metric'] = ['AdvancedNormalizedCorrelation']

    elastix.SetParameterMap(pm)
    elastix.SetLogToConsole(False)
    elastix.Execute()

    result = sitk.GetArrayFromImage(elastix.GetResultImage())
    result8 = norm8(result.astype(np.float32))

    ov3 = np.zeros((H_merc, W_merc, 3), dtype=np.uint8)
    ov3[:, :, 1] = result8
    ov3[:, :, 0] = merc8
    ov3[:, :, 2] = merc8

    r3 = cv2.resize(ov3, (DISP, DISP))
    cv2.putText(r3, "ECC + Elastix B-spline", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    panel4 = np.hstack([resized[0], r3, resized[1]])
    cv2.imwrite(os.path.join(OUT, 'affine_vs_elastix_vs_farneback.png'), panel4)
    print(f"Saved {OUT}/affine_vs_elastix_vs_farneback.png")

    # Get transformix point transform for landmark verification
    print("Elastix registration complete")

except ImportError:
    print("\nSimpleITK not available — skipping elastix B-spline")
except Exception as e:
    print(f"\nElastix failed: {e}")

print("\nDone.")
