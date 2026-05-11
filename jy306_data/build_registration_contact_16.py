#!/usr/bin/env python3
"""
Contact sheet: 16 in-vivo slices registered to ex-vivo.
Each row = one in-vivo z-slice:
  Left: ex-vivo stitched slice with cyan landmark crosshairs
  Right: in-vivo 2D-warped into same space with yellow landmark crosshairs
Saves one big PNG contact sheet.
"""
import numpy as np
import tifffile
import cv2
from scipy.ndimage import median_filter

BASE = '/Users/neurolab/neuroinformatics/margaret'

IV_XY_UM = 0.6835
IV_Z_UM = 3.0

# ============================================================
# Load data
# ============================================================
print("Loading IOU-only stitched 1µm isotropic volume...")
with tifffile.TiffFile(f'{BASE}/registration_video/stitched/stitched_gfp_iou_only_1um_isotropic.tif') as tif:
    n_pages = len(tif.pages)
    h, w = tif.pages[0].shape
    ev_vol = np.zeros((n_pages, h, w), dtype=np.uint16)
    for i, page in enumerate(tif.pages):
        ev_vol[i] = page.asarray()
print(f"  Ex-vivo: {ev_vol.shape}")

print("Loading JY306 in-vivo volume...")
iv_vol_raw = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif').astype(np.float32)
nz_iv, ny_iv, nx_iv = iv_vol_raw.shape
print(f"  In-vivo: {iv_vol_raw.shape}")

print("Loading landmarks and affine...")
lm = np.load(f'{BASE}/registration_video/affine_3d_iou_results.npz', allow_pickle=True)
ev_um = lm['ev_stitched_um']
iv_um = lm['iv_um']
A = lm['affine_3x4']
N_LM = ev_um.shape[0]

# ============================================================
# Median filter BG subtraction
# ============================================================
print("Median filter BG subtraction...")
iv_vol = np.zeros_like(iv_vol_raw)
for z in range(nz_iv):
    bg = median_filter(iv_vol_raw[z], size=15)
    iv_vol[z] = np.clip(iv_vol_raw[z] - bg, 0, None)
del iv_vol_raw

# ============================================================
# Build 3D affine in (z,y,x) pixel convention
# ============================================================
sx, sy, sz = IV_XY_UM, IV_XY_UM, IV_Z_UM
M_fwd = np.array([
    [A[2,2]*sz, A[2,1]*sy, A[2,0]*sx],
    [A[1,2]*sz, A[1,1]*sy, A[1,0]*sx],
    [A[0,2]*sz, A[0,1]*sy, A[0,0]*sx],
])
t_fwd = np.array([A[2,3], A[1,3], A[0,3]])
M_inv = np.linalg.inv(M_fwd)
offset_inv = -M_inv @ t_fwd

# XY bounding box
corners_iv = np.array([
    [0,0,0],[0,0,nx_iv-1],[0,ny_iv-1,0],[0,ny_iv-1,nx_iv-1],
    [nz_iv-1,0,0],[nz_iv-1,0,nx_iv-1],[nz_iv-1,ny_iv-1,0],[nz_iv-1,ny_iv-1,nx_iv-1]
], dtype=np.float64)
corners_out = (M_fwd @ corners_iv.T).T + t_fwd
y_lo = max(0, int(np.floor(corners_out[:,1].min())))
y_hi = min(ev_vol.shape[1], int(np.ceil(corners_out[:,1].max())) + 1)
x_lo = max(0, int(np.floor(corners_out[:,2].min())))
x_hi = min(ev_vol.shape[2], int(np.ceil(corners_out[:,2].max())) + 1)
crop_h = y_hi - y_lo
crop_w = x_hi - x_lo

# Predicted in-vivo landmarks in stitched space
iv_h = np.hstack([iv_um, np.ones((N_LM, 1))])
iv_pred = iv_h @ A.T
errors = np.sqrt(np.sum((iv_pred - ev_um)**2, axis=1))

# ============================================================
# DS factor for contact sheet (target ~600px wide per panel)
# ============================================================
DS = max(1, crop_w // 600)
out_w = crop_w // DS
out_h = crop_h // DS
print(f"  XY crop: {crop_h}x{crop_w}, DS{DS} → {out_h}x{out_w}")

# ============================================================
# Normalize helper
# ============================================================
def norm8(img):
    vals = img[img > 0]
    if len(vals) < 100:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

# ============================================================
# Build contact sheet: 4 columns x 16 rows
# Col 1: ex-vivo with landmarks
# Col 2: in-vivo warped with landmarks
# Col 3: overlay (green/magenta)
# Col 4: overlay zoomed on landmark cluster
# ============================================================
LABEL_H = 30
GAP = 4
COLS = 3
panel_w = out_w
panel_h = out_h
row_h = panel_h + LABEL_H + GAP

sheet_w = COLS * panel_w + (COLS + 1) * GAP
sheet_h = nz_iv * row_h + GAP
print(f"\nContact sheet: {sheet_w}x{sheet_h}")

sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

# Header
HEADER_H = 50
sheet_full = np.zeros((HEADER_H + sheet_h, sheet_w, 3), dtype=np.uint8)

for z_iv in range(nz_iv):
    center_iv = np.array([z_iv, ny_iv/2, nx_iv/2])
    center_out = M_fwd @ center_iv + t_fwd
    z_st = int(round(np.clip(center_out[0], 0, ev_vol.shape[0] - 1)))

    print(f"  iv z={z_iv} → stitched z={z_st}", end="")

    # 2D backward warp
    M2d = np.array([
        [M_inv[2,2], M_inv[2,1], M_inv[2,0]*z_st + M_inv[2,1]*y_lo + M_inv[2,2]*x_lo + offset_inv[2]],
        [M_inv[1,2], M_inv[1,1], M_inv[1,0]*z_st + M_inv[1,1]*y_lo + M_inv[1,2]*x_lo + offset_inv[1]],
    ], dtype=np.float64)

    iv_slice = iv_vol[z_iv].astype(np.float32)
    iv_warped = cv2.warpAffine(iv_slice, M2d, (crop_w, crop_h),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    ev_slice = ev_vol[z_st, y_lo:y_hi, x_lo:x_hi].astype(np.float32)

    # Downsample
    ev_ds = cv2.resize(ev_slice, (out_w, out_h), interpolation=cv2.INTER_AREA)
    iv_ds = cv2.resize(iv_warped, (out_w, out_h), interpolation=cv2.INTER_AREA)

    ev_u8 = norm8(ev_ds)
    iv_u8 = norm8(iv_ds)

    # Color panels
    ev_rgb = cv2.cvtColor(ev_u8, cv2.COLOR_GRAY2BGR)
    iv_rgb = cv2.cvtColor(iv_u8, cv2.COLOR_GRAY2BGR)

    # Overlay: green=ev, magenta=iv
    ov_rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    ov_rgb[:,:,1] = ev_u8          # green
    ov_rgb[:,:,0] = iv_u8          # blue (magenta=R+B)
    ov_rgb[:,:,2] = iv_u8          # red

    # Draw landmarks on all panels
    n_lm_here = 0
    for i in range(N_LM):
        # Find landmarks nearest to this stitched z
        ex_z = ev_um[i, 2]
        if abs(ex_z - z_st) > 20:
            continue
        n_lm_here += 1

        # Ex-vivo landmark in DS coords
        ex_x = int(round((ev_um[i, 0] - x_lo) / DS))
        ex_y = int(round((ev_um[i, 1] - y_lo) / DS))
        # In-vivo predicted in DS coords
        iv_x = int(round((iv_pred[i, 0] - x_lo) / DS))
        iv_y = int(round((iv_pred[i, 1] - y_lo) / DS))

        # Cyan crosshairs on ex-vivo panel
        R = 8
        cv2.drawMarker(ev_rgb, (ex_x, ex_y), (255, 255, 0), cv2.MARKER_CROSS, R*2, 2)
        # Yellow crosshairs on in-vivo panel
        cv2.drawMarker(iv_rgb, (iv_x, iv_y), (0, 255, 255), cv2.MARKER_CROSS, R*2, 2)

        # Both on overlay + connecting line
        cv2.drawMarker(ov_rgb, (ex_x, ex_y), (255, 255, 0), cv2.MARKER_CROSS, R*2, 1)
        cv2.drawMarker(ov_rgb, (iv_x, iv_y), (0, 255, 255), cv2.MARKER_CROSS, R*2, 1)
        cv2.line(ov_rgb, (ex_x, ex_y), (iv_x, iv_y), (255, 255, 255), 1)

    print(f"  {n_lm_here} landmarks")

    # Place panels on sheet
    y0 = HEADER_H + z_iv * row_h + GAP
    panels = [ev_rgb, iv_rgb, ov_rgb]
    for ci, p in enumerate(panels):
        x0 = GAP + ci * (panel_w + GAP)
        sheet_full[y0:y0+panel_h, x0:x0+panel_w] = p

    # Label
    label_y = y0 + panel_h + 2
    cv2.putText(sheet_full, f'iv_z={z_iv} -> stitch_z={z_st}  ({n_lm_here} lm)',
                (GAP + 4, label_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

del iv_vol, ev_vol

# Column headers
headers = ['Ex-vivo (cyan=landmark)', 'In-vivo warped (yellow=predicted)', 'Overlay + error arrows']
for ci, h in enumerate(headers):
    x0 = GAP + ci * (panel_w + GAP)
    cv2.putText(sheet_full, h, (x0 + 4, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

# Title
cv2.putText(sheet_full, f'Registration: 16 in-vivo slices -> stitched ex-vivo | {N_LM} landmarks | mean err={errors.mean():.1f}um',
            (GAP + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

out_path = f'{BASE}/registration_overlay.png'
cv2.imwrite(out_path, sheet_full)
print(f"\nSaved: {out_path} ({sheet_full.shape[1]}x{sheet_full.shape[0]})")
