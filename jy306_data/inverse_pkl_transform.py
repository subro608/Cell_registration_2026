"""
Inverse pkl transform: JY306 in-vivo → ex-vivo space (row3_2).

Test:
  1. Round-trip: pcd_invivo → inverse → forward → should recover pcd_invivo (~0px error)
  2. Visual: warp JY306 image into ex-vivo space, overlay with raw GFP MIP

Inverse pipeline = apply stages in REVERSE order (13→0) with inverted ops:
  scale:       p_in = p_out / scale     (image: sample at q*scale)
  bhat:        p_in = (p_out-t)@inv(R)  (image: sample at q@R+t)
  vec_field:   p_in = p_out - vf[p]    (image: sample at q+vf[q])
"""
import pickle
import numpy as np
import cv2
import tifffile
from scipy.ndimage import map_coordinates
import os

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT  = f'{BASE}/png_exports/pkl_transform_test/inverse_transform'
os.makedirs(OUT, exist_ok=True)

with open(f'{BASE}/transformation/3_2_merscope15transformed_alt_20250425102844_type2.pkl','rb') as f:
    d = pickle.load(f)

transforms  = d['transformations']
pcd_i       = d['pcd_invivo']    # (41,3) in JY306/pkl output coord space
pcd_e       = d['pcd_exvivo']    # (41,3) also in pkl output coord space (post-transform)

# ============================================================
# 1. POINT INVERSE — round-trip test
# ============================================================
def interp_vecfield(vf, pts):
    """Sample vf at fractional point coords pts (N,3). Returns (N,3) displacements."""
    out = np.zeros_like(pts)
    for c in range(3):
        out[:, c] = map_coordinates(vf[..., c], pts.T, order=1, mode='nearest')
    return out

def point_inverse(pts, transforms):
    """
    pts: (N,3) in JY306/output space
    Apply inverse stages in reverse order → ex-vivo input space
    """
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
            # Approximate: subtract vf at current position
            disp = interp_vecfield(val, p)
            p = p - disp
    return p

def point_forward(pts, transforms):
    """
    pts: (N,3) in ex-vivo input space
    Apply forward stages in order → JY306/output space
    """
    p = pts.copy().astype(np.float64)
    for t in transforms:
        key = list(t.keys())[0]
        val = t[key]
        if key == 'scale':
            p = p * val
        elif key == 'bhat':
            R, tv = val[:3].astype(np.float64), val[3].astype(np.float64)
            p = p @ R + tv
        elif key == 'vec_field_total':
            disp = interp_vecfield(val, p)
            p = p + disp
    return p

print("=== Round-trip test (pcd_invivo → inverse → forward → pcd_invivo) ===")
p_ex   = point_inverse(pcd_i, transforms)
p_back = point_forward(p_ex, transforms)
errs   = np.linalg.norm(p_back - pcd_i, axis=1)
print(f"  Round-trip error: mean={errs.mean():.3f}  max={errs.max():.3f}  (px in pkl space)")
print(f"  pcd_invivo[0]:  {pcd_i[0]}")
print(f"  → inverse →    {p_ex[0]}")
print(f"  → forward →    {p_back[0]}")
print(f"  pcd_exvivo[0] (stored): {pcd_e[0]}")
print(f"  inverse vs pcd_exvivo: {np.linalg.norm(p_ex[0] - pcd_e[0]):.3f} px")
print()

# ============================================================
# 2. IMAGE INVERSE — warp JY306 into ex-vivo space
# ============================================================
def norm8(img, p_lo=1, p_hi=99.5):
    vals = img[img > 0]
    if len(vals) == 0: return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

def apply_scale_inv(vol, scale):
    """Inverse of scale: sample at q*scale"""
    z,y,x = np.mgrid[:vol.shape[0],:vol.shape[1],:vol.shape[2]].astype(np.float32)
    return map_coordinates(vol, [z*scale, y*scale, x*scale], order=1, mode='constant', cval=0)

def apply_affine_inv(vol, bhat):
    """Inverse of affine: sample at q@R+t"""
    R, tv = bhat[:3].astype(np.float64), bhat[3].astype(np.float64)
    Z,Y,X = vol.shape
    z,y,x = np.mgrid[:Z,:Y,:X]
    pts = np.stack([z.ravel(),y.ravel(),x.ravel()],axis=1).astype(np.float64)
    pts_in = pts @ R + tv
    return map_coordinates(vol, pts_in.T.reshape(3,Z,Y,X), order=1, mode='constant', cval=0)

def apply_vecfield_inv(vol, vf):
    """Inverse of vec_field: sample at q+vf[q]"""
    z,y,x = np.mgrid[:vol.shape[0],:vol.shape[1],:vol.shape[2]].astype(np.float32)
    return map_coordinates(vol,
        [z + vf[...,0], y + vf[...,1], x + vf[...,2]],
        order=1, mode='constant', cval=0)

print("=== Visual test: warping JY306 → ex-vivo space ===")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif')  # (16,658,629)
H_jy, W_jy = jy306.shape[1], jy306.shape[2]

# Upsample JY306 MIP to pkl canvas size (1734x1734x17)
mip_jy = jy306.max(axis=0).astype(np.float32)
OUT_XY, OUT_Z = 1734, 17
jy_up = np.stack([cv2.resize(mip_jy, (OUT_XY,OUT_XY), interpolation=cv2.INTER_LINEAR)]*OUT_Z, axis=0)

print(f"  JY306 upsampled: {jy_up.shape}")
print("  Applying inverse stages (13→0)...")

result = jy_up.copy()
for i, t in enumerate(reversed(transforms)):
    key = list(t.keys())[0]; val = t[key]
    print(f"    Stage [{len(transforms)-1-i:2d}→inv] {key}...", end=' ', flush=True)
    if key == 'scale':            result = apply_scale_inv(result, val)
    elif key == 'bhat':           result = apply_affine_inv(result, val)
    elif key == 'vec_field_total':result = apply_vecfield_inv(result, val)
    print(f"done  nonzero={np.count_nonzero(result)}")

# MIP of inverse-warped JY306
mip_inv = result.max(axis=0)  # (1734,1734)

# Load raw ex-vivo GFP MIP (4200x4200 native)
exvivo_mip = cv2.imread(
    f'{BASE}/png_exports/registration_video/row3_2/GFP_MIP.png',
    cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Resize both to same display size for overlay
DISP = 1000
inv_d = cv2.resize(mip_inv.astype(np.float32), (DISP,DISP), interpolation=cv2.INTER_LINEAR)
ex_d  = cv2.resize(exvivo_mip, (DISP,DISP), interpolation=cv2.INTER_LINEAR)

inv8 = norm8(inv_d)
ex8  = norm8(ex_d)

# Overlay: green=ex-vivo native, magenta=inverted JY306
ov = np.zeros((DISP,DISP,3), dtype=np.uint8)
ov[:,:,1] = ex8    # green  = ex-vivo native MIP
ov[:,:,0] = inv8   # magenta = JY306 warped to ex-vivo space
ov[:,:,2] = inv8

cv2.imwrite(f'{OUT}/jy306_warped_to_exvivo.png',    norm8(inv_d))
cv2.imwrite(f'{OUT}/exvivo_native_MIP.png',          ex8)
cv2.imwrite(f'{OUT}/overlay_jy306_inv_vs_exvivo.png', ov)

print(f"\nSaved to {OUT}/")
print("  jy306_warped_to_exvivo.png — JY306 warped into ex-vivo space")
print("  overlay_jy306_inv_vs_exvivo.png — green=ex-vivo native, magenta=warped JY306")
