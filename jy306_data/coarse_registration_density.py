"""
Coarse registration using cell-density maps.
Detect cell bodies (blobs) in both MIPs, smooth to density maps, NCC on those.
More robust than raw NCC since cell sizes/shapes are comparable across modalities.
"""
import tifffile
import numpy as np
import cv2
import os

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT  = f'{BASE}/png_exports/coarse_registration'
os.makedirs(OUT, exist_ok=True)

print("Loading volumes...")
inv = tifffile.imread(f'{BASE}/registration_video/stitched/invivo_1um_isotropic.tif')
ex  = tifffile.imread(f'{BASE}/registration_video/stitched/stitched_gfp_elastix_1um_isotropic.tif')
print(f"  invivo : {inv.shape}")
print(f"  exvivo : {ex.shape}")

def norm8(v, p_lo=1, p_hi=99.5):
    v = v.astype(np.float32)
    vals = v[v > 0]
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    return np.clip((v - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

def detect_cells(img8, min_area=20, max_area=2000, threshold=60):
    """Threshold + find blobs → return centroid heatmap (same size as img8)."""
    _, thr = cv2.threshold(img8, threshold, 255, cv2.THRESH_BINARY)
    # morphological open to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thr)
    heat = np.zeros(img8.shape, dtype=np.float32)
    n_cells = 0
    for i in range(1, nlabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cy, cx = int(centroids[i][1]), int(centroids[i][0])
            heat[cy, cx] = 1.0
            n_cells += 1
    print(f"    Detected {n_cells} cell blobs (area {min_area}–{max_area}px, thr={threshold})")
    return heat, n_cells

def make_density(heat, sigma_px=20):
    """Gaussian smooth centroid map → density map."""
    k = int(sigma_px * 4) | 1  # odd kernel
    return cv2.GaussianBlur(heat, (k, k), sigma_px)

# ── In-vivo ──────────────────────────────────────────────────────────────────
print("\n--- In-vivo cell detection ---")
mip_inv = norm8(inv.max(axis=0))
heat_inv, n_inv = detect_cells(mip_inv, min_area=10, max_area=1500, threshold=40)
dens_inv = make_density(heat_inv, sigma_px=30)

# ── Ex-vivo ──────────────────────────────────────────────────────────────────
print("--- Ex-vivo cell detection ---")
mip_ex  = norm8(ex.max(axis=0))
heat_ex, n_ex = detect_cells(mip_ex, min_area=10, max_area=1500, threshold=30)
dens_ex = make_density(heat_ex, sigma_px=30)

# save density maps for inspection
cv2.imwrite(f'{OUT}/density_invivo.png',  norm8(dens_inv))
cv2.imwrite(f'{OUT}/density_exvivo.png',  norm8(dens_ex))
print(f"  Saved density maps: density_invivo.png, density_exvivo.png")

# ── NCC template match on density maps ───────────────────────────────────────
print("\n--- NCC on density maps ---")
DS = 4
tmpl = cv2.resize(dens_inv, (dens_inv.shape[1]//DS, dens_inv.shape[0]//DS))
srch = cv2.resize(dens_ex,  (dens_ex.shape[1]//DS,  dens_ex.shape[0]//DS))
print(f"  Template DS={DS}: {tmpl.shape}  Search: {srch.shape}")

res = cv2.matchTemplate(srch.astype(np.float32), tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(res)
y0 = max_loc[1] * DS
x0 = max_loc[0] * DS
print(f"  Best NCC={max_val:.4f}  →  top-left: y={y0}, x={x0}")

# Save NCC heatmap
heatmap8 = norm8(res)
heatmap_disp = cv2.resize(heatmap8, (686, 686))
cv2.imwrite(f'{OUT}/ncc_heatmap_density.png', heatmap_disp)

# Print top-5 peaks for inspection
flat = res.ravel()
idx  = np.argsort(flat)[::-1][:10]
print("  Top-10 NCC peaks:")
for rank, i in enumerate(idx):
    iy, ix = np.unravel_index(i, res.shape)
    print(f"    #{rank+1}: NCC={flat[i]:.4f}  y={iy*DS}  x={ix*DS}")

# ── Overlay ───────────────────────────────────────────────────────────────────
H_inv, W_inv = inv.shape[1], inv.shape[2]
H_ex,  W_ex  = ex.shape[1],  ex.shape[2]

mip_inv8 = norm8(inv.max(axis=0))
mip_ex8  = norm8(ex.max(axis=0))

ov = np.zeros((H_ex, W_ex, 3), dtype=np.uint8)
ov[:, :, 1] = mip_ex8   # green = ex-vivo

y1 = min(y0 + H_inv, H_ex); iy1 = y1 - y0
x1 = min(x0 + W_inv, W_ex); ix1 = x1 - x0
ov[y0:y1, x0:x1, 0] = mip_inv8[:iy1, :ix1]  # R  } magenta
ov[y0:y1, x0:x1, 2] = mip_inv8[:iy1, :ix1]  # B  }
cv2.rectangle(ov, (x0, y0), (x0+W_inv-1, y0+H_inv-1), (0, 255, 255), 4)
cv2.imwrite(f'{OUT}/overlay_density_ncc.png', ov)

# Zoom
PAD = 150
yA=max(0,y0-PAD); yB=min(H_ex,y0+H_inv+PAD)
xA=max(0,x0-PAD); xB=min(W_ex,x0+W_inv+PAD)
crop = ov[yA:yB, xA:xB]
crop_d = cv2.resize(crop, (1000, int(1000*crop.shape[0]/crop.shape[1])))
cv2.imwrite(f'{OUT}/overlay_density_ncc_zoom.png', crop_d)

print(f"\nSaved: overlay_density_ncc.png, overlay_density_ncc_zoom.png")
print(f"Best translation: y={y0}, x={x0}  (z still from XZ NCC = 216 from previous run)")
