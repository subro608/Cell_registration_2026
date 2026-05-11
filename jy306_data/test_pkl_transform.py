"""
Test pkl transform for row3_2 (merscope15).
Load the pkl, inspect the transforms, apply to one GFP z-slice,
overlay with JY306 and save QC image.
"""
import pickle
import numpy as np
import cv2
import tifffile
import os

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/png_exports/pkl_transform_test'
os.makedirs(OUT_DIR, exist_ok=True)

# Load pkl
pkl_path = f'{BASE}/transformation/3_2_merscope15transformed_alt_20250425102844_type2.pkl'
print("Loading pkl...")
with open(pkl_path, 'rb') as f:
    d = pickle.load(f)

print("Keys:", list(d.keys()))
print("transformed shape:", d['transformed'].shape)
print("pcd_exvivo shape:", d['pcd_exvivo'].shape)
print("pcd_invivo shape:", d['pcd_invivo'].shape)
print("transformations count:", len(d['transformations']))
print("transform keys:", [list(t.keys()) for t in d['transformations']])
print()

# Load JY306 in-vivo
print("Loading JY306...")
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif')
print("JY306 shape:", jy306.shape)  # (16, 658, 629)

# The 'transformed' field is already the warped volume in JY306 space
# Shape: (3, 17, 1734, 1734) — likely (channels, z, y, x) but at different resolution
transformed = d['transformed']
print("transformed shape:", transformed.shape)

# Check what the 'transformed' volume looks like — take MIP
# Use channel 0 (GFP likely)
for ch in range(transformed.shape[0]):
    vol_ch = transformed[ch]  # (17, 1734, 1734)
    mip = vol_ch.max(axis=0)
    mip_norm = ((mip - mip.min()) / max(mip.max() - mip.min(), 1e-6) * 255).astype(np.uint8)
    cv2.imwrite(f'{OUT_DIR}/transformed_ch{ch}_MIP.png', mip_norm)
    print(f"  ch{ch}: min={vol_ch.min():.3f} max={vol_ch.max():.3f}")

# Take mid z-slice of transformed and JY306 and overlay
mid_z_tfm = transformed.shape[1] // 2
mid_z_jy306 = jy306.shape[0] // 2

# Use channel 0 of transformed
tfm_slice = transformed[0, mid_z_tfm]  # (1734, 1734)
jy306_slice = jy306[mid_z_jy306]       # (658, 629)

# Normalize both
def norm8(img):
    img = img.astype(np.float32)
    p2, p98 = np.percentile(img[img > 0], [2, 98]) if img.max() > 0 else (0, 1)
    return np.clip((img - p2) / max(p98 - p2, 1e-6) * 255, 0, 255).astype(np.uint8)

tfm_8   = norm8(tfm_slice)
jy306_8 = norm8(jy306_slice)

# Resize transformed to JY306 size for overlay
tfm_resized = cv2.resize(tfm_8, (jy306_8.shape[1], jy306_8.shape[0]))

# Green/magenta overlay
h, w = jy306_8.shape
ov = np.zeros((h, w, 3), dtype=np.uint8)
ov[:,:,1] = jy306_8     # green = JY306 in-vivo
ov[:,:,0] = tfm_resized  # magenta = transformed exvivo
ov[:,:,2] = tfm_resized
cv2.imwrite(f'{OUT_DIR}/overlay_mid_z.png', ov)
print(f"\nSaved overlay_mid_z.png (JY306 z={mid_z_jy306}, transformed z={mid_z_tfm})")

# Also save point correspondences as scatter
from matplotlib import pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
pcd_e = d['pcd_exvivo']
pcd_i = d['pcd_invivo']
axes[0].scatter(pcd_e[:,2], pcd_e[:,1], c=pcd_e[:,0], cmap='viridis', s=20)
axes[0].set_title('pcd_exvivo (y vs x, colored by z)')
axes[0].invert_yaxis()
axes[1].scatter(pcd_i[:,2], pcd_i[:,1], c=pcd_i[:,0], cmap='viridis', s=20)
axes[1].set_title('pcd_invivo (y vs x, colored by z)')
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/point_correspondences.png', dpi=120)
plt.close()
print("Saved point_correspondences.png")
print(f"\nAll outputs: {OUT_DIR}/")
