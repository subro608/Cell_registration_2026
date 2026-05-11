"""
QC panel for pkl transform (row3_2 / merscope15).
Row 1: in-vivo raw | ex-vivo raw
Row 2: in-vivo + landmarks | ex-vivo + landmarks
Row 3: overlay (transformed ex-vivo → JY306) + both landmark sets
"""
import pickle
import numpy as np
import cv2
import tifffile
from matplotlib import pyplot as plt
import os

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = f'{BASE}/png_exports/pkl_transform_test'
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load pkl ----
pkl_path = f'{BASE}/transformation/3_2_merscope15transformed_alt_20250425102844_type2.pkl'
with open(pkl_path, 'rb') as f:
    d = pickle.load(f)

transformed = d['transformed']   # (3, 17, 1734, 1734)
pcd_e = d['pcd_exvivo']          # (41, 3) z,y,x  — in pkl coord space (~JY306 scale)
pcd_i = d['pcd_invivo']          # (41, 3) z,y,x  — in JY306 pixel space

# ---- Load JY306 ----
jy306 = tifffile.imread(f'{BASE}/JY306_in_Vivo_stack_flipped_s80.tif')  # (16, 658, 629)
mid_z = jy306.shape[0] // 2
H, W = jy306.shape[1], jy306.shape[2]   # 658, 629

# ---- Load ex-vivo native MIP ----
exvivo_mip = cv2.imread(
    f'{BASE}/png_exports/registration_video/row3_2/GFP_MIP.png',
    cv2.IMREAD_GRAYSCALE).astype(np.float32)
mip_H, mip_W = exvivo_mip.shape   # 4200, 4200

def norm8(img, p_lo=1, p_hi=99.5):
    vals = img[img > 0]
    if len(vals) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

jy_8   = norm8(jy306[mid_z].astype(np.float32))
ev_8   = norm8(exvivo_mip)

# Transformed ex-vivo cropped to JY306 size
tfm_crop = transformed[0, mid_z][:H, :W]
tfm_8    = norm8(tfm_crop.astype(np.float32))

# Overlay (green = JY306, magenta = transformed ex-vivo)
ov = np.zeros((H, W, 3), dtype=np.uint8)
ov[:, :, 1] = jy_8
ov[:, :, 0] = tfm_8
ov[:, :, 2] = tfm_8

# Scale pcd_exvivo from pkl coord space → native MIP pixels
# pcd_exvivo x: 0–563 maps to JY306 W=629 → MIP W=4200
scale_y = mip_H / H   # ~6.38
scale_x = mip_W / W   # ~6.68

N = len(pcd_i)

# ---- Figure: 5 panels ----
fig = plt.figure(figsize=(22, 14), facecolor='black')

# Row 1: raw images
ax1 = fig.add_subplot(2, 3, 1)   # in-vivo raw
ax2 = fig.add_subplot(2, 3, 2)   # ex-vivo raw

# Row 2: with landmarks + overlay
ax3 = fig.add_subplot(2, 3, 4)   # in-vivo + landmarks
ax4 = fig.add_subplot(2, 3, 5)   # ex-vivo + landmarks
ax5 = fig.add_subplot(1, 3, 3)   # overlay + landmarks (spans both rows, right col)

for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_facecolor('black')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    ax.tick_params(colors='#666')

# --- 1: in-vivo raw ---
ax1.imshow(jy_8, cmap='gray', vmin=0, vmax=255)
ax1.set_title(f'In-vivo (JY306)  z={mid_z}', color='lime', fontsize=13, pad=6)
ax1.axis('off')

# --- 2: ex-vivo raw ---
ax2.imshow(ev_8, cmap='gray', vmin=0, vmax=255)
ax2.set_title('Ex-vivo row3_2  GFP MIP (native)', color='violet', fontsize=13, pad=6)
ax2.axis('off')

# --- 3: in-vivo + landmarks (red circles) ---
ax3.imshow(jy_8, cmap='gray', vmin=0, vmax=255)
for k in range(N):
    zi, yi, xi = pcd_i[k]
    ax3.plot(xi, yi, 'o', color='red', markersize=9,
             markeredgewidth=1.5, markeredgecolor='white', zorder=5)
ax3.set_title('In-vivo + landmarks', color='lime', fontsize=13, pad=6)
ax3.axis('off')

# --- 4: ex-vivo + landmarks (yellow circles, scaled to MIP pixels) ---
ax4.imshow(ev_8, cmap='gray', vmin=0, vmax=255)
for k in range(N):
    ze, ye, xe = pcd_e[k]
    ax4.plot(xe * scale_x, ye * scale_y, 'o', color='yellow', markersize=9,
             markeredgewidth=1.5, markeredgecolor='black', zorder=5)
ax4.set_title('Ex-vivo + landmarks', color='violet', fontsize=13, pad=6)
ax4.axis('off')

# --- 5: overlay + both landmark sets + connecting lines ---
ax5.imshow(ov)
for k in range(N):
    zi, yi, xi = pcd_i[k]
    ze, ye, xe = pcd_e[k]
    # Both point sets are in JY306/pkl coord space — plot directly
    ax5.plot(xi, yi, 'o', color='red',    markersize=9,
             markeredgewidth=1.5, markeredgecolor='white', zorder=6, label='in-vivo' if k==0 else '')
    ax5.plot(xe, ye, 's', color='yellow', markersize=9,
             markeredgewidth=1.5, markeredgecolor='black',  zorder=6, label='ex-vivo' if k==0 else '')
    ax5.plot([xi, xe], [yi, ye], '-', color='white', alpha=0.6, linewidth=0.9, zorder=5)
ax5.legend(loc='upper right', fontsize=10, facecolor='#111', labelcolor='white',
           markerscale=0.8, framealpha=0.7)
ax5.set_title('Overlay: transformed ex-vivo (M) + JY306 (G)\nRed=in-vivo landmarks  Yellow=ex-vivo landmarks',
              color='white', fontsize=11, pad=6)
ax5.axis('off')

plt.tight_layout(pad=0.8)
plt.savefig(f'{OUT_DIR}/qc_panel.png', dpi=150, bbox_inches='tight',
            facecolor='black', edgecolor='none')
plt.close()
print(f"Saved: {OUT_DIR}/qc_panel.png")
