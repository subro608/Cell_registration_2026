"""
Pairwise Distance Correlation: in-vivo vs ex-vivo in JY306 space (3D, µm).

Both pcd_invivo and pcd_exvivo are in JY306 master space (z, y, x).
Convert to µm: XY * 0.685, Z * 3.0 (JY306 pixel sizes).

Output: png_exports/coarse_registration/contact_sheet/distance_matrix_correlation_3d.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import glob
import os

BASE = '/Users/neurolab/neuroinformatics/margaret'
OUT_DIR = os.path.join(BASE, 'png_exports/coarse_registration/contact_sheet')

# JY306 pixel sizes
XY_UM = 0.685   # µm per pixel XY
Z_UM = 3.0      # µm per z-step

# Discover landmark files
lm_files = sorted(glob.glob(os.path.join(BASE, 'registration_video/landmarks_nd2_native_*.npz')))
# Also check for legacy row2_1 file
legacy = os.path.join(BASE, 'registration_video/landmarks_27_nd2_native.npz')
if os.path.exists(legacy):
    lm_files.append(legacy)

print(f"Found {len(lm_files)} landmark files")

# Collect data per tile
tiles = []
for f in lm_files:
    bn = os.path.basename(f)
    if 'landmarks_27_nd2_native' in bn:
        tile = 'row2_1'
    else:
        tile = bn.replace('landmarks_nd2_native_', '').replace('.npz', '')

    d = np.load(f)
    pcd_iv = d['pcd_invivo_jy306']   # (N, 3) = (z, y, x) in JY306 pixels
    pcd_ev = d['pcd_exvivo_jy306']   # (N, 3) = (z, y, x) in JY306 pixels

    n = pcd_iv.shape[0]
    if n < 3:
        print(f"  Skipping {tile}: only {n} cells")
        continue

    # Convert to µm
    iv_um = pcd_iv.copy()
    iv_um[:, 0] *= Z_UM   # z
    iv_um[:, 1] *= XY_UM   # y
    iv_um[:, 2] *= XY_UM   # x

    ev_um = pcd_ev.copy()
    ev_um[:, 0] *= Z_UM
    ev_um[:, 1] *= XY_UM
    ev_um[:, 2] *= XY_UM

    # Pairwise distances (3D Euclidean)
    d_iv = pdist(iv_um)
    d_ev = pdist(ev_um)

    r = np.corrcoef(d_iv, d_ev)[0, 1]
    slope = np.polyfit(d_iv, d_ev, 1)[0]

    tiles.append({
        'tile': tile,
        'n': n,
        'r': r,
        'slope': slope,
        'd_iv': d_iv,
        'd_ev': d_ev,
        'dm_iv': squareform(d_iv),
        'dm_ev': squareform(d_ev),
    })
    print(f"  {tile}: n={n}, r={r:.6f}, slope={slope:.4f}")

# Sort tiles
tiles.sort(key=lambda x: x['tile'])

# Plot: per-tile panels with distance matrices + scatter
n_tiles = len(tiles)
fig, axes = plt.subplots(n_tiles, 3, figsize=(14, 4 * n_tiles))
if n_tiles == 1:
    axes = axes[np.newaxis, :]

fig.suptitle('Pairwise Distance Correlation: in-vivo (JY306) vs ex-vivo (JY306) — 3D µm',
             fontsize=14, fontweight='bold', y=0.998)

for i, t in enumerate(tiles):
    # In-vivo distance matrix
    ax0 = axes[i, 0]
    im0 = ax0.imshow(t['dm_iv'], cmap='viridis')
    ax0.set_title(f"In-vivo (JY306)", fontsize=9)
    ax0.set_xlabel('cell')
    ax0.set_ylabel('cell')
    plt.colorbar(im0, ax=ax0, shrink=0.7, label='µm')

    # Ex-vivo distance matrix
    ax1 = axes[i, 1]
    im1 = ax1.imshow(t['dm_ev'], cmap='viridis')
    ax1.set_title(f"Ex-vivo (JY306)", fontsize=9)
    ax1.set_xlabel('cell')
    plt.colorbar(im1, ax=ax1, shrink=0.7, label='µm')

    # Scatter
    ax2 = axes[i, 2]
    ax2.scatter(t['d_iv'], t['d_ev'], s=2, alpha=0.3, c='green')
    mx = max(t['d_iv'].max(), t['d_ev'].max()) * 1.05
    ax2.plot([0, mx], [0, mx], 'r-', lw=1, alpha=0.5, label='y=x')
    ax2.set_xlim(0, mx)
    ax2.set_ylim(0, mx)
    ax2.set_xlabel('d(in-vivo) µm')
    ax2.set_ylabel('d(ex-vivo) µm')
    ax2.set_aspect('equal')
    ax2.legend(fontsize=7)

    # Stats text
    stats = f"{t['tile']}\nn={t['n']} cells\nr={t['r']:.4f}\nslope={t['slope']:.4f}\n+/-{1-t['slope']:.2f}"
    ax2.text(0.98, 0.02, stats, transform=ax2.transAxes, fontsize=8,
             va='bottom', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'distance_matrix_correlation_3d.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved to {out_path}")
plt.close()
