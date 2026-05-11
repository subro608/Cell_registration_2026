"""
Build a single stitched 3D volume from per-tile assets.
Each tile's dense slices are center-pasted onto a common canvas width,
stacked in z-order. MERSCOPE gene dots are composited onto each slice.
Saves 'stitched_volume' and 'stitched_z' into scene5b_assets_v3.pkl.
"""
import numpy as np, pickle, os

BASE = '/Users/neurolab/neuroinformatics/margaret'

with open(f'{BASE}/animation/scene5b_assets_v3.pkl', 'rb') as f:
    assets = pickle.load(f)

# Sort tiles by z-offset
tiles = sorted([t for t in assets.keys() if not t.startswith('_')],
               key=lambda t: assets[t]['stitch_z_offset'])
max_w = max(assets[t]['cell_w'] for t in tiles)
cell_h = 400

print(f"Building stitched volume from {len(tiles)} tiles")
print(f"Canvas: {max_w} x {cell_h}, tiles sorted by z-offset")

all_slices = []
all_z = []

for t in tiles:
    a = assets[t]
    dense = a['dense']  # (n_dense, h, w, 3)
    dense_z = a['dense_z']
    ms = a.get('merscope')  # (h, w, 3) or None
    cw = a['cell_w']
    z_off = a['stitch_z_offset']
    z_spacing = a['tile_z_spacing']

    # Pad to common width (center)
    pad_l = (max_w - cw) // 2
    pad_r = max_w - cw - pad_l

    for i in range(len(dense)):
        sl = dense[i].copy()
        # No MERSCOPE dots — stitched volume is pure in-vivo + ex-vivo overlay

        # Pad width
        if pad_l > 0 or pad_r > 0:
            sl = np.pad(sl, ((0, 0), (pad_l, pad_r), (0, 0)), mode='constant')

        all_slices.append(sl)
        # Global z position: tile z-offset (in native slices) + within-tile z
        global_z = z_off + dense_z[i] / z_spacing  # convert dense_z back to native slices
        all_z.append(global_z)

    print(f"  {t}: {len(dense)} slices, z=[{z_off}, {z_off + a['n_slices']-1}], padded to {max_w}")

stitched = np.array(all_slices, dtype=np.uint8)
stitched_z = np.array(all_z, dtype=np.float64)

print(f"\nStitched volume: {stitched.shape} ({stitched.nbytes / 1e6:.0f} MB)")
print(f"Z range: {stitched_z[0]:.1f} to {stitched_z[-1]:.1f}")

# Save into assets
assets['_stitched'] = {
    'volume': stitched,
    'z': stitched_z,
    'width': max_w,
    'height': cell_h,
}

with open(f'{BASE}/animation/scene5b_assets_v3.pkl', 'wb') as f:
    pickle.dump(assets, f)

sz = os.path.getsize(f'{BASE}/animation/scene5b_assets_v3.pkl') / 1e6
print(f"Saved ({sz:.0f} MB)")
