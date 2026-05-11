"""
Build a stitched volume with MERSCOPE gene dots composited in.
Each tile's MERSCOPE dots are placed at the tile's center z-slice
in the stitched volume, at the correct XY position.

Saves to scene5b_assets_v3.pkl under '_stitched_with_merscope' key.
"""
import numpy as np, cv2, pickle, os

BASE = '/Users/neurolab/neuroinformatics/margaret'

TILES = ['row1_3',
         'row2_1', 'row2_2', 'row2_3', 'row2_4', 'row2_5',
         'row3_1', 'row3_2', 'row3_3', 'row3_4', 'row3_5', 'row3_6',
         'row4_1', 'row4_2', 'row4_3', 'row4_4', 'row4_5', 'row4_6',
         'row5_1']

print("Loading assets...")
with open(f'{BASE}/animation/scene5b_assets_v3.pkl', 'rb') as f:
    assets = pickle.load(f)

stitch = assets['_stitched']
volume = stitch['volume'].copy()  # (nz, h, w, 3) uint8 BGR
z_vals = stitch['z']
vol_w = stitch['width']
nz, vol_h, _, _ = volume.shape
print(f"Stitched volume: {volume.shape}, z=[{z_vals[0]:.0f},{z_vals[-1]:.0f}]")

# Build the stitched volume builder's tile placement info
# From scene5b_build_stitched.py: tiles are placed at canvas coords
# The stitched volume places each tile centered horizontally with padding
# and at its stitch_z_offset vertically in z

tile_list = [t for t in TILES if t in assets]

# Figure out the z-index mapping: stitch_z_offset → index in volume
# z_vals contains the actual z values for each slice
# stitch_z_offset is the starting z-index for each tile in the canvas

# For each tile, find the z-slice in the volume closest to the tile's center z
for tile in tile_list:
    a = assets[tile]
    ms = a.get('merscope')
    if ms is None:
        print(f"  {tile}: no MERSCOPE data, skipping")
        continue

    cell_w = a['cell_w']
    cell_h = a['cell_h']
    z_offset = a['stitch_z_offset']
    dense_z = a['dense_z']
    center_z = a['center_z']

    # Find center z-slice in the volume
    # stitch_z_offset is the starting z-index in the stitched volume
    # center slice is at offset + n_slices // 2
    n_slices = len(dense_z)
    center_z_idx = min(nz - 1, z_offset + n_slices // 2)

    # Resize MERSCOPE to match dense slice dimensions if needed
    if ms.shape[0] != cell_h or ms.shape[1] != cell_w:
        ms = cv2.resize(ms, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

    # Place horizontally centered (same as stitched volume builder)
    pad_l = (vol_w - cell_w) // 2

    # Composite: overlay gene dots where they have signal
    # Place on center z-slice and ±1 neighboring slices for visibility
    for dz in [-1, 0, 1]:
        zi = center_z_idx + dz
        if zi < 0 or zi >= nz:
            continue
        region = volume[zi, :cell_h, pad_l:pad_l + cell_w]
        ms_mask = np.max(ms, axis=2) > 10  # gene dot pixels
        alpha = 0.8 if dz == 0 else 0.4  # center slice stronger
        region[ms_mask] = np.clip(
            region[ms_mask].astype(np.float32) * (1 - alpha) + ms[ms_mask].astype(np.float32) * alpha,
            0, 255
        ).astype(np.uint8)

    print(f"  {tile}: MERSCOPE composited at z-slice {center_z_idx} (center_z={center_z:.0f})")

print(f"\nSaving stitched volume with MERSCOPE...")

# Save as separate pkl to avoid modifying the main assets file (212MB already)
out_path = f'{BASE}/animation/stitched_with_merscope.pkl'
with open(out_path, 'wb') as f:
    pickle.dump({
        'volume': volume,
        'z': z_vals,
        'width': vol_w,
        'height': vol_h,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)

fsize = os.path.getsize(out_path) / 1e6
print(f"Saved to {out_path} ({fsize:.0f} MB)")

# Also save a quick MIP preview
mip = np.max(volume, axis=0)
cv2.imwrite(f'{BASE}/animation/stitched_with_merscope_mip.png', mip)
print(f"MIP preview saved to stitched_with_merscope_mip.png")
