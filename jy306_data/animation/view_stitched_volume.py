"""Quick napari viewer for the stitched 3D volume in scene5b_assets_v3.pkl."""
import numpy as np, pickle

BASE = '/Users/neurolab/neuroinformatics/margaret'
with open(f'{BASE}/animation/scene5b_assets_v3.pkl', 'rb') as f:
    assets = pickle.load(f)

vol = assets['_stitched']['volume']   # (418, 400, 452, 3) uint8
z = assets['_stitched']['z']
print(f"Volume: {vol.shape}, z range: {z[0]:.1f} to {z[-1]:.1f}")

import napari
viewer = napari.Viewer(title='Stitched 3D Volume')
viewer.add_image(vol, rgb=True, name='stitched', scale=[2, 1, 1])  # stretch z for visibility
napari.run()
