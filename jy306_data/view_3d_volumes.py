"""
View nd2 GFP + exvivo_combined in napari for 3D landmark marking.

nd2 GFP:          (12, 4200, 4200) — native ex-vivo confocal
exvivo_combined:   (16, 658, 629)  — registered cell labels in JY306 space

Toggle 3D view: cube button (bottom-left) or Ctrl+Y

Mark matching points:
  1. Select 'nd2_points' layer, click on features in nd2
  2. Select 'exvivo_points' layer, click on matching features in exvivo
  3. Keep points in matching order
  4. Press S to save, Q to quit without saving

Run with: python3 view_3d_volumes.py
"""

import numpy as np
import nd2
import tifffile
import napari

# ============================================================
# Load volumes
# ============================================================
print("Loading nd2 (GFP channel)...")
with nd2.ND2File('registration_video/1.nd2') as f:
    nd2_data = f.asarray()  # (12, 2, 4200, 4200)
nd2_gfp = nd2_data[:, 1].astype(np.float32)  # GFP channel
print(f"  nd2 GFP: {nd2_gfp.shape}, dtype={nd2_gfp.dtype}")

print("Loading exvivo_combined...")
ev = tifffile.imread('jy306_registered files/exvivo_combined.tif').astype(np.float32)
print(f"  exvivo_combined: {ev.shape}, dtype={ev.dtype}")

# ============================================================
# Launch napari
# ============================================================
viewer = napari.Viewer(title='3D Landmark Picker — nd2 vs exvivo_combined')

viewer.add_image(nd2_gfp, name='nd2 GFP (native exvivo)',
                 colormap='green', blending='additive',
                 contrast_limits=[np.percentile(nd2_gfp[nd2_gfp>0], 2),
                                  np.percentile(nd2_gfp[nd2_gfp>0], 99.5)])

viewer.add_image(ev, name='exvivo_combined (registered)',
                 colormap='magenta', blending='additive',
                 visible=False,
                 contrast_limits=[0, np.percentile(ev[ev>0], 99)])

# Points layers
nd2_pts = viewer.add_points(ndim=3, name='nd2_points',
                             face_color='green', edge_color='white',
                             size=30, symbol='cross')

ev_pts = viewer.add_points(ndim=3, name='exvivo_points',
                            face_color='magenta', edge_color='white',
                            size=8, symbol='cross')

# ============================================================
# Key bindings
# ============================================================
out_path = 'registration_video/landmarks_3d_manual.npz'

@viewer.bind_key('s')
def save(viewer):
    src = nd2_pts.data   # (N, 3) z, row, col
    tgt = ev_pts.data
    n = min(len(src), len(tgt))
    if n < 1:
        print("No points to save!")
        return
    src = src[:n]
    tgt = tgt[:n]
    np.savez(out_path,
             nd2_points=src,      # (N, 3) z, row, col in nd2 space (4200x4200)
             exvivo_points=tgt,   # (N, 3) z, row, col in exvivo space (658x629)
             nd2_shape=nd2_gfp.shape,
             exvivo_shape=ev.shape)
    print(f"\nSaved {n} pairs to {out_path}")
    print("nd2 points (z, row, col):")
    for i, p in enumerate(src):
        print(f"  {i+1}: z={p[0]:.1f}, row={p[1]:.1f}, col={p[2]:.1f}")
    print("exvivo points (z, row, col):")
    for i, p in enumerate(tgt):
        print(f"  {i+1}: z={p[0]:.1f}, row={p[1]:.1f}, col={p[2]:.1f}")
    viewer.close()

@viewer.bind_key('q')
def quit_nosave(viewer):
    print("Quit without saving")
    viewer.close()

print("""
=== 3D VOLUME VIEWER ===

Two volumes loaded:
  Green  = nd2 GFP (12 z, 4200x4200) — native ex-vivo confocal
  Magenta = exvivo_combined (16 z, 658x629) — registered cell labels

NOTE: These are at very different scales!
  Toggle layers with the eye icon to switch between them.
  Toggle 3D view: cube icon or Ctrl+Y

TO MARK POINTS:
  1. Select 'nd2_points' layer
  2. Press P (Add Points mode)
  3. Click on a feature in the nd2 volume
  4. Select 'exvivo_points' layer
  5. Toggle to exvivo_combined (eye icon)
  6. Click on the SAME feature
  7. Repeat — keep in matching order!

S = save    Q = quit without saving
""")

napari.run()
