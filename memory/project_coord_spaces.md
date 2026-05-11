---
name: Coordinate Spaces — Full Map
description: All 4 coordinate spaces, what lives where, pixel sizes, and transform chains between them
type: project
---

## Space 1 — Native Ex-Vivo Confocal (nd2 tiles)
- Per-tile: (12z, 4200, 4200) at **0.645 µm/px XY**, 2 µm z-step
- 22 tiles (row1–row5), each imaging same hippocampal region at different depths
- Files: `png_exports/registration_video/{row}/{tile}/GFP_z*.png`
- VIA masks (`via_masks_v4.npz`) are in this space

## Space 2 — JY306 Registered Space (MASTER REFERENCE)
- Shape: **(16, 658, 629)** — this is the master coordinate system
- z: 0–15, y: 0–657, x: 0–628, ~0.685 µm/px XY, 3 µm z-step
- Files:
  - `JY306_in_Vivo_stack_flipped_s80.tif` — in-vivo confocal
  - `jy306_registered/exvivo_combined.tif` — GFP intensity
  - `jy306_registered/antirat_combined.tif` — Anti-Rat staining (uint64 but IS intensity, max=1091)
  - `jy306_registered/exvivo_total.tif` — cell LABELS (uint64, NOT intensity!)
- **ALL point clouds in pkl files are in this space**: pcd_invivo AND pcd_exvivo (both post-transform)
- `transformation/*.pkl` `transformed` field is (3, 17, 1734, 1734) — UPSAMPLED JY306 canvas (scale ≈ 2.64×)
- landmark tgt_points (JY306 side) are in this space
- landmark src_points (nd2 side) are in nd2 native 4200px space (NOT JY306)

## Space 3 — MERSCOPE Space
- Per-FOV: ~1627×1627 canvas, 3 z-slices
- 26 FOVs covering hippocampus (FOV 4–26)
- Files in this space:
  - `exvivo_merscope_combined/*.tif` — ex-vivo confocal GFP warped INTO MERSCOPE space, shape (3z, 1627, 1627, 3ch). ch0 = main GFP, ch1/ch2 near-zero. 23 tile pairs covering all rows (1-5).
  - Full MERSCOPE mosaic: `mosaic_Anti-Rat_z4.tif` at 18875×22558 (all FOVs stitched)
- FOV numbering inverted vs tile numbering: tile 3_2→FOV15, 3_3→FOV14, ..., 3_6→FOV11

## Space 4 — Stitched 1µm Isotropic
- `stitched_gfp_elastix_1um_isotropic.tif` — **(516, 2748, 2748)** at 1 µm isotropic
- `stitched_gfp_elastix_1um_isotropic_feathered.tif` — **(504, 2748, 2748)** with soft-edge masks
- `invivo_1um_isotropic.tif` — **(189, 1229, 1177)** — in-vivo resampled to 1 µm
- Built from all 22 nd2 tiles stitched together (IOU rigid → elastix B-spline → z-stack → resample)
- Used for 3D viewers (4x or 8x downsampled) and publication figures

---

## Transform Chains

### JY306 → nd2 native (row2_1)
```
p_nd2 = M_inv @ [y_jy, x_jy, 1]
M_inv = cv2.invertAffineTransform(affine_nd2_to_exvivo.npy)
```
Note: affine_nd2_to_exvivo.npy is in (y,x) convention, use directly with cv2.warpAffine

### nd2 native → stitched 1µm isotropic
```
pos_global = tile_positions[row2_1] + nd2_pixel * 0.645  # µm
pos_stitched = pos_global / 1.0  # already in µm
```
tile_positions.npz has global_pos_dict (pixel offsets in large mosaic canvas)

### Forward pkl (ex-vivo tile → JY306)
Apply stages 0→13 in order (backward image warping):
- scale: sample at coords/scale
- bhat: sample at (p - t) @ inv(R)
- vec_field: sample at p - vf[p]

### JY306 → MERSCOPE native (pkl inverse, CORRECT method)
Feed raw JY306 coords (z, y, x) — NO canvas scaling — into pkl backward mapping in reverse stage order:
- scale: `pt = pt / val`
- bhat: `pt = (pt - t) @ R_inv` where R=bhat[:3,:], t=bhat[3,:], (z,y,x) order
- vec_field_total: `pt = pt - vf[z,y,x,:]` where channels are (dz, dy, dx)
Accuracy: ~5px in MERSCOPE space

### MERSCOPE native → nd2 native (USE SIFT, not ECC)
```
M_sift = np.load('registration_video/affine_nd2_to_merscope_sift_{tile}.npy')  # nd2→MERSCOPE
M3_inv = np.linalg.inv(np.vstack([M_sift, [0,0,1]]))[:2,:]  # MERSCOPE→nd2
p_nd2 = M3_inv @ [x_merc, y_merc, 1]
```
SIFT affine, image-based (no landmarks needed). Scale ≈ 0.3873, zero rotation.
Per-tile: each tile needs its own SIFT affine with its `exvivo_merscope_combined/*.tif`.

### nd2 native → stitched 1µm isotropic (propagate_point_to_stitched)
1. IOU rigid warp (pair-wise tile alignment)
2. Elastix inverse displacement field
3. Cumulative IOU to canvas coords (5510×5496 native pixels)
4. ×0.645 to convert to 1µm isotropic → (3554×3545)

### Full chain: JY306 → stitched 1µm isotropic (ex-vivo landmarks)
```
pcd_exvivo (JY306 s80, 658×629) → pkl inverse → MERSCOPE (1627)
  → SIFT inv affine → nd2 native (4200×4200)
  → IOU rigid → elastix → canvas (5510×5496)
  → ×0.645 → stitched 1µm iso (3554×3545)
```
Total accuracy: ~13px in nd2 space (~8µm). Bottleneck is pkl inverse (~5px MERSCOPE).

### In-vivo landmarks
- `pcd_invivo` (JY306 s80) used directly — landmarks are in s80 pixel space, same as JY306_in_Vivo_stack_flipped_s80.tif

### PKL point clouds
- `pcd_invivo` and `pcd_exvivo` are BOTH in JY306 s80 pixel space (16, 658, 629) @ 0.6835µm XY, 3µm Z
- Differences between them are ~1-2px (registration residual)
- 19 pkl files in `transformation/` directory

### Landmark files
- `registration_video/landmarks_nd2_native_{tile}.npz` — keys: `ev_nd2`, `pcd_invivo_jy306`, `pcd_exvivo_jy306`
- `registration_video/landmarks_stitched_v5_{tile}.npz` — keys: `stitched_coords`, `pcd_invivo_jy306`, `ev_nd2`, `cell_nd2_z`
