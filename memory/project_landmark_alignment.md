---
name: Landmark-based Cell Matching
description: Interactive landmark picker for row2_1↔JY306 cell matching, overlay visualization, and Erdem's full registration pipeline
type: project
---

## Current State (2026-03-28)
4 cells matched between row2_1 (ex-vivo) and JY306 (in-vivo) in native space. Overlay with 27 pkl cells verified general alignment. Next: identify more cells in native space.

**Why:** For Nature Science publication — need precise cell-level 3D correspondences in native space to build the final stitched↔in-vivo transform.

## Erdem's Full Pipeline (2026-03-28)
1. **Find matched cells in native space** — Use the 4-point affine to predict where pkl cells are in native nd2 and native JY306. Visually verify each, click precise (x,y,z) in both native images. Goal: go from 4 to 27+ verified native-space correspondences.
2. **Transfer ex-vivo native → stitched volume** — Map verified nd2 cell positions through the stitching pipeline (tile positioning → IOU rigid → elastix B-spline → resample to 1µm isotropic) to get their coords in stitched space (2748×2748×516).
3. **Now have two 3D coordinate sets:**
   - In-vivo cells in native JY306 space (658×629×16)
   - Ex-vivo cells in stitched 1µm isotropic space (2748×2748×516)
4. **Compute 3D affine** between stitched ex-vivo and JY306 in-vivo using matched cells.
5. **Final elastix pass** — Use affine as initialization, run elastix B-spline deformable registration between stitched volume and in-vivo volume.

**Key insight:** Erdem wants native-space coordinates, NOT JY306-registered pkl positions. The pkl positions went through a registration pipeline that may have errors. Ground-truth = clicking on cells in the raw images.

## Landmark Picker Script
`landmark_picker_row21_jy306.py` — Interactive cv2 GUI for cell matching.
- **Run**: `/usr/bin/python3 landmark_picker_row21_jy306.py`
- Left panel: ex-vivo tiles (all rows, switch with **1/2**)
- Right panel: JY306 in-vivo or exvivo_combined (switch with **9/0**)
- **A/D**: left z-nav, **J/L**: right z-nav
- **C/V**: left contrast, **N/M**: right contrast
- Click left then right to mark a pair. **Z**: undo, **S**: save, **Q**: quit
- Currently starts on row3_2 (change `left_idx` line to switch default tile)
- Saves to `registration_video/landmarks_row21_jy306.npz`

## Saved Landmark Data
`registration_video/landmarks_row21_jy306.npz` — 4 matched cell pairs:

| Cell | row2_1 nd2 (col, row, z) | JY306 in-vivo (col, row, z) |
|------|--------------------------|----------------------------|
| 1 | (1040.7, 2184.0, z=0) | (117.0, 375.1, z=2) |
| 2 | (1292.7, 2431.3, z=0) | (228.8, 488.4, z=2) |
| 3 | (1059.3, 1824.7, z=0) | (133.8, 239.8, z=2) |
| 4 | (1106.0, 1843.3, z=0) | (148.4, 245.7, z=2) |

- src: nd2 native 4200×4200, tgt: JY306 658×629, z_idx = z-slice index

## Overlay Scripts & Images
- `overlay_jy306_to_row21.py` — Standalone script. Warps JY306 MIP to nd2 space via inverted affine. `JY306_BOOST=8.0` for visibility. Green=row2_1, Magenta=JY306.
  - Output: `png_exports/coarse_registration/jy306mip_warped_to_row21_side.png`, `*_overlay.png`
- RBF warp (31 points = 4 clicked + 27 pkl): `png_exports/coarse_registration/jy306mip_rbf31_to_row21_overlay.png`
- Affine vs RBF comparison: `png_exports/coarse_registration/jy306_affine_vs_rbf_overlay.png`
- Z-slice overlays (original contrast, no boost): `landmarks_jy306z2_vs_row21z0_native.png`, `*_transformed.png`

## Key Transforms
- `registration_video/affine_nd2_to_exvivo.npy` — 2×3, maps (col, row, 1) nd2→JY306
- Inverse: `M_jy_to_nd2 = inv(vstack(M, [0,0,1]))[:2,:]` — JY306→nd2
- Verified: affine-predicted vs clicked positions match within ~5-10px

## Related Data (pkl / existing matches)
- `matched_cells_3d.npz` — 27 cells: `pcd_invivo` (27×2), `pcd_exvivo` (27×2) both in JY306 space, `M_ev_to_iv` (3×4)
- `matched_landmarks.npz` — 13 pairs (iv/ev in JY306 2D space)
- These are in JY306 REGISTERED space, not native — Erdem wants native-space positions instead
- Use the affine to predict where these 27 cells are in nd2, then visually verify and click precise positions

## Next Steps
1. Use affine to predict nd2 positions of 27 pkl cells → show in picker as guides
2. Click precise native-space positions for each verifiable cell
3. Expand to other tiles (row1, row3, etc.)
4. Build coordinate transfer: nd2 native → stitched 1µm isotropic (via tile_positions + IOU rigid + elastix transforms)
5. Compute 3D affine: stitched ex-vivo ↔ JY306 in-vivo
6. Final elastix deformable pass
