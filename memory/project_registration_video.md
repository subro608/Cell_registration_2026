---
name: Registration Video Data and Pipeline
description: Key files, coordinate spaces, computed transforms (2D+3D), and all saved landmarks for nd2-to-exvivo registration
type: project
---

## registration_video/ folder

Collaborator-provided folder with all raw data for the deformation animation:

- `1.nd2` — (12, 2, 4200, 4200) uint16, channels: DAPI (ch0) + GFP (ch1). Native pre-registration spinning-disk confocal (Plan Apo 10x, 0.645 µm/px xy, 2.0 µm z-step, 24 µm total z-range). Only nd2 file — 12 slices is all we have.
- `exvivo21_merscope17_combined.tif` — (4719, 5640, 3) float32, native MERSCOPE Anti-Rat intensity.
- `exvivo21_merscope17_combined_seg.npy` — Cellpose segmentation, 44 cells, masks (4719, 5640)
- `JY306_in_Vivo_stack_s80.tif` — (63, 1798, 1720) uint16, full-res JY306 in-vivo (NOT aligned with nd2)
- `mosaic_Anti-Rat_z4.tif` — (18875, 22558) uint16, full MERSCOPE mosaic
- `mosaic_DAPI_z4.tif` — (18875, 22558) uint16, DAPI mosaic

## Saved transforms and landmarks (KEEP SAFE)

- `landmarks.npz` — 4 manually marked 2D landmark pairs (src=nd2 GFP 4200x4200, tgt=exvivo_combined 658x629), with z_idx
- `landmarks_3d.npz` — 6 manually marked 3D landmark pairs (col, row, z) between nd2 and exvivo_combined
- `affine_nd2_to_exvivo.npy` — 2D affine (2x3): nd2 GFP (4200x4200) → exvivo_combined (658x629). Scale ~0.38, rotation 8.33°, translation (-278, -541), mean reprojection error 2.19px from 4 landmarks.
- `affine_3d.npz` — 3D affine from 6 landmark pairs (lower quality, not used)
- `affine_3d_cells.npz` — 3D affine from 27 auto-detected cell centroids (exvivo→invivo in JY306 space). M_ev_to_iv (3x4), mean error 2.04px.
- `matched_cells_3d.npz` — 27 matched cell 3D centroids (iv_3d, ev_3d) found via intensity-weighted z-centroid in invivo and antirat_combined stacks.

## Coordinate spaces

- **JY306 space** = (16, 658, 629) — in-vivo two-photon coordinate system. exvivo_combined.tif and antirat_combined.tif are registered INTO this space.
- **JY316 space** = (19, 578, 599) — different mouse/registration target, NOT JY306
- **pkl canvas** = (16, 1704, 1704) — intermediate registration space in pkl
- **nd2 native** = (12, 4200, 4200) — native confocal resolution before registration
- pcd_invivo/pcd_exvivo landmarks (27 neurons) are in JY306 space (r range [81, 638])
- The 27 matched cells are between invivo and antirat_combined (both JY306 space), NOT confirmed to be identifiable in nd2 native

## jy306_registered files/ (4 files)

- `2_1_merscope17transformed_20250424104024.pkl` (5.57GB) — registration transforms + output. Contains: pcd_invivo, pcd_exvivo (27 matched neurons), transformed[0] (16, 1704, 1704), bhat matrices, scale factors, vector fields.
- `JY306_in_Vivo_stack_flipped_s80 (2).tif` — (16, 658, 629) JY306 in-vivo
- `antirat_combined.tif` — (16, 658, 629) uint64, Anti-Rat INTENSITY registered to JY306 space (IS intensity despite uint64, max=1091)
- `exvivo_combined.tif` — (16, 658, 629) uint64, ex-vivo cell LABELS (max=771, NOT intensity)

## PNG exports created

- `png_exports/registration_video/nd2_1/` — 12 z-slices of nd2 ch1 GFP (4200x4200)
- `png_exports/registration_video/nd2_full/` — All nd2 channels: DAPI_z*.png, GFP_z*.png, DAPI_MIP.png, GFP_MIP.png
- `png_exports/exvivo_combined_registered/` — 16 z-slices + MIP.png (658x629)
- `png_exports/nd2_transformed_to_exvivo/` — 12 warped z-slices + side_by_side/ + overlays/
- `png_exports/nd2_3d_aligned/` — 3D-aligned outputs, MIP overlay, per-z overlays, 3D confirmation plots
- `png_exports/overlays/` — Various overlay verification images (GFP_MIP_vs_invivo_MIP.png, etc.)

## 3D alignment attempts and findings

- nd2 is a "thin slab" — all 12 z-slices look nearly identical, NCC cross-correlation shows all best-match to invivo z02
- SimpleITK 3D rigid registration failed (z-translation pushed data out of range)
- Auto z-centroid approach with 27 cells gave mean 2.04px error but prof wants MANUAL 3D point marking
- Prof's request: manually locate matched cells in 3D in both volumes, find centroids across z-slices, align 3D coords, transform stacks, show angular 3D view for confirmation

**Why:** Building deformation animation for Nature Science pub.

**How to apply:** The 2D affine (affine_nd2_to_exvivo.npy) is the most reliable transform so far. For 3D alignment, prof wants manual cell localization in both nd2 and exvivo_combined volumes using napari (view_3d_volumes.py). The two target volumes for alignment are nd2 (1.nd2) and exvivo_combined.tif — NOT antirat.
