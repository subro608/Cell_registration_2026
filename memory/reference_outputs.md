---
name: Generated Output Locations
description: Locations of all generated PNG exports, QC images, comparison collages, masks, and registration results
type: reference
---

## Masks
- `registration_video/via_masks_v4.npz` — **CURRENT**: from CSV polylines via `cv2.fillPoly`. All 21 tiles.
- `registration_video/via_masks_v3.npz` — Superseded (PIL polygon fill)
- `registration_video/via_masks_v2.npz` — Superseded (used as base for v4)

## Transform Files
- `registration_video/auto_align_transforms_iou_v4.json` — **CURRENT** IOU rigid transforms (rows 2-4 rerun with v4 masks, row1+row5 kept from old)
- `registration_video/auto_align_transforms_iou.json` — older IOU rigid transforms (all 20 pairs, v2/v3 masks)
- `registration_video/auto_align_transforms.json` — ECC v1 transforms
- `registration_video/deformable_results_v3.json` — Farneback v3 stats

## V5 Pipeline Outputs (CURRENT — all 20 pairs, v4 masks)
- `png_exports/z_stitch_comparison_v5/` — **CURRENT** 5-column collages per pair + ALL_PAIRS_contact_sheet.png
  - Col 1: Raw + annotation contours (centroid shift, no masking)
  - Col 2: IOU Rigid
  - Col 3: Farneback v3
  - Col 4: Elastix 500 iter
  - Col 5: Elastix 2000 iter
  - NCC shown per cell, MAGENTA_BOOST=2.0 for visibility
  - `results_v5.json` — all NCC values per pair per method
- Script: `comparison_collage_v5.py`

## V4 Pipeline Outputs (v4 masks, rows 2-4 reprocessed + row1/row5 copied from old)
- `png_exports/z_stitch_qc_iou_v4/` — IOU rigid QC overlays per pair (green=ref, magenta=mov)
- `png_exports/z_stitch_qc_iou/` — IOU rigid QC (also used for v4 run output)
- `png_exports/z_stitch_elastix_v4/` — Elastix B-spline (DS=2, 2000 iter, v4 masks). All 20 pairs (row1+row5 copied from old elastix)
- `png_exports/z_stitch_elastix_v4_500iter/` — Elastix B-spline (DS=2, 500 iter, v4 masks). 17/20 pairs (rows 2-4 only)
- `png_exports/z_stitch_farneback_v4/` — Farneback v3 all 20 pairs, v4 masks. `farneback_results_v4.json`
- `png_exports/z_stitch_comparison_v4/` — 3-column: IOU Rigid | Farneback v3 | Elastix 2000iter

## Elastix Configs Summary
| Version | DS | Grid | Res | Iter | Masks | Pairs | Avg NCC |
|---|---|---|---|---|---|---|---|
| `z_stitch_elastix/` (original) | 2 | 32px | 4 | 500 | v2 | 20 | ~0.46 |
| `z_stitch_elastix_highiter/` | 2 | 32px | 4 | 2000 | v3 | 20 | ~0.52 |
| `z_stitch_elastix_v4/` | 2 | 32px | 4 | 2000 | v4 | 20 | ~0.51 |
| `z_stitch_elastix_v4_500iter/` | 2 | 32px | 4 | 500 | v4 | 17 | ~0.49 |
- All use: AdvancedNormalizedCorrelation, 4096 spatial samples, NewSamplesEveryIteration=true, BSplineInterpolationOrder=3

## Annotation QC
- `via_annotations/annotated_qc_new/{key}/GFP_*_annotated.png` — **CURRENT**: CSV annotations on all GFP slices (273 images). Green outline=ref mask, magenta=mov mask.
- `via_annotations/annotated_qc/` — older MIP-only
- `via_annotations/annotated_qc_all/` — older per-tile z-slices (unannotated)

## VIA QC Overlays
- `png_exports/z_stitch_qc_via_v4/` — **CURRENT**: 3-panel centroid-aligned overlays with v4 mask contours (all 20 pairs + contact sheet)
- `png_exports/z_stitch_qc_via/` — older version (v3 masks)

## V3 Pipeline Outputs
- `png_exports/z_stitch_comparison_v3/` — IOU Rigid | Farneback v3 | Elastix 2000iter (8 row groups)
- `png_exports/z_stitch_elastix_fullres_4000iter/` — DS=1, 4000 iter (cancelled at 13/20 pairs)

## V2 Pipeline Outputs
- `png_exports/z_stitch_comparison_v2/` — IOU Rigid | Farneback v3 | Elastix
- `png_exports/z_stitch_comparison/` — 6-method collages (original)
- `png_exports/z_stitch_qc_deformable_v3/` — Farneback v3 QC

## Stitched 3D Volumes (in `registration_video/stitched/`)
- `stitched_gfp_fullres_v5.tif` — **LATEST**: (258, 5510, 5496) native res, IOU rigid + full-res elastix (v5), 15.6 GB
- `stitched_gfp_fullres_v5_1um_isotropic.tif` — **LATEST**: (516, 3554, 3545) 1µm isotropic. NOTE: saved as per-page tif, load with TiffFile page-by-page loop, not imread
- `stitch_v5_params.json` — cumulative IOU transforms, tile z-offsets, canvas offsets for landmark propagation
- `stitched_gfp_elastix_1um_isotropic.tif` — OLD: (516, 2748, 2748) DS=2 elastix
- `stitched_gfp_elastix.tif` — OLD: DS=2 native (258, 2100, 2100) before resampling
- `stitched_gfp_masked_1um_isotropic.tif` — rigid-only IOU version, masked
- `stitched_gfp_rigid.tif` — rigid-only, unmasked (includes background)
- `invivo_1um_isotropic.tif` — (189, 1229, 1177) 1µm isotropic in-vivo, resampled from (63, 1798, 1720) @ 0.6835×0.6845×3µm
- Stitching chain: IOU rigid warp (full res) → DS=2 → elastix B-spline (TransformParameters from `z_stitch_elastix_v4_500iter/`) → stack in z → resample
- Row1_4 = 6-slice black slab (12µm gap)
- **Requires `/usr/bin/python3`** for SimpleITK (not venv python)

## 3D HTML Viewers
- `3d_viewer/viewer.html` — **ORIGINAL**: Three.js point cloud (8.7MB). 4x subsampled (129×687×687), /4000 normalization, base64 PNGs
- `3d_viewer/viewer_copy.html` — Exact recreation of viewer.html via `build_viewer_copy.py`
- `3d_viewer/viewer_equalized.html` — per-slice equalized + proper GP regression with RBF kernel. Via `build_viewer_equalized.py`
- `3d_viewer/viewer_rbf.html` — Clone of viewer.html + naive RBF spreading (not true GP) via `build_rbf_viewer.py`
- `3d_viewer/viewer_feathered.html` — Feathered masks + intensity equalization version (252×687×687)
- `3d_viewer/viewer_gapfree_fullz.html` — Full 504 z-slices, 8x XY DS (504×343×343) via `build_gapfree_viewer.py`
- `3d_viewer/viewer_stitched_v5.html` — Stitched v5 (1µm iso) volume. Per-slice equalized + GP RBF. 4x DS (129×888×886). Via `build_viewer_stitched_v5.py`
- `3d_viewer/viewer_dual_3d.html` — **DEPLOYED**: Dual volume viewer (stitched ex-vivo + JY306 s80 in-vivo). 878 matched cell lines. Via `build_viewer_dual_3d.py`. **Live at: https://invivo-exvivo-cell-registration.vercel.app**
- `3d_viewer/viewer_warped_invivo_3d_v4.html` — **LATEST**: Warped in-vivo overlaid on stitched ex-vivo. PKL direct registration. 878 landmarks (all 19 tiles incl row3_1/3_5). Toggle: Filtered(401)/All(878)/None. Cyan=filtered, Orange=unfiltered. ~84MB. Via `build_viewer_warped_invivo_3d_v4.py`
- `3d_viewer/viewer_warped_invivo_3d_v3.html` — TPS 3D viewer, 320 blob-filtered landmarks only. Via `build_viewer_warped_invivo_3d_v3.py` (~36MB)
- `3d_viewer/viewer_warped_invivo_3d_v3_elastix.html` — Elastix 3D viewer, 378 landmarks. Via `build_viewer_warped_invivo_3d_v3_elastix.py` (~41MB)
- `3d_viewer/patch_strip_v4.png` + `cell_info_v4.json` — Cached patch data for v4 viewer (speeds up HTML rebuild)
- Features: drag rotate, scroll zoom, shift+drag pan, threshold/opacity/colormap sliders, auto-rotate, additive blending

## PKL Direct Registration (in `png_exports/registration_per_tile_pkl/`)
- `<tile>/pkl_transform_<tile>.npz` — Per-tile transforms: `M2d_jy306_to_nd2` (2x3), `ev_nd2`, `iv_nd2`, `pcd_invivo_jy306`, `nd2_z_gauss`, `pkl_dist_um`. All 19 tiles.
- `3d_viewer/landmark_patches_pkl_direct.html` — **LATEST**: 5-column patch viewer (63.7MB). 878 lm, 401 pass blob filter. Filtered/All toggle. Via `build_patches_pkl_direct_all.py`
- `3d_viewer/landmark_patches_pkl_row31_35.html` — Row3_1/3_5 only version (6.6MB). Via `build_patches_pkl_direct_row31_35.py`

## Contact Sheets (in `png_exports/coarse_registration/contact_sheet/`)
- `single_z/{tile}_contact_sheet.png` — zoomed patches use exact z-slice per cell (19 tiles)
- `mip_pm2/{tile}_contact_sheet.png` — zoomed patches use MIP ±2 z-slices (19 tiles)
- Both: overview image at top uses full MIP. Green crosshairs mark matched cells.
- `distance_matrix_correlation.png` — 2D XY pairwise distance scatter (pixel units)
- `distance_matrix_correlation_um.png` — 2D XY pairwise distance scatter (µm units)
- `distance_matrix_correlation_3d.png` — 3D (XYZ) pairwise distance in JY306 µm space
- `stitched_v5/{tile}_contact_sheet.png` — stitched ex-vivo (1µm iso, fullres elastix) vs JY306 in-vivo (19 tiles)
- Landmarks in stitched space: `registration_video/landmarks_stitched_v5_{tile}.npz`
- Shared with Cliodna and Jason for verification (2026-03-29)

## MERSCOPE → Confocal Transformation PKLs (in `transformation/`)
- `3_2_merscope15transformed_alt_*.pkl` — row3_2 ↔ MERSCOPE FOV 15 (8.6GB)
- `3_3_merscope14transformed_*.pkl` — row3_3 ↔ MERSCOPE FOV 14 (6.1GB)
- `3_4_merscope13transformed_*.pkl` — row3_4 ↔ MERSCOPE FOV 13 (5.7GB)
- `3_5_merscope12transformed_*.pkl` — row3_5 ↔ MERSCOPE FOV 12 (5.6GB)
- `3_6_merscope11transformed_*.pkl` — row3_6 ↔ MERSCOPE FOV 11 (7.3GB)
- Each contains: `pcd_invivo` (38-41×3), `pcd_exvivo` (38-41×3), `transformations` (13-step chain: scale→bhat→RBF warp), `transformed` (3ch, 17z, 1734×1734 float64)

## 3D MIP Views (in `png_exports/stitched_3d_views/`)
- `MIP_XY_elastix.png`, `MIP_XZ_elastix.png` — elastix stitched volume
- `MIP_XY_masked.png`, `MIP_XZ_masked.png` — rigid masked version
- `MIP_XY_topdown.png`, `MIP_XZ_side.png`, `MIP_YZ_front.png` — unmasked rigid
- `three_view_panel.png` — combined XY+XZ+YZ panel
- `slice_z*.png` — sample XY slices at various depths

## Key Notes
- Elastix always beats Farneback on NCC for most pairs
- Farneback best params: winsize=76, blur_k=13, fsk=65, DS=2, levels=5, iter=15
- Elastix best config: DS=2, grid=32px, 4 res, 2000 iter, AdvancedNormalizedCorrelation
- 2000 iter meaningfully better than 500 iter (~0.51 vs ~0.49 avg NCC)
- All v4+ scripts load PNGs (GFP_z000.png / GFP_z011.png), NOT nd2 — avoids OOM
- MAGENTA_BOOST=2.0 used in v5 collage for better visual contrast
