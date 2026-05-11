---
name: Key Scripts Created
description: Scripts built for registration, landmark picking, animation, mask generation, z-stitch alignment, QC, and 3D visualization
type: reference
---

## Landmark & Visualization
- `landmark_picker.py` — cv2 interactive tool for 2D landmark marking between nd2 GFP z-slices and exvivo_combined. Side-by-side, A/D J/L z-nav, alternating click, saves landmarks.npz.
- `landmark_picker_3d.py` — cv2 interactive 3D landmark marking. Saves landmarks_3d.npz.
- `landmark_picker_row21_jy306.py` — **CURRENT**: cv2 interactive cell matcher, ex-vivo tiles ↔ JY306 in-vivo. Switchable left tile (1/2), switchable right set (9/0: invivo/exvivo_combined). Independent z-nav (A/D, J/L) and contrast (C/V, N/M). Lazy-loads tiles. Saves `registration_video/landmarks_row21_jy306.npz`.
- `overlay_jy306_to_row21.py` — Warps JY306 MIP to row2_1 nd2 space via inverted affine. JY306_BOOST param (default 8.0). Green=row2_1, Magenta=JY306. Output: `png_exports/coarse_registration/jy306mip_warped_to_row21_*.png`
- `view_3d_volumes.py` — napari viewer: nd2 GFP + exvivo_combined.tif as 3D volumes. S to save, Q to quit.

## 3D Viewer Build Scripts
- `build_viewer_copy.py` — Recreates viewer.html exactly (4x subsample, /4000 norm). Output: `3d_viewer/viewer_copy.html`
- `build_viewer_equalized.py` — per-slice equalized + proper GP regression (RBF kernel, Cholesky solve). Output: `3d_viewer/viewer_equalized.html`
- `build_rbf_viewer.py` — viewer.html + naive RBF point spreading. Output: `3d_viewer/viewer_rbf.html`
- `build_gapfree_viewer.py` — Full 504 z-slices, 8x XY DS. Output: `3d_viewer/viewer_gapfree_fullz.html`
- `build_viewer_stitched_v5.py` — Stitched v5 (1µm iso) volume. Output: `3d_viewer/viewer_stitched_v5.html`
- `build_viewer_dual_3d.py` — Dual 3D viewer: stitched ex-vivo v5 + JY306 s80 in-vivo, 878 cell lines. Output: `3d_viewer/viewer_dual_3d.html` (~25MB)
- `build_viewer_warped_invivo_3d_v4.py` — **LATEST**: Warped in-vivo overlaid on stitched ex-vivo. PKL direct registration (replaces 3D affine+TPS). All 19 tiles (incl row3_1/3_5). 878 landmarks with Filtered(401)/All/None toggle. Loads `pkl_transform_{tile}.npz` per tile. Patch strip cached. Output: `3d_viewer/viewer_warped_invivo_3d_v4.html` (~84MB)
- `build_viewer_warped_invivo_3d_v3.py` — TPS 3D viewer, 320 blob-filtered landmarks. Output: `3d_viewer/viewer_warped_invivo_3d_v3.html` (~36MB)
- `build_viewer_warped_invivo_3d_v3_elastix.py` — Elastix 3D viewer, 378 landmarks. Output: `3d_viewer/viewer_warped_invivo_3d_v3_elastix.html` (~41MB)

## Animation
- `invivo_to_exvivo_animation.py` — In-vivo → ex-vivo warp animation with 27 matched neurons. Pink=in-vivo, Green=ex-vivo. Outputs invivo_to_exvivo_transform.mp4.
- `native_to_registered_animation.py` — nd2 native → affine warp → exvivo_combined animation.

## Mask Generation
- `generate_masks_from_csv.py` — **CORRECT method**: reads `via_annotations/via_export_csv.csv`, fills polylines with `cv2.fillPoly`, saves `registration_video/via_masks_v4.npz`. Loads v2 as base.
- `generate_masks_from_images.py` — **DO NOT USE**: auto-Otsu threshold from GFP MIPs. User rejected — masks were wrong.
- `draw_annotations_qc.py` — Draws CSV polylines on all GFP images (MIP + z000–z011). Output: `via_annotations/annotated_qc_new/{key}/`

## QC Overlays
- `make_qc_via_v4.py` — 3-panel centroid-aligned overlays with v4 mask contours for all 20 pairs. Output: `png_exports/z_stitch_qc_via_v4/`

## Contact Sheets & Cell Matching (JY306 ↔ nd2 native)
- `make_contact_sheet.py` — **CURRENT**: Full pipeline for all 19 pkl tiles. SIFT affine (nd2↔MERSCOPE) + iterative pkl inverse (JY306→MERSCOPE) + contact sheets with green crosshairs. Generates 2 sets: single z-slice and MIP±2 z-slices. Big overview image always uses full MIP. Output: `contact_sheet/single_z/` and `contact_sheet/mip_pm2/` + `registration_video/landmarks_nd2_native_{tile}.npz`
- `distance_correlation_3d.py` — Pairwise distance correlation: in-vivo vs ex-vivo in JY306 space (3D µm). Output: `contact_sheet/distance_matrix_correlation_3d.png`
- `inverse_pkl_transform.py` — pkl inverse round-trip test + JY306 image warp to ex-vivo space. Output: `png_exports/pkl_transform_test/inverse_transform/`

## PKL Direct Registration (all 19 tiles)
- `build_patches_pkl_direct_all.py` — **CURRENT**: PKL direct pipeline for all 19 tiles. Fits 2D affine per tile from pkl-derived correspondences (JY306→nd2), warps in-vivo slices, 5-column patch viewer with Filtered/All toggle. Saves transforms to `png_exports/registration_per_tile_pkl/<tile>/pkl_transform_<tile>.npz`. Output: `3d_viewer/landmark_patches_pkl_direct.html` (63.7MB)
- `build_patches_pkl_direct_row31_35.py` — Earlier version for row3_1 & row3_5 only. Output: `3d_viewer/landmark_patches_pkl_row31_35.html`

## PKL Transform Pipeline (MERSCOPE → JY306)
- `test_pkl_transform.py` — inspect pkl contents, save MIP per channel + mid-z overlay with JY306 + point correspondence scatter. Output: `png_exports/pkl_transform_test/`
- `pkl_qc_panel.py` — 6-panel QC figure: raw in-vivo | raw ex-vivo MIP | in-vivo+landmarks | transformed ex-vivo+landmarks | overlay+correspondences. Output: `png_exports/pkl_transform_test/qc_panel.png`
- `apply_pkl_transform.py` — re-applies all 14 pkl transform stages (scale/bhat affine/vec_field deformable) to raw GFP PNGs using scipy.ndimage backward warping. Compares with stored pkl output. Output: `png_exports/pkl_transform_test/applied_transform/`

## Full-Res Elastix Stitching (v5)
- `elastix_fullres_v5.py` — Full-res (4200×4200) elastix B-spline for all 20 pairs. Grid=64, 1000 iter, ~7s/pair. Uses `/usr/bin/python3`. Output: `png_exports/z_stitch_elastix_fullres_v5/`
- `build_stitched_fullres_v5.py` — Builds stitched volume: IOU rigid + full-res elastix per tile, stacks in z, resamples to 1µm iso. Uses `/usr/bin/python3`. Output: `registration_video/stitched/stitched_gfp_fullres_v5*.tif` + `stitch_v5_params.json`
- `contact_sheet_stitched_v5.py` — Contact sheets: stitched ex-vivo (1µm iso) vs JY306 in-vivo. Propagates landmarks through IOU+elastix chain. Uses `/usr/bin/python3`. Output: `contact_sheet/stitched_v5/`

## Z-Stitch Alignment (v5 — CURRENT, all 20 pairs)
- `comparison_collage_v5.py` — **CURRENT**: 5-column collage per pair + high-res ALL_PAIRS contact sheet.
  - Col 1: Raw + annotation contours | Col 2: IOU Rigid | Col 3: Farneback v3 | Col 4: Elastix 500iter | Col 5: Elastix 2000iter
  - MAGENTA_BOOST=2.0, CELL_W/H=600, saves `results_v5.json`
  - Output: `png_exports/z_stitch_comparison_v5/`

## Z-Stitch Alignment (v4 — rows 2-4 reprocessed)
- `auto_align_iou_v4.py` — 3-stage IOU rigid: coarse(±20°,±400px,ds=4) / medium(±3°,±40px,ds=2) / fine(±1°,±16px,ds=1). 17/20 pairs. PNG loading per-pair. Saves `auto_align_transforms_iou_v4.json`, QC to `png_exports/z_stitch_qc_iou/`
- `elastix_all_pairs_v4.py` — SimpleElastix B-spline DS=2, configurable iter (2000 default → 500 also run), 4 res, grid=32px. Output: `png_exports/z_stitch_elastix_v4/` (2000) and `z_stitch_elastix_v4_500iter/` (500)
- `farneback_all_pairs_v4.py` — Farneback v3 all 20 pairs, v4 masks + v4 IOU transforms. Output: `png_exports/z_stitch_farneback_v4/`
- `comparison_collage_v4.py` — 3-column: IOU Rigid | Farneback v3 | Elastix 2000iter. Output: `png_exports/z_stitch_comparison_v4/`

## Calcium Movie Pipeline
- `build_viewer_warped_invivo_3d_v2.py` — Base builder for warped invivo 3D viewer (v2)
- Calcium registration: inline in session — SIFT affine between avg movie frame (flipped YX) and JY306 z=3
- Patch video generation: ffmpeg H.264, 878 patches cached at `/tmp/patch_b64s_h264.npy`
- Sprite sheet alt (not used): `/tmp/sprite_b64s_jpg.npy` (JPEG, 111 frames, 64×64)
- **mp4v (cv2 default) NOT browser-compatible — always use H.264 via ffmpeg for embedded video**

## Z-Stitch Alignment (earlier versions)
- `auto_align_iou_v3.py` / `auto_align_iou_only.py` — older IOU scripts (v3 masks)
- `comparison_collage_v2.py` / `comparison_collage_v3.py` — older 3-column collages
- `elastix_all_pairs_highiter.py` — DS=2, 2000 iter, v3 masks
- `elastix_all_pairs_fullres.py` — DS=1, 4000 iter, cancelled
- `elastix_all_pairs.py` — original, DS=2, 500 iter
- `tune_deformable.py` — per-pair Farneback parameter tuning
