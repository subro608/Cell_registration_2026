---
name: Per-Tile Registration Pipeline (3D Affine + TPS)
description: Full per-tile in-vivo→ex-vivo registration: Gaussian z-fit, 3D affine, TPS+IRLS correction — 17 tiles (row3_1/3_5 excluded), 786 landmarks. TPS reduces error 29% over affine alone.
type: project
---

**Approach:** Two registration methods available per tile:
1. **PKL direct (CURRENT for v4 viewer):** pkl inverse + SIFT affine chain → fit 2D affine per tile (JY306→nd2). Works for all 19 tiles. Mean 2D error 1.0–5.1µm.
2. **3D affine + TPS+IRLS (legacy):** Gaussian z-fit → 3D affine → TPS nonrigid correction. 17 tiles only (row3_1/3_5 excluded). Mean 4.0µm after TPS.

**Why:** PKL direct covers all 19 tiles and gives comparable accuracy without needing good z-overlap. 3D affine+TPS was better for nonrigid correction but failed on row3_1/3_5 (only 1 z-slice overlap).

**How to apply:** v4 3D viewer now uses PKL direct. TPS transforms still available at `registration_per_tile_tps/` for comparison.

---

## TPS+IRLS Results (2026-04-02)

- **878 total landmarks across 19 tiles** (786 after excluding row3_1/row3_5)
- **Affine mean: 5.6µm → TPS mean: 4.0µm (29% reduction, LOO cross-validated)**
- Best tiles: row4_1 (47%), row3_1 (43%), row3_6 (42%), row2_2 (40%)
- Worst: row1_3 (-31%, only 13 landmarks), row2_5 (6%)

## Excluded Tiles
- **row3_1** (55 lm): only 1 nd2 z-slice maps to valid iv range (z_nd2=9→z_iv=15). 0/55 pass blob filter.
- **row3_5** (37 lm): only 1 nd2 z-slice maps to valid iv range (z_nd2=9→z_iv=11). 2/37 pass blob filter.
- Both excluded from v4 3D viewer due to insufficient in-vivo overlap.

## Landmark Filtering

### Blob-distance filter (CURRENT — used in 3D viewers)
- **Single filter:** warp Gaussian blob through affine+TPS, measure centroid distance from ex-vivo landmark
- Threshold: 5µm → 318/786 kept (after row3_1/3_5 exclusion)
- Most direct/honest filter — measures actual warp accuracy

### Legacy filters (landmark_patches.html)
- Edge z-match: gaussian z < 0.5 or > 10.5
- 2D XY affine error > 10µm
- Z-mapping outlier (local linear z-fit residual)

## 3D Viewers (warped in-vivo overlaid on stitched ex-vivo)

### `viewer_warped_invivo_3d_v4.html` — LATEST (2026-04-02, updated with PKL direct)
- **Registration:** PKL direct 2D affine (replaces 3D affine + TPS). Uses `pkl_transform_{tile}.npz` per tile.
- **All 19 tiles** included (row3_1/row3_5 no longer excluded)
- **Landmarks:** 878 total, 401 filtered. Toggle: Filtered/All/None buttons
- **Colors:** Cyan = filtered (<5µm), Orange = unfiltered, Yellow = hover, Green = select
- **Landmark positions:** from `landmarks_stitched_v5_{tile}.npz` (proper v5 pipeline)
- **Volume warp:** pkl M2d (JY306→nd2) per tile, z-mapping from landmark correspondences
- **Patch strip cached** to `3d_viewer/patch_strip_v4.png` + `cell_info_v4.json`
- **Voxels:** 361K ex-vivo + 258K in-vivo warped
- Script: `build_viewer_warped_invivo_3d_v4.py`
- Output: `3d_viewer/viewer_warped_invivo_3d_v4.html` (~84 MB)

### `viewer_warped_invivo_3d_v3.html` — TPS blob filter only
- 320 landmarks (blob filter only, no unfiltered shown)
- Landmark positions via cum_iou (skips pair elastix B-spline — slightly misaligned)
- Script: `build_viewer_warped_invivo_3d_v3.py`
- Output: `3d_viewer/viewer_warped_invivo_3d_v3.html` (~36 MB)

### `viewer_warped_invivo_3d_v3_elastix.html` — Elastix variant
- Uses elastix B-spline instead of TPS for volume warp + blob filter
- 378 landmarks
- Script: `build_viewer_warped_invivo_3d_v3_elastix.py`
- Output: `3d_viewer/viewer_warped_invivo_3d_v3_elastix.html` (~41 MB)

## Patch Viewers (2D landmark patches)

- `landmark_patches_warp_err.html` — TPS, blob filter, 323 kept. Script: `build_landmark_patches_viewer_warp_err.py`
- `landmark_patches_elastix_warp_err.html` — Elastix, blob filter, 376 kept. Script: `build_landmark_patches_viewer_elastix_warp_err.py`
- `landmark_patches.html` — TPS, multi-filter, 643 kept. Script: `build_landmark_patches_viewer.py`
- `landmark_patches_elastix.html` — Elastix, multi-filter, 639 kept. Script: `build_landmark_patches_viewer_elastix.py`

## Transform Files

- **TPS per tile:** `png_exports/registration_per_tile_tps/<tile>/tps_transform_<tile>.npz`
  - Forward: W, a_coeff, src_pts | Inverse: W_inv, a_inv, src_inv
  - Per-landmark: weights, aff_errs, loo_errs, pred_pts, actual_pts
  - Per z-pair: ncc_affine, ncc_tps, z_iv_labels
  - Config: lam=20000, n_irls=10
- **3D affine + landmarks:** `png_exports/registration_per_tile_elastix/<tile>/transform_<tile>.npz`
- **Stitched landmark coords:** `registration_video/landmarks_stitched_v5_{tile}.npz`
  - Keys: `stitched_coords` (N,3) [z_um, y_um, x_um] in 1µm iso, `pcd_invivo_jy306`, `ev_nd2`, `cell_nd2_z`

## Scripts
- `build_viewer_warped_invivo_3d_v4.py` — **LATEST** 3D viewer with filtered/all/none toggle, stitched_v5 coords, patch caching
- `build_viewer_warped_invivo_3d_v3.py` — TPS 3D viewer with blob filter
- `build_viewer_warped_invivo_3d_v3_elastix.py` — Elastix 3D viewer
- `build_landmark_patches_viewer_warp_err.py` — TPS 2D patches with blob filter
- `build_landmark_patches_viewer_elastix_warp_err.py` — Elastix 2D patches with blob filter
- `build_all_tiles_tps.py` — Fits TPS+IRLS per tile, saves transforms + metrics plots
- `build_all_tiles_elastix_v2.py` — Contact sheets per tile

## Evolution
- 2D affine → 3D affine argmax (broken z) → 3D affine Gaussian z → +RBF → +Elastix → **3D affine + TPS+IRLS (current best)**
