---
name: PKL Direct Registration Pipeline
description: All-tile registration using direct pkl deformation field (SIFT+pkl inverse chain), 878 landmarks, 401 pass blob filter, per-tile 2D affine transforms saved
type: project
---

**Approach:** Direct pkl deformation field pipeline for in-vivo→ex-vivo registration. Uses pre-computed pkl inverse + SIFT affine chain (from `make_contact_sheet.py`) to map JY306 in-vivo positions to nd2 native space. Fits a global 2D affine per tile from all pkl-derived correspondences (JY306 xy → nd2 xy), warps full in-vivo z-slices to nd2 space.

**Why:** Alternative to 3D affine + TPS pipeline. Works for ALL 19 tiles including row3_1/row3_5 (which fail with 3D affine due to only 1 z-slice overlap). PKL mean 2D error ranges 1.0–5.1µm across tiles.

**How to apply:** Use pkl_transform npz files for per-tile JY306→nd2 2D affine. Blob filter at 5µm for quality gating.

---

## Pipeline Chain
JY306 in-vivo → pkl inverse (iterative) → MERSCOPE space → SIFT affine inverse → nd2 native 4200px

## Results (2026-04-02)
- **878 total landmarks** across 19 tiles
- **401 passed** blob filter (<5µm), 46% pass rate
- Best tiles: row2_5 (1.0µm), row1_3 (2.2µm), row3_1 (2.4µm)
- Worst tiles: row5_1 (5.1µm), row4_3 (4.8µm), row4_6 (4.8µm)

## Per-Tile Summary
| Tile | Landmarks | PKL err (µm) | Passed/Total |
|------|-----------|---------------|--------------|
| row1_3 | 13 | 2.2 | 8/13 |
| row2_1 | 27 | 3.0 | 16/27 |
| row2_2 | 40 | 3.1 | 10/40 |
| row2_3 | 43 | 3.1 | 22/43 |
| row2_4 | 38 | 3.4 | 23/38 |
| row2_5 | 32 | 1.0 | 10/32 |
| row3_1 | 55 | 2.4 | 17/55 |
| row3_2 | 41 | 2.8 | 16/41 |
| row3_3 | 38 | 4.3 | 13/38 |
| row3_4 | 43 | 3.4 | 21/43 |
| row3_5 | 37 | 2.5 | 20/37 |
| row3_6 | 47 | 3.1 | 9/47 |
| row4_1 | 70 | 4.7 | 32/70 |
| row4_2 | 66 | 3.6 | 35/66 |
| row4_3 | 73 | 4.8 | 29/73 |
| row4_4 | 67 | 3.1 | 43/67 |
| row4_5 | 51 | 2.9 | 29/51 |
| row4_6 | 58 | 4.8 | 27/58 |
| row5_1 | 39 | 5.1 | 21/39 |

## Transforms Saved
- `png_exports/registration_per_tile_pkl/<tile>/pkl_transform_<tile>.npz`
  - Keys: `M2d_jy306_to_nd2` (2x3 affine), `M_lstsq` (3x2 lstsq coeffs), `ev_nd2`, `iv_nd2`, `pcd_invivo_jy306`, `nd2_z_gauss`, `pkl_dist_um`, `n_landmarks`

## Scripts
- `build_patches_pkl_direct_all.py` — All 19 tiles, saves transforms, HTML viewer with Filtered/All toggle
- `build_patches_pkl_direct_row31_35.py` — Row3_1 & row3_5 only (earlier version)

## Viewer
- `3d_viewer/landmark_patches_pkl_direct.html` (63.7 MB) — 5-column format: Ex-vivo nd2 | In-vivo warped (pkl) | Overlay | In-vivo raw | Warp diagnostic. Tile selector, size controls, crosshair toggle, Filtered(401)/All(878) buttons. Blob filter threshold: 5µm.

## Comparison with TPS+IRLS Pipeline
- TPS+IRLS: 786 landmarks (17 tiles, excludes row3_1/3_5), affine mean 5.6µm → TPS mean 4.0µm
- PKL direct: 878 landmarks (19 tiles, includes row3_1/3_5), mean 2D error ~3.3µm
- PKL direct covers all tiles; TPS+IRLS gives better nonrigid correction for tiles with good z-overlap
