---
name: 4-Modality 2×2 Grid Viewer
description: Website viewer with ex-vivo/in-vivo/calcium/MERSCOPE in 2×2 Three.js grid, built for Nature Science editor review
type: project
---

## Overview
4-modality 3D viewer for Nature Science editor (Erdem's request). Built by `build_viewer_dual_3d_v5.py`, outputs to `invivo-exvivo-cell-registration/dual_v5.html` (~98MB).

## Architecture
- 2×2 CSS grid, each cell has its own Three.js scene/renderer/camera
- Synchronized rotation/zoom across all 4 quadrants
- 2D overlay canvas for cross-quadrant landmark connection lines
- Base64-embedded data (volumes + patches + calcium strip)

## Quadrants (colors match Jason's convention, updated 2026-04-07)
| Position | Modality | 3D Cloud Color | Source |
|----------|----------|-------|--------|
| Q0 (top-left) | Ex-vivo Z-stack | Magenta | nd2 tiles DS5, /4000 norm |
| Q1 (top-right) | In-vivo Z-stack | Green | JY306 s80 |
| Q2 (bottom-left) | In-vivo Calcium | Green | movie_rolling_avg temporal std |
| Q3 (bottom-right) | MERSCOPE mRNA | Rainbow (550 per-gene HSV, most_common) | 19 tiles, tissue-masked, IOU-aligned |

## Each modality in native coordinate space
- Ex-vivo/MERSCOPE: nd2 pixel space (4200×4200 per tile)
- In-vivo/calcium: movie native space (512×512)
- `buildCloud()` auto-normalizes each quadrant independently

## Patches (bottom panel on landmark click)
- **Order**: CALCIUM | IN VIVO | info | EX VIVO | MERSCOPE
- 878 landmarks across 19 tiles (including row2_1 via stitched_v5 fallback)
- Ex-vivo: single z-slice in nd2 space, CROP_RADIUS=130 nd2 px (~84µm), magenta
- In-vivo: single z-slice in JY306 native space, CROP_JY_R=61 px (~42µm, halved for visual zoom match), green
- Calcium: **animated video** — 60 frames warped movie→JY306, cropped per-landmark, plays at 10fps in magenta
- MERSCOPE: scene7 transform chain (microns→mosaic→fliplr→×0.108→PKL affine inverse→nd2), per-gene colored dots on black canvas
- MERSCOPE patch colors: per-cell `most_common()` gene ranking (scene7 approach), BGR palette, converted to RGB for PIL strip
- Crosshairs: toggleable via checkbox

## Bugs Fixed (2026-04-07)
1. **Color swap**: 3D clouds and labels were swapped vs Jason's convention (ex-vivo was green, in-vivo was magenta). Fixed to: ex-vivo=magenta, in-vivo=green everywhere.
2. **Landmark index mismatch**: Source HTML had 878 landmarks (with row2_1) but npz patch data only had 851 (no `landmarks_nd2_native_row2_1.npz`). Every patch after index 13 was shifted by 27 positions. Fixed by loading row2_1 from `landmarks_stitched_v5_row2_1.npz` as fallback.
3. **MERSCOPE dot colors blueish**: GENE_PALETTE stored as RGB, converted to BGR for dot canvas, then PIL read as RGB → double swap. Fixed by storing palette as BGR (matching scene7), converting to RGB only at PIL save.
4. **MERSCOPE 3D cloud whitish**: Gene colors assigned alphabetically (spread across rainbow → additive blend = white). Fixed by using `most_common()` ranking so frequent genes get warm HSV colors (red/orange).
5. **Label**: Changed "EX VIVO STRUCTURAL" → "EX VIVO Z-STACK"

## Interaction
- Landmark connection lines between Q0 (ex-vivo) and Q1 (in-vivo)
- Tile filter dropdown, hover detection (2D point-to-segment distance, 15px threshold)
- Multi-select (shift-click), per-modality opacity/point-size sliders
- Close button stops calcium animation

## Key Data
- 878 landmarks from npz files (ev_nd2 + pcd_invivo_jy306), row2_1 from stitched_v5 fallback
- 1,655,686 ex-vivo voxels, 216,870 in-vivo voxels, 173,637 calcium voxels, 150,000 MERSCOPE dots
- MERSCOPE 3D: per-gene rainbow via HSV palette, most_common() ranking
- Calcium strip: 60 frames × 629×658 grayscale PNG (~10MB base64)

## Deployment
- GitHub: subro608/invivo-exvivo-cell-registration (main + dev branches)
- Vercel: https://invivo-exvivo-cell-registration.vercel.app
- No separate test repo; use dev branch for preview before pushing to main
