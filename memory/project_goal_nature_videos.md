---
name: Nature Science Video/Animation Goal
description: Prof Erdem's 4-step directive for 3D alignment, current stitching status, and remaining registration steps
type: project
---

The user needs to generate animations and videos showing the multimodal cell-matching data for a Nature Science submission.

**Why:** Publication requirement — supplementary videos and figure panels showing the registration and matching pipeline visually.

**How to apply:** Prioritize proper 3D alignment first, then build publication-quality animations.

## Prof Erdem's directive (2026-03-25)

1. **Locate matched cells in 3D** in both volumes (each cell spans multiple z-slices, find true centroid) — NOT DONE. Prof wants MANUAL marking, not auto-detected.
2. **Align 3D coordinates** from in-vivo to ex-vivo — NOT DONE. Need seed points first.
3. **Transform entire 3D stacks** to each other in 3D — NOT DONE.
4. **Show 3D alignment at angular view** for confirmation — NOT DONE.

## Stitching pipeline completed (2026-03-27)

### What was done
1. All 22 nd2 tiles (row1/1-4, row2/1-5, row3/1-6, row4/1-6, row5/1) identified as sequential z-sections
2. VIA annotations (v4) used to mask hippocampus region in each tile
3. Pairwise alignment: IOU rigid (centroid+rotation+translation) → elastix B-spline (500 iter, DS=2)
4. Elastix transform files saved: `png_exports/z_stitch_elastix_v4_500iter/{pair}/TransformParameters.0.txt`
5. Volume stitched: mask applied → IOU rigid warp at full res → downsample to DS=2 → elastix B-spline → stack in z
6. Row1_4 replaced with 6-slice black slab (12µm gap, no annotation available)
7. Resampled to 1µm isotropic (both ex-vivo and in-vivo)
8. **Requires `/usr/bin/python3` for SimpleITK** (not the venv python)

### Output files (in `registration_video/stitched/`)
- `stitched_gfp_elastix_1um_isotropic.tif` — **(516, 2748, 2748) 1µm isotropic, elastix B-spline aligned, masked** — THIS IS THE FINAL 3D VOLUME
- `stitched_gfp_elastix.tif` — native DS=2 resolution (258, 2100, 2100) before resampling
- `stitched_gfp_masked_1um_isotropic.tif` — rigid-only version (for comparison)
- `stitched_gfp_rigid.tif` — rigid-only, unmasked (background included)
- `invivo_1um_isotropic.tif` — **(189, 1229, 1177) 1µm isotropic in-vivo** — resampled from (63, 1798, 1720) at 0.6835×0.6845×3µm

### 3D HTML viewer
- `3d_viewer/viewer.html` — Three.js point cloud volumetric viewer (8.7MB self-contained HTML)
- 4x downsampled (129×687×687), 129 PNG slices embedded as base64
- Features: drag rotate, scroll zoom, threshold/opacity/colormap sliders, auto-rotate
- Open with: `open /Users/neurolab/neuroinformatics/margaret/3d_viewer/viewer.html`

### MIP exports
- `png_exports/stitched_3d_views/MIP_XY_elastix.png` — top-down MIP of elastix volume
- `png_exports/stitched_3d_views/MIP_XZ_elastix.png` — side view MIP
- `png_exports/stitched_3d_views/MIP_XY_masked.png` — rigid masked version

## Erdem's latest (2026-03-27)
- "Cool now let's resample this such that voxels are 1 x 1 x 1 um where they are currently 0.6544 x 0.6544 x 2" — DONE, the final volume is `stitched_gfp_elastix_1um_isotropic.tif`
- **Focus is on ex-vivo stitched volume**, NOT in-vivo at this stage. The in-vivo resampling was done proactively but Erdem has not asked for it yet.

## Stitching pipeline detail
The final ex-vivo volume was built from elastix B-spline transforms, NOT just rigid:
1. Mask each tile (v4 VIA annotations)
2. IOU rigid warp at full res (4200×4200)
3. Downsample to DS=2 (2100×2100)
4. Apply elastix B-spline transform (TransformParameters from `z_stitch_elastix_v4_500iter/`)
5. Stack 258 slices in z (21 tiles × 12 + 6 slab)
6. Resample from 1.3088×1.3088×2µm → 1×1×1 µm isotropic

## Next steps (remaining from Erdem's directive)
1. User identifies matched cells in row2_1 as seed points (cells visible in both ex-vivo and in-vivo)
2. Compute rough 3D affine from seed points between stitched ex-vivo and in-vivo (both now at 1µm isotropic)
3. Search around affine solution for better cross-correlation
4. Build angular 3D confirmation view showing alignment quality
