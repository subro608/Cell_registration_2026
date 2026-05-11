---
name: MERSCOPE Overlay Pipeline
description: Correct coordinate transform for MERFISH gene dots → ex-vivo nd2 space, full-FOV overlay generation
type: project
---

## MERFISH Transcript → Ex-Vivo Coordinate Transform

The correct pipeline to map MERFISH transcripts onto the ex-vivo confocal (nd2) images:

1. **Global microns → mosaic pixels**: `x_mos = scale * gx + tx`, `y_mos = scale * gy + ty` (from `micron_to_mosaic_pixel_transform.csv`)
2. **Flip LR**: `x_flip = W_mosaic - 1 - x_mos`
3. **Zoom by 0.108**: `merc_x = x_flip * 0.108`, `merc_y = y_mos * 0.108` → MERSCOPE source coords
4. **PKL affine inverse**: `exvivo = R_3_inv @ (merc - offset_3)` → ex-vivo TIF coords (1627×1627)
5. **Scale to nd2**: `nd2 = tif * (4200/1627)` → nd2 native coords (4200×4200 at 0.645 µm/px)

**Why:** Just using `zoom(0.108)` (steps 1-3) without the PKL affine (step 4) gives wrong alignment — the gene dots are too large and mispositioned. The PKL affine accounts for scale (~0.598), rotation (~2°), and translation between MERSCOPE and ex-vivo spaces.

## PKL Transform Details

- `merscope_exvivo/` PKLs map MERSCOPE → ex-vivo (NOT ex-vivo → MERSCOPE as initially assumed)
- `pcd_fixed` and `pcd_moving` are BOTH in the output (ex-vivo) space
- The affine B is built from: scale(0.5965) → bhat(rotation+translation) → scale(1.003) → bhat(fine) → vec_field(~zero)
- `R_3 = inv(B[:3,:3]).T` maps output→input; `R_3_inv = inv(R_3)` maps input→output (what we need)
- `offset_3 = -B[-1,:-1] @ inv(B[:3,:3])`
- `exvivo_merscope_combined` TIFs are in ex-vivo space, not MERSCOPE space

## Outputs

### Full-FOV Overlays
- Script: `build_merscope_overlay.py`
- Output: `png_exports/merscope_overlay/region_{id}_{tile}.png`
- 22 regions, 8408×4232 px (side-by-side: confocal DAPI+GCaMP | confocal + gene dots)
- nd2 native resolution (4200×4200 at 0.645 µm/px), single-pixel dots, fully opaque

### Cell-Centered Patches
- Script: `build_merscope_cell_patches.py`
- Output: `png_exports/merscope_overlay/cell_patches_region_{id}_{tile}.png`
- 22 regions, ~119K total cells (cells with ≥50 transcripts from `cell_id` in detected_transcripts.csv)
- Each cell: DAPI+GCaMP (left) | DAPI+GCaMP+gene dots (right), 80×80 px display, 60px crop radius in nd2 space
- 10 patches per row, sorted by transcript count (highest first)

### nd2 Tile Paths
- `registration_video/row{r}/{c}.nd2` (shape: 12, 2, 4200, 4200), ch0=GCaMP, ch1=DAPI
- Row 5 exception: `registration_video/row5/Row5/{c}.nd2`

## How to apply
- When mapping MERFISH data to ex-vivo space, always use the full 5-step pipeline (not just zoom 0.108)
- The PKL affine inverse is essential — without it coordinates are wrong
