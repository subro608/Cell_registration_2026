---
name: Landmark Patches PKL Direct Viewer
description: build_patches_pkl_direct_all.py generates landmark_patches_pkl_direct.html — 5-panel QC patches for in-vivo→ex-vivo registration via PKL deformation fields
type: project
---

## Script: `build_patches_pkl_direct_all.py`

Generates `3d_viewer/landmark_patches_pkl_direct.html` — interactive HTML viewer for QC of in-vivo → ex-vivo registration.

### Pipeline
1. Load landmark files: `png_exports/registration_per_tile_pkl/<tile>/landmarks_nd2_native_*.npz`
   - Contains: `ev_nd2` (ex-vivo nd2 coords), `iv_nd2` (in-vivo in nd2 space), `pcd_invivo_jy306` (JY306 coords)
2. Load JY306 in-vivo stack (`invivo_jy306_12z_uint16.tif`, shape 12×512×512)
3. Load nd2 ex-vivo slices (from `png_exports/nd2_slices/<tile>_z{z}_ch{c}.png`)
4. Fit per-tile 2D affine: `M_jy_to_nd2` from lstsq of `pcd_invivo_jy306` → `iv_nd2`
5. Warp in-vivo slices to nd2 space using `cv2.warpAffine`
6. Generate 5 panels per landmark:
   - Ex-vivo nd2 | In-vivo warped (pkl) | Overlay (green/magenta) | In-vivo raw | Warp diagnostic (blob)
7. Blob filter at 5µm radius for pass/fail classification

### Outputs
- `3d_viewer/landmark_patches_pkl_direct.html` — interactive viewer with tile selector, size controls, pass/fail filter, crosshair toggle
- `png_exports/registration_per_tile_pkl/<tile>/pkl_transform_<tile>.npz` — saved affine transforms per tile

### Key Details
- Uses **all 19 tiles** including row3_1 and row3_5 (which are excluded from other pipelines)
- 878 total landmarks, 401 pass the 5µm blob filter
- In-vivo is flipped YX before SIFT matching
- Patch radius: 40px in nd2 space (~26µm at 0.645µm/px)

**Why:** This is the primary QC tool for verifying in-vivo → ex-vivo cell correspondence via PKL deformation fields.
**How to apply:** When user asks about landmark registration quality or wants to modify the QC viewer, this is the script to edit.

## Landmark + Gene Dot Contact Sheets

### Script: `build_landmark_genedot_patches.py`

Generates 4-panel contact sheets per tile combining in-vivo, ex-vivo, and MERFISH gene dots at each landmark.

### Panels (per landmark)
1. **IV-zoomed** — in-vivo raw (JY306 space, 2x zoom, ~25µm radius, green)
2. **IV-warped** — in-vivo warped to nd2 space via pkl M2d affine (green)
3. **EV-nd2** — ex-vivo GFP from nd2 (green)
4. **EV+genedots** — ex-vivo GFP dimmed to 35% + MERFISH gene dots (multicolored, 2px radius)

### Gene Dot Pipeline
- Tile→MERSCOPE region mapping via `exvivo_merscope_combined/` filenames (e.g. `2_1_merscope17.tif` → row2_1 = region 17)
- Transcripts transformed to nd2 space using the full 5-step pipeline (microns→mosaic→fliplr→zoom(0.108)→PKL affine inverse→nd2 scale)
- Dots subsampled to max 300 per patch to avoid saturation (actual counts range 2K-50K per 156×156 patch)
- Top 12 genes coloured distinctly, rest grey

### Filtering
- Only passing landmarks (<5µm blob error)
- Sorted by dot count, keep least-saturated 50%
- ~340 candidates total across 19 tiles

### Output
- `png_exports/landmark_genedot_patches/{tile}.png` — one contact sheet per tile (19 files)
- 10 patches per row, labels with LM#, tile, z-mapping, error, dot count
