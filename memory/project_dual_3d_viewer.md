---
name: Dual 3D Viewer Versions (Ex-vivo + In-vivo)
description: Multiple dual-volume 3D HTML viewers (v2-v5) with matched cell lines, depth-coded MIP patches, per-volume scaling
type: project
---

## Viewer Versions

| Version | Script | Output | Key Features |
|---------|--------|--------|-------------|
| v2 | `build_viewer_dual_3d_v2.py` | `viewer_dual_3d_v2.html` (65MB) | Stitched ex-vivo DS3+GP2x, raw nd2 patches |
| v3 | `build_viewer_dual_3d_v3.py` | `viewer_dual_3d_v3.html` | Same as v2 + depth-coded MIP patches |
| v4 | `build_viewer_dual_3d_v4.py` | `viewer_dual_3d_v4.html` (66MB) | Raw nd2 stacked (no IOU/elastix), DS5, 851 cells |
| v4_masked | `build_viewer_dual_3d_v4_masked.py` | `viewer_dual_3d_v4_masked.html` (62MB) | v4 + v4 masks applied |
| **v5** | `build_viewer_dual_3d_v5.py` | `viewer_dual_3d_v5.html` (~93MB) | **LATEST**: v3 base + per-volume scaling + centroid alignment + depth MIP |

## v5 Details (LATEST)

### Volumes
- **Ex-vivo**: Stitched v5 (516, 3554, 3545) → DS3+GP2x (343×1184×1181), per-slice equalized, sparse voxels
- **In-vivo**: JY306 s80 (16, 658, 629) → 1µm iso (48, 450, 430), median filter BG sub, threshold 50

### Per-volume normalization (Erdem's request 2026-03-30)
- Each volume normalized to its own max grid dimension (not shared global_max)
- Ex-vivo rendered at 4x scale (`ex_sx=ex_sy=ex_sz=4.0`) to appear larger than in-vivo
- Both volumes centroid-aligned (shift so voxel centroid = 0.5) to sit on same plane
- Spacing = 2.5 between volumes
- Point size: ex-vivo 0.024 (4x base to compensate for 4x scale), iv 0.006

### Landmark normalization
- Ex-vivo: `(stitched_coord / DS_EX) / ex_span + centroid_shift`
- In-vivo: `(native_px * pixel_size / DS_IV) / iv_span + centroid_shift`
- Critical: ex-vivo z must divide by DS_EX (bug found: without this, lines shoot outside volume)

### Depth-coded MIP patches (4-column strip)
- Layout: [ex-gray, ex-depth, iv-gray, iv-depth] per cell row
- Depth = argmax along z: which of the ±2 slices is brightest at each pixel
- Color: `depth_color(t)` — t=0 green (shallow), t=0.5 yellow, t=1 red (deep)
- Depth fraction is relative to ±2 MIP window (not full tile/volume)
- Ex-vivo patches from raw nd2 PNGs (not stitched), in-vivo from s80 native
- Both use 100µm FOV: PHYS_RADIUS=50µm (radius), full FOV=100µm
- CROP_JY=73px (50µm / 0.6835 µm/px) — straightforward
- CROP_ND2=184px (50µm / 0.272 µm/px) — NOT 0.645 µm/px! ev_nd2 coords go through MERSCOPE (0.701 µm/px) then SIFT inverse (×2.582), so effective pixel size = 0.701/2.582 = 0.272 µm/px
- Bug history: was wrongly set to 78px (using 0.645) then 150/200/300px manually — 184px is the correct physics-derived value
- Green crosshairs on grayscale, white crosshairs on depth patches

### Default config (user-approved 2026-03-30)
- Ex: threshold 8, opacity 57, pt size 1
- IV: threshold 50, opacity 74, pt size 1
- Lines: opacity 30, default tile row1_3
- Zoom: 8.0, no auto-rotate

## Critical Notes
- **USE**: `JY306_in_Vivo_stack_flipped_s80.tif` — landmarks are in this pixel space
- **DO NOT USE**: `invivo_1um_isotropic.tif` — from a DIFFERENT source volume
- All viewers must be <100MB for Vercel deployment
- Stitched canvas (5510×5496 native px) is only slightly larger than single tile (4200) because tiles overlap in XY (sequential z-sections, not spatial mosaic)
- Confocal ex-vivo depth MIP tends to be mostly green (argmax picks shallowest slice due to signal attenuation with depth)
- Weighted average depth was tried but reverted — user preferred argmax

## Deployment
- **v5 was the deployed version** but `build_viewer_dual_3d_v5.py` has been repurposed for the 4-modality viewer (see project_4modality_viewer.md)
- The original dual viewer source is preserved in `3d_viewer/viewer_dual_3d_v5.html`
- GitHub: subro608/invivo-exvivo-cell-registration
- Vercel: https://invivo-exvivo-cell-registration.vercel.app
