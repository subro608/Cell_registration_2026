---
name: Calcium Movie Registration
description: 2-photon calcium movie (movie_rolling_avg_win12_step3.tif), flip correction, registration to JY306 z=3, warped movie, and 3D viewer with per-landmark patch videos
type: project
---

## Source file
`jy306_registered files/movie_rolling_avg_win12_step3.tif`
- Shape: (663, 512, 512) int16
- Rolling average window=12, step=3 (original ~2000 frames)
- Needs **flip Y+X** to match JY306 orientation (NCC=0.96 after flip+resample)

## Registration to JY306
- Best matching z-slice: **z=3** (NCC=0.75, 113 SIFT matches, 95 RANSAC inliers)
- Method: SIFT_create(5000) → BFMatcher → Lowe 0.75 → estimateAffine2D RANSAC 5px
- Affine: scale=0.881 (movie FOV ~13% larger), rotation=0.045°, translation=(61.7, 89.3)px
- Mean reprojection error: **0.66px**
- 214 of 878 landmarks are in focal plane (JY306 z=3 ±2)

**Why:** Erdem requested registering the calcium movie to the in-vivo volume as part of the multimodal pipeline.

**How to apply:** Always flip movie Y+X before registration. Use SIFT affine to warp into JY306 space.

## Outputs
- `png_exports/native_invivo/movie_warped_to_jy306.mp4` — all 663 frames warped into JY306 space (629×658)
- `png_exports/native_invivo/movie_flipped_yx.mp4` — flipped movie (not warped)
- `png_exports/native_invivo/movie_histmatch_avg_frame.mp4` — histogram matched to avg frame
- `png_exports/native_invivo/movie_to_jy306_z3_flipped.png` — 4-panel registration QC image
- `png_exports/native_invivo/sidebyside_all16_v3.png` — JY306 vs native confocal resampled+flipped, all 16 z-slices

## Native confocal (s80.tif)
- `jy306_registered files/JY306_in_Vivo_stack_s80.tif` — (63, 1798, 1720) uint16, raw full-res
- `JY306_in_Vivo_stack_flipped_s80.tif` (root) — (16, 658, 629) float64, registered+flipped — THIS is used by all scripts and pkl landmarks
- To match: resample raw to (16,658,629) via zoom then flip Y+X (NCC=0.96)

## 3D Viewer with Calcium Patches
- `3d_viewer/viewer_warped_invivo_3d_v3.html` (123MB) — LATEST viewer
- **Fixed floating calcium panel** (bottom-right corner) — always visible, auto-plays first in-focus landmark on page load
- Click any landmark → updates the floating video panel with that landmark's patch
- Cyan spheres (larger) = in focal plane (z=3±2, 214 landmarks); orange spheres (dim) = out of plane
- "Only focal plane" checkbox filters to cyan landmarks only
- Patches: 878 H.264 videos, 80×80px crop from warped movie at each landmark's JY306 (y,x)
- Cached at `/tmp/patch_b64s_h264.npy` — regenerate with ffmpeg if /tmp cleared
- **mp4v codec NOT browser-compatible — must use H.264 (libx264) via ffmpeg**
- "In focal plane" = JY306 z=3±2 (movie only captures one focal plane, z=3 best match)
- "Out of plane" = landmark at different z-depth, movie didn't image that cell

## Key scripts
- `png_exports/native_invivo/movie_warped_to_jy306.mp4` — mp4v (source, not browser-playable)
- `png_exports/native_invivo/movie_warped_h264.mp4` — H.264 re-encoded, browser-playable
- Re-encode command: `ffmpeg -y -i movie_warped_to_jy306.mp4 -vcodec libx264 -pix_fmt yuv420p -crf 23 -preset fast movie_warped_h264.mp4`

## Patch files (separate, for GitHub deployment)
- `/Users/neurolab/neuroinformatics/invivo-exvivo-cell-registration/patches/` — 878 H.264 patch MP4s
- Files: `patch_0.mp4` → `patch_877.mp4`, total 33MB
- One file per landmark (same order as landmarks array in viewer)
- Saved from `/tmp/patch_b64s_h264.npy` — regenerate with ffmpeg if needed
- Next step: update HTML to load patches by path (`src="patches/patch_idx.mp4"`) instead of base64, then push to GitHub Pages
