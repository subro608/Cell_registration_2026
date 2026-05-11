---
name: Animation Feedback v2 (2026-04-04)
description: Erdem's latest feedback — static angle for tile mosaic, real tile merge animation, scene7 no zoom-out, MERSCOPE dots, correct z-depth
type: feedback
---

## Point 2: Tile mosaic — no spinning rotation
- When showing all 19 tiles, they don't need to rotate much
- Camera should approach the mosaic at an angle so we see thickness
- Static/near-static view, not spinning
- Implemented: STATIC_ROT_Y=0.25 (fixed angle)

## Point 3: Tile merge — real spatial animation, not crossfade
- Tiles should physically move from grid positions (all z=0) to their final stacked positions
- Each tile has known x, y alignment in the volume
- z is from real PKL landmark medians (med_z per tile)
- Render incremental translations, NOT fade-in animation
- Implemented: Phase C with per-tile z_offsets from real data

## Point 4: Scene 7 — no zoom-out for 4-view patches
- Don't zoom out at the end of the 4-view panel
- Merge cell patch views at the zoomed-in level
- Do this for multiple cells (bunch of cells)

## Point 5: MERSCOPE gene dots on tiles
- Show aligned MERSCOPE gene expression dots overlaid on tiles
- Filter for bright saturated dots only (not DAPI/GCaMP background)
- Implemented: B1.5 fade-in + B2 fly-by with MERSCOPE visible

## Point 6: Z-axis correct scaling
- Z-axis was not scaled correctly with arbitrary Z_SPACING
- Fixed: per-tile z_spacing computed from real pixel sizes (0.645 µm/px XY, 3.0 µm z-step)
- Merge animation uses exaggerated z (visible), then transitions to real z-depth in Phase D

**Why:** Erdem wants physically accurate, intuitive animations — real spatial movements rather than visual effects. Publication quality for Nature Science.

**How to apply:** Always prefer geometric/spatial animations (translations, rotations) over opacity fades/crossfades when transitioning between spatial arrangements. All z-spacing values must be derived from real data.
