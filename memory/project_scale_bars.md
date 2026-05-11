---
name: Scale Bars for Animation
description: Scale bars on all scenes (1-5, 5b, 7). Per-phase µm/disp_px computed from real data. Post-processed onto scene5b frames.
type: project
---

Jason asked for scale bars in the video (2026-04-04). Added to all scenes.

**Raw pixel sizes (corrected 2026-04-05):**
- In-vivo confocal: **0.82 µm/px** XY
- nd2 ex-vivo confocal: **0.65 µm/px** XY
- Calcium movie (512×512): **0.776 µm/px** (may also need update)

**Scene 5b scale values (computed from assets):**
- `avg_um_per_dpx = 3.19` (stitched volume pixel size)
- GRID_SCALE=1.0 (multi_tile_3d): ~3.2 µm/disp_px → 500 µm bar
- 3-column grid (GRID_SCALE≈0.3): ~10.7 µm/disp_px → 1000 µm bar
- VOLUME_SCALE=1.8: ~1.77 µm/disp_px → 200 µm bar
- VOL_SCALE_SPLIT≈1.19: ~2.67 µm/disp_px → 500 µm bar
- Merged grid (MERGED_SCALE≈0.575): ~5.55 µm/disp_px

**Implementation:** `animation/add_scale_bars_5b.py` post-processes existing frames. Per-phase µm/disp_px with smooth interpolation during transitions. No scale bars on fade-to-black frames.

**Scene 7:** Scale bar at 0.65 µm/disp_px (CANVAS_UM_PER_PX).

**Convention:** Scale bars shown even during 3D rotation (represents XY scale at front face).

**Why:** Nature Science publication needs physical scale reference.
