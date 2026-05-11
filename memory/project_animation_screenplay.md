---
name: Animation Screenplay v1.4
description: Registration animation scenes 1-7. Combined script for 1-3-5. Scene 7 v5 (4 cells, 1565fr). 6x speedup on middle tiles.
type: project
---

## Scene-by-Scene Status

### Scene 1+2 — Calcium movie → in vivo alignment ✅ DONE
- **Script**: `animation/scene1_2_movie_to_zstack.py`
- **Output**: `animation/scene1_2_h264.mp4` (11.5s, 276 frames)
- Subtitles updated: no "confocal", "in vivo"/"ex vivo" in italic (Helvetica Oblique via PIL)

### Scene 3 — In vivo z-stack 3D rotation ✅ DONE
- **Script**: `animation/scene3_zstack_3d.py`
- **Output**: `animation/scene3_h264.mp4` (12.5s, 300 frames)
- Caption: "IN VIVO Z-STACK" throughout (no "16 SLICES")
- Scene 3f: 36-frame slide transition (in-vivo moves center → left)

### Scene 4 — 3D→z=3→move left ✅ DONE
- **Script**: `animation/scene4_landmarks.py`
- Note: still needs text_utils/italic update

### Scene 5 — Per-tile registration walkthrough ✅ DONE (19 tiles)
- **Script**: `animation/scene5_all_tiles_v2.py`
- ML/AP axis widget in bottom-left corner
- Italic "in vivo"/"ex vivo" via text_utils

### Combined Script: Scenes 1+3+5 ✅
- **Script**: `animation/scene_1_3_5_combined.py` (1374 lines)
- Merges all 3 scripts, shared constants/utilities, deduplicated functions
- Functions: `render_scene1_2(vw)`, `render_scene3(vw)`, `render_scene5(vw)`
- Outputs: `merged_scenes_1_3_5_h264.mp4` (full) + `merged_scenes_1_3_5_6x_h264.mp4` (6x middle)

### Merged Scenes 1-3-5 with 6x speedup ✅
- **Output**: `animation/merged_scenes_1-3-5_fast_mid.mp4` (~99s)
- Normal speed: row2_1 + row1_3 (first 2 tiles) and row4_6 + row5_1 (last 2 tiles)
- 6x fast: row2_2 through row4_5 (15 middle tiles)
- Timestamps: normal 0-47s, 6x 47-219.5s, normal 219.5-end

### Scene 5b — Multi-tile 3D + three stacks + rotation ✅ COMBINED
- **Combined frames**: `animation/frames_scene5b_combined/` (1025 frames @ 12fps)
- **Combined video**: `animation/scene5b_combined_preview.mp4`

#### Sub-scripts and frame sources:
1. **multi_tile_3d** (`scene5b_multi_tile_3d_parallel.py`) → `frames_multi_tile_3d/` (846fr)
2. **three_stacks_v2** (`scene5b_three_stacks_v2.py`) → `frames_three_stacks_v2/` (372fr)
3. **rotation transition** (`gen_rotate_transition.py`) → `/tmp/_rotate_transition/` (84fr)

### Scene 6 — TODO

### Scene 7 — Cell identity cards ✅ v5 (4 cells)
- **Script**: `animation/scene7_cell_cards.py`
- **Frames**: `animation/frames_scene7_v5/` (1565 frames)
- **Video**: `animation/scene7_v5_4cells_h264.mp4`
- **4 cells**: row1_3 #6, row1_3 #9, row2_1 #5, row2_1 #0 (5th cell row2_1 #4 REMOVED)
- **4 panels**: Calcium IN VIVO FUNCTIONAL | GCaMP IN VIVO STATIC | GCaMP EX VIVO STATIC | MERSCOPE mRNA EXPRESSION
- **Axis widget**: ML/AP/DV trident
- **Gene dots**: 24 colors (from overlay PNGs)
- Labels use italic "in vivo"/"ex vivo" via text_utils

### Full Combined Video (outdated — needs rebuild with 4-cell scene 7)
- Previous: `animation/scenes_1_to_5b_7.mp4` (468s)

---

## Key Assets

| File | Contents |
|------|----------|
| `scene7_assets.pkl` (724MB) | Pre-computed volume, per-cell panels, calcium frames, dot positions |
| `scene5b_assets_v3.pkl` (212MB) | 19 tiles raw dense+merscope, stitched volume |
| `scene5b_three_stacks_assets.pkl` (1.5GB) | Per-tile: dense/dense_with_ms/invivo/exvivo/merscope. Stitched volumes. |
| `stitched_with_merscope.pkl` (227MB) | Stitched volume with MERSCOPE at correct z-offsets |
