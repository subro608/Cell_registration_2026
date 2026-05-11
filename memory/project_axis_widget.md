---
name: 3D Axis Widget
description: AP/ML/DV rotating axis trident on all 3D volume scenes (5b, 7). Colors: ML=red, AP=dark, DV=blue.
type: project
---

**3D axis widget** drawn in bottom-left corner of all 3D volume renders. Rotates with the volume using same rot_y/rot_x.

**Axes:**
- **ML** (medial-lateral): red (BGR 0,0,180), horizontal in slice → X
- **AP** (anterior-posterior): dark/black (BGR 40,40,40), vertical in slice (up) → Y  
- **DV** (dorsal-ventral): blue (BGR 200,80,0), depth/z-stack → Z

**Position:** cx=120, cy=H-120 (bottom-left), ax_len=70px

**Added to scripts (2026-04-05):**
- `scene5b_multi_tile_3d_parallel.py` — `draw_axes()` in `render_frame()`
- `scene5b_three_stacks_v2.py` — `draw_axes()` in `render_frame()`
- `gen_rotate_transition.py` — `draw_axes()` in `render_frame()`
- `scene7_cell_cards.py` — already had it (inline, lines 217-250)

**Why:** Erdem requested anatomical orientation. Confirmed axes: row1→5 = dorsal→ventral = oriens→pyramidale.
