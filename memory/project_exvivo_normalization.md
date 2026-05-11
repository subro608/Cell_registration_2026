---
name: Ex-Vivo Per-Tile Brightness Normalization
description: Ventral tiles (row4_5-row5_1) 3x dimmer than dorsal — per-tile p99 normalization applied to all scene5b assets
type: project
---

**Problem:** Ex-vivo (magenta) intensity varies 3x across tiles — dorsal tiles (row1-2) have p99≈255, ventral tiles (row4_5, row4_6, row5_1) have p99=142-194. The pyramidale cell layer "ring" is invisible in ventral tiles.

**Fix:** Per-tile normalization targeting the median p99 (243) across all tiles.

**Boosts applied:**
| Tile | p99 | Boost |
|------|-----|-------|
| row2_4 | 199 | 1.22x |
| row3_1 | 197 | 1.23x |
| row3_2 | 210 | 1.16x |
| row3_5 | 228 | 1.07x |
| row4_5 | 174 | 1.40x |
| row4_6 | 194 | 1.25x |
| row5_1 | 142 | 1.71x |

**Where applied:**
1. `build_three_stacks_assets.py` — normalizes B+R channels of per-tile dense slices AND stitched volume z-chunks before saving to `scene5b_three_stacks_assets.pkl`
2. `scene5b_multi_tile_3d_parallel.py` — normalizes at load time in `load_tiles()` (reads from `scene5b_assets_v3.pkl` which is NOT modified)

**Why:** Erdem confirmed row1→5 = dorsal→ventral, oriens→pyramidale. The pyramidale ring must be visible to appreciate ex-vivo/MERSCOPE spatial match.

**How to apply:** If assets are rebuilt, normalization is baked in. If new tiles or different pkl is used, re-run build_three_stacks_assets.py.
