---
name: MERSCOPE to Confocal Transformations
description: 5 pkl files mapping row3 ex-vivo tiles to MERSCOPE FOVs — 13-step transform chains, matched cell landmarks, warped volumes
type: project
---

12 pkl files in `transformation/` align ex-vivo confocal tiles to JY306 in-vivo space (rows 1–3). Row4/row5 have NO pkls.

**Why:** Multimodal registration — map gene expression (MERSCOPE) onto the confocal hippocampal volume for the Nature Science submission.

**How to apply:** Each pkl is a dict with keys: `pcd_invivo`, `pcd_exvivo`, `transformations`, `transformed`.

### All 12 pkl files
| File | Tile | Matched cells |
|---|---|---|
| `1_3_merscope23transformed_*` | row1_3 | 13 |
| `2_1_merscope17transformed_*` | row2_1 | 27 |
| `2_2_merscope18transformed_*` | row2_2 | ~40 |
| `2_3_merscope19transformed_*` | row2_3 | ~43 |
| `2_4_merscope20transformed_*` | row2_4 | ~38 |
| `2_5_merscope21transformed_alt_*` | row2_5 | ~32 |
| `3_1_merscope16transformed_alt_*` | row3_1 | ~51 |
| `3_2_merscope15transformed_alt_*` | row3_2 | ~41 |
| `3_3_merscope14transformed_*` | row3_3 | 38 |
| `3_4_merscope13transformed_*` | row3_4 | ~38 |
| `3_5_merscope12transformed_*` | row3_5 | ~38 |
| `3_6_merscope11transformed_*` | row3_6 | ~38 |

### PKL structure
- `pcd_invivo` (N×3 float64) — matched cell 3D positions in in-vivo/confocal space
- `pcd_exvivo` (N×3 float64) — corresponding matched cell positions in MERSCOPE/ex-vivo space
- `transformations` (list of 13 dicts) — progressive alignment chain:
  - `scale` — rescaling step
  - `bhat` — rigid/affine transform
  - `vec_field_total` — nonlinear RBF warp displacement field
  - Pattern: scale → bhat → scale → bhat → bhat → RBF → bhat → bhat → RBF → bhat → RBF → bhat → RBF
- `transformed` (3, 17, 1734, 1734) float64 — MERSCOPE volume warped into confocal space
  - ch0: main signal (range ~-1200 to ~11500, mean ~4)
  - ch1: near-zero
  - ch2: all zeros (in the inspected file)

### Coordinate spaces
- `pcd_invivo` and `pcd_exvivo` are BOTH stored in JY306 in-vivo coordinate space (post-transform). pcd_exvivo = where ex-vivo landmarks ended up after registration. ~4px residual between them.
- The `transformed` canvas (1734×1734) maps to JY306 (658×629) via resize (not crop). Scale factors: y≈2.635, x≈2.757.
- Input to the pkl pipeline was raw ex-vivo at ~600×600 (7× downsampled from 4200px native).

### QC & exploration scripts
- `test_pkl_transform.py` — loads pkl, saves MIP per channel, mid-z overlay with JY306, point correspondence scatter. Output: `png_exports/pkl_transform_test/`
- `pkl_qc_panel.py` — 6-panel QC: in-vivo raw | ex-vivo MIP | in-vivo+landmarks | transformed ex-vivo+landmarks | overlay+correspondences. Output: `png_exports/pkl_transform_test/qc_panel.png`
- `apply_pkl_transform.py` — re-applies the 14 pkl transform stages to raw GFP PNGs from scratch using scipy.ndimage. Compares result with stored pkl output. Output: `png_exports/pkl_transform_test/applied_transform/`

### How to re-apply transforms to a new image
1. Load raw GFP PNGs, downsample to ~600×600 (7× from 4200px), stack to (17, 600, 600)
2. Upsample to (17, 1734, 1734) pkl canvas
3. For each of 14 stages in `transformations`:
   - `scale`: backward sample at coords/scale using `map_coordinates`
   - `bhat` (4×3): backward affine — p_in = (p_out - t) @ inv(R)
   - `vec_field_total` (17,1734,1734,3): backward warp — sample at (z-dz, y-dy, x-dx)
4. Result is the ex-vivo image in JY306 space

### Notes
- All files are for row 3 tiles only (row3_2 through row3_6)
- MERSCOPE FOV numbering is inverted: tile 3_2→FOV15, 3_3→FOV14, ..., 3_6→FOV11
- `transformed` is float64 — very large on disk (~5-8GB each)
- Requires `pickle.load()` to read (standard Python pickle)
- ch2 of `transformed` is all zeros; ch1 has near-zero values; ch0 is the main signal (GFP)
- The stripe/banding artifact in `transformed` comes from the deformable warping pipeline — not a display issue
