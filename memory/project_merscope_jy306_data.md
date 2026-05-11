---
name: JY306 MERSCOPE Data (jy306_varol)
description: MERSCOPE transcript data, mosaic transform, and ex-vivo↔MERSCOPE alignment PKLs for JY306
type: project
---

## Key Files

### jy306_varol/region_0_resegmentation/ (extracted from zip)
- `detected_transcripts.csv` — 3.28M transcripts, columns: `global_x, global_y, global_z, x, y, fov, gene, transcript_id, cell_id`
- `micron_to_mosaic_pixel_transform.csv` — 3×3 affine: x_px = 9.2595×x_µm − 92548, y_px = 9.2597×y_µm − 1701
- `manifest.json` — experiment metadata

### manifest key values
- `microns_per_pixel`: **0.108 µm/px** (NOT 0.701 as previously assumed)
- Mosaic: 22537 × 20720 pixels = 2.43mm × 2.24mm tissue
- 12×11 FOV grid (132 FOVs total)
- Stains: Anti-Rat, DAPI, PolyT, Anti-Chicken, Cellbound1/2/3 (z0-z7)
- bbox_microns: x=[9994.9, 12428.8], y=[183.7, 2421.4]

**Why:** The `registration_video/mosaic_Anti-Rat_z4.tif` (18875×22558) is a DIFFERENT dataset (likely JY316 or different region). The JY306 mosaic has dimensions 22537×20720.

## Transcript Data Usage
- `x, y` = FOV-relative pixel coordinates (within each FOV) — use directly for per-cell gene dot patches
- `fov` = FOV number (matches PKL filenames: reg4-reg26 in exvivo_merscope_combined/)
- `global_x/y` = physical microns → use micron_to_mosaic_pixel_transform to get mosaic pixels
- `cell_id` = cell assignment from resegmentation (-1 = unassigned)

## merscope_exvivo/ PKLs (ex-vivo GCaMP → MERSCOPE alignment)
- 22 PKLs: `{row}_{col}_reg{fov}_transformed_{timestamp}.pkl`
- Structure: `transformed` (1, 2, 1632, 1632), `pcd_fixed` (MERSCOPE FOV coords), `pcd_moving` (GCaMP coords), `transformations` (5-step affine chain)
- Alignment offset: < 3px (~2µm) — nearly perfect co-registration
- Transformation: scale (0.597) + rotation (~2.1°) + translation + minor correction — **purely affine** (vec_field = all zeros)
- `pcd_fixed` y-range: 598-1136, x-range: 396-1074 (within 1632×1632 FOV)

## exvivo_merscope_combined/ TIFs
- Shape: (3, 1627, 1627, 3) = (z_slices, H, W, RGB)
- ch0 (R): ex-vivo GFP warped to MERSCOPE FOV space (max ~8839)
- ch1 (G): ex-vivo DAPI warped to MERSCOPE FOV space (max ~0.16, normalized)
- ch2 (B): zero

## Gene-to-Cell Mapping
- `cell_id` column in detected_transcripts.csv assigns each transcript to a segmented cell
- region_17: 10,979 unique cells, 550 genes, 60% of transcripts assigned, mean 49 unique genes/cell, max 221
- Filter by `cell_id` → count per gene = per-cell expression profile
- `cell_id = -1` = unassigned transcripts

## Cell Segmentation Files (region_17 / tile row2_1)
- **`Merscope_video/exvivo/exvivo21_merscope17_invivo.seg.npy`**: dict with keys `filename`, `masks` (19, 578, 599) float32, `contours_3d`. Cell labels in **in-vivo stack space** (JY306 s80 coordinates). 52 matched cells.
- **`Merscope_video/transcripts/reg17_gene_invivo.tif`**: shape (19, 578, 599, 3) float32, max 151. All-gene density volume already transformed into **in-vivo space**. ch0+ch1 have data, ch2 zero.
- These two files are already co-registered to in-vivo — no MERSCOPE coordinate transform needed for gene visualization in in-vivo space.

## merscope 2/reg_17/ (MERSCOPE mosaic space, likely JY316 not JY306)
- `gene_scatter_single_channel.tif`: (15013, 20569) uint8 — gene positions rendered in mosaic space
- `mosaic_Anti-Rat_z4.tif`: (15013, 20569) uint16 — Anti-Rat stain mosaic
- Dimensions 15013×20569 ≠ region_17 manifest (18875×22558) — this is JY316 data, not JY306

## How to apply: Per-cell MERFISH dot patches
- For gene dots in in-vivo space: use `reg17_gene_invivo.tif` + `exvivo21_merscope17_invivo.seg.npy` (already aligned, no coordinate transform)
- For per-cell gene expression profiles: filter `detected_transcripts.csv` by `cell_id`, count per gene
- For gene dots in ex-vivo/MERSCOPE space: use global_x,y from detected_transcripts.csv with coordinate mapping (user has separate transform code)
