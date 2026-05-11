---
name: MERSCOPE Gene Color Convention
description: HSV rainbow palette (550 colors), must use most_common() gene ranking and BGR storage for cv2 pipeline
type: project
---

## Palette
- `make_rainbow_palette(550)`: HSV sweep H=0→180, varied S/V
- H=0 = red/orange (most common genes), H=180 = blue/purple (rare genes)
- Stored as **BGR tuples** (raw cv2 output) throughout Python pipeline
- Convert to RGB only at final output: PIL save (patches) or JS export (3D cloud)

## Color Assignment
- **Per-cell patches** (scene7 + dual_v5): `Counter(genes_in_crop).most_common()` — each cell gets its own ranking, most frequent gene = index 0 = red/orange
- **3D cloud** (dual_v5): `gene_counter.most_common()` globally — most frequent gene across all tiles = index 0 = red/orange
- scene5b also uses 550 rainbow from CSV

## Why most_common matters
- Alphabetical sorting spreads colors evenly across rainbow → additive blend in 3D = whitish
- most_common puts frequent genes at warm end (H≈0) → cloud looks orange, matching scene7

## BGR vs RGB Pitfall (bug found 2026-04-07)
- cv2 uses BGR, PIL uses RGB, browser JS uses RGB
- GENE_PALETTE must be BGR for dot canvas rendering (cv2.resize etc)
- Convert BGR→RGB for: `Image.fromarray()` (PIL strip), `json.dumps()` (JS genePalette)
- Previous bug: double BGR↔RGB swap made MERSCOPE patches look blueish

## Scene7 now uses 550-color CSV rendering
- Previously used only 24 colors from overlay PNGs — switched to 550-color rainbow from CSV (same as scene5b)
- `scene7_precompute.py` renders dots directly from detected_transcripts.csv
