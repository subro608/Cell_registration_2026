---
name: Twitter Showcase Pipeline
description: build_twitter_final.py — tissue + cell contact sheets for Jason's Twitter post. GIF + static overlay + 3-cell grid per tile.
type: project
---

## Goal
Jason requested showcase images for Twitter showing in-vivo → ex-vivo → MERSCOPE registration quality. Two examples: one oriens, one pyramidale.

## Final Layout (per tile)
Matches Erdem's template design:
- **Left (GIF)**: Tissue-level cycling through IN VIVO → EX VIVO → MERSCOPE mRNA → REGISTERED with crossfade
- **Middle (static)**: Tissue overlay (all 3 registered, blended)
- **Right**: 3 cells × 3 columns (In vivo green | Ex vivo magenta | MERSCOPE rainbow dots)

## Selected Cells
- **row2_1 (ORIENS)**: LM#7 (1.79µm), LM#5 (0.46µm), LM#17 (0.61µm)
- **row3_6 (PYRAMIDALE)**: LM#0 (0.42µm), LM#36 (0.04µm), LM#8 (0.90µm)

## Key Scripts
| Script | Purpose |
|--------|---------|
| `build_twitter_final.py` | Final layout: static PNG + GIF frames per tile |
| `build_twitter_tissue.py` | Earlier tissue-only 4-column contact sheets |
| `build_cell_contact_sheet.py` | Cell candidate picker (10 cells × 4 cols) |
| `build_twitter_showcase.py` | Earlier cell-level showcase (deprecated) |

## Outputs
- `png_exports/twitter_showcase/twitter_oriens_row2_1_static.png`
- `png_exports/twitter_showcase/twitter_pyramidale_row3_6_static.png`
- `png_exports/twitter_showcase/twitter_oriens_row2_1.gif`
- `png_exports/twitter_showcase/twitter_pyramidale_row3_6.gif`
- `png_exports/twitter_showcase/gif_frames_*/` — individual PNG frames

## Design Decisions
- Tissue panels use via_masks_v4.npz (4200×4200 binary) for clean boundaries
- In-vivo: MIP ±2z around median, warped via M2d to nd2 space
- Colors: in-vivo=green, ex-vivo=magenta, MERSCOPE=rainbow (550 HSV, most_common ranking)
- Blank genes filtered from MERSCOPE (~10K per tile)
- Overlay blend: 50% in-vivo, 50% ex-vivo, 60% dots
- GIF: 12fps, 24-frame hold + 12-frame crossfade per modality
- Brightness boost 1.8× on in-vivo/ex-vivo for cell visibility

**Why:** Twitter/social media post to accompany Nature Science submission, showcasing multimodal registration quality.
**How to apply:** Run `python3 build_twitter_final.py` to regenerate. Adjust TILES and cell_lms lists to change selections.
