---
name: Via Masks for Video Rendering
description: V4 binary tissue masks (21 tiles, 4200×4200) needed for scene5b stacked tile view — apply before stacking to show real tissue shape
type: project
---

## Mask Files
- **Current**: `registration_video/via_masks_v4.npz` — 21 tiles, 4200×4200, uint8 binary
- **Source annotations**: `via_annotations/via_export_csv.csv` (VIA polylines)
- **Generation script**: `generate_masks_from_csv.py`

## Why needed for video
Scene 5b stacks tiles at real XY+Z positions from stitched canvas. Without masks, rectangular crops overlap badly. Masks show actual tissue boundaries per tile.

## How to apply
Load mask per tile from npz, crop to same bounds (crop_x0:crop_y1), downscale to (cell_w, cell_h), multiply with overlay slices before rendering. Should be applied during asset generation or at render time.
