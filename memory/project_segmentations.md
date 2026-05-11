---
name: Cell Segmentation Inventory
description: All cellpose segmentation files — 17 ex-vivo MERSCOPE regions (899 cells), 6 behaviour masks (256 cells), 1 native combined seg (44 cells)
type: project
---

## Ex-vivo MERSCOPE segmentations (17 regions)

Location: `exvivo/exvivoNN_merscopeNN_invivo.seg.npy`
Format: cellpose .seg.npy, keys: filename, masks, contours_3d
Shape: all (19, 578, 599) float32 — **JY316 space**
Total: **899 cells** across 17 regions (range: 11–99 per region)

Duplicated in `Merscope_video/exvivo/`.

Top regions by cell count: exvivo41_merscope5 (99), exvivo42_merscope4 (95), exvivo35_merscope12 (88), exvivo34_merscope11 (71), exvivo43_merscope3 (70).

## Behaviour masks (6 files)

Location: `behaviour masks/{or,pyr}_plane{0,1,2}_seg.npy`
Format: cellpose .seg.npy, keys: filename, masks, contours_3d, chan_choose, ismanual, flows
Shape: all (1, 19, 578, 599) float32 — **JY316 space** (note extra leading dim)
- Oriens: 16 + 38 + 29 = **83 cells**
- Pyramidal: 45 + 85 + 43 = **173 cells**
- Total: **256 behaviour-labelled cells**

Duplicated in `Merscope_video/behaviour masks/`.

## Native pre-registration segmentation

Location: `registration_video/exvivo21_merscope17_combined_seg.npy`
Format: full cellpose output (outlines, colors, masks, flows, model_path, etc.)
Shape: (4719, 5640) uint16 — native MERSCOPE resolution
Cells: **44**

## Registered combined label map

`jy306_registered files/exvivo_combined.tif` — (16, 658, 629) uint64, max=771 — cell LABELS in JY306 space (NOT intensity).

**Why:** Understanding which segmentations exist and their coordinate spaces is critical for any cell-matching or animation work.

**How to apply:** Ex-vivo and behaviour segs are in JY316 space (578×599), not JY306 (658×629). The native combined seg is at full MERSCOPE resolution (4719×5640). Always check which coordinate space a mask is in before overlaying.
