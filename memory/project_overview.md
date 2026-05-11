---
name: Margaret Project Overview
description: Multimodal neuroscience project combining in-vivo two-photon, ex-vivo MERSCOPE spatial transcriptomics, and behaviour imaging for mouse hippocampus
type: project
---

## What this project is

Margaret is a **multimodal cell-matching pipeline** for mouse hippocampus (CA1 — pyramidal and oriens layers). The goal is to link functional (in vivo behaviour) and molecular (MERSCOPE spatial transcriptomics) identity of the same neurons.

**Why:** Preparing figures, animations, and videos for a Nature Science publication.

**How to apply:** All visualization/animation work should emphasize the cross-modal registration story — behaviour cells matched to gene expression profiles in the same tissue.

## Scientific workflow

1. **In vivo two-photon imaging** during behaviour → cell segmentation (mice JY306, JY316)
2. **Post-hoc MERSCOPE** spatial transcriptomics on the same tissue (17 regions: reg3-5, 9-22)
3. **Registration** of ex vivo MERSCOPE coordinates to in vivo stack
4. **Cell matching** across modalities — 27 matched neurons confirmed in JY306
5. **Visualization** via napari for publication figures/videos

## Key data

- **In vivo stacks:** `JY306_in_Vivo_stack_flipped_s80.tif`, `JY316_in_Vivo_stack_flipped.tif`
- **Ex vivo registered:** `exvivo_combined.tif` (cell labels), `antirat_combined.tif` (Anti-Rat intensity) — both in JY306 space (16, 658, 629)
- **Native ex-vivo:** `registration_video/exvivo21_merscope17_combined.tif` (4719x5640x3), `registration_video/1.nd2` (12, 2, 4200, 4200)
- **Transcripts:** `transcripts/regN_gene_invivo.tif` (17 regions)
- **Behaviour masks:** `behaviour masks/or_planeN_seg.npy` and `pyr_planeN_seg.npy`
- **MERSCOPE raw:** `merscope/reg_N/mosaic_Anti-Rat_z4.tif`
- **Pickle files:** `2_1_merscope17transformed_*.pkl` (~5.5GB, transforms + 27 matched landmarks)

## Notebook

`Merscope_video/Matching_Antiratmask_behaviour_simulataneous_final.ipynb` — Main visualization notebook using napari. Color coding:
- Red/Green: in vivo stack
- Blue (bop blue): gene transcripts / ex vivo
- Green: behaviour masks
- Magenta: matched cell segmentation masks

## Folder variants

`Merscope_video/` through `Merscope_video 5/` are iterations of data subsets prepared for different video compositions.
