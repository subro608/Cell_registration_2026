---
name: nd2 ↔ MERSCOPE ↔ JY306 Pipeline
description: Correct 3-space transform chain, SIFT affine for nd2↔MERSCOPE, iterative pkl inversion, and accuracy limits
type: project
---

## Key Discovery (2026-03-29)
The pkl transforms do NOT map nd2 directly to JY306. There is a missing intermediate space:

```
nd2 (4200) → [affine] → MERSCOPE native (1627) → [pkl stages] → JY306 (658×629)
```

**Why:** Originally assumed pkl input = nd2 scaled by 4200/1704, giving 90px error. The actual input is MERSCOPE native space. The `exvivo_merscope_combined/*.tif` files ARE the nd2 tiles already warped into MERSCOPE space — that's the pkl input.

## Correct pkl Point Inversion (JY306 → MERSCOPE native)
- Feed raw JY306 coords directly (no canvas scaling!) in **(z, y, x)** order
- Apply backward mapping of each stage in **reverse order** (last stage first)
- `scale`: `pt = pt / val`
- `bhat`: `pt = (pt - t) @ R_inv` where R=bhat[:3,:], t=bhat[3,:], coords are (z,y,x)
- `vec_field_total`: **Use iterative fixed-point inversion** (not single-step subtraction)
  - Solve `p_out = p_in + vf[p_in]` iteratively: `p_in^{k+1} = p_out - vf[p_in^k]`
  - Converges in ~5-10 iterations to machine precision (0.000000 px round-trip)
  - Single-step (`pt = pt - vf[pt]`) gives ~0.15px error — works but iterative is free accuracy
- Accuracy limit: **~5px in MERSCOPE space** (pkl registration quality, NOT inversion error)

## nd2 ↔ MERSCOPE Affine — USE SIFT, NOT ECC
- **ECC failed for 9/12 tiles**: converged to wrong local minima with distorted scales (up to 20% off) and spurious rotations (up to -21°). Only row2_1 was correct.
- **SIFT feature matching** works perfectly for all 12 tiles: pure scale 0.3873, zero rotation, sub-pixel translation, 1200-1800 RANSAC inliers each.
- Files: `registration_video/affine_nd2_to_merscope_ecc_{tile}.npy` (overwritten with SIFT results)
- Also saved as `affine_nd2_to_merscope_sift_{tile}.npy`
- Method: SIFT(5000 features) → BFMatcher → Lowe ratio 0.7 → estimateAffine2D(RANSAC, 5px)
- Applied between nd2 MIP downsampled to 1627 and exvivo_merscope_combined MIP

## Full Pipeline Accuracy
| Stage | Error |
|-------|-------|
| pkl inverse (JY306→MERSCOPE) | ~5 px in MERSCOPE space (registration quality limit) |
| SIFT affine (MERSCOPE→nd2) | ~0 px (pure scale, sub-pixel) |
| Total (JY306→nd2) | ~13 px in nd2 space (~8 µm) — dominated by pkl registration |

## What Didn't Help
- **ECC image correlation**: diverges to wrong local minima for most tiles
- **Farneback optical flow** on top of affine: ~0px improvement
- **Elastix B-spline** (grid 16/32/64): slightly worse than affine alone
- **Per z-slice ECC**: diverges to bad local minima, MIP-based is more stable
- **Iterative pkl inverse**: mathematically perfect but only shifts positions ~0.4px in nd2 space

## PKL Files (from Cliodna's registration tool)
- `pcd_invivo` and `pcd_exvivo` are BOTH in JY306 s80 pixel space (16, 658, 629) @ 0.6835µm XY, 3µm Z
- Differences between them are ~1-2px (registration residual)
- 19 pkl files in `transformation/` directory

## Contact Sheets — make_contact_sheet.py: JY306 → nd2 native
- Generated for all 12 tiles: `png_exports/coarse_registration/contact_sheet/row{tile}_contact_sheet.png`
- Pipeline:
  1. Load pkl → get `pcd_invivo`, `pcd_exvivo` (both JY306 s80 space)
  2. SIFT affine between nd2 MIP and MERSCOPE MIP → `M3` (nd2→MERSCOPE), `M3_inv` (MERSCOPE→nd2)
  3. `point_inverse_iterative(pcd_ev, transforms)` → `merc_ev` (JY306 → MERSCOPE via pkl inverse)
  4. `M3_inv @ merc_ev` → `ev_nd2` (MERSCOPE → nd2 native 4200×4200)
- Saves: `registration_video/landmarks_nd2_native_{tile}.npz` with keys: `ev_nd2`, `pcd_invivo_jy306`, `pcd_exvivo_jy306`
- Note: ex-vivo GFP confocal shows dense neuropil, not clear cell bodies like JY306 in-vivo. Visual verification is limited by modality difference.

## Stitched Contact Sheets — contact_sheet_stitched_v5.py: nd2 native → stitched 1µm iso
1. Load `ev_nd2` from landmarks_nd2_native files (nd2 native 4200×4200 coords)
2. `propagate_point_to_stitched()`:
   - IOU rigid warp (pair-wise)
   - Elastix inverse displacement field
   - Cumulative IOU to canvas coords (5510×5496 native pixels)
   - ×0.645 to convert to 1µm isotropic
3. Saves: `registration_video/landmarks_stitched_v5_{tile}.npz` with keys: `stitched_coords`, `pcd_invivo_jy306`, `ev_nd2`, `cell_nd2_z`

## Full Ex-Vivo Landmark Chain
```
pcd_exvivo (JY306 s80) → pkl inverse → MERSCOPE → SIFT inverse → nd2 native (4200×4200)
  → IOU rigid → elastix → canvas (5510×5496) → ×0.645 → stitched 1µm iso (3554×3545)
```

## In-Vivo Side
- `pcd_invivo` (JY306 s80) → used directly (landmarks are in s80 pixel space, same as JY306_in_Vivo_stack_flipped_s80.tif)

## z-Correspondence (row2_1)
- All 3 MERSCOPE z-slices best match nd2 z≈9 (NCC peaks at z=8-10)
- No z-dependent XY drift → 2D MIP-based affine is sufficient
