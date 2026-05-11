# Workflow Audit — 4-Modality & Registration Viewers

**Date:** 2026-04-07
**Scripts audited:**
- `build_viewer_dual_3d_v5.py` (4-Modality viewer → `dual_v5.html`)
- `build_viewer_warped_invivo_3d_v5.py` (Registration viewer → `warped_v5.html`)
- Landmark/transform data files

---

## CRITICAL Issues

### 1. Trailing space in `merscope_exvivo ` directory
- **Location:** Both scripts, line ~29
- **Code:** `PKL_MERC_DIR = os.path.join(BASE, 'merscope_exvivo ')`
- **Impact:** Works now because the directory literally has a trailing space on disk. But any copy, rename, archive, or git operation could silently strip the space, causing ALL MERSCOPE loading to fail with no error. Every `os.path.isdir(PKL_MERC_DIR)` would just return False.

### 2. MERSCOPE dots skip pair elastix in warped_v5
- **Location:** warped build, lines 679-684
- **Code:** MERSCOPE dots use only `cum_iou[tile]` (cumulative IOU transform). No pair-wise rigid or pair elastix deformation is applied.
- **Compare:** In-vivo warping goes through full `stitch_tile_v5()` (pair IOU + elastix + cumulative placement).
- **Impact:** MERSCOPE dots may be offset by several pixels vs other modalities in the Registration viewer. The pair elastix correction is skipped entirely.

### 3. ~66 GB calcium warp cache (OOM risk)
- **Location:** warped build, lines 1141-1147
- **Code:** Caches all warped calcium frames (4200x4200 uint8) per tile. ~220 frames x 17.6 MB x up to 17 tiles = ~66 GB.
- **Impact:** Likely OOM on most machines. In practice, not all tiles may have pkl transforms, reducing the count, but still a major memory concern.

---

## HIGH Issues

### 4. Three incompatible gene color schemes in dual_v5
- **Location:** dual build, lines 296 vs 409 vs 525
- **Details:**
  - **3D cloud:** global `most_common()` ranking (frequent genes = warm red/orange)
  - **Patch background gene_to_bgr:** alphabetical sorting (line 409-413) — dead code but confusing
  - **Per-cell patch dots:** local `most_common()` per crop (line 525-528) — this is what actually renders
- **Impact:** Same gene gets different colors in 3D cloud vs patch dots. The 3D cloud uses global ranking, patches use per-cell ranking. For Nature publication, gene colors should be consistent.

### 5. Per-frame calcium normalization kills temporal dynamics
- **Location:** dual build, lines 362-364
- **Code:** `p99 = np.percentile(warped[warped > 0], 99)` per frame, then normalize to 255.
- **Impact:** Each frame independently normalized to its own p99. A bright calcium transient frame and a dim baseline frame will look identical in intensity. Defeats the purpose of calcium activity visualization. Should use a single global p99 across all frames.

### 6. 26 landmarks with >10 um PKL error
- **Location:** Landmark data, predominantly row4/row5 (ventral tiles)
- **Worst offenders:**
  - row4_2 lm#7: **31.35 um**
  - row5_1 lm#12: **26.56 um**
  - row4_2 lm#6: **25.36 um**
  - row5_1 lm#13: **25.06 um**
  - row4_3 lm#2: **17.75 um**
  - row4_4 lm#61: **16.59 um**
- **Impact:** Patches at these landmarks show wrong locations. The PKL deformation field is less reliable in ventral tiles. These landmarks should be flagged or excluded from the filtered set.

### 7. Silent elastix failure
- **Location:** warped build, line 127
- **Code:** `except Exception: sl_deformed = sl_rigid`
- **Impact:** Any elastix failure (corrupt transform, version mismatch, missing file) silently falls back to rigid-only result. No warning printed. Could produce misaligned tiles with no diagnostic output.

---

## MEDIUM Issues

### 8. MERSCOPE affine ignores z-coupling terms
- **Location:** warped build, lines 662-664
- **Code:** Only uses `R3i[2,1]`, `R3i[2,2]`, `R3i[1,1]`, `R3i[1,2]` — drops the [0] index (z contribution to x/y).
- **Impact:** If the pkl affine has any z-rotation coupling, this introduces systematic x/y offset in MERSCOPE dot positions. Acceptable if z-coupling is negligible.

### 9. ivz_hi hardcoded to 15
- **Location:** warped build, line 1060
- **Code:** `ivz_hi = min(15, z_iv + DZ_SLICES)`
- **Impact:** Assumes 16 in-vivo z-slices. The actual count `nz_iv` (loaded at line 268) is not used. If JY306 has a different number of slices, the z-range metadata in `cell_info_js` is wrong.

### 10. MERSCOPE dots all at single z per tile
- **Location:** warped build, lines 688-689
- **Code:** `z_um = (z_offset + 6) * ND2_Z_UM` — every transcript placed at tile middle (z-slice 6 of 12).
- **Impact:** MERSCOPE data has z-position info (`global_z` in CSV) but it's not used. Dots form flat 2D layers per tile in 3D, not a true 3D distribution.

### 11. Normalize-then-denormalize intensity distortion
- **Location:** warped build, lines 111-124
- **Code:** Input to elastix is percentile-stretched to 0-255, output is denormalized back using original percentiles.
- **Impact:** Values above p99.5 get clipped to 255 before elastix, then denormalized back to p99.5. Dynamic range above p99.5 is permanently lost. Values below p2 are also compressed.

### 12. Assumes isotropic MERSCOPE scaling
- **Location:** dual build, lines 394-395
- **Code:** `sc_m = m2m[0, 0]` used for both x and y transformations.
- **Impact:** If `micron_to_mosaic` has different x/y scales (`m2m[0,0] != m2m[1,1]`), the y-axis transformation is wrong. The 0.108 um/px value suggests isotropic, but this is an assumption.

### 13. `or` chain treats 0.0 as falsy
- **Location:** dual build, lines 93, 96
- **Code:** `EX_SZ = extract_const('EX_SZ') or 0.195`
- **Impact:** If the extracted value is `0.0` (a valid float), Python's `or` treats it as falsy and falls through to the hardcoded default. Unlikely for scaling factors but violates the "no arbitrary constants" principle.

### 14. Asymmetric boundary crops for edge landmarks
- **Location:** dual build, lines 457-458
- **Code:** `y0, y1 = max(0, y_nd2 - cr), min(page.shape[0], y_nd2 + cr)`
- **Impact:** Edge landmarks get clipped crops stretched to PATCH_SZ. The MERSCOPE dot canvas uses full `cr*2` size regardless, so MERSCOPE and ex-vivo patches have different effective FOVs at boundaries.

---

## LOW Issues

### 15. Column clipped against height
- **Location:** warped build, line 355
- **Code:** `c = int(round(np.clip(x, 10, nd2_h - 11)))` — should be `nd2_w - 11`
- **Impact:** Works only because tiles are square (4200x4200). Would break for non-square tiles.

### 16. Gaussian z-crop off-center by 1px
- **Location:** warped build, line 357
- **Code:** `nd2_slices[z][r-10:r+10, c-10:c+10]` gives 20x20 crop, center pixel at position [10] of 20 — off by 0.5px.

### 17. Weak-signal z-fit returns noise argmax
- **Location:** warped build, line 144
- **Code:** When total signal < 1e-6, returns `argmax(intensities)`. If all values are nearly identical, argmax is noise-dependent.

### 18. Nearest-neighbor z-resampling
- **Location:** warped build, lines 497-503
- **Code:** `z_int = int(round(z_native))` — no interpolation between adjacent z-slices.
- **Impact:** Z-axis has staircase artifacts. Every other output slice maps to the same input slice (nd2 z=2um, output z=1um).

### 19. Calcium 2D cloud z-jitter scales with image size
- **Location:** dual build, line 184
- **Code:** `cal_vz = rng_cal.uniform(-0.02, 0.02, n_cal_vox) * max(cal_w, cal_h)`
- **Impact:** For 512x512 movie, z-jitter is [-10.24, 10.24]. Minor cosmetic issue.

### 20. `landmarks_nd2_native_row2_1.npz` missing
- **Location:** `registration_video/` directory
- **Impact:** Only tile missing from nd2_native set. Fallback to `landmarks_stitched_v5_row2_1.npz` works but has different key structure (no `iv_nd2`, has `stitched_coords` and `cell_nd2_z` instead).

---

## Data Integrity Summary

| Item | Value |
|------|-------|
| Total landmarks | 878 across 19 tiles |
| Tile range | row1_3 (13 lm) to row4_3 (73 lm) |
| Missing nd2_native npz | row2_1 only (stitched_v5 fallback) |
| pkl_transform files | All 19 tiles present, consistent keys |
| M2d scale range | ~2.67 (dorsal/row1) to ~2.40 (ventral/row5) |
| Movie-to-JY306 affine | scale 0.88, NCC=0.76 at z=3 |
| MERSCOPE pkl files | 22 files (rows 1-5) |
| High-error landmarks (>10um) | 26, mostly row4/row5 |

---

## Coordinate Space Verification

### Registration viewer (warped_v5.html) — All in ex-vivo stitched space
| Modality | 3D Cloud | Patches | Transform Chain |
|----------|----------|---------|-----------------|
| Ex Vivo | Native stitched GFP (DS4) — reference frame | nd2 crop at (x_nd2, y_nd2) +/-78px | None (reference) |
| In Vivo | JY306 -> M2d -> IOU -> elastix -> ex-vivo grid | JY306 slice warped to nd2 via M2d, cropped at nd2 coords | M2d_jy306_to_nd2 + stitch_tile_v5 |
| Calcium | movie -> M_m2j -> M2d -> IOU -> elastix -> ex-vivo grid | Frames warped movie->JY306->nd2 via composed affine, cropped at nd2 coords | M_m2j x M_j2n + stitch |
| MERSCOPE | microns->mosaic->fliplr->x0.108->pkl affine inv->nd2->cum_iou->ex-vivo grid | Dots in nd2 space at (x_nd2, y_nd2) +/-78px | Full transform chain (but NO pair elastix for 3D) |

### 4-Modality viewer (dual_v5.html) — Each in native space
| Modality | 3D Cloud Space | Patches |
|----------|---------------|---------|
| Ex Vivo (Q0) | ND2 native/stitched (DS5) | nd2 crop +/-130px |
| In Vivo (Q1) | JY306 native (unwarped) | JY306 crop +/-62px |
| Calcium (Q2) | Native movie space (temporal std + z jitter) | Animated frames in JY306 space +/-62px |
| MERSCOPE (Q3) | ND2 stitched space (transformed from microns) | Dots in nd2 space +/-130px |

---

## Z-Matching Method

Per tile, per landmark:
1. Sample 20x20 patch intensity across 12 nd2 z-slices at landmark (x, y)
2. Fit Gaussian to intensity profile -> `z_nd2` (sub-slice precision)
3. Landmark's in-vivo z from `pcd_iv[i, 0]` (JY306 z-slice index from landmark file)
4. Group all landmarks by rounded `z_nd2` (0-11)
5. For each `z_nd2`, take median of all corresponding `z_iv` values
6. Unmapped z-slices use nearest mapped z (nearest-neighbor fill)
7. For each `z_nd2` in 0-11, pull matched `z_iv` JY306 slice, warp with M2d to nd2 space

---

## Transform Files

| File | Location | Shape | Purpose |
|------|----------|-------|---------|
| `pkl_transform_{tile}.npz` | `png_exports/registration_per_tile_pkl/{tile}/` | M2d: (2,3) | JY306 -> nd2 per-tile 2D affine |
| `movie_avi_to_jy306_affine.npz` | `animation/` | M_affine: (2,3) | Movie -> JY306 SIFT affine |
| `landmarks_nd2_native_{tile}.npz` | `png_exports/registration_video/` | ev_nd2: (N,3) | Per-tile landmark coords in nd2 |
| `landmarks_stitched_v5_{tile}.npz` | `png_exports/registration_video/` | ev_nd2: (N,3) | Per-tile landmark coords (stitched, fallback) |
| MERSCOPE pkl files | `merscope_exvivo /` | varies | Per-tile MERSCOPE->nd2 deformation fields |
