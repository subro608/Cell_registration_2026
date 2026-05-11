# Margaret — Multimodal Cell-Matching Pipeline

Matching individual neurons across in-vivo two-photon, ex-vivo confocal, and MERSCOPE spatial transcriptomics for mouse hippocampus (JY306/JY316).

---

## Coordinate Spaces

| Space | Volume | Dimensions | Pixel Size | Source |
|-------|--------|------------|------------|--------|
| **JY306 s80** | `JY306_in_Vivo_stack_flipped_s80.tif` | (16, 658, 629) | 0.6835 x 0.6835 x 3.0 µm | Two-photon in-vivo |
| **nd2 native** | `registration_video/{tile}/GFP_z*.png` | (12, 4200, 4200) per tile | 0.645 x 0.645 x 2.0 µm | Confocal ex-vivo (22 tiles) |
| **MERSCOPE** | `transformation/*.pkl` `transformed` array | (17, 1734, 1734) per FOV | ~0.108 µm | MERSCOPE spatial transcriptomics |
| **Stitched 1µm iso** | `registration_video/stitched/stitched_gfp_fullres_v5_1um_isotropic.tif` | (516, 3554, 3545) | 1.0 x 1.0 x 1.0 µm | Stitched from 22 nd2 tiles |

---

## Full Landmark Propagation Chain

### Source: PKL transform files (`transformation/*.pkl`)
- From Cliodna's registration tool (MERSCOPE platform)
- `pcd_invivo` — matched cell coords in **JY306 s80 pixel space**
- `pcd_exvivo` — matched cell coords in **JY306 s80 pixel space** (same space, ~1-2px residual)
- `transformations` — 13-14 stage chain (scale → bhat affine → vec_field deformable)
- 19 pkl files, 878 total matched cells across all tiles

### Step 1: JY306 → nd2 native (`make_contact_sheet.py`)

```
pcd_exvivo (JY306 s80)
    │
    ▼  point_inverse_iterative() — reverses all 13-14 pkl stages
    │
merc_ev (MERSCOPE space)
    │
    ▼  M3_inv @ merc_ev — inverse SIFT affine
    │
ev_nd2 (nd2 native, 4200x4200)
```

**Saved:** `registration_video/landmarks_nd2_native_{tile}.npz`
- `ev_nd2` — (N, 3) cell positions in nd2 native pixels (col, row, z_merc)
- `pcd_invivo_jy306` — (N, 3) in-vivo positions in JY306 s80 pixels (z, y, x)
- `pcd_exvivo_jy306` — (N, 3) ex-vivo positions in JY306 s80 pixels (z, y, x)

### Step 2: nd2 native → stitched 1µm iso (`contact_sheet_stitched_v5.py`)

```
ev_nd2 (nd2 native, 4200x4200)
    │
    ▼  IOU rigid warp (pair-wise, from auto_align_transforms_iou_v4.json)
    │
    ▼  Elastix inverse displacement field (from z_stitch_elastix_fullres_v5/)
    │
    ▼  Cumulative IOU to canvas coords (5510x5496 native pixels)
    │
    ▼  × 0.645 (native_xy_um)
    │
stitched_coords (1µm isotropic, 3554x3545)
```

**Saved:** `registration_video/landmarks_stitched_v5_{tile}.npz`
- `stitched_coords` — (N, 3) positions in stitched 1µm iso space (z, y, x)
- `pcd_invivo_jy306` — (N, 3) in-vivo positions in JY306 s80 pixels (z, y, x)
- `ev_nd2` — (N, 3) nd2 native positions
- `cell_nd2_z` — (N,) best nd2 z-slice per cell (brightness search)

### In-vivo side
`pcd_invivo_jy306` is used directly — coordinates are already in JY306 s80 pixel space, which is the native space of `JY306_in_Vivo_stack_flipped_s80.tif`.

---

## Stitching Pipeline (v5)

### Step 1: Pairwise alignment (`auto_align_iou_v4.py`)
- 3-stage IOU rigid: coarse (±20°, ±400px, ds=4) → medium (±3°, ±40px, ds=2) → fine (±1°, ±16px, ds=1)
- 20 consecutive pairs, v4 masks from CSV annotations
- Output: `registration_video/auto_align_transforms_iou_v4.json`

### Step 2: Deformable registration (`elastix_fullres_v5.py`)
- Full-res (4200x4200) elastix B-spline for all 20 pairs
- Grid=64, 1000 iter, AdvancedNormalizedCorrelation
- Output: `png_exports/z_stitch_elastix_fullres_v5/` (TransformParameters per pair)

### Step 3: Build stitched volume (`build_stitched_fullres_v5.py`)
- Apply IOU rigid (full res) → elastix B-spline per tile
- Stack in z on canvas (5510x5496 native pixels, 258 z-slices)
- Resample to 1µm isotropic: (258, 5510, 5496) @ 0.645µm → (516, 3554, 3545) @ 1µm
- Output: `registration_video/stitched/stitched_gfp_fullres_v5_1um_isotropic.tif` (15.6 GB)
- Params: `registration_video/stitch_v5_params.json`

---

## 3D Visualization

### Single-volume viewers (ex-vivo only)
| Viewer | Script | Key Features |
|--------|--------|-------------|
| `viewer.html` | (original) | 4x DS, /4000 norm, 8.7MB |
| `viewer_equalized.html` | `build_viewer_equalized.py` | Per-slice equalized + GP RBF interpolation, interactive sliders |
| `viewer_stitched_v5.html` | `build_viewer_stitched_v5.py` | Stitched v5 volume, per-slice equalized + GP RBF |
| `viewer_gapfree_fullz.html` | `build_gapfree_viewer.py` | Full 504 z-slices, 8x DS |

### Dual-volume viewer (ex-vivo + in-vivo)
| Viewer | Script | Key Features |
|--------|--------|-------------|
| `viewer_dual_3d.html` | `build_viewer_dual_3d.py` | **LATEST**: Side-by-side stitched ex-vivo + JY306 in-vivo, 878 matched cell lines |

#### `viewer_dual_3d.html` details:
- **Ex-vivo**: Stitched v5 (516, 3554, 3545) → DS4 (129, 888, 886). Per-slice equalized, slice PNGs + JS-side GP RBF interpolation
- **In-vivo**: JY306 s80 (16, 658, 629) → resampled to 1µm iso (48, 450, 430). Median filter background subtraction, sparse voxels
- **Physical proportions**: Shared global scale. Ex-vivo ~3554µm, in-vivo ~450µm (true 2P FOV vs full hippocampus)
- **Interaction**: Hover lines (yellow glow + endpoint spheres), click to select (green glow + MIP patches below), click empty to deselect all
- **Controls**: Separate opacity/size per volume, threshold, GP length/interp, tile selector, colormap picker
- **Default config**: threshold 8, ex opacity 26, iv opacity 5, pt size 1, GP length 10, interp 2, tile row1_3

---

## Contact Sheets

### Per-tile cell matching (`make_contact_sheet.py`)
- 19 tiles with pkl transforms
- SIFT affine (nd2 ↔ MERSCOPE) + iterative pkl inverse (JY306 → MERSCOPE)
- Green crosshairs on matched cells
- Output: `contact_sheet/single_z/` and `contact_sheet/mip_pm2/`

### Stitched verification (`contact_sheet_stitched_v5.py`)
- Propagates landmarks through IOU+elastix chain to stitched space
- Loads stitched volume + JY306 in-vivo for side-by-side patches
- Output: `contact_sheet/stitched_v5/`

---

## Z-Stitch Comparison (v5)

`comparison_collage_v5.py` — 5-column collage per pair:
1. Raw + annotation contours (centroid shift, no masking)
2. IOU Rigid
3. Farneback v3
4. Elastix 500 iter
5. Elastix 2000 iter

Output: `png_exports/z_stitch_comparison_v5/` + `results_v5.json`

---

## Masks

- **CURRENT**: `registration_video/via_masks_v4.npz` — from CSV polylines via `cv2.fillPoly` (`generate_masks_from_csv.py`)
- Source: `via_annotations/via_export_csv.csv`
- **DO NOT USE** auto-threshold masks from `generate_masks_from_images.py`

---

## Key Files

| File | Location | Description |
|------|----------|-------------|
| Stitched ex-vivo (native) | `registration_video/stitched/stitched_gfp_fullres_v5.tif` | (258, 5510, 5496) uint16 |
| Stitched ex-vivo (1µm iso) | `registration_video/stitched/stitched_gfp_fullres_v5_1um_isotropic.tif` | (516, 3554, 3545) uint16 |
| In-vivo (s80) | `JY306_in_Vivo_stack_flipped_s80.tif` | (16, 658, 629) float64 |
| Stitch params | `registration_video/stitch_v5_params.json` | Cumulative IOU transforms, z-offsets, canvas offsets |
| IOU transforms | `registration_video/auto_align_transforms_iou_v4.json` | Pairwise rigid transforms |
| V4 masks | `registration_video/via_masks_v4.npz` | Binary masks per tile |
| PKL transforms | `transformation/*.pkl` | 19 files, MERSCOPE ↔ JY306 transforms |
| Landmarks (nd2) | `registration_video/landmarks_nd2_native_*.npz` | 19 tiles, nd2 + JY306 coords |
| Landmarks (stitched) | `registration_video/landmarks_stitched_v5_*.npz` | 19 tiles, stitched + JY306 coords |

---

## Tile Ordering (sequential z-sections)

```
row1/1 → row1/2 → row1/3 →
row2/1 → row2/2 → row2/3 → row2/4 → row2/5 →
row3/1 → row3/2 → row3/3 → row3/4 → row3/5 → row3/6 →
row4/1 → row4/2 → row4/3 → row4/4 → row4/5 → row4/6 →
row5/1
```
22 tiles, 12 z-slices each = 264 native z-slices = 528µm depth (at 2µm z-step).
Row1_3→Row2_1 gap: 6-slice black slab (12µm).

---

## Important Notes

- `invivo_1um_isotropic.tif` (189, 1229, 1177) is from a DIFFERENT larger source volume (63, 1798, 1720), NOT from s80. Do not use for landmark visualization.
- Elastix requires `/usr/bin/python3` (SimpleITK not in venv)
- All v4+ scripts load PNGs (GFP_z000.png / GFP_z011.png), NOT nd2 directly — avoids OOM
- Normalize 3D viewers by /4000, not /p99
- Full-res (3554×3545) 3D viewer is too slow for browser GP interpolation
