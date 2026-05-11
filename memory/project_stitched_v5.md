---
name: Stitched V5 Pipeline
description: Full-res elastix stitching (DS=1), landmark propagation to stitched space, and stitched vs in-vivo contact sheets
type: project
---

## Full-Res Elastix Stitching (v5, 2026-03-30)
Built a new stitched volume at full resolution (no DS=2) with elastix B-spline deformation.

**Why:** Erdem requested contact sheets showing stitched/masked ex-vivo images with landmarks propagated through the elastix deformation, to verify matches in the stitched space before proceeding to 3D affine + deformable cross-modality registration.

**How to apply:** Use v5 stitched volumes and stitch_v5_params.json for any future landmark propagation or visualization in stitched space.

## Pipeline
1. `elastix_fullres_v5.py` — Elastix B-spline at 4200×4200 (grid=64, 1000 iter) for all 20 consecutive pairs. ~7s/pair, all succeeded.
2. `build_stitched_fullres_v5.py` — Per tile: pair IOU rigid → pair elastix (transformix) → cumulative placement on canvas → stack z → resample to 1µm isotropic
3. `contact_sheet_stitched_v5.py` — Propagate landmarks through same chain, extract patches from stitched 1µm volume

## Output Files
- `stitched_gfp_fullres_v5.tif` — (258, 5510, 5496) native res, 15.6 GB
- `stitched_gfp_fullres_v5_1um_isotropic.tif` — (516, 3554, 3545) 1µm iso. **Load with TiffFile page loop, not imread** (saved as per-page tif)
- `stitch_v5_params.json` — tile z-offsets, cumulative IOU transforms, canvas offset
- `landmarks_stitched_v5_{tile}.npz` — propagated landmarks per tile (stitched_coords, pcd_invivo_jy306)
- `contact_sheet/stitched_v5/` — 19 contact sheets

## Key Differences from Previous Stitch
| | Old (elastix v4) | New (v5) |
|---|---|---|
| Elastix resolution | DS=2 (2100×2100) | DS=1 (4200×4200) |
| Grid spacing | 32px at DS=2 | 64px at full res |
| Iterations | 500 | 1000 |
| Native volume | (258, 2100, 2100) | (258, 5510, 5496) |
| 1µm isotropic | (516, 2748, 2748) | (516, 3554, 3545) |
| Canvas larger because cumulative IOU rotations expand bounds |

## Note on Elastix Point Propagation
The elastix deformation is applied to images via transformix, but for POINTS the inverse direction is needed (which transformix doesn't directly provide). Current landmark propagation uses IOU rigid only for XY mapping (elastix deformation is ~10-20px, small vs cell spacing). If higher precision needed, could implement B-spline evaluation from TransformParameters.
