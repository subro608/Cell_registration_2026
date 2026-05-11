# Cell Registration 2026

Cross-modality neuron re-identification for mouse hippocampus — pairing the same cell across in-vivo two-photon calcium, ex-vivo confocal, and MERSCOPE spatial transcriptomics.

Two complementary codebases live here:

| Subdirectory | What it is |
|---|---|
| [`jy306_data/`](jy306_data/) | Working pipeline for the JY306/JY316 datasets — registration, stitching, 4-modality 3D viewer, Nature Science animation. See [`jy306_data/README.md`](jy306_data/README.md). |
| [`cellinvariance/`](cellinvariance/) | Published method (Erdem Varol's lab): frozen DINOv2 features + MLP projector trained with SupCon + Sinkhorn for landmark-free cross-modality re-ID. Standalone reproducible eval on the Sparrow_3 / zstack pair. See [`cellinvariance/README.md`](cellinvariance/README.md). |
| [`memory/`](memory/) | Project notes accumulated across iterations — 47 files of project state, feedback, conventions, and gotchas. |

---

## Quick map

**jy306_data** is the experimental pipeline. The full landmark chain is:

```
JY306 in-vivo (658x629x16, 0.6835x0.6835x3.0 µm)
   |  pkl inverse  (per-tile 14-stage scale -> bhat -> vec_field)
MERSCOPE space (1704x1704 canvas)
   |  inv(SIFT 2D affine)
nd2 native (4200x4200 per tile, 0.645x0.645x2.0 µm, 22 tiles)
   |  pair IOU rigid + elastix B-spline + cumulative IOU
Stitched 1 µm isotropic (516, 3554, 3545)
```

End products:
- **3D viewer** — 4-modality 2x2 grid (ex-vivo confocal / in-vivo z-stack / in-vivo calcium / MERSCOPE gene dots) deployed at `invivo-exvivo-cell-registration.vercel.app`. Built by `build_v5_website.py` -> `build_viewer_dual_3d_v5.py` -> `make_v4_deploy.py`.
- **Animation** — scene-by-scene screenplay video for the Nature Science figure. Built by per-scene asset scripts under `jy306_data/animation/` -> `make_registration_animation_v7.py` -> `animation/edit_combined_video.py`. Screenplay in [`jy306_data/animation_script_v3.md`](jy306_data/animation_script_v3.md), pipeline writeup in [`jy306_data/website_video_pipeline.md`](jy306_data/website_video_pipeline.md).

**cellinvariance** is the published cross-modality re-ID model. Best config (`w8`) reaches LOOCV R@5 = 0.25 on the Sparrow_3/zstack landmark pair; within-modality bal-knn = 99.8%. The bridge is the open problem.

---

## Conventions

- **Color**: in-vivo = green, ex-vivo = magenta (Jason's convention, applied in 3D clouds, panels, viewers, captions).
- **Captions**: `IN VIVO`, `EX VIVO`, `MERSCOPE mRNA` — no hyphens, no "confocal" with in vivo.
- **MERSCOPE gene colors**: ordered by `Counter.most_common()`, stored BGR, converted to RGB only at output.
- **Display vs. registration**: ex-vivo display is raw intensity; median filter is for registration *input* only.

More in [`memory/feedback_*.md`](memory/).

---

## Repository scope

Only `.py` source and `.md` documentation/notes are tracked. Raw data (`.pkl`, `.tif`, `.npz`, `.mp4`, etc.) and derived artifacts (`.png`, `.html`, `.json`, `.bin`) are excluded via `.gitignore` — they live on the HPC filesystem next to the code.

- 187 `.py` (178 jy306 + 9 cellinvariance)
- 48 `.md` in `memory/` plus per-subdir READMEs
