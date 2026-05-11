---
name: 3D Viewer Build Parameters
description: How to correctly build the 3D HTML point cloud viewer — normalization must use /4000, not /p99; subsampling not block averaging
type: feedback
---

When building 3D HTML viewers from the stitched volume, normalize by dividing by **4000** (not by p99 of nonzero).

**Why:** Using p99 (~788) makes nearly all pixels bright, washing out individual cells into a blurry blob. The /4000 normalization keeps most pixels dark so only the brightest cells punch through — producing the sharp, distinct cell appearance seen in viewer.html. This was confirmed by decoding actual PNG slices from the original viewer.html (slice mean=2.2, p99=48 at 8-bit).

**How to apply:**
- `vol_u8 = np.clip(vol / 4000 * 255, 0, 255).astype(np.uint8)` — correct
- `vol_u8 = np.clip(vol / p99 * 255, 0, 255).astype(np.uint8)` — WRONG, too bright
- Downsampling method: subsampling (`vol[::4, ::4, ::4]`) works. Block averaging smooths out cell peaks.
- Original viewer.html: 4x downsample all axes → 129×687×687, ASPECT_Z = 0.187773
- Rotation: object rotation (points.rotation.y/x), NOT camera orbit
- Y-axis flipped: `pos[i*3+1] = -(v.y/NY - 0.5) * 2`
- Material: `size: ps*0.01, AdditiveBlending, depthWrite:false`, no sizeAttenuation
- Default: threshold=15, opacity=30, pointSize=2

### GP Regression (proper implementation in viewer_equalized.html)
- Per-slice intensity equalization BEFORE normalization (scale each slice's nonzero mean to global mean)
- True GP with RBF kernel: `μ(z*) = k(z*, Z) · K⁻¹ · y` where K_ij = exp(-(z_i-z_j)²/2l²) + σ_n²δ_ij
- K is 129×129 (one per z-grid) — factorize via Cholesky ONCE, shared across all (x,y) columns
- Interpolated z-positions inserted between originals (interp factor 1-4×)
- JS code: store raw data as Uint8Array, recompute GP weights on slider change, apply per-column
- Naive RBF spreading (viewer_rbf.html) is NOT true GP — just copies points at ±dz with Gaussian weight

### Dual viewer lessons (build_viewer_dual_3d.py)
- Pre-extracted sparse voxels look noisy/grainy — always prefer slice PNGs + JS-side GP interpolation for quality
- DS8 gives far too few voxels (53K) — use DS4 minimum to match viewer_equalized quality (~400K+ voxels)
- For in-vivo two-photon: median filter background subtraction (size=15) then threshold — removes neuropil haze
- When showing two volumes at physical scale: use shared global_max, NOT independent scaling per volume
- Hit detection radius for line selection: 0.12 is good; 0.25 is too large (can't deselect); 0.08 is too small
- Click on empty space must clear all selections — don't only handle clicks that hit a line
- Full res (3554×3545 per slice) is too slow for browser GP: ~12.6M columns, ~100MB HTML, ~1.6GB RAM
