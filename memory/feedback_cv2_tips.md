---
name: CV2 and Animation Feedback
description: User corrections about animation quality and cv2 usage
type: feedback
---

- Don't just crossfade between images — actually warp/deform with cv2.warpAffine using interpolated affine matrices. User said "u r not warping properly" when given a simple crossfade.
- cv2.remap requires float32 maps — numpy operations produce float64, must cast with `.astype(np.float32)`.
- User prefers side-by-side comparisons and red/green translucent overlays for verifying registration quality.
- User wants to see ALL z-slices transformed, not just a subset.
- **Calcium movie display**: use raw TIF divided by global max — do NOT use norm_u8, green LUT, or contrast boost. Keep **grayscale**.
- **In-vivo display**: ALWAYS use **hot colormap** (cv2.COLORMAP_HOT) — red/hot everywhere invivo appears, including overlays. Never green for invivo.
- **Confocal z-stack / ex-vivo display**: use **green** channel.
- **Transition style**: when showing two images aligning, the second image should **emerge/fade in place** on top of the first (showing cell matching), NOT slide in from the side.
- **Avoid jarring resizes**: do warp interpolation in **display space** (1920×1080 canvas) so image shrinks smoothly rather than jumping sizes. Pre-render **1 image per frame** at the exact target size — don't interpolate between a few pre-cached sizes, render all of them.
- **Animation format**: 1920×1080, 24fps, black background. Match `registration_animation_v4.mp4` style.

**Why:** The user is building publication-quality animations for Nature Science and expects proper spatial transformations, not visual shortcuts. The v4 animation is the reference style.

- **Ex-vivo visibility during warp**: When warping in-vivo onto ex-vivo, keep the ex-vivo at full opacity throughout — never let it disappear or fade out. The user needs to see the target the whole time.
- **M2d_jy306_to_nd2 is (x,y) convention**: Directly usable by cv2.warpAffine — no row/column swap needed. pcd_invivo_jy306 is (z,y,x), ev_nd2 is (x,y,z).
- **landmarks.npz (4 manual pts)** maps nd2_full→JY306, NOT a specific tile. Don't use it for per-tile warps.

- **No heavy filtering for display**: Don't apply median filter, percentile thresholding, or background subtraction when displaying volumes. Just raw values / max (or very light lo=0.5 percentile at most).
- **Scene 5b stitching**: Use raw nd2 GFP PNGs (not isotropic tif) + full v5 stitch (including elastix). The normalize_u8_f is OK for display since we norm to uint8 anyway. Mask exvivo to invivo coverage area to avoid rectangular tile edges.

**How to apply:** Always use proper geometric transforms (warpAffine/warpPerspective) for registration animations. Generate verification overlays for all slices. Cast map arrays to float32 for cv2.remap. Use display-space affine interpolation to avoid resize jumps. Keep target image visible during warp transitions. Show test images before committing to full pipeline runs.
