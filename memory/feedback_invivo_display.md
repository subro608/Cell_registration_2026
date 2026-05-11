---
name: In-vivo display without thresholding
description: Use raw (unfiltered) in-vivo for display patches, median filter only for elastix registration
type: feedback
---

Do not threshold or filter the in-vivo image for display/visualization. Use raw in-vivo for patches and overlays.

**Why:** The median-filtered background-subtracted in-vivo doesn't look good in the patch viewer. The filtering is only needed for elastix registration to work well (MI metric).

**How to apply:** When building display patches or contact sheets, warp the raw in-vivo through the same transform (affine + elastix). Keep median filter only for the elastix optimization step.
