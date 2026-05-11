---
name: Save transforms with outputs
description: Always save affine/transform matrices in the same folder as contact sheet outputs
type: feedback
---

Always save the transformation matrix (affine, etc.) in the same output folder as the contact sheet PNG/HTML.

**Why:** User wants results and the transforms that produced them co-located for easy sharing and reproducibility.

**How to apply:** When generating registration contact sheets, save the `.npz` with affine matrix, errors, landmarks, etc. alongside the PNG/HTML in the same output directory.
