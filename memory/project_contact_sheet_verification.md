---
name: Contact Sheet Verification Status
description: Contact sheets shared with Cliodna and Jason for cell match verification, awaiting feedback
type: project
---

## Status (2026-03-29)
Contact sheets for all 19 pkl tiles shared with Cliodna and Jason for verification.

**Why:** Erdem requested independent verification of cell matches before proceeding with downstream analysis. Also flagged that pairwise distance scatter lines look "too straight" — expected some deformation.

**How to apply:** Wait for Cliodna/Jason feedback before finalizing landmarks. If matches look wrong for specific tiles, may need to revisit pkl inverse or SIFT affine for those tiles.

## What was shared
- MIP±2 z-slice contact sheets for all 19 tiles (from `mip_pm2/`)
- Also generated but not shared: `single_z/` and `combined/` versions
- Each shows ex-vivo nd2 (left) ↔ in-vivo JY306 (right) with green crosshairs marking landmarks
- Message sent: "Ex-vivo landmarks in native space were obtained by inverting the transformation files in the transformation/ folder" — keep descriptions simple for collaborators, don't mention SIFT/pkl/MERSCOPE details

## Pairwise distance correlation
- 3D version (XYZ in JY306 µm space): r > 0.999 for all tiles, slope ~1.0 — but this is trivially good because both point clouds are in same space (~1.3px apart)
- 2D native-space version (JY306 XY vs nd2 XY): r > 0.993, slope ~2.3 (from pkl spatial scaling, not deformation)
- Erdem noted lines too straight — because pcd_invivo and pcd_exvivo traverse same pkl inverse path, deformation cancels
