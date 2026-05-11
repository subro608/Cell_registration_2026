---
name: Coordinate Space Pitfalls
description: Hard-won lessons about confusing coordinate spaces and file types in the margaret project
type: project
---

Critical gotchas discovered during registration work:

1. **exvivo_total.tif is LABELS not intensity** — uint64, max=771, cell segmentation masks. MIP looks plausible due to normalization but is NOT a tissue image. The actual ex-vivo intensity in JY306 space is `antirat_combined.tif`.

2. **antirat_combined.tif IS intensity despite uint64** — max=1091, Anti-Rat staining registered to JY306 space. Don't be fooled by the dtype.

3. **JY306 vs JY316** — Two different mice/coordinate systems. JY306=(16, 658, 629), JY316=(19, 578, 599). pcd landmarks are in JY306 space (r range goes up to 638 > 578).

4. **MERSCOPE mosaic vs FOV mismatch** — The full MERSCOPE tile (mosaic_Anti-Rat_z4.tif at 18875x22558) is at completely different scale than the JY306 FOV. Don't try to directly compare/overlay them.

5. **pkl canvas is NOT JY306 space** — transformed[0] in the pkl is (16, 1704, 1704), an intermediate canvas. The final registered outputs (exvivo_combined, antirat_combined) are (16, 658, 629).

**Why:** These caused multiple false starts and wrong animations during the registration video work.

**How to apply:** Always verify image dtype, shape, and value range before assuming what a file contains. Check coordinate space compatibility before overlaying.
