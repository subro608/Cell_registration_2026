#!/usr/bin/env python3
"""Merge the 19 per-tile ``landmarks_stitched_v5_*.npz`` files from
``jy306_data/registration_video/`` into a single ``jy306_landmarks.json`` in
the Sparrow layout (``[id, [iv_x, iv_y, iv_z], [ex_x, ex_y, ex_z]]``).

Sources expected at:
    <NPZ_DIR>/landmarks_stitched_v5_<tile>.npz

Each npz must contain ``pcd_invivo_jy306`` (N, 3) (z, y, x) in JY306 in-vivo
pixel coords and ``stitched_coords`` (N, 3) (z, y, x) in stitched 1 µm
isotropic ex-vivo voxel coords. We swap (z, y, x) → (x, y, z) to match
runner.load_landmarks().

Override the input/output paths with env vars:
    JY306_LANDMARK_NPZ_DIR  (default: jy306_data/registration_video/)
    JY306_LANDMARK_OUT      (default: cellinvariance/data/jy306/jy306_landmarks.json)

Usage:
    python scripts/build_jy306_landmarks.py
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NPZ_DIR = Path(
    os.environ.get(
        "JY306_LANDMARK_NPZ_DIR",
        "/scratch/sd5963/cell_registration_2026/jy306_data/registration_video",
    )
)
DEFAULT_OUT = Path(
    os.environ.get(
        "JY306_LANDMARK_OUT",
        str(PROJECT_ROOT / "data" / "jy306" / "jy306_landmarks.json"),
    )
)


def build(npz_dir: Path, out_path: Path) -> int:
    files = sorted(npz_dir.glob("landmarks_stitched_v5_*.npz"))
    if not files:
        raise SystemExit(f"No landmarks_stitched_v5_*.npz under {npz_dir}")
    values: list = []
    skipped: list[str] = []
    for f in files:
        d = np.load(str(f), allow_pickle=True)
        if "pcd_invivo_jy306" not in d or "stitched_coords" not in d:
            skipped.append(f.name)
            continue
        iv = np.asarray(d["pcd_invivo_jy306"], dtype=np.float64)
        ex = np.asarray(d["stitched_coords"], dtype=np.float64)
        tile = f.stem.replace("landmarks_stitched_v5_", "")
        for i in range(iv.shape[0]):
            values.append([
                f"{tile}_{i:03d}",
                [float(iv[i, 2]), float(iv[i, 1]), float(iv[i, 0])],  # x, y, z
                [float(ex[i, 2]), float(ex[i, 1]), float(ex[i, 0])],  # x, y, z
            ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "fixed": "stitched_gfp_fullres_v5_1um_isotropic",
        "landmarks": {
            "next_index": len(values) + 1,
            "transform_mode": "Thinplate",
            "values": values,
        },
        "mode": "registration",
        "stacks": [
            "./JY306_in_Vivo_stack_flipped_s80.tif",
            "./stitched_gfp_fullres_v5_1um_isotropic.tif",
        ],
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(values)} landmarks across {len(files) - len(skipped)} tiles -> {out_path}")
    if skipped:
        print(f"Skipped (missing keys): {skipped}")
    return len(values)


if __name__ == "__main__":
    build(DEFAULT_NPZ_DIR, DEFAULT_OUT)
