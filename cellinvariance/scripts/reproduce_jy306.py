#!/usr/bin/env python3
"""Run the cellinvariance pipeline on JY306.

Mirrors ``reproduce_w8.py`` but pulls per-subject patch scales from
``runner.SUBJECTS["jy306"]`` and accepts an ``--epochs`` / ``--augs`` override
so the first JY306 run can be small.

Usage:
    CELLINVARIANCE_SUBJECT=jy306 \\
        python scripts/reproduce_jy306.py --epochs 50 --augs 100
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CELLINVARIANCE_SUBJECT", "jy306")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTRASTIVE  = PROJECT_ROOT / "src" / "contrastive.py"
CACHE_DIR    = PROJECT_ROOT / ".feature_cache"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cellinvariance on JY306.")
    parser.add_argument("--epochs", type=int, default=200, help="MLP training epochs.")
    parser.add_argument("--augs", type=int, default=800, help="Augmentations per cell.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Output tag (default: jy306_e<E>_a<A>).")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda-align", type=float, default=0.7)
    args = parser.parse_args()

    # Pull subject-specific patch scales from the registry.
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import runner as nr

    cfg = nr.SUBJECTS[nr.SUBJECT]
    xy_ums = ",".join(str(x) for x in cfg["multiscale_xy_ums"])
    z_ums  = ",".join(str(x) for x in cfg["multiscale_z_ums"])

    tag = args.tag or (
        f"jy306_e{args.epochs}_a{args.augs}_xy{cfg['patch_xy_um']}z{cfg['patch_z_um']}_skh07"
    )
    emb_cache = CACHE_DIR / f"emb_cache_{tag}"

    flags = [
        "--mode",               "full_iv_ex_eval",
        "--train-source",       "joint",
        "--epochs",             str(args.epochs),
        "--batch",              str(args.batch),
        "--lr",                 str(args.lr),
        "--weight-decay",       "0.0001",
        "--temperature",        "0.1",
        "--knn-k",              "5",
        "--seed",               "23",
        "--aug-per-preset",     str(args.augs),
        "--aug-presets",        "hard_intra",
        "--snapshot-every",     "50",
        "--dino-version",       "v2",
        "--embed-batch-size",   "8",
        "--mlp-out-dim",        "64",
        "--alignment-loss",     "sinkhorn",
        "--lambda-align",       str(args.lambda_align),
        "--lambda-cross-pair",  "0",
        "--geom-neighbor-k",    "0",
        "--patch-xy-um",        str(cfg["patch_xy_um"]),
        "--patch-z-um",         str(cfg["patch_z_um"]),
        "--spatial-jitter-vox", "0",
        "--multiscale-enable",
        "--multiscale-xy-ums",  xy_ums,
        "--multiscale-z-ums",   z_ums,
        "--output-tag",         tag,
        "--emb-cache",          str(emb_cache),
    ]
    log(f"subject={nr.SUBJECT}  tag={tag}")
    log(f"  patch scales: xy={xy_ums} µm, z={z_ums} µm")
    log(f"  data dir: {nr.DATASET_DIR}")
    log(f"  emb cache: {emb_cache}")
    cmd = [sys.executable, str(CONTRASTIVE), *flags]
    log("Launching contrastive.py ...")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
