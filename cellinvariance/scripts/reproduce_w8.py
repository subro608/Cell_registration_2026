#!/usr/bin/env python3
"""Reproduce the w8 cross-modality cell re-id experiment end-to-end.

Pipeline:
  1. Compute frozen DINOv2-small multi-scale features for the 115 paired
     in-vivo / ex-vivo cells with `hard_intra` augmentations (800 per cell).
     Cached under .feature_cache/emb_cache_w8/. First run is slow
     (~12-20h on MPS); subsequent runs reuse the cache.
  2. Train an MLP projector with supervised contrastive + Sinkhorn alignment
     for 200 epochs (~30 min after the cache exists).
  3. Install the trained projector + results JSON into results/.
  4. Regenerate viz/dino_repr_interactive.html.

Usage:
    uv run python scripts/reproduce_w8.py
    uv run python scripts/reproduce_w8.py --skip-train  # only regen viz
    uv run python scripts/reproduce_w8.py --skip-viz    # only train
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTRASTIVE  = PROJECT_ROOT / "src" / "contrastive.py"
GEN_DATA     = PROJECT_ROOT / "viz" / "generate_data.py"
GEN_HTML     = PROJECT_ROOT / "viz" / "generate_html.py"
RESULTS_DIR  = PROJECT_ROOT / "results"
CACHE_DIR    = PROJECT_ROOT / ".feature_cache"

TAG          = "w8_e200_ms3_hi_ap800_j0_xy80z24_skh07_dv2_g0"
EMB_CACHE    = CACHE_DIR / "emb_cache_w8"

TRAINED_PROJ    = CACHE_DIR / f"loocv_contrastive_mlp_{TAG}_proj.pt"
TRAINED_RESULTS = CACHE_DIR / f"loocv_contrastive_mlp_{TAG}_results.json"
TRAINED_FOLDS   = CACHE_DIR / f"loocv_contrastive_mlp_{TAG}_folds.jsonl"

INSTALLED_PROJ    = RESULTS_DIR / "w8_projector.pt"
INSTALLED_RESULTS = RESULTS_DIR / "w8_results.json"
INSTALLED_FOLDS   = RESULTS_DIR / "w8_folds.jsonl"

# These mirror the CLI invocation that produced w8 in the original autoresearch
# loop. Changing any of these gives you a different experiment.
BASE_FLAGS = [
    "--mode",               "full_iv_ex_eval",
    "--train-source",       "joint",
    "--epochs",             "200",
    "--batch",              "1024",
    "--lr",                 "0.001",
    "--weight-decay",       "0.0001",
    "--temperature",        "0.1",
    "--knn-k",              "5",
    "--seed",               "23",
    "--aug-per-preset",     "800",
    "--aug-presets",        "hard_intra",
    "--snapshot-every",     "50",
    "--dino-version",       "v2",
    "--embed-batch-size",   "8",
    "--mlp-out-dim",        "64",
    "--alignment-loss",     "sinkhorn",
    "--lambda-align",       "0.7",
    "--lambda-cross-pair",  "0",
    "--geom-neighbor-k",    "0",
    "--patch-xy-um",        "80",
    "--patch-z-um",         "24",
    "--spatial-jitter-vox", "0",
    "--multiscale-enable",
    "--multiscale-xy-ums",  "40,80,120",
    "--multiscale-z-ums",   "12,24,36",
    "--output-tag",         TAG,
    "--emb-cache",          str(EMB_CACHE),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train() -> None:
    log("Training w8 (multi-scale 3x + sinkhorn + 800 augs/cell)...")
    log(f"  CACHE_DIR: {CACHE_DIR}")
    log(f"  TAG: {TAG}")
    cmd = [sys.executable, str(CONTRASTIVE), *BASE_FLAGS]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        log("ERROR: training failed.")
        sys.exit(1)
    if not TRAINED_PROJ.exists():
        log(f"ERROR: expected projector at {TRAINED_PROJ} but it was not produced.")
        sys.exit(1)
    log("Training complete.")


def install_artifacts() -> None:
    """Copy fresh training output into results/ as the blessed artifacts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        (TRAINED_PROJ,    INSTALLED_PROJ),
        (TRAINED_RESULTS, INSTALLED_RESULTS),
        (TRAINED_FOLDS,   INSTALLED_FOLDS),
    ]:
        if src.exists():
            shutil.copyfile(src, dst)
            log(f"Installed {src.name} -> {dst.relative_to(PROJECT_ROOT)}")


def regenerate_viz() -> None:
    log("Regenerating visualization...")
    r = subprocess.run([sys.executable, str(GEN_DATA)])
    if r.returncode != 0:
        log("ERROR: viz data generation failed.")
        sys.exit(1)
    r = subprocess.run([sys.executable, str(GEN_HTML)])
    if r.returncode != 0:
        log("ERROR: viz HTML generation failed.")
        sys.exit(1)
    html = PROJECT_ROOT / "viz" / "dino_repr_interactive.html"
    if html.exists():
        log(f"Visualization ready: {html} ({html.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce w8 end-to-end.")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; use the bundled results/w8_projector.pt instead.",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip regenerating viz/dino_repr_interactive.html.",
    )
    args = parser.parse_args()

    t0 = time.time()
    if args.skip_train:
        if not INSTALLED_PROJ.exists():
            log(f"ERROR: --skip-train requires {INSTALLED_PROJ} to exist.")
            sys.exit(1)
        log(f"Skipping training; using bundled {INSTALLED_PROJ.name}.")
    else:
        train()
        install_artifacts()

    if not args.skip_viz:
        regenerate_viz()

    elapsed_min = (time.time() - t0) / 60
    log(f"Done in {elapsed_min:.1f} min.")


if __name__ == "__main__":
    main()
