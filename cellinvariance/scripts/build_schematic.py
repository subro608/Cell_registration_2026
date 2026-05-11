#!/usr/bin/env python3
"""Build the README schematic illustrating the w8 pipeline.

Stages (left -> right):
  1. Source volumes (in-vivo / ex-vivo) with landmark dots
  2. Patches around 5 example cells (paired iv/ex rows)
  3. Augmentations of one example patch (`hard_intra` preset)
  4. Architecture + losses (DINOv2 frozen -> MLP -> SupCon + Sinkhorn)
  5. Final 64-d embedding scatter (PCA-2D of trained MLP output)

Prerequisites:
  uv run python scripts/download_data.py          # downloads TIFs + landmarks
  uv run python scripts/reproduce_w8.py --skip-train  # produces viz/dino_repr_data.json

Output: viz/schematic.png
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import augmentations as A  # noqa: E402
import runner as nr  # noqa: E402

# Inputs (resolved via the same logic the rest of the repo uses)
ZSTACK    = nr.ZSTACK_PATH
EXVIVO    = nr.EXVIVO_PATH
LM_JSON   = nr.LANDMARKS_PATH
DATA_JSON = PROJECT_ROOT / "viz" / "dino_repr_data.json"

# 0-indexed channel; landmarks were annotated on channel 2 (1-indexed)
INVIVO_CHANNEL = max(0, nr.INVIVO_CHANNEL_1BASED - 1)

# Output
OUT_PNG = PROJECT_ROOT / "viz" / "schematic.png"

# Pick 5 example cells with reasonably spread-out IV positions.
# Indices into the cell_ids list from DATA_JSON.
EXAMPLE_INDICES = [0, 8, 24, 65, 89]
HIGHLIGHT_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]

FIG_W = 24.0  # inches
FIG_H = 8.0
DPI   = 200


def b64_to_rgb(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def percentile_normalize(img: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(img, [low, high])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def load_iv_slice(z: int) -> np.ndarray:
    with tifffile.TiffFile(ZSTACK) as tf:
        # zstack.tif is ZCYX with 4 channels per z; flat page index is z*4 + c
        page_idx = z * 4 + INVIVO_CHANNEL
        return tf.pages[page_idx].asarray()


def load_ex_slice(z: int) -> np.ndarray:
    with tifffile.TiffFile(EXVIVO) as tf:
        return tf.pages[z].asarray()


def get_landmarks() -> list[tuple[str, list[int], list[int]]]:
    raw = json.load(open(LM_JSON))
    return [(name, A_, B_) for name, A_, B_ in raw["landmarks"]["values"]]


def panel_volumes(ax_iv, ax_ex, landmarks, example_lm_names) -> None:
    iv_zs = [a[2] for _, a, _ in landmarks]
    ex_zs = [b[2] for _, _, b in landmarks]
    iv_z  = int(np.median(iv_zs))
    ex_z  = int(np.median(ex_zs))

    iv = percentile_normalize(load_iv_slice(iv_z))
    ex = percentile_normalize(load_ex_slice(ex_z))

    ax_iv.imshow(iv, cmap="gray")
    ax_ex.imshow(ex, cmap="gray")

    for name, iv_xyz, ex_xyz in landmarks:
        iv_alpha = max(0.15, 1.0 - abs(iv_xyz[2] - iv_z) / 30.0)
        ex_alpha = max(0.15, 1.0 - abs(ex_xyz[2] - ex_z) / 30.0)
        if name in example_lm_names:
            color = HIGHLIGHT_COLORS[example_lm_names.index(name)]
            ax_iv.scatter(iv_xyz[0], iv_xyz[1], s=140, c=color,
                          edgecolors="white", linewidths=1.5, zorder=5)
            ax_ex.scatter(ex_xyz[0], ex_xyz[1], s=140, c=color,
                          edgecolors="white", linewidths=1.5, zorder=5)
        else:
            ax_iv.scatter(iv_xyz[0], iv_xyz[1], s=18, c="#ffd60a",
                          alpha=iv_alpha * 0.55, edgecolors="none", zorder=3)
            ax_ex.scatter(ex_xyz[0], ex_xyz[1], s=18, c="#ffd60a",
                          alpha=ex_alpha * 0.55, edgecolors="none", zorder=3)

    for ax, label in [(ax_iv, "in-vivo  zstack"), (ax_ex, "ex-vivo  registered")]:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=11, fontweight="bold", pad=6)
        for spine in ax.spines.values():
            spine.set_color("#666"); spine.set_linewidth(0.8)


def panel_patches(fig, gs_patches, data, example_indices) -> None:
    inner = GridSpecFromSubplotSpec(2, 5, subplot_spec=gs_patches, wspace=0.08, hspace=0.10)
    for col, idx in enumerate(example_indices):
        for row, source, label in [(0, "iv", "iv"), (1, "ex", "ex")]:
            ax = fig.add_subplot(inner[row, col])
            ax.imshow(b64_to_rgb(data[source]["disp_b64"][idx]))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(HIGHLIGHT_COLORS[col])
                spine.set_linewidth(2.5)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, va="center", ha="right", labelpad=8)


def panel_augmentations(fig, gs_aug, data, chosen_idx, color) -> None:
    inner = GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_aug, wspace=0.08, hspace=0.10)
    base = b64_to_rgb(data["iv"]["disp_b64"][chosen_idx])
    rng = np.random.default_rng(23)
    for k in range(8):
        ax = fig.add_subplot(inner[k // 4, k % 4])
        aug = np.clip(A.augment_patch(base, rng, preset="hard_intra"), 0, 1)
        ax.imshow(aug)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color); spine.set_linewidth(1.5)


def panel_architecture(ax) -> None:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, label, fill="#f0f4fa", edge="#345080",
            textsize=10, badge=None, badge_color="#0a7a3b"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
            linewidth=1.5, facecolor=fill, edgecolor=edge,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=textsize, fontweight="bold", color="#222")
        if badge:
            ax.text(x + w - 0.15, y + h - 0.20, badge, fontsize=9, ha="right",
                    color=badge_color, fontweight="bold", style="italic")

    def arrow(x1, y1, x2, y2, label=None):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
            color="#345080", linewidth=1.5,
        ))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label,
                    ha="center", fontsize=8, color="#345080")

    box(0.3, 5.3, 2.6, 1.6, "DINOv2-small\n(3 scales)", badge="frozen",
        badge_color="#bf8a00", fill="#fff2cc", edge="#bf8a00")
    arrow(2.95, 6.1, 4.0, 6.1, "1152-d")
    box(4.0, 5.3, 2.4, 1.6, "MLP\n1152→256→64\n(L2-norm)", badge="trained",
        fill="#e8f5e8", edge="#0a7a3b")
    arrow(6.45, 6.1, 7.5, 6.1, "64-d")

    box(7.5, 7.0, 2.4, 1.0, "SupCon (intra-modality)", textsize=9,
        fill="#fff", edge="#345080")
    box(7.5, 4.6, 2.4, 1.0, "Sinkhorn (cross-modality)", textsize=9,
        fill="#fff", edge="#345080")
    arrow(8.7, 6.0, 8.7, 7.0)
    arrow(8.7, 6.0, 8.7, 5.6)

    ax.text(5.0, 9.3, "encoder & losses", ha="center",
            fontsize=11, fontweight="bold")
    ax.text(5.0, 2.0, "frozen backbone   |   only the MLP is trained",
            ha="center", fontsize=9, color="#666", style="italic")


def panel_embeddings(ax, data, example_indices) -> None:
    iv_pca = np.array(data["iv"]["mlp_pca"])
    ex_pca = np.array(data["ex"]["mlp_pca"])
    n = iv_pca.shape[0]

    other_color = "#bbbbbb"
    other_alpha = 0.55
    for i in range(n):
        if i in example_indices:
            continue
        ax.scatter(iv_pca[i, 0], iv_pca[i, 1], s=22, c=other_color,
                   alpha=other_alpha, marker="o", edgecolors="none", zorder=2)
        ax.scatter(ex_pca[i, 0], ex_pca[i, 1], s=24, c=other_color,
                   alpha=other_alpha, marker="D", edgecolors="none", zorder=2)

    for slot, idx in enumerate(example_indices):
        c = HIGHLIGHT_COLORS[slot]
        ax.scatter(iv_pca[idx, 0], iv_pca[idx, 1], s=120, c=c, marker="o",
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.scatter(ex_pca[idx, 0], ex_pca[idx, 1], s=120, c=c, marker="D",
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.plot([iv_pca[idx, 0], ex_pca[idx, 0]],
                [iv_pca[idx, 1], ex_pca[idx, 1]],
                color=c, alpha=0.5, linewidth=1.2, zorder=4)

    ax.set_title("trained 64-d embedding (PCA)",
                 fontsize=11, fontweight="bold", pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("●  in-vivo       ◆  ex-vivo       same color = same cell",
                  fontsize=9, color="#444")
    for spine in ax.spines.values():
        spine.set_color("#666"); spine.set_linewidth(0.8)


def _check_inputs() -> None:
    missing = []
    for label, p in [
        ("in-vivo zstack",      ZSTACK),
        ("ex-vivo registered",  EXVIVO),
        ("landmarks JSON",      LM_JSON),
    ]:
        if not p.exists():
            missing.append(f"  - {label}: {p}")
    if missing:
        print("ERROR: dataset files not found:\n" + "\n".join(missing))
        print("\nRun:  uv run python scripts/download_data.py")
        sys.exit(1)
    if not DATA_JSON.exists():
        print(f"ERROR: representation data JSON not found at\n  {DATA_JSON}")
        print("\nRun:  uv run python scripts/reproduce_w8.py --skip-train")
        print("(this generates the JSON + the bundled HTML viz)")
        sys.exit(1)


def main() -> None:
    _check_inputs()

    print("Loading representation data JSON...")
    data = json.load(open(DATA_JSON))
    cell_ids = data["cell_ids"]

    print("Loading landmarks...")
    landmarks = get_landmarks()
    lm_by_name = {n: (a, b) for n, a, b in landmarks}

    example_lm_names = [cell_ids[i] for i in EXAMPLE_INDICES]
    print(f"Highlighted cells: {example_lm_names}")

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    gs = GridSpec(1, 5, figure=fig,
                  width_ratios=[2.2, 2.2, 1.8, 3.0, 2.5],
                  left=0.01, right=0.99, top=0.92, bottom=0.06, wspace=0.18)

    # Stage 1
    gs1 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.18)
    ax_iv = fig.add_subplot(gs1[0])
    ax_ex = fig.add_subplot(gs1[1])
    panel_volumes(
        ax_iv, ax_ex,
        [(n,) + lm_by_name[n] for n in cell_ids if n in lm_by_name],
        example_lm_names,
    )

    # Stage 2
    panel_patches(fig, gs[1], data, EXAMPLE_INDICES)

    # Stage 3
    chosen_slot = 2
    panel_augmentations(fig, gs[2], data,
                        EXAMPLE_INDICES[chosen_slot],
                        HIGHLIGHT_COLORS[chosen_slot])

    # Stage 4
    ax_arch = fig.add_subplot(gs[3])
    panel_architecture(ax_arch)

    # Stage 5
    ax_emb = fig.add_subplot(gs[4])
    panel_embeddings(ax_emb, data, EXAMPLE_INDICES)

    # Section titles
    fig.text(0.10,  0.95, "source volumes",                    ha="center", fontsize=11, fontweight="bold")
    fig.text(0.32,  0.95, "patches around cells (80 µm)",      ha="center", fontsize=11, fontweight="bold")
    fig.text(0.495, 0.95, "augmentations  (×800/cell)",        ha="center", fontsize=11, fontweight="bold")

    # Inter-stage right-arrows in the gap rows
    def stage_arrow(x_start, x_end, y=0.5):
        fig.add_artist(FancyArrowPatch(
            (x_start, y), (x_end, y),
            arrowstyle="-|>", mutation_scale=28,
            color="#345080", linewidth=2.0,
            transform=fig.transFigure, shrinkA=0, shrinkB=0,
        ))

    for x1, x2 in [(0.205, 0.230), (0.418, 0.443), (0.563, 0.585), (0.770, 0.792)]:
        stage_arrow(x1, x2)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_PNG}  ({OUT_PNG.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
