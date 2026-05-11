#!/usr/bin/env python3
"""
LOOCV contrastive MLP on frozen in-vivo DINO CLS embeddings (default: DINOv3 ViT-L,
same model id as native_runner — single view per patch, not orbit-mean).

For each left-out cell:
  - hold out that cell (+all its augmentations) as test
  - split remaining cells 80/20 (cell-level) into train/val
  - train MLP projector with supervised-contrastive loss
  - select epoch by validation kNN accuracy
  - evaluate held-out test samples by nearest-neighbor identity behavior

Also supports one selected fold animation over all epochs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

import runner as nr
from augmentations import augment_patch
from alignment import (
    DomainDiscriminator,
    dann_alpha_schedule,
    dann_loss,
    get_alignment_loss,
)

# `contrastive.py` lives at <project>/src/contrastive.py — project root is two levels up.
ROOT = Path(__file__).parent.parent
OUT_DIR = Path(
    os.environ.get("CELLINVARIANCE_CACHE_DIR", ROOT / ".feature_cache")
)

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl_cache"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".hf_cache" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".hf_cache" / "transformers"))

TORCH_DEVICE = nr.choose_device()
DEVICE = str(TORCH_DEVICE)
XY_UM, Z_UM = 200, 28
DEFAULT_MULTISCALE_XY_UMS = (100, 200, 300)
DEFAULT_MULTISCALE_Z_UMS = (14, 28, 42)
PATCH_CACHE_FORMAT_VERSION = 1

DEFAULT_AUG_PRESETS = [
    "heavy",
    "medium",
    "rotation_only",
    "domain_gap",
    "geometry_strong",
    "geometry_plus_medium",
    "rotation_crop_heavy",
    "hard_intra",
]
DEFAULT_AUG_PER_PRESET = 4


def output_paths(tag: str) -> dict[str, Path]:
    """Tagged filenames avoid overwriting prior runs (e.g. DINOv2 vs DINOv3)."""
    t = (tag or "").strip()
    suf = f"_{t}" if t else ""
    return {
        "results": OUT_DIR / f"loocv_contrastive_mlp{suf}_results.json",
        "folds_jsonl": OUT_DIR / f"loocv_contrastive_mlp{suf}_folds.jsonl",
        "anim_html": OUT_DIR / f"loocv_contrastive_mlp{suf}_fold_animation.html",
        "anim_json": OUT_DIR / f"loocv_contrastive_mlp{suf}_fold_animation.json",
        "summary_html": OUT_DIR / f"loocv_contrastive_mlp{suf}_summary.html",
        "coembed_html": OUT_DIR / f"loocv_contrastive_mlp{suf}_coembed.html",
        "proj_ckpt": OUT_DIR / f"loocv_contrastive_mlp{suf}_proj.pt",
    }


def resolve_dino_model_id(version: str, model_id: str) -> str:
    if model_id.strip():
        return model_id.strip()
    v = version.lower().strip()
    if v == "v3":
        return nr.DINO_MODEL_ID
    if v == "v2":
        return "facebook/dinov2-small"
    if v == "eupe":
        return "facebook/EUPE-ViT-S"
    raise ValueError(f"Unsupported dino version: {version}")


def landmark_3d_positions(lm: list[dict], fuse: str) -> np.ndarray:
    """Stack (N,3) landmark coordinates; fuse in {midpoint, invivo, exvivo}."""
    n = len(lm)
    pos = np.zeros((n, 3), dtype=np.float32)
    for i, d in enumerate(lm):
        if fuse == "midpoint":
            pos[i, 0] = (d["invivo_x"] + d["exvivo_x"]) * 0.5
            pos[i, 1] = (d["invivo_y"] + d["exvivo_y"]) * 0.5
            pos[i, 2] = (d["invivo_z"] + d["exvivo_z"]) * 0.5
        elif fuse == "invivo":
            pos[i, 0] = d["invivo_x"]
            pos[i, 1] = d["invivo_y"]
            pos[i, 2] = d["invivo_z"]
        elif fuse == "exvivo":
            pos[i, 0] = d["exvivo_x"]
            pos[i, 1] = d["exvivo_y"]
            pos[i, 2] = d["exvivo_z"]
        else:
            raise ValueError(f"Unknown --geom-fuse mode: {fuse}")
    return pos


def compute_geom_knn_offsets(lm: list[dict], neighbor_k: int, fuse: str) -> np.ndarray:
    """
    Per-cell KNN structure in fused 3D space: flattened relative offsets to K neighbors.
    Returns (n_cells, 3 * K), scaled by median nearest-neighbor distance over cells.
    """
    pos = landmark_3d_positions(lm, fuse)
    n = pos.shape[0]
    k = int(neighbor_k)
    d2 = np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=2).astype(np.float64)
    np.fill_diagonal(d2, np.inf)
    nn_dist = np.min(d2, axis=1)
    scale = float(np.sqrt(np.median(nn_dist[np.isfinite(nn_dist)])))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    feats = np.zeros((n, 3 * k), dtype=np.float32)
    for i in range(n):
        nn = np.argsort(d2[i])[:k]
        rel = pos[nn] - pos[i]
        feats[i] = (rel.reshape(-1) / scale).astype(np.float32)
    return feats


def stack_sample_geom(samples: list[dict], geom_by_cell: np.ndarray) -> np.ndarray:
    return np.stack([geom_by_cell[int(s["cell_idx"])] for s in samples], axis=0).astype(np.float32)


def torch_eval_geom_zero(x: torch.Tensor, geom_dim: int, do_zero: bool) -> torch.Tensor:
    """Zero last geom_dim columns for val/eval/snapshots when training used fused CLS+geom."""
    if not do_zero or geom_dim <= 0:
        return x
    out = x.clone()
    out[:, -geom_dim:] = 0
    return out


def np_eval_geom_zero(emb: np.ndarray, geom_dim: int, do_zero: bool) -> np.ndarray:
    if not do_zero or geom_dim <= 0:
        return emb
    out = emb.copy()
    out[:, -geom_dim:] = 0
    return out


def maybe_reduce_embeddings(train_emb: np.ndarray, target_emb: np.ndarray, out_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Optionally compress frozen DINO embeddings before MLP."""
    if out_dim <= 0 or out_dim >= train_emb.shape[1]:
        return train_emb, target_emb
    p = PCA(n_components=out_dim, random_state=23)
    train_z = p.fit_transform(train_emb)
    target_z = p.transform(target_emb)
    return train_z.astype(np.float32), target_z.astype(np.float32)


@torch.no_grad()
def embed_patches_cls(embedder: nr.FrozenVisionEmbedder, patches_rgb01: list[np.ndarray], batch_size: int) -> np.ndarray:
    """CLS embeddings, L2-normalized (DINO or EUPE via embedder.embed_cls)."""
    return embedder.embed_cls(patches_rgb01, batch_size=max(1, int(batch_size)))


def embed_multiscale_samples_cls(
    embedder: nr.FrozenVisionEmbedder,
    samples: list[dict],
    batch_size: int,
    use_multiscale: bool,
) -> np.ndarray:
    if not use_multiscale:
        return embed_patches_cls(embedder, [s["patch"] for s in samples], batch_size)
    n_scales = len(samples[0]["patches_ms"])
    scale_embs = []
    for scale_idx in range(n_scales):
        scale_patches = [s["patches_ms"][scale_idx] for s in samples]
        scale_embs.append(embed_patches_cls(embedder, scale_patches, batch_size))
    return np.concatenate(scale_embs, axis=1).astype(np.float32)


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        z = self.fc2(F.relu(self.fc1(x)))
        return F.normalize(z, dim=1)


def supervised_contrastive_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    SupCon loss over integer labels in a batch.

    For joint IV+EX training, labels are usually disjoint across modalities (IV uses
    0..N-1, EX uses N..2N-1), so positives are only same-cell augmentations *within*
    in-vivo or *within* ex-vivo—never IV↔EX matched pairs. Cross-domain same-landmark
    pulls are opt-in via cross_domain_pair_loss or shared label spaces.
    """
    b = z.shape[0]
    sim = torch.mm(z, z.t()) / temperature
    sim = sim - torch.max(sim, dim=1, keepdim=True).values
    mask_self = torch.eye(b, device=z.device, dtype=torch.bool)
    y_eq = (y[:, None] == y[None, :]) & (~mask_self)
    exp_sim = torch.exp(sim) * (~mask_self)
    log_prob = sim - torch.log(torch.clamp(exp_sim.sum(dim=1, keepdim=True), min=1e-12))
    pos_count = y_eq.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (log_prob * y_eq).sum(dim=1) / pos_count
    return -mean_log_prob_pos.mean()


def cross_domain_pair_loss(
    z: torch.Tensor,
    d_batch: torch.Tensor,
    landmark_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Pull IV/EX embeddings together when the same landmark cell_idx appears in both
    halves of a minibatch (stochastic pairing; no global label merge).
    z: (B,D) L2-normalized; d_batch: (B,) 0=iv 1=ex; landmark_batch: (B,) cell indices.
    """
    iv_mask = d_batch == 0
    ex_mask = d_batch == 1
    if not bool(iv_mask.any()) or not bool(ex_mask.any()):
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    z_iv = z[iv_mask]
    z_ex = z[ex_mask]
    lm_iv = landmark_batch[iv_mask]
    lm_ex = landmark_batch[ex_mask]
    match = lm_iv.unsqueeze(1) == lm_ex.unsqueeze(0)
    if not bool(match.any()):
        return torch.tensor(0.0, device=z.device, dtype=z.dtype)
    sim = z_iv @ z_ex.t()
    pair_sim = sim.masked_select(match)
    return (1.0 - pair_sim).mean()


@torch.no_grad()
def knn_predict(
    z_ref: torch.Tensor,
    y_ref: torch.Tensor,
    z_q: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    # cosine via normalized vectors dot-product
    sim = z_q @ z_ref.t()
    topk = torch.topk(sim, k=min(k, z_ref.shape[0]), dim=1).indices  # (Q,k)
    nn_labels = y_ref[topk]  # (Q,k)
    # mode vote
    preds = []
    for i in range(nn_labels.shape[0]):
        vals, counts = torch.unique(nn_labels[i], return_counts=True)
        preds.append(vals[torch.argmax(counts)])
    return torch.stack(preds, dim=0)


def knn_majority_labels(nn_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # nn_labels: (N,k) integer class ids
    votes = F.one_hot(nn_labels, num_classes=num_classes).sum(dim=1)
    return votes.argmax(dim=1)


def _slab_half_extents_vox(xy_um: int, z_um: int, vox_dims: tuple[float, float, float]) -> tuple[float, float, float]:
    """Max |offset| in voxel units from center to slab border along x,y,z (vol order z,y,x)."""
    vx, vy, vz = vox_dims
    xc = yc = int(xy_um)
    zc = int(z_um)
    ext_x = (xc - 1) / (2.0 * max(vx, 1e-6))
    ext_y = (yc - 1) / (2.0 * max(vy, 1e-6))
    ext_z = (zc - 1) / (2.0 * max(vz, 1e-6))
    return ext_x, ext_y, ext_z


def _jitter_center_xyz(
    cx: float,
    cy: float,
    cz: float,
    rng: np.random.Generator,
    jitter_vox: float,
    vol_shape: tuple[int, ...],
    vox_dims: tuple[float, float, float],
    xy_um: int,
    z_um: int,
    margin: float = 2.0,
) -> tuple[float, float, float]:
    if jitter_vox <= 0:
        return cx, cy, cz
    nz, ny, nx = int(vol_shape[0]), int(vol_shape[1]), int(vol_shape[2])
    ext_x, ext_y, ext_z = _slab_half_extents_vox(xy_um, z_um, vox_dims)
    lo_x, hi_x = ext_x + margin, nx - ext_x - margin
    lo_y, hi_y = ext_y + margin, ny - ext_y - margin
    lo_z, hi_z = ext_z + margin, nz - ext_z - margin
    if lo_x >= hi_x or lo_y >= hi_y or lo_z >= hi_z:
        return cx, cy, cz
    j = float(jitter_vox)
    return (
        float(np.clip(cx + rng.uniform(-j, j), lo_x, hi_x)),
        float(np.clip(cy + rng.uniform(-j, j), lo_y, hi_y)),
        float(np.clip(cz + rng.uniform(-j, j), lo_z, hi_z)),
    )


def extract_patch_iv_at(
    iv_vol: np.ndarray,
    cx: float,
    cy: float,
    cz: float,
    xy_um: int,
    z_um: int,
) -> np.ndarray:
    h = int(xy_um)
    slab = nr.extract_isotropic_slab(
        iv_vol,
        (cx, cy, cz),
        nr.INVIVO_VOXEL_DIMS_UM,
        int(xy_um),
        int(z_um),
    )
    rgb = nr.slab_to_depth_lut(slab, nr.norm_perc_linear)
    if rgb is None:
        return np.zeros((h, h, 3), dtype=np.float32)
    return rgb.astype(np.float32)


def extract_patch_ex_at(
    ex_vol: np.ndarray,
    cx: float,
    cy: float,
    cz: float,
    ev_vox: tuple[float, float, float],
    xy_um: int,
    z_um: int,
) -> np.ndarray:
    h = int(xy_um)
    slab = nr.extract_isotropic_slab(
        ex_vol,
        (cx, cy, cz),
        ev_vox,
        int(xy_um),
        int(z_um),
    )
    rgb = nr.slab_to_depth_lut(slab, nr.norm_perc_linear)
    if rgb is None:
        return np.zeros((h, h, 3), dtype=np.float32)
    return rgb.astype(np.float32)


def parse_multiscale_um_list(text: str, expected_len: int, flag_name: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if len(vals) != expected_len:
        raise ValueError(f"{flag_name} must contain exactly {expected_len} comma-separated integers")
    if min(vals) < 8:
        raise ValueError(f"{flag_name} values must be >= 8")
    return vals


def _extract_patch_triplet_iv(
    iv_vol: np.ndarray,
    cx: float,
    cy: float,
    cz: float,
    xy_ums: tuple[int, ...],
    z_ums: tuple[int, ...],
) -> list[np.ndarray]:
    return [extract_patch_iv_at(iv_vol, cx, cy, cz, int(xy), int(z)) for xy, z in zip(xy_ums, z_ums)]


def _extract_patch_triplet_ex(
    ex_vol: np.ndarray,
    cx: float,
    cy: float,
    cz: float,
    ev_vox: tuple[float, float, float],
    xy_ums: tuple[int, ...],
    z_ums: tuple[int, ...],
) -> list[np.ndarray]:
    return [extract_patch_ex_at(ex_vol, cx, cy, cz, ev_vox, int(xy), int(z)) for xy, z in zip(xy_ums, z_ums)]


def build_dataset(
    rng: np.random.Generator,
    aug_presets: list[str],
    aug_per_preset: int,
    patch_xy_um: int = 200,
    patch_z_um: int = 28,
    spatial_jitter_vox: float = 0.0,
    multiscale_enable: bool = False,
    multiscale_xy_ums: tuple[int, ...] = DEFAULT_MULTISCALE_XY_UMS,
    multiscale_z_ums: tuple[int, ...] = DEFAULT_MULTISCALE_Z_UMS,
):
    lm = nr.load_landmarks()
    iv_vol, _ = nr.load_volumes()
    cell_ids = [str(x.get("id", i)) for i, x in enumerate(lm)]
    xy_u, z_u = int(patch_xy_um), int(patch_z_um)
    xy_scales = tuple(int(x) for x in (multiscale_xy_ums if multiscale_enable else (xy_u,)))
    z_scales = tuple(int(z) for z in (multiscale_z_ums if multiscale_enable else (z_u,)))
    jitter_xy = max(xy_scales)
    jitter_z = max(z_scales)
    base_patches = [
        _extract_patch_triplet_iv(iv_vol, float(x["invivo_x"]), float(x["invivo_y"]), float(x["invivo_z"]), xy_scales, z_scales)
        for x in lm
    ]
    samples = []
    default_idx = len(xy_scales) // 2
    for ci, p_ms in enumerate(base_patches):
        sample = {"cell_idx": ci, "cell_id": cell_ids[ci], "preset": "base", "patch": p_ms[default_idx]}
        if multiscale_enable:
            sample["patches_ms"] = p_ms
        samples.append(sample)
        x = lm[ci]
        for preset in aug_presets:
            for _ in range(aug_per_preset):
                cx, cy, cz = _jitter_center_xyz(
                    float(x["invivo_x"]),
                    float(x["invivo_y"]),
                    float(x["invivo_z"]),
                    rng,
                    spatial_jitter_vox,
                    iv_vol.shape,
                    nr.INVIVO_VOXEL_DIMS_UM,
                    jitter_xy,
                    jitter_z,
                )
                raw_ms = _extract_patch_triplet_iv(iv_vol, cx, cy, cz, xy_scales, z_scales)
                aug_ms = [augment_patch(p, rng, preset) for p in raw_ms]
                sample = {"cell_idx": ci, "cell_id": cell_ids[ci], "preset": preset, "patch": aug_ms[default_idx]}
                if multiscale_enable:
                    sample["patches_ms"] = aug_ms
                samples.append(sample)
    return lm, cell_ids, samples


def build_iv_ex_datasets(
    rng: np.random.Generator,
    aug_presets: list[str],
    aug_per_preset: int,
    patch_xy_um: int = 200,
    patch_z_um: int = 28,
    spatial_jitter_vox: float = 0.0,
    multiscale_enable: bool = False,
    multiscale_xy_ums: tuple[int, ...] = DEFAULT_MULTISCALE_XY_UMS,
    multiscale_z_ums: tuple[int, ...] = DEFAULT_MULTISCALE_Z_UMS,
    *,
    log_progress: bool = True,
):
    lm = nr.load_landmarks()
    iv_vol, ex_vol = nr.load_volumes()
    ev_vox = nr.fit_exvivo_voxel_dims(lm)
    cell_ids = [str(x.get("id", i)) for i, x in enumerate(lm)]
    n_cells = len(lm)
    xy_u, z_u = int(patch_xy_um), int(patch_z_um)
    xy_scales = tuple(int(x) for x in (multiscale_xy_ums if multiscale_enable else (xy_u,)))
    z_scales = tuple(int(z) for z in (multiscale_z_ums if multiscale_enable else (z_u,)))
    jitter_xy = max(xy_scales)
    jitter_z = max(z_scales)
    default_idx = len(xy_scales) // 2
    iv_samples = []
    ex_samples = []
    n_augs_per_cell = len(aug_presets) * aug_per_preset
    expected_pairs = n_cells * (1 + n_augs_per_cell)
    t_build0 = time.time()
    if log_progress:
        print(
            f"Building IV/EX augmented patches: {n_cells} cells, multiscale={multiscale_enable}, "
            f"base + {len(aug_presets)} preset(s)×{aug_per_preset} aug → {expected_pairs} IV/EX pairs each",
            flush=True,
        )
    for ci, x in enumerate(lm):
        p_iv_ms = _extract_patch_triplet_iv(
            iv_vol, float(x["invivo_x"]), float(x["invivo_y"]), float(x["invivo_z"]), xy_scales, z_scales
        )
        p_ex_ms = _extract_patch_triplet_ex(
            ex_vol, float(x["exvivo_x"]), float(x["exvivo_y"]), float(x["exvivo_z"]), ev_vox, xy_scales, z_scales
        )
        iv_sample = {"cell_idx": ci, "cell_id": cell_ids[ci], "preset": "base", "patch": p_iv_ms[default_idx]}
        ex_sample = {"cell_idx": ci, "cell_id": cell_ids[ci], "preset": "base", "patch": p_ex_ms[default_idx]}
        if multiscale_enable:
            iv_sample["patches_ms"] = p_iv_ms
            ex_sample["patches_ms"] = p_ex_ms
        iv_samples.append(iv_sample)
        ex_samples.append(ex_sample)
        for preset in aug_presets:
            for _ in range(aug_per_preset):
                cix, ciy, ciz = _jitter_center_xyz(
                    float(x["invivo_x"]),
                    float(x["invivo_y"]),
                    float(x["invivo_z"]),
                    rng,
                    spatial_jitter_vox,
                    iv_vol.shape,
                    nr.INVIVO_VOXEL_DIMS_UM,
                    jitter_xy,
                    jitter_z,
                )
                raw_iv_ms = _extract_patch_triplet_iv(iv_vol, cix, ciy, ciz, xy_scales, z_scales)
                cex, cey, cez = _jitter_center_xyz(
                    float(x["exvivo_x"]),
                    float(x["exvivo_y"]),
                    float(x["exvivo_z"]),
                    rng,
                    spatial_jitter_vox,
                    ex_vol.shape,
                    ev_vox,
                    jitter_xy,
                    jitter_z,
                )
                raw_ex_ms = _extract_patch_triplet_ex(ex_vol, cex, cey, cez, ev_vox, xy_scales, z_scales)
                aug_iv_ms = [augment_patch(p, rng, preset) for p in raw_iv_ms]
                aug_ex_ms = [augment_patch(p, rng, preset) for p in raw_ex_ms]
                iv_sample = {"cell_idx": ci, "cell_id": cell_ids[ci], "preset": preset, "patch": aug_iv_ms[default_idx]}
                ex_sample = {"cell_idx": ci, "cell_id": cell_ids[ci], "preset": preset, "patch": aug_ex_ms[default_idx]}
                if multiscale_enable:
                    iv_sample["patches_ms"] = aug_iv_ms
                    ex_sample["patches_ms"] = aug_ex_ms
                iv_samples.append(iv_sample)
                ex_samples.append(ex_sample)
        if log_progress:
            done = ci + 1
            elapsed = time.time() - t_build0
            eta_s = (elapsed / done) * (n_cells - done) if done < n_cells else 0.0
            print(
                f"  IV/EX patch build: cells {done}/{n_cells} ({100.0 * done / n_cells:.1f}%) "
                f"samples={len(iv_samples)} elapsed={elapsed:.0f}s eta~{eta_s:.0f}s",
                flush=True,
            )
    if log_progress:
        print(
            f"  IV/EX patch build done: {len(iv_samples)} pairs in {time.time() - t_build0:.1f}s",
            flush=True,
        )
    return cell_ids, iv_samples, ex_samples


def iv_ex_recipe_dict(
    aug_presets: list[str],
    *,
    seed: int,
    aug_per_preset: int,
    patch_xy_um: int,
    patch_z_um: int,
    spatial_jitter_vox: float,
    multiscale_enable: bool,
    multiscale_xy_ums: tuple[int, ...],
    multiscale_z_ums: tuple[int, ...],
) -> dict:
    return {
        "format_version": PATCH_CACHE_FORMAT_VERSION,
        "seed": int(seed),
        "aug_presets": list(aug_presets),
        "aug_per_preset": int(aug_per_preset),
        "patch_xy_um": int(patch_xy_um),
        "patch_z_um": int(patch_z_um),
        "spatial_jitter_vox": float(spatial_jitter_vox),
        "multiscale_enable": bool(multiscale_enable),
        "multiscale_xy_ums": list(int(x) for x in multiscale_xy_ums),
        "multiscale_z_ums": list(int(z) for z in multiscale_z_ums),
    }


def _normalize_recipe(r: dict) -> dict:
    out = dict(r)
    out["format_version"] = int(out.get("format_version", 0))
    out["seed"] = int(out["seed"])
    out["aug_presets"] = list(out["aug_presets"])
    out["aug_per_preset"] = int(out["aug_per_preset"])
    out["patch_xy_um"] = int(out["patch_xy_um"])
    out["patch_z_um"] = int(out["patch_z_um"])
    out["spatial_jitter_vox"] = float(out["spatial_jitter_vox"])
    out["multiscale_enable"] = bool(out["multiscale_enable"])
    out["multiscale_xy_ums"] = [int(x) for x in out["multiscale_xy_ums"]]
    out["multiscale_z_ums"] = [int(z) for z in out["multiscale_z_ums"]]
    return out


def _recipe_mismatch_message(expected: dict, got: dict) -> str:
    keys = sorted(set(expected) | set(got))
    lines = []
    for k in keys:
        if expected.get(k) != got.get(k):
            lines.append(f"  {k}: cache={got.get(k)!r} expected={expected.get(k)!r}")
    return "Patch cache recipe mismatch:\n" + "\n".join(lines) if lines else "unknown mismatch"


def save_iv_ex_patch_cache(
    path: Path,
    cell_ids: list[str],
    iv_samples: list[dict],
    ex_samples: list[dict],
    recipe: dict,
    *,
    log_progress: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ni, ne = len(iv_samples), len(ex_samples)
    if ni != ne:
        raise ValueError(f"iv_samples ({ni}) and ex_samples ({ne}) length mismatch")
    n = ni
    if len(cell_ids) == 0:
        raise ValueError("empty cell_ids")
    if log_progress:
        print(f"Patch cache: stacking {n} IV/EX sample pairs into arrays…", flush=True)
    t0 = time.time()
    cell_idx = np.array([int(s["cell_idx"]) for s in iv_samples], dtype=np.int32)
    cell_id = np.array([str(s["cell_id"]) for s in iv_samples], dtype=object)
    preset = np.array([str(s["preset"]) for s in iv_samples], dtype=object)
    ms = bool(recipe["multiscale_enable"])
    arrays: dict[str, np.ndarray] = {
        "cell_idx": cell_idx,
        "cell_id": cell_id,
        "preset": preset,
    }
    if ms:
        for s in range(3):
            arrays[f"iv_ms_{s}"] = np.stack([iv_samples[i]["patches_ms"][s] for i in range(n)], axis=0).astype(
                np.float32, copy=False
            )
            arrays[f"ex_ms_{s}"] = np.stack([ex_samples[i]["patches_ms"][s] for i in range(n)], axis=0).astype(
                np.float32, copy=False
            )
    else:
        arrays["iv_patch"] = np.stack([iv_samples[i]["patch"] for i in range(n)], axis=0).astype(np.float32, copy=False)
        arrays["ex_patch"] = np.stack([ex_samples[i]["patch"] for i in range(n)], axis=0).astype(np.float32, copy=False)

    if log_progress:
        print(f"Patch cache: stacked in {time.time() - t0:.1f}s; compressing {path.name}…", flush=True)
    t1 = time.time()
    np.savez_compressed(path, **arrays)
    if log_progress:
        print(f"Patch cache: wrote compressed npz in {time.time() - t1:.1f}s", flush=True)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({"recipe": recipe}, indent=2), encoding="utf-8")


def load_iv_ex_patch_cache(
    path: Path,
    recipe_expected: dict,
    *,
    log_progress: bool = True,
) -> tuple[list[str], list[dict], list[dict]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"patch cache not found: {path}")
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"patch cache meta missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    recipe_file = _normalize_recipe(meta.get("recipe", {}))
    recipe_exp = _normalize_recipe(recipe_expected)
    if recipe_file != recipe_exp:
        raise ValueError(_recipe_mismatch_message(recipe_exp, recipe_file))

    if log_progress:
        try:
            mb = path.stat().st_size / (1024 * 1024)
            print(f"Patch cache: reading {path.name} ({mb:.1f} MiB compressed)…", flush=True)
        except OSError:
            print(f"Patch cache: reading {path.name}…", flush=True)
    t_load = time.time()
    data = np.load(path, allow_pickle=True)
    if log_progress:
        print(f"Patch cache: decompressed into memory in {time.time() - t_load:.1f}s", flush=True)
    try:
        cell_idx = data["cell_idx"]
        n = int(cell_idx.shape[0])
        cell_ids_row = [str(x) for x in data["cell_id"].tolist()]
        presets = [str(x) for x in data["preset"].tolist()]
        ms = bool(recipe_exp["multiscale_enable"])
        default_idx = 1 if ms else 0
        iv_samples: list[dict] = []
        ex_samples: list[dict] = []
        step = max(1, n // 10) if n > 5000 else max(1, n)
        t_rec = time.time()
        if ms:
            stacks_iv = [data[f"iv_ms_{s}"] for s in range(3)]
            stacks_ex = [data[f"ex_ms_{s}"] for s in range(3)]
            for i in range(n):
                ci = int(cell_idx[i])
                p_iv = [np.asarray(stacks_iv[s][i], dtype=np.float32).copy() for s in range(3)]
                p_ex = [np.asarray(stacks_ex[s][i], dtype=np.float32).copy() for s in range(3)]
                cid = cell_ids_row[i]
                pr = presets[i]
                iv_samples.append(
                    {"cell_idx": ci, "cell_id": cid, "preset": pr, "patch": p_iv[default_idx], "patches_ms": p_iv}
                )
                ex_samples.append(
                    {"cell_idx": ci, "cell_id": cid, "preset": pr, "patch": p_ex[default_idx], "patches_ms": p_ex}
                )
                if log_progress and n > 2000 and ((i + 1) % step == 0 or i + 1 == n):
                    print(f"  Patch cache: reconstructed {i + 1}/{n} sample pairs…", flush=True)
        else:
            iv_st = data["iv_patch"]
            ex_st = data["ex_patch"]
            for i in range(n):
                ci = int(cell_idx[i])
                cid = cell_ids_row[i]
                pr = presets[i]
                iv_samples.append(
                    {
                        "cell_idx": ci,
                        "cell_id": cid,
                        "preset": pr,
                        "patch": np.asarray(iv_st[i], dtype=np.float32).copy(),
                    }
                )
                ex_samples.append(
                    {
                        "cell_idx": ci,
                        "cell_id": cid,
                        "preset": pr,
                        "patch": np.asarray(ex_st[i], dtype=np.float32).copy(),
                    }
                )
                if log_progress and n > 2000 and ((i + 1) % step == 0 or i + 1 == n):
                    print(f"  Patch cache: reconstructed {i + 1}/{n} sample pairs…", flush=True)
        if log_progress:
            print(f"Patch cache: reconstructed {n} sample pairs in {time.time() - t_rec:.1f}s", flush=True)
    finally:
        data.close()

    if not iv_samples:
        return [], [], []
    max_ci = max(int(s["cell_idx"]) for s in iv_samples)
    cell_ids_out = [""] * (max_ci + 1)
    for s in iv_samples:
        ci = int(s["cell_idx"])
        if cell_ids_out[ci] == "":
            cell_ids_out[ci] = str(s["cell_id"])
    if any(x == "" for x in cell_ids_out):
        raise ValueError("patch cache: missing cell_id for some cell_idx values")
    return cell_ids_out, iv_samples, ex_samples


# ── Embedding cache helpers ────────────────────────────────────────────────

def _emb_cache_is_valid(emb_cache_dir: str, recipe: dict) -> bool:
    """True if the directory contains embeddings computed from a matching recipe."""
    d = Path(emb_cache_dir)
    if not (d / "iv_emb.npy").is_file() or not (d / "ex_emb.npy").is_file():
        return False
    if not (d / "iv_cell_idx.npy").is_file() or not (d / "ex_cell_idx.npy").is_file():
        return False
    rp = d / "emb_recipe.json"
    if not rp.is_file():
        return False
    try:
        cached = _normalize_recipe(json.loads(rp.read_text(encoding="utf-8")))
        return cached == _normalize_recipe(recipe)
    except Exception:
        return False


def _save_emb_cache(
    emb_cache_dir: str,
    iv_emb: np.ndarray,
    ex_emb: np.ndarray,
    iv_cell_idx: list[int],
    ex_cell_idx: list[int],
    cell_ids: list[str],
    recipe: dict,
) -> None:
    d = Path(emb_cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "iv_emb.npy", iv_emb.astype(np.float32))
    np.save(d / "ex_emb.npy", ex_emb.astype(np.float32))
    np.save(d / "iv_cell_idx.npy", np.array(iv_cell_idx, dtype=np.int32))
    np.save(d / "ex_cell_idx.npy", np.array(ex_cell_idx, dtype=np.int32))
    (d / "cell_ids.json").write_text(json.dumps(cell_ids), encoding="utf-8")
    (d / "emb_recipe.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    sz = sum((d / f).stat().st_size for f in ["iv_emb.npy", "ex_emb.npy"]) / 1e6
    print(f"Emb cache: saved {sz:.0f} MB to {d}", flush=True)


def _load_emb_cache(
    emb_cache_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    d = Path(emb_cache_dir)
    iv_emb = np.load(d / "iv_emb.npy")
    ex_emb = np.load(d / "ex_emb.npy")
    iv_cell_idx = np.load(d / "iv_cell_idx.npy")
    ex_cell_idx = np.load(d / "ex_cell_idx.npy")
    cell_ids = json.loads((d / "cell_ids.json").read_text(encoding="utf-8"))
    print(f"Emb cache: loaded iv={iv_emb.shape} ex={ex_emb.shape} from {d}", flush=True)
    return iv_emb, ex_emb, iv_cell_idx, ex_cell_idx, cell_ids


def split_iv_train_val_cells(iv_samples: list[dict], seed: int, val_frac: float = 0.2):
    cells = sorted({s["cell_idx"] for s in iv_samples})
    rng = np.random.default_rng(seed)
    perm = np.array(cells)
    rng.shuffle(perm)
    n_val = max(1, int(round(val_frac * len(perm))))
    val_cells = set(perm[:n_val].tolist())
    train_cells = set(perm[n_val:].tolist())
    tr_idx = [i for i, s in enumerate(iv_samples) if s["cell_idx"] in train_cells]
    va_idx = [i for i, s in enumerate(iv_samples) if s["cell_idx"] in val_cells]
    return tr_idx, va_idx, train_cells, val_cells


def make_split_indices(samples: list[dict], left_out_cell: int, seed: int):
    # Cell-level split among non-held-out cells.
    non_hold_cells = sorted({s["cell_idx"] for s in samples if s["cell_idx"] != left_out_cell})
    rng = np.random.default_rng(seed)
    perm = np.array(non_hold_cells)
    rng.shuffle(perm)
    n_train = int(0.8 * len(perm))
    train_cells = set(perm[:n_train].tolist())
    val_cells = set(perm[n_train:].tolist())
    tr_idx = [i for i, s in enumerate(samples) if s["cell_idx"] in train_cells]
    va_idx = [i for i, s in enumerate(samples) if s["cell_idx"] in val_cells]
    te_idx = [i for i, s in enumerate(samples) if s["cell_idx"] == left_out_cell]
    return tr_idx, va_idx, te_idx, train_cells, val_cells


def run_fold(
    emb: np.ndarray,
    samples: list[dict],
    left_out_cell: int,
    epochs: int,
    batch: int,
    lr: float,
    wd: float,
    temp: float,
    k: int,
    seed: int,
    capture_snapshots: bool = False,
    num_total_cells: int = 115,
    mlp_out_dim: int = 64,
):
    tr_idx, va_idx, te_idx, train_cells, val_cells = make_split_indices(samples, left_out_cell, seed=seed + left_out_cell)

    Xtr = torch.tensor(emb[tr_idx], dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor([samples[i]["cell_idx"] for i in tr_idx], dtype=torch.long, device=DEVICE)
    Xva = torch.tensor(emb[va_idx], dtype=torch.float32, device=DEVICE)
    yva_cell = torch.tensor([samples[i]["cell_idx"] for i in va_idx], dtype=torch.long, device=DEVICE)
    Xte = torch.tensor(emb[te_idx], dtype=torch.float32, device=DEVICE)
    Xtv = torch.tensor(emb[tr_idx + va_idx], dtype=torch.float32, device=DEVICE)
    ytv_cell = torch.tensor([samples[i]["cell_idx"] for i in (tr_idx + va_idx)], dtype=torch.long, device=DEVICE)
    Xall_ref = torch.tensor(emb[tr_idx + va_idx + te_idx], dtype=torch.float32, device=DEVICE)
    yall_cell = torch.tensor([samples[i]["cell_idx"] for i in (tr_idx + va_idx + te_idx)], dtype=torch.long, device=DEVICE)

    proj = MLPProjector(in_dim=emb.shape[1], hidden=256, out_dim=mlp_out_dim).to(DEVICE)
    opt = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=wd)

    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    hist = []
    snaps = []

    for ep in range(1, epochs + 1):
        proj.train()
        perm = torch.randperm(Xtr.shape[0], device=DEVICE)
        losses = []
        eff_batch = Xtr.shape[0] if (batch <= 0 or batch >= Xtr.shape[0]) else batch
        for s in range(0, Xtr.shape[0], eff_batch):
            idx = perm[s:s + eff_batch]
            z = proj(Xtr[idx])
            loss = supervised_contrastive_loss(z, ytr[idx], temperature=temp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        proj.eval()
        with torch.no_grad():
            ztv = proj(Xtv)
            n_tr = len(tr_idx)
            zq = ztv[n_tr:]  # val queries
            sim = zq @ ztv.t()  # (n_val, n_tv)
            # exclude exact self matches in train+val pool
            if len(va_idx) > 0:
                row_idx = torch.arange(len(va_idx), device=DEVICE)
                col_idx = n_tr + row_idx
                sim[row_idx, col_idx] = -1e9
            topk = torch.topk(sim, k=min(k, ztv.shape[0] - 1), dim=1).indices
            nn_labels = ytv_cell[topk]
            preds = knn_majority_labels(nn_labels, num_classes=num_total_cells)
            va_acc = float((preds == yva_cell).float().mean().item())
        tr_loss = float(np.mean(losses)) if losses else float("nan")
        hist.append({"epoch": ep, "train_loss": tr_loss, "val_knn_acc": va_acc})
        if va_acc > best["val_acc"]:
            best = {
                "val_acc": va_acc,
                "epoch": ep,
                "state": {n: p.detach().cpu().clone() for n, p in proj.state_dict().items()},
            }
        if capture_snapshots:
            with torch.no_grad():
                zall = proj(torch.tensor(emb, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            snaps.append({"epoch": ep, "z": zall})

    # load best state
    proj.load_state_dict(best["state"])
    proj.eval()
    with torch.no_grad():
        zall = proj(Xall_ref)
        # Test protocol: include train+val+test, exclude exact query sample;
        # measure whether nearest neighbors mostly belong to held-out cell.
        n_ref_wo_test = len(tr_idx) + len(va_idx)
        zte = zall[n_ref_wo_test:]
        sim = zte @ zall.t()  # (n_test, n_all)
        if len(te_idx) > 0:
            row_idx = torch.arange(len(te_idx), device=DEVICE)
            col_idx = n_ref_wo_test + row_idx
            sim[row_idx, col_idx] = -1e9
        topk = torch.topk(sim, k=min(k, zall.shape[0] - 1), dim=1).indices
        nn_cells = yall_cell[topk]
        pred_cell = knn_majority_labels(nn_cells, num_classes=num_total_cells)
        test_true = torch.full((len(te_idx),), left_out_cell, dtype=torch.long, device=DEVICE)
        sample_top1_acc = float((pred_cell == test_true).float().mean().item())
        purity = float((nn_cells == left_out_cell).float().mean().item())

    cell_correct = bool(sample_top1_acc >= 0.5)
    out = {
        "left_out_cell": left_out_cell,
        "best_epoch": int(best["epoch"]),
        "best_val_knn_acc": float(best["val_acc"]),
        "test_sample_top1_acc": sample_top1_acc,
        "test_nn_purity_topk": purity,
        "cell_correct_majority": cell_correct,
        "n_train_cells": len(train_cells),
        "n_val_cells": len(val_cells),
        "test_pred_cells": [int(x) for x in pred_cell.detach().cpu().tolist()],
        "test_true_cells": [int(left_out_cell)] * len(te_idx),
        "history": hist,
    }
    if capture_snapshots:
        out["snapshots"] = snaps
        out["split"] = {
            "train_idx": tr_idx,
            "val_idx": va_idx,
            "test_idx": te_idx,
        }
    return out


def run_full_iv_training(
    emb_iv: np.ndarray,
    emb_ex: np.ndarray,
    iv_samples: list[dict],
    ex_samples: list[dict],
    epochs: int,
    batch: int,
    lr: float,
    wd: float,
    temp: float,
    k: int,
    seed: int,
    snapshot_every: int,
    num_cells: int,
    train_source: str,
    mlp_out_dim: int = 64,
    alignment_loss_name: str = "none",
    lambda_align: float = 0.0,
    dann_schedule: str = "none",
    shared_labels: bool = False,
    lambda_cross_pair: float = 0.0,
    eval_geom_zero: bool = False,
    geom_dim: int = 0,
):
    iv_tr_idx, iv_va_idx, iv_train_cells, iv_val_cells = split_iv_train_val_cells(iv_samples, seed=seed, val_frac=0.2)
    ex_tr_idx, ex_va_idx, ex_train_cells, ex_val_cells = split_iv_train_val_cells(ex_samples, seed=seed + 1000, val_frac=0.2)

    # Disjoint label spaces (unless shared_labels): IV 0..N-1, EX N..2N-1 ⇒ SupCon positives only within each modality.
    y_iv = np.array([s["cell_idx"] for s in iv_samples], dtype=np.int64)
    if shared_labels:
        y_ex = np.array([s["cell_idx"] for s in ex_samples], dtype=np.int64)
    else:
        y_ex = np.array([s["cell_idx"] + num_cells for s in ex_samples], dtype=np.int64)
    y_ex_landmark = np.array([s["cell_idx"] for s in ex_samples], dtype=np.int64)

    if train_source == "iv_only":
        tr_idx_global = [("iv", i) for i in iv_tr_idx]
    elif train_source == "ex_only":
        tr_idx_global = [("ex", i) for i in ex_tr_idx]
    else:
        tr_idx_global = [("iv", i) for i in iv_tr_idx] + [("ex", i) for i in ex_tr_idx]

    Xtr = np.concatenate(
        [emb_iv[[i for m, i in tr_idx_global if m == "iv"]], emb_ex[[i for m, i in tr_idx_global if m == "ex"]]],
        axis=0,
    ).astype(np.float32)
    ytr = np.concatenate(
        [y_iv[[i for m, i in tr_idx_global if m == "iv"]], y_ex[[i for m, i in tr_idx_global if m == "ex"]]],
        axis=0,
    ).astype(np.int64)
    dtr = np.array([0 if m == "iv" else 1 for m, _ in tr_idx_global], dtype=np.int64)
    lm_tr = np.array(
        [iv_samples[i]["cell_idx"] if m == "iv" else ex_samples[i]["cell_idx"] for m, i in tr_idx_global],
        dtype=np.int64,
    )

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    dtr_t = torch.tensor(dtr, dtype=torch.long, device=DEVICE)
    lmtr_t = torch.tensor(lm_tr, dtype=torch.long, device=DEVICE)
    Xiv_all = torch.tensor(emb_iv, dtype=torch.float32, device=DEVICE)
    Xex_all = torch.tensor(emb_ex, dtype=torch.float32, device=DEVICE)
    yiv_all_landmark = torch.tensor([s["cell_idx"] for s in iv_samples], dtype=torch.long, device=DEVICE)
    yex_all_landmark = torch.tensor(y_ex_landmark, dtype=torch.long, device=DEVICE)

    proj = MLPProjector(in_dim=emb_iv.shape[1], hidden=256, out_dim=mlp_out_dim).to(DEVICE)
    opt = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=wd)
    align_fn = get_alignment_loss(alignment_loss_name)
    disc = None
    opt_disc = None
    if alignment_loss_name == "dann":
        disc = DomainDiscriminator(in_dim=mlp_out_dim).to(DEVICE)
        opt_disc = torch.optim.AdamW(disc.parameters(), lr=lr, weight_decay=wd)
    hist = []
    snapshots = []
    best = {"val_acc": -1.0, "epoch": -1, "state": None}

    for ep in range(1, epochs + 1):
        proj.train()
        if disc is not None:
            disc.train()
        perm = torch.randperm(Xtr_t.shape[0], device=DEVICE)
        losses = []
        align_losses = []
        cross_pair_losses = []
        eff_batch = Xtr_t.shape[0] if (batch <= 0 or batch >= Xtr_t.shape[0]) else batch
        for s in range(0, Xtr_t.shape[0], eff_batch):
            idx = perm[s:s + eff_batch]
            d_batch = dtr_t[idx]
            z = proj(Xtr_t[idx])
            sup_loss = supervised_contrastive_loss(z, ytr_t[idx], temperature=temp)
            z_align = z
            align_loss = torch.tensor(0.0, device=DEVICE)
            if train_source == "joint" and alignment_loss_name != "none" and lambda_align > 0:
                iv_mask = d_batch == 0
                ex_mask = d_batch == 1
                if bool(iv_mask.any()) and bool(ex_mask.any()):
                    z_iv = z_align[iv_mask]
                    z_ex = z_align[ex_mask]
                    if alignment_loss_name == "dann":
                        alpha = dann_alpha_schedule(ep, epochs, schedule=dann_schedule)
                        align_loss = dann_loss(z_iv, z_ex, discriminator=disc, alpha=alpha)
                    else:
                        align_loss = align_fn(z_iv, z_ex)
            cp_loss = torch.tensor(0.0, device=DEVICE)
            if train_source == "joint" and lambda_cross_pair > 0 and not shared_labels:
                cp_loss = cross_domain_pair_loss(z_align, d_batch, lmtr_t[idx])
            loss = sup_loss + (lambda_align * align_loss) + (lambda_cross_pair * cp_loss)
            opt.zero_grad()
            if opt_disc is not None:
                opt_disc.zero_grad()
            loss.backward()
            opt.step()
            if opt_disc is not None:
                opt_disc.step()
            losses.append(float(loss.item()))
            align_losses.append(float(align_loss.item()))
            cross_pair_losses.append(float(cp_loss.item()))

        proj.eval()
        with torch.no_grad():
            Xe_iv = torch_eval_geom_zero(Xiv_all, geom_dim, eval_geom_zero)
            Xe_ex = torch_eval_geom_zero(Xex_all, geom_dim, eval_geom_zero)
            ziv_all = proj(Xe_iv)
            zex_all = proj(Xe_ex)

            # All similarity evals are batched on CPU to avoid MPS OOM for large N.
            _EVAL_BATCH = 2048

            # IV val kNN: query iv_val vs iv_train+iv_val with self exclusion
            ztv_iv = ziv_all[iv_tr_idx + iv_va_idx].cpu()
            ytv_iv_cpu = torch.tensor([iv_samples[i]["cell_idx"] for i in (iv_tr_idx + iv_va_idx)], dtype=torch.long)
            n_iv_tr = len(iv_tr_idx)
            zq_iv_cpu = ztv_iv[n_iv_tr:]
            _k_iv = min(k, ztv_iv.shape[0] - 1)
            _topk_iv_chunks: list[torch.Tensor] = []
            for _bs in range(0, zq_iv_cpu.shape[0], _EVAL_BATCH):
                _sim_b = zq_iv_cpu[_bs:_bs + _EVAL_BATCH] @ ztv_iv.t()
                # self-exclusion: each val query is at offset n_iv_tr+_bs in ztv_iv
                for _qi in range(_sim_b.shape[0]):
                    _sim_b[_qi, n_iv_tr + _bs + _qi] = -1e9
                _topk_iv_chunks.append(torch.topk(_sim_b, k=_k_iv, dim=1).indices)
            topk_iv_cpu = torch.cat(_topk_iv_chunks, dim=0)
            nn_iv = ytv_iv_cpu[topk_iv_cpu]
            pred_iv = knn_majority_labels(nn_iv.to(DEVICE), num_classes=num_cells).cpu()
            y_iv_val_cpu = torch.tensor([iv_samples[i]["cell_idx"] for i in iv_va_idx], dtype=torch.long)
            iv_val_acc = float((pred_iv == y_iv_val_cpu).float().mean().item())

            # EX val kNN: query ex_val vs ex_train+ex_val with self exclusion
            ztv_ex = zex_all[ex_tr_idx + ex_va_idx].cpu()
            ytv_ex_cpu = torch.tensor([ex_samples[i]["cell_idx"] for i in (ex_tr_idx + ex_va_idx)], dtype=torch.long)
            n_ex_tr = len(ex_tr_idx)
            zq_ex_cpu = ztv_ex[n_ex_tr:]
            _k_ex = min(k, ztv_ex.shape[0] - 1)
            _topk_ex_chunks: list[torch.Tensor] = []
            for _bs in range(0, zq_ex_cpu.shape[0], _EVAL_BATCH):
                _sim_b = zq_ex_cpu[_bs:_bs + _EVAL_BATCH] @ ztv_ex.t()
                for _qi in range(_sim_b.shape[0]):
                    _sim_b[_qi, n_ex_tr + _bs + _qi] = -1e9
                _topk_ex_chunks.append(torch.topk(_sim_b, k=_k_ex, dim=1).indices)
            topk_exv_cpu = torch.cat(_topk_ex_chunks, dim=0)
            nn_exv = ytv_ex_cpu[topk_exv_cpu]
            pred_exv = knn_majority_labels(nn_exv.to(DEVICE), num_classes=num_cells).cpu()
            y_ex_val_cpu = torch.tensor([ex_samples[i]["cell_idx"] for i in ex_va_idx], dtype=torch.long)
            ex_val_acc = float((pred_exv == y_ex_val_cpu).float().mean().item())

            # EX->IV identification (all EX queries against all IV refs)
            _zex_cpu = zex_all.cpu()
            _ziv_cpu = ziv_all.cpu()
            _k_eval  = min(k, _ziv_cpu.shape[0])
            _topk_chunks: list[torch.Tensor] = []
            for _bs in range(0, _zex_cpu.shape[0], _EVAL_BATCH):
                _sim_b = _zex_cpu[_bs : _bs + _EVAL_BATCH] @ _ziv_cpu.t()
                _topk_chunks.append(torch.topk(_sim_b, k=_k_eval, dim=1).indices)
            topk_ex_to_iv = torch.cat(_topk_chunks, dim=0).to(DEVICE)
            nn_ex_to_iv = yiv_all_landmark[topk_ex_to_iv]
            pred_ex_to_iv = knn_majority_labels(nn_ex_to_iv, num_classes=num_cells)
            ex_id_acc = float((pred_ex_to_iv == yex_all_landmark).float().mean().item())

            # model selection by balanced IV/EX val accuracy
            val_bal_acc = 0.5 * (iv_val_acc + ex_val_acc)

        train_loss = float(np.mean(losses)) if losses else float("nan")
        align_loss_epoch = float(np.mean(align_losses)) if align_losses else 0.0
        cross_pair_epoch = float(np.mean(cross_pair_losses)) if cross_pair_losses else 0.0
        hist.append(
            {
                "epoch": ep,
                "train_loss": train_loss,
                "alignment_loss": align_loss_epoch,
                "cross_pair_loss": cross_pair_epoch,
                "iv_val_knn_acc": iv_val_acc,
                "ex_val_knn_acc": ex_val_acc,
                "ex_to_iv_knn_acc": ex_id_acc,
                "val_balanced_acc": val_bal_acc,
            }
        )
        print(
            f"[full_iv_ex/frozen] epoch {ep}/{epochs} "
            f"loss={train_loss:.4f} iv_val={iv_val_acc:.4f} ex_val={ex_val_acc:.4f} "
            f"bal_val={val_bal_acc:.4f} ex2iv={ex_id_acc:.4f} align={align_loss_epoch:.4f} "
            f"cpair={cross_pair_epoch:.4f}",
            flush=True,
        )
        if val_bal_acc > best["val_acc"]:
            best = {
                "val_acc": val_bal_acc,
                "epoch": ep,
                "iv_val_acc": iv_val_acc,
                "ex_val_acc": ex_val_acc,
                "state": {n: p.detach().cpu().clone() for n, p in proj.state_dict().items()},
            }
        if ep % snapshot_every == 0 or ep == 1 or ep == epochs:
            snapshots.append({"epoch": ep, "ziv": ziv_all.detach().cpu().numpy(), "zex": zex_all.detach().cpu().numpy()})

    proj.eval()
    proj.load_state_dict(best["state"])
    return proj, hist, best, snapshots, iv_tr_idx, iv_va_idx, iv_train_cells, iv_val_cells, ex_tr_idx, ex_va_idx, ex_train_cells, ex_val_cells


def _run_tail_on_cache_batches(
    cache: torch.Tensor,
    tail_fn,
    batch_sz: int = 32,
) -> torch.Tensor:
    """Run tail on cache in batches, return (N, D) CLS tensor."""
    out = []
    for s in range(0, cache.shape[0], batch_sz):
        h = cache[s : s + batch_sz].to(DEVICE).float()  # float16 cache → float32 for transformer
        with torch.no_grad():
            cls = tail_fn(h)
        out.append(F.normalize(cls, dim=-1).cpu())
    return torch.cat(out, dim=0)


def run_full_iv_ex_lora_training(
    cache_iv: torch.Tensor | None,
    cache_ex: torch.Tensor | None,
    iv_samples: list[dict],
    ex_samples: list[dict],
    epochs: int,
    batch: int,
    lr: float,
    wd: float,
    temp: float,
    k: int,
    seed: int,
    snapshot_every: int,
    num_cells: int,
    train_source: str,
    mlp_out_dim: int,
    tail_fn,
    proj: nn.Module,
    opt: torch.optim.Optimizer,
    best_state_callback,
    lora_model=None,
    alignment_loss_name: str = "none",
    lambda_align: float = 0.0,
    dann_schedule: str = "none",
    shared_labels: bool = False,
    lambda_cross_pair: float = 0.0,
    geom_feat_iv: np.ndarray | None = None,
    geom_feat_ex: np.ndarray | None = None,
    eval_geom_zero: bool = False,
    geom_dim: int = 0,
    backbone_model=None,
    backbone_processor=None,
    n_frozen_blocks: int = 0,
    otf_embed_batch: int = 8,
):
    """
    Joint IV+EX contrastive training with LoRA backbone + MLP.
    tail_fn: callable(h) -> (B, D) CLS from cached hidden states.
    best_state_callback: callable() -> dict to save model/proj state.
    Returns same structure as run_full_iv_training.
    """
    G_iv_t = G_ex_t = None
    if geom_feat_iv is not None and geom_feat_ex is not None:
        G_iv_t = torch.tensor(geom_feat_iv, dtype=torch.float32, device=DEVICE)
        G_ex_t = torch.tensor(geom_feat_ex, dtype=torch.float32, device=DEVICE)

    iv_tr_idx, iv_va_idx, iv_train_cells, iv_val_cells = split_iv_train_val_cells(iv_samples, seed=seed, val_frac=0.2)
    ex_tr_idx, ex_va_idx, ex_train_cells, ex_val_cells = split_iv_train_val_cells(ex_samples, seed=seed + 1000, val_frac=0.2)

    y_iv = np.array([s["cell_idx"] for s in iv_samples], dtype=np.int64)
    if shared_labels:
        y_ex = np.array([s["cell_idx"] for s in ex_samples], dtype=np.int64)
    else:
        y_ex = np.array([s["cell_idx"] + num_cells for s in ex_samples], dtype=np.int64)
    y_ex_landmark = np.array([s["cell_idx"] for s in ex_samples], dtype=np.int64)

    if train_source == "iv_only":
        tr_idx_global = [("iv", i) for i in iv_tr_idx]
    elif train_source == "ex_only":
        tr_idx_global = [("ex", i) for i in ex_tr_idx]
    else:
        tr_idx_global = [("iv", i) for i in iv_tr_idx] + [("ex", i) for i in ex_tr_idx]

    tr_iv_idx = [i for m, i in tr_idx_global if m == "iv"]
    tr_ex_idx = [i for m, i in tr_idx_global if m == "ex"]
    ytr = np.concatenate(
        [y_iv[tr_iv_idx], y_ex[tr_ex_idx]],
        axis=0,
    ).astype(np.int64)
    dtr = np.array([0 if m == "iv" else 1 for m, _ in tr_idx_global], dtype=np.int64)
    lm_tr = np.array(
        [iv_samples[i]["cell_idx"] if m == "iv" else ex_samples[i]["cell_idx"] for m, i in tr_idx_global],
        dtype=np.int64,
    )
    n_tr = len(tr_idx_global)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    dtr_t = torch.tensor(dtr, dtype=torch.long, device=DEVICE)
    lmtr_t = torch.tensor(lm_tr, dtype=torch.long, device=DEVICE)
    yiv_all_landmark = torch.tensor([s["cell_idx"] for s in iv_samples], dtype=torch.long, device=DEVICE)
    yex_all_landmark = torch.tensor(y_ex_landmark, dtype=torch.long, device=DEVICE)

    hist = []
    snapshots = []
    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    eff_batch = n_tr if (batch <= 0 or batch >= n_tr) else min(batch, n_tr)
    align_fn = get_alignment_loss(alignment_loss_name)
    disc = None
    opt_disc = None
    if alignment_loss_name == "dann":
        disc = DomainDiscriminator(in_dim=mlp_out_dim).to(DEVICE)
        opt_disc = torch.optim.AdamW(disc.parameters(), lr=lr, weight_decay=wd)

    for ep in range(1, epochs + 1):
        if lora_model is not None:
            lora_model.train()
        proj.train()
        if disc is not None:
            disc.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        losses = []
        align_losses = []
        cross_pair_losses = []
        for s in range(0, n_tr, eff_batch):
            idx = perm[s : s + eff_batch].cpu().numpy()
            if backbone_model is not None:
                # On-the-fly mode: run frozen prefix per mini-batch, no huge pre-cache
                raw_patches = []
                for i in idx:
                    mod, ii = tr_idx_global[i]
                    p = iv_samples[ii]["patch"] if mod == "iv" else ex_samples[ii]["patch"]
                    raw_patches.append((np.clip(p * 255, 0, 255)).astype("uint8"))
                inp = backbone_processor(images=raw_patches, return_tensors="pt")
                inp = {kk: vv.to(DEVICE) for kk, vv in inp.items()}
                with torch.no_grad():
                    h = backbone_model.embeddings(inp["pixel_values"])
                    for _bi in range(n_frozen_blocks):
                        h = backbone_model.encoder.layer[_bi](h)
                cls_batch = tail_fn(h)  # LoRA tail gets gradients
            else:
                batch_h = []
                for i in idx:
                    mod, ii = tr_idx_global[i]
                    if mod == "iv":
                        batch_h.append(cache_iv[ii])
                    else:
                        batch_h.append(cache_ex[ii])
                h_batch = torch.stack(batch_h, dim=0).to(DEVICE).float()
                cls_batch = tail_fn(h_batch)
            if G_iv_t is not None:
                geom_rows = []
                for ii in idx:
                    mod, j = tr_idx_global[int(ii)]
                    geom_rows.append(G_iv_t[j] if mod == "iv" else G_ex_t[j])
                geom_b = torch.stack(geom_rows, dim=0)
                inp_batch = torch.cat([cls_batch, geom_b], dim=-1)
            else:
                inp_batch = cls_batch
            z = proj(inp_batch)
            idx_t = torch.as_tensor(idx, device=DEVICE)
            batch_y = ytr_t[idx_t]
            sup_loss = supervised_contrastive_loss(z, batch_y, temperature=temp)
            align_loss = torch.tensor(0.0, device=DEVICE)
            d_batch = dtr_t[idx_t]
            if train_source == "joint" and alignment_loss_name != "none" and lambda_align > 0:
                iv_mask = d_batch == 0
                ex_mask = d_batch == 1
                if bool(iv_mask.any()) and bool(ex_mask.any()):
                    z_iv = z[iv_mask]
                    z_ex = z[ex_mask]
                    if alignment_loss_name == "dann":
                        alpha = dann_alpha_schedule(ep, epochs, schedule=dann_schedule)
                        align_loss = dann_loss(z_iv, z_ex, discriminator=disc, alpha=alpha)
                    else:
                        align_loss = align_fn(z_iv, z_ex)
            cp_loss = torch.tensor(0.0, device=DEVICE)
            if (
                train_source == "joint"
                and lambda_cross_pair > 0
                and not shared_labels
            ):
                cp_loss = cross_domain_pair_loss(z, d_batch, lmtr_t[idx_t])
            loss = sup_loss + (lambda_align * align_loss) + (lambda_cross_pair * cp_loss)
            opt.zero_grad()
            if opt_disc is not None:
                opt_disc.zero_grad()
            loss.backward()
            opt.step()
            if opt_disc is not None:
                opt_disc.step()
            losses.append(float(loss.item()))
            align_losses.append(float(align_loss.item()))
            cross_pair_losses.append(float(cp_loss.item()))

        if lora_model is not None:
            lora_model.eval()
        proj.eval()
        with torch.no_grad():
            if backbone_model is not None:
                # On-the-fly eval: process patches in small batches → proj → CPU
                # Use a larger batch for eval (no grad = less MPS memory pressure).
                _eval_embed_batch = max(otf_embed_batch, 128)
                def _otf_embed(samples):
                    _out = []
                    patches_all = [s["patch"] for s in samples]
                    for _s in range(0, len(patches_all), _eval_embed_batch):
                        _chunk = patches_all[_s:_s + _eval_embed_batch]
                        _imgs = [(np.clip(p * 255, 0, 255)).astype("uint8") for p in _chunk]
                        _inp = backbone_processor(images=_imgs, return_tensors="pt")
                        _inp = {kk: vv.to(DEVICE) for kk, vv in _inp.items()}
                        _h = backbone_model.embeddings(_inp["pixel_values"])
                        for _bi in range(n_frozen_blocks):
                            _h = backbone_model.encoder.layer[_bi](_h)
                        _cls = tail_fn(_h)
                        _z = F.normalize(proj(_cls), dim=-1)
                        _out.append(_z.cpu())
                    return torch.cat(_out, dim=0)
                ziv_all = _otf_embed(iv_samples)
                zex_all = _otf_embed(ex_samples)
            else:
                cls_iv = _run_tail_on_cache_batches(cache_iv, tail_fn)
                cls_ex = _run_tail_on_cache_batches(cache_ex, tail_fn)
                if G_iv_t is not None:
                    inp_iv = torch.cat([cls_iv, G_iv_t.cpu()], dim=-1).to(DEVICE)
                    inp_ex = torch.cat([cls_ex, G_ex_t.cpu()], dim=-1).to(DEVICE)
                    inp_iv = torch_eval_geom_zero(inp_iv, geom_dim, eval_geom_zero)
                    inp_ex = torch_eval_geom_zero(inp_ex, geom_dim, eval_geom_zero)
                    ziv_all = proj(inp_iv).cpu().to(DEVICE)
                    zex_all = proj(inp_ex).cpu().to(DEVICE)
                else:
                    ziv_all = proj(cls_iv.to(DEVICE)).cpu().to(DEVICE)
                    zex_all = proj(cls_ex.to(DEVICE)).cpu().to(DEVICE)

            # Use CPU-batched eval to avoid MPS OOM on large embeddings
            _EVAL_BATCH = 2048

            ztv_iv = ziv_all[iv_tr_idx + iv_va_idx]
            ytv_iv = torch.tensor([iv_samples[i]["cell_idx"] for i in (iv_tr_idx + iv_va_idx)], dtype=torch.long, device=DEVICE)
            n_iv_tr = len(iv_tr_idx)
            zq_iv = ztv_iv[n_iv_tr:]
            ztv_iv_cpu = ztv_iv.cpu()
            zq_iv_cpu = zq_iv.cpu()
            ytv_iv_cpu = ytv_iv.cpu()
            topk_iv_chunks = []
            for _bs in range(0, zq_iv_cpu.shape[0], _EVAL_BATCH):
                _sim_b = zq_iv_cpu[_bs:_bs + _EVAL_BATCH] @ ztv_iv_cpu.t()
                for _qi in range(_sim_b.shape[0]):
                    _sim_b[_qi, n_iv_tr + _bs + _qi] = -1e9
                topk_iv_chunks.append(torch.topk(_sim_b, k=min(k, ztv_iv_cpu.shape[0] - 1), dim=1).indices)
            topk_iv = torch.cat(topk_iv_chunks, dim=0)
            nn_iv = ytv_iv_cpu[topk_iv]
            pred_iv = knn_majority_labels(nn_iv.to(DEVICE), num_classes=num_cells).cpu()
            y_iv_val = torch.tensor([iv_samples[i]["cell_idx"] for i in iv_va_idx], dtype=torch.long)
            iv_val_acc = float((pred_iv == y_iv_val).float().mean().item())

            ztv_ex = zex_all[ex_tr_idx + ex_va_idx]
            ytv_ex = torch.tensor([ex_samples[i]["cell_idx"] for i in (ex_tr_idx + ex_va_idx)], dtype=torch.long, device=DEVICE)
            n_ex_tr = len(ex_tr_idx)
            zq_ex = ztv_ex[n_ex_tr:]
            ztv_ex_cpu = ztv_ex.cpu()
            zq_ex_cpu = zq_ex.cpu()
            ytv_ex_cpu = ytv_ex.cpu()
            topk_exv_chunks = []
            for _bs in range(0, zq_ex_cpu.shape[0], _EVAL_BATCH):
                _sim_b = zq_ex_cpu[_bs:_bs + _EVAL_BATCH] @ ztv_ex_cpu.t()
                for _qi in range(_sim_b.shape[0]):
                    _sim_b[_qi, n_ex_tr + _bs + _qi] = -1e9
                topk_exv_chunks.append(torch.topk(_sim_b, k=min(k, ztv_ex_cpu.shape[0] - 1), dim=1).indices)
            topk_exv = torch.cat(topk_exv_chunks, dim=0)
            nn_exv = ytv_ex_cpu[topk_exv]
            pred_exv = knn_majority_labels(nn_exv.to(DEVICE), num_classes=num_cells).cpu()
            y_ex_val = torch.tensor([ex_samples[i]["cell_idx"] for i in ex_va_idx], dtype=torch.long)
            ex_val_acc = float((pred_exv == y_ex_val).float().mean().item())

            # EX→IV identification: CPU-batched to avoid OOM
            ziv_all_cpu = ziv_all.cpu()
            zex_all_cpu = zex_all.cpu()
            yiv_all_landmark_cpu = yiv_all_landmark.cpu()
            topk_ex_to_iv_chunks = []
            for _bs in range(0, zex_all_cpu.shape[0], _EVAL_BATCH):
                _sim_b = zex_all_cpu[_bs:_bs + _EVAL_BATCH] @ ziv_all_cpu.t()
                topk_ex_to_iv_chunks.append(torch.topk(_sim_b, k=min(k, ziv_all_cpu.shape[0]), dim=1).indices)
            topk_ex_to_iv = torch.cat(topk_ex_to_iv_chunks, dim=0)
            nn_ex_to_iv = yiv_all_landmark_cpu[topk_ex_to_iv]
            pred_ex_to_iv = knn_majority_labels(nn_ex_to_iv.to(DEVICE), num_classes=num_cells).cpu()
            yex_all_landmark_cpu = yex_all_landmark.cpu()
            ex_id_acc = float((pred_ex_to_iv == yex_all_landmark_cpu).float().mean().item())

            val_bal_acc = 0.5 * (iv_val_acc + ex_val_acc)

        train_loss = float(np.mean(losses)) if losses else float("nan")
        align_loss_epoch = float(np.mean(align_losses)) if align_losses else 0.0
        cross_pair_epoch = float(np.mean(cross_pair_losses)) if cross_pair_losses else 0.0
        hist.append({
            "epoch": ep,
            "train_loss": train_loss,
            "alignment_loss": align_loss_epoch,
            "cross_pair_loss": cross_pair_epoch,
            "iv_val_knn_acc": iv_val_acc,
            "ex_val_knn_acc": ex_val_acc,
            "ex_to_iv_knn_acc": ex_id_acc,
            "val_balanced_acc": val_bal_acc,
        })
        print(
            f"[full_iv_ex/lora] epoch {ep}/{epochs} "
            f"loss={train_loss:.4f} iv_val={iv_val_acc:.4f} ex_val={ex_val_acc:.4f} "
            f"bal_val={val_bal_acc:.4f} ex2iv={ex_id_acc:.4f} align={align_loss_epoch:.4f} "
            f"cpair={cross_pair_epoch:.4f}",
            flush=True,
        )
        if val_bal_acc > best["val_acc"]:
            best = {
                "val_acc": val_bal_acc,
                "epoch": ep,
                "iv_val_acc": iv_val_acc,
                "ex_val_acc": ex_val_acc,
                "state": best_state_callback(),
            }
        if ep % snapshot_every == 0 or ep == 1 or ep == epochs:
            snapshots.append({"epoch": ep, "ziv": ziv_all.detach().cpu().numpy(), "zex": zex_all.detach().cpu().numpy()})

    proj.eval()
    best_state = best["state"]
    if best_state and "proj_state" in best_state:
        proj.load_state_dict(best_state["proj_state"])
    return proj, hist, best, snapshots, iv_tr_idx, iv_va_idx, iv_train_cells, iv_val_cells, ex_tr_idx, ex_va_idx, ex_train_cells, ex_val_cells


@torch.no_grad()
def eval_ex_to_iv_knn(
    proj: nn.Module,
    emb_iv: np.ndarray,
    emb_ex: np.ndarray,
    y_iv: np.ndarray,
    y_ex: np.ndarray,
    k: int,
    num_cells: int,
    cell_proto_top_k: int = 5,
    patch_knn_top_k: int = 5,
):
    ziv = proj(torch.tensor(emb_iv, dtype=torch.float32, device=DEVICE))
    zex = proj(torch.tensor(emb_ex, dtype=torch.float32, device=DEVICE))
    y_iv_t = torch.tensor(y_iv, dtype=torch.long, device=DEVICE)
    yex_t  = torch.tensor(y_ex, dtype=torch.long, device=DEVICE)

    # Use CPU-batched evaluation to avoid GPU/MPS OOM on large embedding sets.
    # The full sim matrix (N_ex × N_iv) at float32 can easily exceed 30 GB for
    # 800-aug runs (92k × 92k); 2048-row CPU chunks peak at ~750 MB instead.
    _EVAL_BATCH = 2048
    _ziv_cpu = ziv.cpu()
    _zex_cpu = zex.cpu()
    _y_iv_cpu = y_iv_t.cpu()
    _y_ex_cpu = yex_t.cpu()

    pk = min(int(patch_knn_top_k), int(_ziv_cpu.shape[0]))
    k_eval = min(k, _ziv_cpu.shape[0])

    topk_patch_list: list[torch.Tensor] = []
    topk_list:       list[torch.Tensor] = []
    for _bs in range(0, _zex_cpu.shape[0], _EVAL_BATCH):
        _sim_b = _zex_cpu[_bs:_bs + _EVAL_BATCH] @ _ziv_cpu.t()
        topk_patch_list.append(torch.topk(_sim_b, k=pk,     dim=1).indices)
        topk_list.append(      torch.topk(_sim_b, k=k_eval, dim=1).indices)

    topk_patch_cpu = torch.cat(topk_patch_list, dim=0)
    topk_cpu       = torch.cat(topk_list,       dim=0)

    neigh_lbl      = _y_iv_cpu[topk_patch_cpu]
    top5_patch_hits = (neigh_lbl == _y_ex_cpu[:, None]).any(dim=1).float().mean().item()

    nn_labels = _y_iv_cpu[topk_cpu]
    pred      = knn_majority_labels(nn_labels.to(DEVICE), num_classes=num_cells).cpu()
    acc       = float((pred == _y_ex_cpu).float().mean().item())
    purity    = float((nn_labels == _y_ex_cpu[:, None]).float().mean().item())

    cm = np.zeros((num_cells, num_cells), dtype=np.int64)
    y_true = yex_t.detach().cpu().numpy()
    y_pred = pred.detach().cpu().numpy()
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    # Cell-prototype retrieval: one mean projected IV vector per cell, rank cells per EX query.
    proto_rows = []
    for c in range(num_cells):
        m = y_iv_t == c
        if bool(m.any()):
            pv = ziv[m].mean(dim=0, keepdim=True)
            proto_rows.append(F.normalize(pv, dim=1).squeeze(0))
        else:
            proto_rows.append(torch.zeros(ziv.shape[1], device=DEVICE, dtype=ziv.dtype))
    protos = torch.stack(proto_rows, dim=0)
    zex_n = F.normalize(zex, dim=1)
    protos_n = F.normalize(protos, dim=1)
    sim_pc = zex_n @ protos_n.t()
    sidx = torch.argsort(sim_pc, dim=1, descending=True)
    nq = sim_pc.shape[0]
    ranks: list[int] = []
    top_hits: list[bool] = []
    rrs: list[float] = []
    tk = int(cell_proto_top_k)
    for qi in range(nq):
        tc = int(yex_t[qi].item())
        row = sidx[qi]
        pos = int((row == tc).nonzero(as_tuple=True)[0][0].item())
        r = pos + 1
        ranks.append(r)
        top_hits.append(pos < tk)
        rrs.append(1.0 / float(r))

    return {
        "acc_top1": acc,
        "topk_purity": purity,
        "pred_cells": y_pred.tolist(),
        "confusion": cm.tolist(),
        "ziv": ziv.detach().cpu().numpy(),
        "zex": zex.detach().cpu().numpy(),
        "ex_to_iv_top5_acc_cell_proto": float(np.mean(top_hits)),
        "ex_to_iv_mean_rank_cell_proto": float(np.mean(ranks)),
        "ex_to_iv_mrr_cell_proto": float(np.mean(rrs)),
        "ex_to_iv_top5_acc_patch_knn": float(top5_patch_hits),
    }


def render_coembed_report(
    paths: dict[str, Path],
    cell_ids: list[str],
    iv_samples: list[dict],
    ex_samples: list[dict],
    snapshots: list[dict],
    train_hist: list[dict],
    confusion: list[list[int]],
    tr_idx: list[int],
    best_epoch: int,
    best_val_acc: float,
    ex_to_iv_top1_patch_knn: float | None = None,
    ex_to_iv_top5_patch_knn: float | None = None,
    ex_to_iv_top5_cell_proto: float | None = None,
    ex_to_iv_mean_rank_cell_proto: float | None = None,
    ex_to_iv_mrr_cell_proto: float | None = None,
):
    zstack = np.concatenate([np.concatenate([s["ziv"], s["zex"]], axis=0) for s in snapshots], axis=0)
    p2 = PCA(n_components=2, random_state=23)
    xystack = p2.fit_transform(zstack)
    n_iv = len(iv_samples)
    n_ex = len(ex_samples)
    n_all = n_iv + n_ex
    for i, s in enumerate(snapshots):
        xy = xystack[i * n_all : (i + 1) * n_all]
        s["xy_iv"] = xy[:n_iv]
        s["xy_ex"] = xy[n_iv:]

    xy_iv0 = snapshots[0]["xy_iv"]
    xy_ex0 = snapshots[0]["xy_ex"]
    iv_labels = [s["cell_id"] for s in iv_samples]
    ex_labels = [s["cell_id"] for s in ex_samples]
    iv_cell = [s["cell_idx"] for s in iv_samples]
    ex_cell = [s["cell_idx"] for s in ex_samples]
    epochs = [h["epoch"] for h in train_hist]
    loss = [h["train_loss"] for h in train_hist]
    iv_vacc = [h["iv_val_knn_acc"] for h in train_hist]
    ex_vacc = [h["ex_val_knn_acc"] for h in train_hist]
    ex_id = [h["ex_to_iv_knn_acc"] for h in train_hist]
    tr_set = set(tr_idx)
    iv_symbol = ["circle" if i in tr_set else "diamond" for i in range(n_iv)]

    frames = []
    for s in snapshots:
        frames.append(
            {
                "name": str(s["epoch"]),
                "traces": [0, 1],
                "data": [
                    {
                        "type": "scatter",
                        "mode": "markers",
                        "x": s["xy_iv"][:, 0].tolist(),
                        "y": s["xy_iv"][:, 1].tolist(),
                        "text": iv_labels,
                        "marker": {
                            "size": 5,
                            "opacity": 0.65,
                            "symbol": iv_symbol,
                            "color": iv_cell,
                            "colorscale": "Blues",
                        },
                    },
                    {
                        "type": "scatter",
                        "mode": "markers",
                        "x": s["xy_ex"][:, 0].tolist(),
                        "y": s["xy_ex"][:, 1].tolist(),
                        "text": ex_labels,
                        "marker": {"size": 5, "opacity": 0.65, "symbol": "diamond-open", "color": ex_cell, "colorscale": "Oranges"},
                    },
                ],
            }
        )
    steps = [
        {
            "label": str(s["epoch"]),
            "method": "animate",
            "args": [[str(s["epoch"])], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
        }
        for s in snapshots
    ]

    def _fmt4(x: float | None) -> str:
        return f"{x:.4f}" if x is not None else "n/a"

    def _fmt2(x: float | None) -> str:
        return f"{x:.2f}" if x is not None else "n/a"

    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Full-IV train, EX infer co-embedding</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ background:#0f1116; color:#e6e6e6; font-family:-apple-system,Segoe UI,sans-serif; margin:0; }}
.wrap {{ max-width:1280px; margin:0 auto; padding:16px; }}
#joint {{ height:60vh; }} #loss1 {{ height:30vh; margin-top:14px; }} #loss2 {{ height:30vh; margin-top:14px; }} #cm {{ height:58vh; margin-top:14px; }}
</style></head><body><div class="wrap">
<h3>Full-IV MLP training + EX co-embedding</h3>
<p>Best epoch by balanced IV/EX val kNN: {best_epoch} (val acc={best_val_acc:.4f})</p>
<p>EX→IV at best checkpoint: patch kNN top-1={_fmt4(ex_to_iv_top1_patch_knn)}, top-5={_fmt4(ex_to_iv_top5_patch_knn)} | cell-prototype top-5={_fmt4(ex_to_iv_top5_cell_proto)}, mean rank={_fmt2(ex_to_iv_mean_rank_cell_proto)}, MRR={_fmt4(ex_to_iv_mrr_cell_proto)}</p>
<div id="joint"></div><div id="loss1"></div><div id="loss2"></div><div id="cm"></div>
</div><script>
Plotly.newPlot("joint", [
{{type:"scatter", mode:"markers", name:"In-vivo (train circle / val diamond)", x:{xy_iv0[:,0].tolist()}, y:{xy_iv0[:,1].tolist()},
 text:{iv_labels}, marker:{{size:5, opacity:0.65, symbol:{iv_symbol}, color:{iv_cell}, colorscale:"Blues"}}}},
{{type:"scatter", mode:"markers", name:"Ex-vivo", x:{xy_ex0[:,0].tolist()}, y:{xy_ex0[:,1].tolist()},
 text:{ex_labels}, marker:{{size:5, opacity:0.65, symbol:"diamond-open", color:{ex_cell}, colorscale:"Oranges"}}}}
], {{title:"Joint IV/EX co-embedding (common PCA basis)", paper_bgcolor:"#171b24", plot_bgcolor:"#171b24", font:{{color:"#e6e6e6"}},
xaxis:{{title:"PC1", gridcolor:"#2d3648"}}, yaxis:{{title:"PC2", gridcolor:"#2d3648"}},
updatemenus:[{{type:"buttons",direction:"left",x:0,y:1.15,buttons:[
{{label:"Play",method:"animate",args:[null,{{fromcurrent:true,frame:{{duration:250,redraw:true}},transition:{{duration:0}}}}]}},
{{label:"Pause",method:"animate",args:[[null],{{mode:"immediate",frame:{{duration:0,redraw:false}},transition:{{duration:0}}}}]}}
]}}],
sliders:[{{active:0,currentvalue:{{prefix:"Snapshot epoch: "}},steps:{json.dumps(steps)}}}]
}}, {{responsive:true}}).then(()=>Plotly.addFrames("joint", {json.dumps(frames)}));

Plotly.newPlot("loss1", [
{{x:{epochs}, y:{loss}, type:"scatter", mode:"lines", name:"train contrastive loss", yaxis:"y1"}},
{{x:{epochs}, y:{iv_vacc}, type:"scatter", mode:"lines", name:"in-vivo val kNN acc", yaxis:"y2"}}
],{{title:"In-vivo train loss + in-vivo val kNN acc", paper_bgcolor:"#171b24", plot_bgcolor:"#171b24", font:{{color:"#e6e6e6"}},
xaxis:{{title:"Epoch", gridcolor:"#2d3648"}}, yaxis:{{title:"Contrastive loss", gridcolor:"#2d3648"}},
yaxis2:{{title:"Accuracy", overlaying:"y", side:"right", rangemode:"tozero"}}, legend:{{orientation:"h", y:1.12}}}}, {{responsive:true}});

Plotly.newPlot("loss2", [
{{x:{epochs}, y:{iv_vacc}, type:"scatter", mode:"lines", name:"in-vivo val kNN acc"}},
{{x:{epochs}, y:{ex_vacc}, type:"scatter", mode:"lines", name:"ex-vivo val kNN acc"}},
{{x:{epochs}, y:{ex_id}, type:"scatter", mode:"lines", name:"ex->iv kNN id acc"}}
],{{title:"Validation + EX identification accuracy", paper_bgcolor:"#171b24", plot_bgcolor:"#171b24", font:{{color:"#e6e6e6"}},
xaxis:{{title:"Epoch", gridcolor:"#2d3648"}}, yaxis:{{title:"Accuracy", gridcolor:"#2d3648", rangemode:"tozero"}},
legend:{{orientation:"h", y:1.12}}}}, {{responsive:true}});

Plotly.newPlot("cm", [{{type:"heatmap", z:{json.dumps(confusion)}, x:{json.dumps(cell_ids)}, y:{json.dumps(cell_ids)},
colorscale:"Viridis", colorbar:{{title:"Count"}}}}],
{{title:"EX→IV confusion (rows=true EX cell, cols=predicted IV cell)", paper_bgcolor:"#171b24", plot_bgcolor:"#171b24",
font:{{color:"#e6e6e6"}}, xaxis:{{title:"Predicted IV class", tickangle:45, scaleanchor:"y", scaleratio:1}},
yaxis:{{title:"True EX class"}}, margin:{{l:110,r:40,t:70,b:130}}}}, {{responsive:true}});
</script></body></html>"""
    paths["coembed_html"].write_text(html, encoding="utf-8")


def render_fold_animation(
    samples: list[dict],
    fold_res: dict,
    paths: dict[str, Path],
    confusion_matrix: list[list[int]] | None = None,
    cell_ids: list[str] | None = None,
):
    snaps = fold_res["snapshots"]
    split = fold_res["split"]
    tr_set = set(split["train_idx"])
    va_set = set(split["val_idx"])
    te_set = set(split["test_idx"])

    # common PCA basis
    Zstack = np.concatenate([s["z"] for s in snaps], axis=0)
    p2 = PCA(n_components=2, random_state=23)
    XYstack = p2.fit_transform(Zstack)
    n = len(samples)
    for i, s in enumerate(snaps):
        s["xy"] = XYstack[i * n : (i + 1) * n]

    cell_idx = np.array([s["cell_idx"] for s in samples])
    labels = [s["cell_id"] for s in samples]
    split_shape = []
    for i in range(n):
        if i in te_set:
            split_shape.append("x")
        elif i in va_set:
            split_shape.append("diamond")
        else:
            split_shape.append("circle")

    frames = []
    for s in snaps:
        xy = s["xy"]
        frames.append(
            {
                "name": str(s["epoch"]),
                "traces": [0, 1],
                "data": [
                    {
                        "type": "scatter",
                        "mode": "markers",
                        "x": xy[list(tr_set | va_set), 0].tolist(),
                        "y": xy[list(tr_set | va_set), 1].tolist(),
                        "text": [labels[i] for i in list(tr_set | va_set)],
                        "marker": {
                            "size": 6,
                            "symbol": [split_shape[i] for i in list(tr_set | va_set)],
                            "color": cell_idx[list(tr_set | va_set)].tolist(),
                            "colorscale": "Turbo",
                            "opacity": 0.65,
                        },
                    },
                    {
                        "type": "scatter",
                        "mode": "markers",
                        "x": xy[list(te_set), 0].tolist(),
                        "y": xy[list(te_set), 1].tolist(),
                        "text": [labels[i] for i in list(te_set)],
                        "marker": {"size": 10, "symbol": "x", "color": "red", "line": {"color": "white", "width": 1}},
                    },
                ],
            }
        )

    init_xy = snaps[0]["xy"]
    non_test = list(tr_set | va_set)
    test = list(te_set)
    steps = [
        {
            "label": str(s["epoch"]),
            "method": "animate",
            "args": [[str(s["epoch"])], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
        }
        for s in snaps
    ]
    hist = fold_res["history"]

    heatmap_div = ""
    heatmap_script = ""
    if confusion_matrix is not None and cell_ids is not None:
        heatmap_div = '<div id="cmPlot" style="height:42vh; margin-top:18px;"></div>'
        heatmap_script = f"""
const cm = {json.dumps(confusion_matrix)};
const cmLabels = {json.dumps(cell_ids)};
Plotly.newPlot("cmPlot", [{{
  type:"heatmap",
  z: cm,
  x: cmLabels,
  y: cmLabels,
  colorscale:"Viridis",
  colorbar:{{title:"Count"}}
}}], {{
  title:"LOOCV augmentation-level confusion matrix (true cell i -> predicted class j)",
  height: 980,
  paper_bgcolor:"#171b24", plot_bgcolor:"#171b24", font:{{color:"#e6e6e6"}},
  xaxis:{{title:"Predicted class", tickangle:45, scaleanchor:"y", scaleratio:1}},
  yaxis:{{title:"True class"}},
  margin:{{l:110, r:40, t:70, b:130}}
}}, {{responsive:true}});
"""

    html = f"""<!doctype html>
<html><head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>LOOCV contrastive MLP fold animation</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ background:#0f1116; color:#e6e6e6; font-family:-apple-system,Segoe UI,sans-serif; margin:0; }}
    .wrap {{ max-width:1200px; margin:0 auto; padding:16px; }}
    #embedPlot {{ height:68vh; }}
    #curvePlot {{ height:34vh; margin-top:18px; }}
  </style>
</head>
<body>
<div class="wrap">
  <h3>LOOCV fold embedding trajectory</h3>
  <p>Left-out cell idx: {fold_res["left_out_cell"]} | best val kNN acc: {fold_res["best_val_knn_acc"]:.4f} @ epoch {fold_res["best_epoch"]}</p>
  <div id="embedPlot"></div>
  <div id="curvePlot"></div>
  {heatmap_div}
</div>
<script>
const frames = {json.dumps(frames)};
const hist = {json.dumps(hist)};
const layout = {{
  paper_bgcolor:"#171b24", plot_bgcolor:"#171b24", font:{{color:"#e6e6e6"}},
  xaxis:{{title:"PC1", gridcolor:"#2d3648"}}, yaxis:{{title:"PC2", gridcolor:"#2d3648"}},
  title:"Projected embeddings (train/val by shape; test as red x)",
  updatemenus:[{{type:"buttons",direction:"left",x:0,y:1.15,buttons:[
    {{label:"Play",method:"animate",args:[null,{{fromcurrent:true,frame:{{duration:180,redraw:true}},transition:{{duration:0}}}}]}},
    {{label:"Pause",method:"animate",args:[[null],{{mode:"immediate",frame:{{duration:0,redraw:false}},transition:{{duration:0}}}}]}}
  ]}}],
  sliders:[{{active:0,currentvalue:{{prefix:"Epoch: "}},steps:{json.dumps(steps)}}}]
}};
const data = [
  {{
    type:"scatter", mode:"markers",
    x:{init_xy[non_test, 0].tolist()},
    y:{init_xy[non_test, 1].tolist()},
    text:{[labels[i] for i in non_test]},
    marker:{{size:6, symbol:{[split_shape[i] for i in non_test]}, color:{cell_idx[non_test].tolist()}, colorscale:"Turbo", opacity:0.65}},
    name:"train+val"
  }},
  {{
    type:"scatter", mode:"markers",
    x:{init_xy[test, 0].tolist()},
    y:{init_xy[test, 1].tolist()},
    text:{[labels[i] for i in test]},
    marker:{{size:10, symbol:"x", color:"red", line:{{color:"white", width:1}}}},
    name:"test"
  }}
];
Plotly.newPlot("embedPlot", data, layout).then(() => Plotly.addFrames("embedPlot", frames));

const epochs = hist.map(r => r.epoch);
const loss = hist.map(r => r.train_loss);
const vacc = hist.map(r => r.val_knn_acc);
Plotly.newPlot("curvePlot", [
  {{x:epochs, y:loss, type:"scatter", mode:"lines", name:"train contrastive loss", line:{{color:"#4da3ff"}}, yaxis:"y1"}},
  {{x:epochs, y:vacc, type:"scatter", mode:"lines", name:"val kNN acc", line:{{color:"#ff9f43"}}, yaxis:"y2"}}
], {{
  title:"Training curves",
  paper_bgcolor:"#171b24", plot_bgcolor:"#171b24", font:{{color:"#e6e6e6"}},
  xaxis:{{title:"Epoch", gridcolor:"#2d3648"}},
  yaxis:{{title:"Contrastive loss", gridcolor:"#2d3648"}},
  yaxis2:{{title:"Validation kNN acc", overlaying:"y", side:"right", rangemode:"tozero"}},
  legend:{{orientation:"h", y:1.12}}
}}, {{responsive:true}});
{heatmap_script}
</script>
</body></html>"""
    paths["anim_html"].write_text(html, encoding="utf-8")
    paths["anim_json"].write_text(
        json.dumps(
            {
                "left_out_cell": fold_res["left_out_cell"],
                "best_epoch": fold_res["best_epoch"],
                "best_val_knn_acc": fold_res["best_val_knn_acc"],
                "animation_html": str(paths["anim_html"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="loocv", choices=["loocv", "full_iv_ex_eval"])
    ap.add_argument("--train-source", type=str, default="iv_only", choices=["iv_only", "ex_only", "joint"])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=0, help="0 means full-batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--max-folds", type=int, default=0, help="0 means all folds")
    ap.add_argument("--animation-cell", type=str, default="L95")
    ap.add_argument("--only-cell", type=str, default="", help="Run only this left-out cell ID (e.g., L95)")
    ap.add_argument("--aug-per-preset", type=int, default=DEFAULT_AUG_PER_PRESET)
    ap.add_argument(
        "--aug-presets",
        type=str,
        default=",".join(DEFAULT_AUG_PRESETS),
        help="Comma-separated presets: heavy,medium,rotation_only,domain_gap",
    )
    ap.add_argument("--snapshot-every", type=int, default=1, help="Capture animation snapshots every N epochs")
    ap.add_argument("--dino-version", type=str, default="v3", choices=["v2", "v3", "eupe"])
    ap.add_argument("--dino-model-id", type=str, default="", help="HF model id for frozen CLS features (optional override)")
    ap.add_argument("--frozen-embed-dim", type=int, default=0, help="If >0, PCA-compress frozen DINO embeddings before MLP")
    ap.add_argument(
        "--lora-bottleneck-dim",
        type=int,
        default=0,
        help="If >0 (backbone-mode=lora), trainable Linear(cls_dim→d) before MLP; 0 = MLP on full CLS",
    )
    ap.add_argument("--mlp-out-dim", type=int, default=64, help="Contrastive projection (MLP output) dimension")
    ap.add_argument("--backbone-mode", type=str, default="frozen", choices=["frozen", "lora"])
    ap.add_argument("--lora-blocks", type=int, default=2, help="Last N blocks with LoRA (backbone-mode=lora)")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument(
        "--alignment-loss",
        type=str,
        default="none",
        choices=["none", "coral", "mmd", "sinkhorn", "swd", "dann", "shared_labels"],
        help="Optional cross-domain alignment strategy in joint training.",
    )
    ap.add_argument(
        "--lambda-align",
        type=float,
        default=0.0,
        help="Weight for alignment loss term (ignored when --alignment-loss none).",
    )
    ap.add_argument(
        "--dann-schedule",
        type=str,
        default="none",
        choices=["none", "warmup", "constant_low", "constant_high"],
        help="Schedule for DANN gradient reversal alpha.",
    )
    ap.add_argument(
        "--shared-labels",
        action="store_true",
        help="Merge IV/EX into one label space so SupCon treats same cell_idx across modalities as positives (oracle cross-domain supervision).",
    )
    ap.add_argument(
        "--lambda-cross-pair",
        type=float,
        default=0.0,
        help="Optional explicit in-batch cosine pull between IV and EX rows with the same landmark cell_idx (0=off; default is intra-modality-only SupCon).",
    )
    ap.add_argument(
        "--geom-neighbor-k",
        type=int,
        default=0,
        help="If >0, concat KNN landmark offset features (3*K dims) to CLS before MLP; 0 disables.",
    )
    ap.add_argument(
        "--geom-fuse",
        type=str,
        default="midpoint",
        choices=["midpoint", "invivo", "exvivo"],
        help="3D frame for neighbor graph: IV/EX midpoint, invivo-only, or exvivo-only coords.",
    )
    ap.add_argument(
        "--eval-geom-zero",
        action="store_true",
        help="Zero geom suffix during val kNN, ex→iv eval, co-embed snapshots (train still uses full fused input).",
    )
    ap.add_argument(
        "--embed-batch-size",
        type=int,
        default=nr.BATCH_SIZE,
        help="Forward batch size when embedding patches (lower if ViT-L OOMs)",
    )
    ap.add_argument(
        "--patch-xy-um",
        type=int,
        default=200,
        help="In-plane slab size (same units as native_runner extract_isotropic_slab xy_um).",
    )
    ap.add_argument(
        "--patch-z-um",
        type=int,
        default=28,
        help="Axial slab depth (z_um passed to extract_isotropic_slab).",
    )
    ap.add_argument(
        "--spatial-jitter-vox",
        type=float,
        default=0.0,
        help="Uniform [-j,j] voxel jitter per axis on landmark center for augmented samples only (IV/EX independent).",
    )
    ap.add_argument(
        "--multiscale-enable",
        action="store_true",
        help="Encode tight/default/wide crops with the same frozen DINO backbone and concatenate CLS features.",
    )
    ap.add_argument(
        "--multiscale-xy-ums",
        type=str,
        default="100,200,300",
        help="Comma-separated XY slab sizes for multiscale mode; middle value must match --patch-xy-um.",
    )
    ap.add_argument(
        "--multiscale-z-ums",
        type=str,
        default="14,28,42",
        help="Comma-separated Z slab sizes for multiscale mode; middle value must match --patch-z-um.",
    )
    ap.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Output filename suffix; empty + dinov3 in model id → dinov3_vitl",
    )
    ap.add_argument(
        "--patch-cache",
        type=str,
        default="",
        help="Load IV/EX augmented patches from .npz + .meta.json (full_iv_ex_eval only); skips volume I/O.",
    )
    ap.add_argument(
        "--export-patch-cache",
        type=str,
        default="",
        help="After building IV/EX datasets, write patch cache and exit before embedding (full_iv_ex_eval only).",
    )
    ap.add_argument(
        "--emb-cache",
        type=str,
        default="",
        help=(
            "Directory to load/save pre-computed frozen DINO embeddings (full_iv_ex_eval only). "
            "If the directory contains a valid cache (matching recipe), skips patch build + DINO entirely. "
            "Otherwise builds patches, embeds with DINO, saves to this directory, then continues to training."
        ),
    )
    args = ap.parse_args()

    args.dino_model_id = resolve_dino_model_id(args.dino_version, args.dino_model_id)
    if args.backbone_mode == "lora" and (
        args.dino_version == "eupe" or nr.is_eupe_vit_model_id(args.dino_model_id)
    ):
        raise ValueError("EUPE ViT does not support --backbone-mode lora; use --backbone-mode frozen.")
    output_tag = args.output_tag.strip()
    if not output_tag:
        mid = args.dino_model_id.lower()
        if nr.is_eupe_vit_model_id(args.dino_model_id):
            if "vit-t" in mid:
                output_tag = "eupe_vitt"
            elif "vit-s" in mid:
                output_tag = "eupe_vits"
            elif "vit-b" in mid:
                output_tag = "eupe_vitb"
            else:
                output_tag = "eupe_vit"
        elif "dinov3" in mid:
            output_tag = "dinov3_vitl"
    paths = output_paths(output_tag)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths["folds_jsonl"].write_text("", encoding="utf-8")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    aug_presets = [x.strip() for x in args.aug_presets.split(",") if x.strip()]
    valid_presets = set(DEFAULT_AUG_PRESETS)
    bad = [x for x in aug_presets if x not in valid_presets]
    if bad:
        raise ValueError(f"Invalid aug preset(s): {bad}; valid={sorted(valid_presets)}")
    if args.aug_per_preset < 1:
        raise ValueError("--aug-per-preset must be >= 1")
    if args.snapshot_every < 1:
        raise ValueError("--snapshot-every must be >= 1")
    if args.embed_batch_size < 1:
        raise ValueError("--embed-batch-size must be >= 1")
    if args.frozen_embed_dim < 0:
        raise ValueError("--frozen-embed-dim must be >= 0")
    if args.mlp_out_dim < 1:
        raise ValueError("--mlp-out-dim must be >= 1")
    if args.lora_bottleneck_dim < 0:
        raise ValueError("--lora-bottleneck-dim must be >= 0")
    if args.lambda_align < 0:
        raise ValueError("--lambda-align must be >= 0")
    if args.lambda_cross_pair < 0:
        raise ValueError("--lambda-cross-pair must be >= 0")
    if args.geom_neighbor_k < 0:
        raise ValueError("--geom-neighbor-k must be >= 0")
    if args.patch_xy_um < 32:
        raise ValueError("--patch-xy-um must be >= 32")
    if args.patch_z_um < 8:
        raise ValueError("--patch-z-um must be >= 8")
    if args.spatial_jitter_vox < 0:
        raise ValueError("--spatial-jitter-vox must be >= 0")
    args.multiscale_xy_ums = parse_multiscale_um_list(args.multiscale_xy_ums, 3, "--multiscale-xy-ums")
    args.multiscale_z_ums = parse_multiscale_um_list(args.multiscale_z_ums, 3, "--multiscale-z-ums")
    if args.multiscale_enable:
        if args.backbone_mode != "frozen":
            raise ValueError("--multiscale-enable currently supports only --backbone-mode frozen")
        if args.multiscale_xy_ums[1] != int(args.patch_xy_um):
            raise ValueError("Middle multiscale XY must match --patch-xy-um")
        if args.multiscale_z_ums[1] != int(args.patch_z_um):
            raise ValueError("Middle multiscale Z must match --patch-z-um")
    if args.alignment_loss == "shared_labels":
        args.shared_labels = True
        args.alignment_loss = "none"

    patch_cache_in = args.patch_cache.strip()
    std_env = os.environ.get("AUTORESEARCH_STANDARD_PATCH_CACHE", "").strip()
    patch_cache_from_env = False
    if not patch_cache_in and std_env:
        p_std = Path(std_env).expanduser()
        if not p_std.is_absolute():
            p_std = (ROOT / p_std).resolve()
        else:
            p_std = p_std.resolve()
        if p_std.is_file():
            patch_cache_in = str(p_std)
            patch_cache_from_env = True
            print(f"Using AUTORESEARCH_STANDARD_PATCH_CACHE: {patch_cache_in}", flush=True)
    patch_cache_out = args.export_patch_cache.strip()
    if patch_cache_in and patch_cache_out:
        raise ValueError("--patch-cache and --export-patch-cache are mutually exclusive")
    if (patch_cache_in or patch_cache_out) and args.mode != "full_iv_ex_eval":
        raise ValueError("--patch-cache and --export-patch-cache require --mode full_iv_ex_eval")

    iv_ex_recipe = iv_ex_recipe_dict(
        aug_presets,
        seed=args.seed,
        aug_per_preset=args.aug_per_preset,
        patch_xy_um=args.patch_xy_um,
        patch_z_um=args.patch_z_um,
        spatial_jitter_vox=args.spatial_jitter_vox,
        multiscale_enable=bool(args.multiscale_enable),
        multiscale_xy_ums=args.multiscale_xy_ums,
        multiscale_z_ums=args.multiscale_z_ums,
    )

    emb_cache_dir = args.emb_cache.strip() if hasattr(args, "emb_cache") else ""
    t0 = time.time()

    if args.mode == "full_iv_ex_eval":
        # ── Try embedding cache first (skips patch build + DINO entirely) ──
        _emb_from_cache = False
        if emb_cache_dir and _emb_cache_is_valid(emb_cache_dir, iv_ex_recipe):
            print(f"Loading pre-computed embeddings from cache: {emb_cache_dir}", flush=True)
            emb_iv, emb_ex, iv_cell_idx_arr, ex_cell_idx_arr, cell_ids = _load_emb_cache(emb_cache_dir)
            n_cells = len(cell_ids)
            # Lightweight sample list — only cell_idx needed for training
            iv_samples = [{"cell_idx": int(c), "cell_id": cell_ids[int(c)], "preset": "cached"} for c in iv_cell_idx_arr]
            ex_samples = [{"cell_idx": int(c), "cell_id": cell_ids[int(c)], "preset": "cached"} for c in ex_cell_idx_arr]
            cls_dim = emb_iv.shape[1]
            geom_dim_emb = 0
            print(
                f"Embeddings from cache: iv={emb_iv.shape} ex={emb_ex.shape} "
                f"(frozen dim={cls_dim}, multiscale={args.multiscale_enable})",
                flush=True,
            )
            _emb_from_cache = True

        if not _emb_from_cache:
            if patch_cache_in:
                try:
                    cell_ids, iv_samples, ex_samples = load_iv_ex_patch_cache(
                        Path(patch_cache_in).expanduser(), iv_ex_recipe
                    )
                except ValueError as e:
                    if patch_cache_from_env:
                        raise ValueError(
                            f"{e}\n"
                            "Unset AUTORESEARCH_STANDARD_PATCH_CACHE or align CLI flags with the cache "
                            "recipe in the sidecar .meta.json next to the .npz."
                        ) from e
                    raise
                print(f"Loaded IV/EX samples from patch cache: {patch_cache_in}", flush=True)
            else:
                cell_ids, iv_samples, ex_samples = build_iv_ex_datasets(
                    rng,
                    aug_presets=aug_presets,
                    aug_per_preset=args.aug_per_preset,
                    patch_xy_um=args.patch_xy_um,
                    patch_z_um=args.patch_z_um,
                    spatial_jitter_vox=args.spatial_jitter_vox,
                    multiscale_enable=bool(args.multiscale_enable),
                    multiscale_xy_ums=args.multiscale_xy_ums,
                    multiscale_z_ums=args.multiscale_z_ums,
                )
            n_cells = len(cell_ids)
            print(
                f"Prepared IV/EX samples: iv={len(iv_samples)} ex={len(ex_samples)} cells={n_cells} "
                f"patch=({args.patch_xy_um},{args.patch_z_um}) jitter_vox={args.spatial_jitter_vox} "
                f"multiscale={args.multiscale_enable}",
                flush=True,
            )

            if patch_cache_out:
                out_p = Path(patch_cache_out).expanduser()
                save_iv_ex_patch_cache(out_p, cell_ids, iv_samples, ex_samples, iv_ex_recipe)
                print(f"Wrote patch cache: {out_p} (+ .meta.json)", flush=True)
                sys.exit(0)

            lm_list = nr.load_landmarks()
            geom_feat_iv = geom_feat_ex = None
            geom_dim_emb = 0
            if args.geom_neighbor_k > 0:
                G_cell = compute_geom_knn_offsets(lm_list, args.geom_neighbor_k, args.geom_fuse)
                geom_feat_iv = stack_sample_geom(iv_samples, G_cell)
                geom_feat_ex = stack_sample_geom(ex_samples, G_cell)
                geom_dim_emb = int(G_cell.shape[1])
                print(
                    f"Geometry KNN: K={args.geom_neighbor_k} fuse={args.geom_fuse} feat_dim={geom_dim_emb}",
                    flush=True,
                )

            if args.eval_geom_zero and geom_dim_emb <= 0:
                print("[warn] --eval-geom-zero has no effect when --geom-neighbor-k is 0", flush=True)

            if args.backbone_mode == "frozen":
                embedder = nr.make_frozen_embedder(args.dino_version, args.dino_model_id, TORCH_DEVICE)
                emb_iv_raw = embed_multiscale_samples_cls(embedder, iv_samples, args.embed_batch_size, bool(args.multiscale_enable))
                emb_ex_raw = embed_multiscale_samples_cls(embedder, ex_samples, args.embed_batch_size, bool(args.multiscale_enable))
                emb_iv_cls, emb_ex_cls = maybe_reduce_embeddings(emb_iv_raw, emb_ex_raw, args.frozen_embed_dim)
                if geom_feat_iv is not None:
                    emb_iv = np.concatenate([emb_iv_cls, geom_feat_iv], axis=1).astype(np.float32)
                    emb_ex = np.concatenate([emb_ex_cls, geom_feat_ex], axis=1).astype(np.float32)
                else:
                    emb_iv, emb_ex = emb_iv_cls, emb_ex_cls
                cls_dim = emb_iv_cls.shape[1]
                print(
                    f"Embeddings shape: iv={emb_iv.shape} ex={emb_ex.shape} "
                    f"(frozen dim={cls_dim}, multiscale={args.multiscale_enable})",
                    flush=True,
                )
                # Save to emb cache if requested
                if emb_cache_dir:
                    _save_emb_cache(
                        emb_cache_dir, emb_iv, emb_ex,
                        [s["cell_idx"] for s in iv_samples],
                        [s["cell_idx"] for s in ex_samples],
                        cell_ids, iv_ex_recipe,
                    )
                # Free raw patch memory — training only needs cell_idx from samples
                for s in iv_samples:
                    s.pop("patch", None); s.pop("patches_ms", None)
                for s in ex_samples:
                    s.pop("patch", None); s.pop("patches_ms", None)

        if args.backbone_mode == "frozen" or _emb_from_cache:
            proj, train_hist, best, snapshots, iv_tr_idx, iv_va_idx, iv_train_cells, iv_val_cells, ex_tr_idx, ex_va_idx, ex_train_cells, ex_val_cells = run_full_iv_training(
                emb_iv=emb_iv,
                emb_ex=emb_ex,
                iv_samples=iv_samples,
                ex_samples=ex_samples,
                epochs=args.epochs,
                batch=args.batch,
                lr=args.lr,
                wd=args.weight_decay,
                temp=args.temperature,
                k=args.knn_k,
                seed=args.seed,
                snapshot_every=max(1, args.snapshot_every),
                num_cells=n_cells,
                train_source=args.train_source,
                mlp_out_dim=args.mlp_out_dim,
                alignment_loss_name=args.alignment_loss,
                lambda_align=args.lambda_align,
                dann_schedule=args.dann_schedule,
                shared_labels=args.shared_labels,
                lambda_cross_pair=args.lambda_cross_pair,
                eval_geom_zero=bool(args.eval_geom_zero),
                geom_dim=geom_dim_emb,
            )
            y_iv = np.array([s["cell_idx"] for s in iv_samples], dtype=np.int64)
            y_ex = np.array([s["cell_idx"] for s in ex_samples], dtype=np.int64)
            ev = eval_ex_to_iv_knn(
                proj=proj,
                emb_iv=np_eval_geom_zero(emb_iv, geom_dim_emb, args.eval_geom_zero),
                emb_ex=np_eval_geom_zero(emb_ex, geom_dim_emb, args.eval_geom_zero),
                y_iv=y_iv,
                y_ex=y_ex,
                k=args.knn_k,
                num_cells=n_cells,
            )
        else:
            from native_dino_lora import (
                inject_lora_dinov2,
                inject_lora_dinov3,
                get_pos_emb_v3,
                compute_cache_v3,
                compute_cache_v2,
                forward_tail_v3,
                forward_tail_v2,
            )

            processor = nr.load_image_processor_for_model(args.dino_model_id)
            model = nr.load_auto_model_for_model(args.dino_model_id, TORCH_DEVICE)

            if args.dino_version == "v3":
                lora_params, cache_block, _ = inject_lora_dinov3(
                    model, args.lora_blocks, args.lora_r, args.lora_alpha, TORCH_DEVICE
                )
                pos_emb = get_pos_emb_v3(model, processor, iv_samples[0]["patch"], TORCH_DEVICE)
                tail_fn = lambda h: forward_tail_v3(h, model, pos_emb, cache_block)
                cache_iv = compute_cache_v3(
                    [s["patch"] for s in iv_samples], model, processor, pos_emb, TORCH_DEVICE, cache_block, args.embed_batch_size
                )
                cache_ex = compute_cache_v3(
                    [s["patch"] for s in ex_samples], model, processor, pos_emb, TORCH_DEVICE, cache_block, args.embed_batch_size
                )
            else:
                lora_params, cache_block, _ = inject_lora_dinov2(
                    model, args.lora_blocks, args.lora_r, args.lora_alpha, TORCH_DEVICE
                )
                tail_fn = lambda h: forward_tail_v2(h, model, cache_block)
                # float16 cache: halves memory from ~36 GB → ~18 GB per modality
                cache_iv = compute_cache_v2(
                    [s["patch"] for s in iv_samples], model, processor, TORCH_DEVICE, cache_block, args.embed_batch_size
                )
                cache_ex = compute_cache_v2(
                    [s["patch"] for s in ex_samples], model, processor, TORCH_DEVICE, cache_block, args.embed_batch_size
                )

            cls_dim = cache_iv.shape[2]
            geom_dim = int(geom_feat_iv.shape[1]) if geom_feat_iv is not None else 0
            in_full = cls_dim + geom_dim
            print(
                f"Cache shape: iv={cache_iv.shape} ex={cache_ex.shape} dtype={cache_iv.dtype} "
                f"(CLS dim={cls_dim}, geom_dim={geom_dim}, "
                f"size={cache_iv.element_size()*cache_iv.numel()/1e9:.1f}+{cache_ex.element_size()*cache_ex.numel()/1e9:.1f} GB)",
                flush=True,
            )

            d_bn = args.lora_bottleneck_dim
            if d_bn > 0:
                if d_bn > in_full:
                    raise ValueError(f"--lora-bottleneck-dim ({d_bn}) must be <= fused input dim ({in_full})")
                proj = nn.Sequential(
                    nn.Linear(in_full, d_bn),
                    MLPProjector(in_dim=d_bn, hidden=256, out_dim=args.mlp_out_dim),
                ).to(DEVICE)
            else:
                proj = MLPProjector(in_dim=in_full, hidden=256, out_dim=args.mlp_out_dim).to(DEVICE)
            all_params = lora_params + list(proj.parameters())
            opt = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)

            def save_state():
                lora_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
                return {"proj_state": {k: v.cpu().clone() for k, v in proj.state_dict().items()}, "lora_state": lora_state}

            proj, train_hist, best, snapshots, iv_tr_idx, iv_va_idx, iv_train_cells, iv_val_cells, ex_tr_idx, ex_va_idx, ex_train_cells, ex_val_cells = run_full_iv_ex_lora_training(
                cache_iv=cache_iv,
                cache_ex=cache_ex,
                iv_samples=iv_samples,
                ex_samples=ex_samples,
                epochs=args.epochs,
                batch=args.batch,
                lr=args.lr,
                wd=args.weight_decay,
                temp=args.temperature,
                k=args.knn_k,
                seed=args.seed,
                snapshot_every=max(1, args.snapshot_every),
                num_cells=n_cells,
                train_source=args.train_source,
                mlp_out_dim=args.mlp_out_dim,
                tail_fn=tail_fn,
                proj=proj,
                opt=opt,
                best_state_callback=save_state,
                lora_model=model,
                alignment_loss_name=args.alignment_loss,
                lambda_align=args.lambda_align,
                dann_schedule=args.dann_schedule,
                shared_labels=args.shared_labels,
                lambda_cross_pair=args.lambda_cross_pair,
                geom_feat_iv=geom_feat_iv,
                geom_feat_ex=geom_feat_ex,
                eval_geom_zero=bool(args.eval_geom_zero),
                geom_dim=geom_dim_emb,
            )

            best_state = best["state"]
            if best_state and "lora_state" in best_state:
                model.load_state_dict(best_state["lora_state"], strict=False)

            with torch.no_grad():
                emb_iv_cls = _run_tail_on_cache_batches(cache_iv, tail_fn).numpy().astype(np.float32)
                emb_ex_cls = _run_tail_on_cache_batches(cache_ex, tail_fn).numpy().astype(np.float32)
            if geom_feat_iv is not None:
                emb_iv = np.concatenate([emb_iv_cls, geom_feat_iv], axis=1).astype(np.float32)
                emb_ex = np.concatenate([emb_ex_cls, geom_feat_ex], axis=1).astype(np.float32)
            else:
                emb_iv, emb_ex = emb_iv_cls, emb_ex_cls

            y_iv = np.array([s["cell_idx"] for s in iv_samples], dtype=np.int64)
            y_ex = np.array([s["cell_idx"] for s in ex_samples], dtype=np.int64)
            ev = eval_ex_to_iv_knn(
                proj=proj,
                emb_iv=np_eval_geom_zero(emb_iv, geom_dim_emb, args.eval_geom_zero),
                emb_ex=np_eval_geom_zero(emb_ex, geom_dim_emb, args.eval_geom_zero),
                y_iv=y_iv,
                y_ex=y_ex,
                k=args.knn_k,
                num_cells=n_cells,
            )
        render_coembed_report(
            paths=paths,
            cell_ids=cell_ids,
            iv_samples=iv_samples,
            ex_samples=ex_samples,
            snapshots=snapshots,
            train_hist=train_hist,
            confusion=ev["confusion"],
            tr_idx=iv_tr_idx,
            best_epoch=int(best["epoch"]),
            best_val_acc=float(best["val_acc"]),
            ex_to_iv_top1_patch_knn=float(ev["acc_top1"]),
            ex_to_iv_top5_patch_knn=float(ev["ex_to_iv_top5_acc_patch_knn"]),
            ex_to_iv_top5_cell_proto=float(ev["ex_to_iv_top5_acc_cell_proto"]),
            ex_to_iv_mean_rank_cell_proto=float(ev["ex_to_iv_mean_rank_cell_proto"]),
            ex_to_iv_mrr_cell_proto=float(ev["ex_to_iv_mrr_cell_proto"]),
        )

        fused_input_dim = int(emb_iv.shape[1])
        out = {
            "mode": args.mode,
            "model_id": args.dino_model_id,
            "cls_dim": fused_input_dim,
            "using_lora_checkpoint": args.backbone_mode == "lora",
            "settings": {
                "epochs": args.epochs,
                "batch": args.batch,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "temperature": args.temperature,
                "knn_k": args.knn_k,
                "seed": args.seed,
                "aug_presets": aug_presets,
                "aug_per_preset": args.aug_per_preset,
                "dino_model_id": args.dino_model_id,
                "dino_version": args.dino_version,
                "train_source": args.train_source,
                "frozen_embed_dim": args.frozen_embed_dim,
                "lora_bottleneck_dim": args.lora_bottleneck_dim,
                "mlp_out_dim": args.mlp_out_dim,
                "backbone_mode": args.backbone_mode,
                "lora_blocks": args.lora_blocks,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "alignment_loss": args.alignment_loss,
                "lambda_align": args.lambda_align,
                "dann_schedule": args.dann_schedule,
                "shared_labels": bool(args.shared_labels),
                "lambda_cross_pair": float(args.lambda_cross_pair),
                "geom_neighbor_k": int(args.geom_neighbor_k),
                "geom_fuse": args.geom_fuse,
                "eval_geom_zero": bool(args.eval_geom_zero),
                "embed_batch_size": args.embed_batch_size,
                "patch_xy_um": int(args.patch_xy_um),
                "patch_z_um": int(args.patch_z_um),
                "spatial_jitter_vox": float(args.spatial_jitter_vox),
                "multiscale_enable": bool(args.multiscale_enable),
                "multiscale_xy_ums": list(args.multiscale_xy_ums),
                "multiscale_z_ums": list(args.multiscale_z_ums),
                "output_tag": output_tag,
                "patch_cache": patch_cache_in or None,
            },
            "num_cells_total": n_cells,
            "num_iv_samples": len(iv_samples),
            "num_ex_samples": len(ex_samples),
            "num_iv_train_cells": len(iv_train_cells),
            "num_iv_val_cells": len(iv_val_cells),
            "num_ex_train_cells": len(ex_train_cells),
            "num_ex_val_cells": len(ex_val_cells),
            "best_epoch_by_balanced_val_knn": int(best["epoch"]),
            "best_balanced_val_knn_acc": float(best["val_acc"]),
            "best_iv_val_knn_acc": float(best["iv_val_acc"]),
            "best_ex_val_knn_acc": float(best["ex_val_acc"]),
            "iv_val_knn_acc": float(best["iv_val_acc"]),
            "ex_val_knn_acc": float(best["ex_val_acc"]),
            "ex_to_iv_top1_acc": ev["acc_top1"],
            "ex_to_iv_topk_purity": ev["topk_purity"],
            "ex_to_iv_top5_acc_patch_knn": ev["ex_to_iv_top5_acc_patch_knn"],
            "ex_to_iv_top5_acc_cell_proto": ev["ex_to_iv_top5_acc_cell_proto"],
            "ex_to_iv_mean_rank_cell_proto": ev["ex_to_iv_mean_rank_cell_proto"],
            "ex_to_iv_mrr_cell_proto": ev["ex_to_iv_mrr_cell_proto"],
            "confusion_matrix_ex_true_vs_iv_pred": ev["confusion"],
            "confusion_matrix_labels": cell_ids,
            "train_history": train_hist,
            "coembed_html": str(paths["coembed_html"]),
            "proj_ckpt": str(paths["proj_ckpt"]),
            "elapsed_s": time.time() - t0,
            "results_json": str(paths["results"]),
        }
        if args.backbone_mode == "frozen" and best.get("state"):
            torch.save(best["state"], paths["proj_ckpt"])
            print(f"Wrote projector checkpoint {paths['proj_ckpt']}", flush=True)
        paths["results"].write_text(json.dumps(out, indent=2), encoding="utf-8")
        paths["folds_jsonl"].write_text(
            "".join(json.dumps(x) + "\n" for x in train_hist),
            encoding="utf-8",
        )
        print(f"Wrote {paths['results']}", flush=True)
        return

    embedder = nr.make_frozen_embedder(args.dino_version, args.dino_model_id, TORCH_DEVICE)
    _, cell_ids, samples = build_dataset(
        rng,
        aug_presets=aug_presets,
        aug_per_preset=args.aug_per_preset,
        patch_xy_um=args.patch_xy_um,
        patch_z_um=args.patch_z_um,
        spatial_jitter_vox=args.spatial_jitter_vox,
        multiscale_enable=bool(args.multiscale_enable),
        multiscale_xy_ums=args.multiscale_xy_ums,
        multiscale_z_ums=args.multiscale_z_ums,
    )
    n_cells = len(cell_ids)
    print(f"Prepared samples: {len(samples)} from {n_cells} cells", flush=True)
    emb = embed_multiscale_samples_cls(embedder, samples, args.embed_batch_size, bool(args.multiscale_enable))
    print(
        f"Embeddings shape: {emb.shape} "
        f"(base CLS dim={embedder.cls_dim}, multiscale={args.multiscale_enable})",
        flush=True,
    )

    fold_indices = list(range(n_cells))
    if args.only_cell:
        if args.only_cell not in cell_ids:
            raise ValueError(f"Unknown --only-cell '{args.only_cell}'")
        fold_indices = [cell_ids.index(args.only_cell)]
    elif args.max_folds > 0:
        fold_indices = fold_indices[: args.max_folds]

    anim_idx = cell_ids.index(args.animation_cell) if args.animation_cell in cell_ids else fold_indices[0]
    all_fold_results = []
    confusion_counts = np.zeros((n_cells, n_cells), dtype=np.int64)
    captured_fold = None

    for f_i, left_out in enumerate(fold_indices, start=1):
        capture = left_out == anim_idx
        print(f"[fold {f_i}/{len(fold_indices)}] left_out={left_out} ({cell_ids[left_out]}) capture={capture}", flush=True)
        fr = run_fold(
            emb=emb,
            samples=samples,
            left_out_cell=left_out,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            wd=args.weight_decay,
            temp=args.temperature,
            k=args.knn_k,
            seed=args.seed,
            capture_snapshots=capture,
            num_total_cells=n_cells,
            mlp_out_dim=args.mlp_out_dim,
        )
        if capture and "snapshots" in fr and args.snapshot_every > 1:
            fr["snapshots"] = [s for i, s in enumerate(fr["snapshots"]) if i % args.snapshot_every == 0]
        if capture:
            captured_fold = fr
        row = {k: v for k, v in fr.items() if k not in ("history", "snapshots", "split")}
        if "test_pred_cells" in fr:
            for p in fr["test_pred_cells"]:
                confusion_counts[left_out, int(p)] += 1
        all_fold_results.append(row)
        with paths["folds_jsonl"].open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"  best_ep={fr['best_epoch']} val={fr['best_val_knn_acc']:.4f} "
            f"test_sample_top1={fr['test_sample_top1_acc']:.4f} purity@{args.knn_k}={fr['test_nn_purity_topk']:.4f}"
            ,
            flush=True,
        )

    cell_top1 = float(np.mean([1.0 if r["cell_correct_majority"] else 0.0 for r in all_fold_results])) if all_fold_results else 0.0
    sample_top1 = float(np.mean([r["test_sample_top1_acc"] for r in all_fold_results])) if all_fold_results else 0.0
    nn_purity = float(np.mean([r["test_nn_purity_topk"] for r in all_fold_results])) if all_fold_results else 0.0

    out = {
        "model_id": args.dino_model_id,
        "cls_dim": int(emb.shape[1]),
        "using_lora_checkpoint": False,
        "settings": {
            "epochs_per_fold": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "knn_k": args.knn_k,
            "seed": args.seed,
            "aug_presets": aug_presets,
            "aug_per_preset": args.aug_per_preset,
            "animation_cell": args.animation_cell,
            "max_folds": args.max_folds,
            "snapshot_every": args.snapshot_every,
            "dino_version": args.dino_version,
            "dino_model_id": args.dino_model_id,
            "mlp_out_dim": args.mlp_out_dim,
            "embed_batch_size": args.embed_batch_size,
            "patch_xy_um": int(args.patch_xy_um),
            "patch_z_um": int(args.patch_z_um),
            "spatial_jitter_vox": float(args.spatial_jitter_vox),
            "multiscale_enable": bool(args.multiscale_enable),
            "multiscale_xy_ums": list(args.multiscale_xy_ums),
            "multiscale_z_ums": list(args.multiscale_z_ums),
            "output_tag": output_tag,
        },
        "num_cells_total": n_cells,
        "num_folds_run": len(fold_indices),
        "loocv_cell_top1_acc": cell_top1,
        "loocv_sample_top1_acc": sample_top1,
        "loocv_test_nn_purity_topk": nn_purity,
        "confusion_matrix": confusion_counts.tolist(),
        "confusion_matrix_labels": cell_ids,
        "elapsed_s": time.time() - t0,
        "folds": all_fold_results,
        "fold_log_jsonl": str(paths["folds_jsonl"]),
        "animation_html": str(paths["anim_html"]) if paths["anim_html"].exists() else None,
        "animation_json": str(paths["anim_json"]) if paths["anim_json"].exists() else None,
        "summary_html": str(paths["summary_html"]),
        "results_json": str(paths["results"]),
    }
    paths["results"].write_text(json.dumps(out, indent=2), encoding="utf-8")
    if captured_fold is not None:
        render_fold_animation(
            samples=samples,
            fold_res=captured_fold,
            paths=paths,
            confusion_matrix=confusion_counts.tolist(),
            cell_ids=cell_ids,
        )
        paths["summary_html"].write_text(paths["anim_html"].read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {paths['results']}", flush=True)


if __name__ == "__main__":
    main()
