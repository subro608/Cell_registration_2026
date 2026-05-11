#!/usr/bin/env python3
"""
Unified experiment runner for native LOOCV cell re-identification.

Supports:
  - Multiple alignment methods (frozen, mean_std, ridge, pca_ridge, procrustes, …)
  - ``alignment: "tps_fusion"`` → geometry fusion with thin-plate-spline position predictor
  - Optional spatial k-NN or geometry-fusion re-ranking
  - ``feature: "multi_channel"`` + ``invivo_channels_1based: [2, 4]`` (two in-vivo zstack channels)
  - LOOCV or val/test split evaluation
  - Feature caching to disk

Usage:
  python native_runner.py --config config.json

  or from Python:

  import native_runner
  result = native_runner.run_config({
      "description": "...",
      "alignment": "pca_ridge",
      "alignment_params": {...},
      "reranking": "spatial_knn",
      "reranking_params": {...},
      "eval_mode": "loocv",
      ...
  })
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from scipy.ndimage import map_coordinates, rotate as nd_rotate
from scipy.spatial.distance import pdist, squareform
from skimage.exposure import equalize_adapthist, equalize_hist
from transformers import AutoImageProcessor, AutoModel

# ---------------------------------------------------------------------------
# Frozen vision embedder protocol (DINOv2/v3 via Transformers, EUPE ViT via torch.hub)
# ---------------------------------------------------------------------------


@runtime_checkable
class FrozenVisionEmbedder(Protocol):
    device: torch.device
    cls_dim: int

    def embed_cls(self, patches_rgb01, batch_size: int | None = None) -> np.ndarray: ...

    def embed_patch_tokens(self, patches_rgb01, pooling: str = "mean", batch_size: int | None = None) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# `runner.py` lives at <project>/src/runner.py — project root is two levels up.
PROJECT_ROOT     = Path(__file__).parent.parent.resolve()
AUTORESEARCH_DIR = PROJECT_ROOT  # alias kept for upstream code that still references it
CELLFIND_ROOT    = Path(
    os.environ.get("CELLFIND_ROOT", "/Users/erdem/Documents/github/cellfind")
)

# Try the bundled config first (configs/rotation_domain_invariance.yaml);
# fall back to the cellfind source tree if it isn't present.
_LOCAL_CFG    = PROJECT_ROOT / "configs" / "rotation_domain_invariance.yaml"
_CELLFIND_CFG = CELLFIND_ROOT / "configs" / "rotation_domain_invariance.yaml"

# ---------------------------------------------------------------------------
# Subject registry
# ---------------------------------------------------------------------------
# Each subject pins the filenames, in-vivo voxel size, and zstack channel
# layout. Select with `CELLINVARIANCE_SUBJECT=<name>` (default: "sparrow").
# Add a new subject by appending an entry here — no other edits required.
SUBJECTS: dict[str, dict] = {
    "sparrow": {
        "zstack":                "zstack.tif",
        "exvivo":                "Sparrow_3_po_488_4x-registered.tif",
        "landmarks":             "slice3_to_invivoLANDMARKS.json",
        "invivo_channel_1based": 2,            # 0 = no channel dim (3D zstack)
        "invivo_voxel_dims_um":  (1.0, 1.0, 4.0),
    },
    "jy306": {
        "zstack":                "JY306_in_Vivo_stack_flipped_s80.tif",
        "exvivo":                "stitched_gfp_fullres_v5_1um_isotropic.tif",
        "landmarks":             "jy306_landmarks.json",
        "invivo_channel_1based": 0,            # zstack is plain (Z, Y, X)
        "invivo_voxel_dims_um":  (0.6835, 0.6835, 3.0),
    },
}

SUBJECT = os.environ.get("CELLINVARIANCE_SUBJECT", "sparrow").lower()
if SUBJECT not in SUBJECTS:
    raise RuntimeError(
        f"Unknown CELLINVARIANCE_SUBJECT={SUBJECT!r}; choose from {sorted(SUBJECTS)}"
    )
_SUBJ_CFG = SUBJECTS[SUBJECT]


def _resolve_dataset_dir() -> Path:
    """Resolve the data directory for the current subject.

    Precedence:
      1. ``CELLINVARIANCE_DATA_DIR`` env var (explicit override, used verbatim).
      2. ``<project_root>/<config dataset_dir>/<subject>``.
      3. ``<project_root>/data/<subject>``.
    """
    env_dir = os.environ.get("CELLINVARIANCE_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    cfg_path = _LOCAL_CFG if _LOCAL_CFG.exists() else _CELLFIND_CFG
    try:
        import yaml
        _cfg = yaml.safe_load(cfg_path.read_text())
        raw = Path(_cfg["dataset"]["dataset_dir"])
    except Exception as _e:
        print(f"[warn] Config fallback ({_e}): defaulting to {PROJECT_ROOT / 'data' / SUBJECT}")
        return PROJECT_ROOT / "data" / SUBJECT

    if not raw.is_absolute():
        raw = (PROJECT_ROOT / raw).resolve()
    return raw / SUBJECT


DATASET_DIR    = _resolve_dataset_dir()
ZSTACK_PATH    = DATASET_DIR / _SUBJ_CFG["zstack"]
EXVIVO_PATH    = DATASET_DIR / _SUBJ_CFG["exvivo"]
LANDMARKS_PATH = DATASET_DIR / _SUBJ_CFG["landmarks"]

# Caches and per-run artifacts live under .feature_cache/ at the project root
# (gitignored). Override with CELLINVARIANCE_CACHE_DIR if you need a different
# location.
OUTPUT_DIR    = Path(
    os.environ.get("CELLINVARIANCE_CACHE_DIR", PROJECT_ROOT / ".feature_cache")
)
CACHE_DIR     = OUTPUT_DIR / "cache"
RESULTS_DIR   = OUTPUT_DIR / "runs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DINO_MODEL_ID         = "facebook/dinov3-vitl16-pretrain-lvd1689m"
INVIVO_CHANNEL_1BASED = _SUBJ_CFG["invivo_channel_1based"]
INVIVO_VOXEL_DIMS_UM  = _SUBJ_CFG["invivo_voxel_dims_um"]
ORBIT_ANGLES_DEG      = tuple(range(0, 360, 15))
BATCH_SIZE            = 8
PAD_VALUE             = 0.0
BOOTSTRAP_N           = 5000   # 5× more resamples → tighter CIs per experiment
BOOTSTRAP_SEED        = 99
SPLIT_SEED            = 23

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def natural_key(s):
    m = re.search(r"(\d+)", s)
    return (int(m.group(1)), s) if m else (math.inf, s)

def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def norm_perc_linear(plane):
    lo, hi = np.percentile(plane, [1.0, 99.8])
    if hi <= lo:
        return np.zeros_like(plane, dtype=np.float32)
    return np.clip((plane - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

def norm_perc_gamma(plane):
    lo, hi = np.percentile(plane, [1.0, 99.8])
    if hi <= lo:
        return np.zeros_like(plane, dtype=np.float32)
    return np.power(np.clip((plane - lo) / (hi - lo), 0.0, 1.0).astype(np.float32), 0.85)

def norm_clahe(plane):
    lo, hi = np.percentile(plane, [0.1, 99.9])
    if hi <= lo:
        return np.zeros_like(plane, dtype=np.float32)
    return equalize_adapthist(np.clip((plane - lo) / (hi - lo), 0.0, 1.0).astype(np.float32), clip_limit=0.03).astype(np.float32)

def norm_histeq(plane):
    return equalize_hist(plane.astype(np.float32)).astype(np.float32)

NORM_FN_MAP = {
    "perc_gamma": norm_perc_gamma,
    "perc_linear": norm_perc_linear,
    "clahe": norm_clahe,
    "histeq": norm_histeq,
}

def slab_to_depth_lut(slab, norm_fn):
    """Render depth slab to RGB using turbo colormap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nz, h, w = slab.shape
    turbo = plt.get_cmap("turbo")(np.linspace(0.0, 1.0, nz))[:, :3].astype(np.float32)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for z in range(nz):
        rgb += norm_fn(slab[z])[..., None] * turbo[z][None, None, :]
    scale = float(np.percentile(rgb, 99.5)) if np.any(rgb > 0) else 1.0
    return np.clip(rgb / max(scale, 1e-8), 0.0, 1.0).astype(np.float32)

def extract_isotropic_slab(vol, center_xyz_vox, vox_dims, xy_um, z_um):
    """Extract isotropic patch centered at landmark."""
    xc = yc = int(xy_um)
    zc = int(z_um)
    x_off = np.arange(xc, dtype=np.float32) - (xc - 1) / 2.0
    y_off = np.arange(yc, dtype=np.float32) - (yc - 1) / 2.0
    z_off = np.arange(zc, dtype=np.float32) - (zc - 1) / 2.0
    zz, yy, xx = np.meshgrid(z_off, y_off, x_off, indexing="ij")
    cx, cy, cz = center_xyz_vox
    vx, vy, vz = vox_dims
    coords = np.vstack([
        cz + zz.reshape(-1) / max(vz, 1e-6),
        cy + yy.reshape(-1) / max(vy, 1e-6),
        cx + xx.reshape(-1) / max(vx, 1e-6)
    ])
    return map_coordinates(
        vol.astype(np.float32), coords, order=1, mode="constant", cval=PAD_VALUE
    ).reshape(zc, yc, xc).astype(np.float32)

def slab_to_brightest_slice_rgb(slab, norm_fn):
    """Find the z-slice with maximum total intensity; render as 3-channel grayscale.

    Removes depth-blur by picking the single sharpest plane — better cross-modality
    comparability when axial resolution differs (4µm in-vivo vs 0.39µm ex-vivo).
    """
    nz, h, w = slab.shape
    z_sums = slab.reshape(nz, -1).sum(axis=1)
    best_z = int(np.argmax(z_sums))
    plane = norm_fn(slab[best_z])   # (H, W) in [0,1]
    rgb = np.stack([plane, plane, plane], axis=-1)
    return rgb.astype(np.float32)


def slab_to_mip_rgb(slab, norm_fn):
    """Max-intensity projection along z → 3-channel grayscale."""
    mip = slab.max(axis=0)
    plane = norm_fn(mip)
    rgb = np.stack([plane, plane, plane], axis=-1)
    return rgb.astype(np.float32)


def slab_to_variance_rgb(slab, norm_fn):
    """Per-pixel variance across z → emphasize depth structure."""
    var_z = slab.var(axis=0)
    plane = norm_fn(var_z)
    rgb = np.stack([plane, plane, plane], axis=-1)
    return rgb.astype(np.float32)


def slab_to_multichannel_rgb_depth(slab, norm_fn):
    """R=MIP, G=mean z-slice intensity, B=depth-of-mass (centroid index)."""
    nz, h, w = slab.shape
    r = norm_fn(slab.max(axis=0))
    g = norm_fn(slab.mean(axis=0))
    z_idx = np.arange(nz, dtype=np.float32)
    z_weighted = np.sum(slab * z_idx[:, None, None], axis=0)
    z_total = slab.sum(axis=0) + 1e-8
    z_centroid = z_weighted / z_total
    b = norm_fn(z_centroid)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb.astype(np.float32)


def slab_to_depth_lut_gaussian(slab, norm_fn):
    """Gaussian-weighted depth LUT: de-emphasizes far-from-center slices.

    The cell soma occupies the central z-slices; Gaussian weights reduce the
    contribution of background above/below, making in-vivo and ex-vivo renders
    more consistent when tissue structure differs axially.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nz, h, w = slab.shape
    turbo = plt.get_cmap("turbo")(np.linspace(0.0, 1.0, nz))[:, :3].astype(np.float32)
    z_idx = np.arange(nz, dtype=np.float32)
    sigma = max(nz / 4.0, 1.0)
    gaus = np.exp(-0.5 * ((z_idx - nz / 2.0) / sigma) ** 2)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for z in range(nz):
        rgb += norm_fn(slab[z])[..., None] * turbo[z][None, None, :] * gaus[z]
    scale = float(np.percentile(rgb, 99.5)) if np.any(rgb > 0) else 1.0
    return np.clip(rgb / max(scale, 1e-8), 0.0, 1.0).astype(np.float32)


def rotate_rgb_patch(p, a):
    """Rotate 3-channel patch."""
    if int(a) % 360 == 0:
        return p.copy()
    return np.clip(
        nd_rotate(p, float(a), axes=(1, 0), reshape=False, order=1, mode="nearest"),
        0.0, 1.0
    ).astype(np.float32)

def fit_exvivo_voxel_dims(lm_df):
    """Fit ex-vivo voxel dimensions using Procrustes alignment."""
    inv = np.array([[l["invivo_x"], l["invivo_y"], l["invivo_z"]] for l in lm_df], np.float64)
    ex  = np.array([[l["exvivo_x"], l["exvivo_y"], l["exvivo_z"]] for l in lm_df], np.float64)
    ex_c = ex.mean(0)
    iv_c = inv.mean(0)
    ex_z = ex - ex_c
    iv_z = inv - iv_c
    U, _, Vt = np.linalg.svd(ex_z.T @ iv_z)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    rotated = ex_z @ R
    scale_xyz = np.sum(rotated * iv_z, 0) / np.maximum(np.sum(rotated ** 2, 0), 1e-8)
    iv = INVIVO_VOXEL_DIMS_UM
    return (
        float(abs(scale_xyz[0]) * iv[0]),
        float(abs(scale_xyz[1]) * iv[1]),
        float(abs(scale_xyz[2]) * iv[2])
    )

def load_landmarks():
    """Load landmarks from JSON."""
    raw = json.loads(LANDMARKS_PATH.read_text())
    rows = raw["landmarks"]["values"]
    out = [
        {
            "id": str(r[0]),
            "invivo_x": float(r[1][0]),
            "invivo_y": float(r[1][1]),
            "invivo_z": float(r[1][2]),
            "exvivo_x": float(r[2][0]),
            "exvivo_y": float(r[2][1]),
            "exvivo_z": float(r[2][2]),
        }
        for r in rows
    ]
    out.sort(key=lambda d: natural_key(d["id"]))
    return out

def load_volumes():
    """Load in-vivo and ex-vivo volumes."""
    log("Loading volumes…")
    zs = tifffile.imread(str(ZSTACK_PATH)).astype(np.float32)
    if INVIVO_CHANNEL_1BASED > 0:
        if zs.ndim != 4:
            raise RuntimeError(
                f"Expected (Z,C,Y,X) when invivo_channel_1based={INVIVO_CHANNEL_1BASED}, got {zs.shape}"
            )
        iv = zs[:, INVIVO_CHANNEL_1BASED - 1, :, :]
    else:
        if zs.ndim != 3:
            raise RuntimeError(
                f"Expected (Z,Y,X) when invivo_channel_1based=0, got {zs.shape}"
            )
        iv = zs
    ex = tifffile.imread(str(EXVIVO_PATH)).astype(np.float32)
    if ex.ndim == 2:
        ex = ex[None]
    log(f"  in-vivo: {iv.shape}  ex-vivo: {ex.shape}")
    return iv, ex

# ---------------------------------------------------------------------------
# Image processor (DINOv2 checkpoints sometimes omit processor files on HF)
# ---------------------------------------------------------------------------

def load_image_processor_for_model(model_id: str):
    """
    Load AutoImageProcessor with fallbacks so DINOv2-small and similar ids work
    when the hub repo lacks preprocessor_config (use another DINOv2 size's processor).
    DINOv3: try ViT-B processor if ViT-L repo has no cached preprocessor.
    """
    candidates = [model_id]
    mid = model_id.lower()
    if "dinov2" in mid:
        for alt in (
            "facebook/dinov2-base",
            "facebook/dinov2-large",
            "facebook/dinov2-small",
        ):
            if alt not in candidates:
                candidates.append(alt)
    if "dinov3" in mid:
        # Do NOT fallback to other DINOv3 repos (can be gated differently).
        # Keep candidate pinned to requested model_id.
        pass
    last_err: Exception | None = None
    # Prefer local HF cache (same blobs as successful prior runs), then hub.
    for local_only in (True, False):
        for cid in candidates:
            for kwargs in ({}, {"use_fast": False}):
                try:
                    proc = AutoImageProcessor.from_pretrained(
                        cid, trust_remote_code=True, local_files_only=local_only, **kwargs
                    )
                    if cid != model_id:
                        log(f"  Image processor: using '{cid}' (fallback for '{model_id}')")
                    elif local_only:
                        log(f"  Image processor: from local cache ({model_id})")
                    return proc
                except Exception as e:
                    last_err = e
    assert last_err is not None
    raise last_err


def load_auto_model_for_model(model_id: str, device):
    """Load AutoModel; try local HF cache first, then download (matches prior embedding runs)."""
    last_err: Exception | None = None
    for local_only in (True, False):
        try:
            m = AutoModel.from_pretrained(
                model_id, trust_remote_code=True, local_files_only=local_only
            ).eval().to(device)
            if local_only:
                log(f"  Backbone: from local cache ({model_id})")
            return m
        except Exception as e:
            last_err = e
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# EUPE ViT (HF-hosted .pt + facebookresearch/eupe architecture via torch.hub)
# ---------------------------------------------------------------------------

EUPE_TORCH_HUB = "facebookresearch/eupe:main"

# Normalized HF repo id -> (exact Hub repo id, torch.hub entrypoint, weight filename)
EUPE_VIT_SPECS: dict[str, tuple[str, str, str]] = {
    "facebook/eupe-vit-t": ("facebook/EUPE-ViT-T", "eupe_vitt16", "EUPE-ViT-T.pt"),
    "facebook/eupe-vit-s": ("facebook/EUPE-ViT-S", "eupe_vits16", "EUPE-ViT-S.pt"),
    "facebook/eupe-vit-b": ("facebook/EUPE-ViT-B", "eupe_vitb16", "EUPE-ViT-B.pt"),
}


def normalize_hf_repo_id(model_id: str) -> str:
    return model_id.strip().lower()


def is_eupe_vit_model_id(model_id: str) -> bool:
    return normalize_hf_repo_id(model_id) in EUPE_VIT_SPECS


def resolve_eupe_vit_hub_and_file(model_id: str) -> tuple[str, str, str, str]:
    """Return (normalized_key, hf_repo_id, hub_entrypoint, weights_filename)."""
    key = normalize_hf_repo_id(model_id)
    if key not in EUPE_VIT_SPECS:
        raise ValueError(
            f"Unknown EUPE ViT model_id {model_id!r}; expected one of: {sorted(EUPE_VIT_SPECS)}"
        )
    repo_exact, entry, fname = EUPE_VIT_SPECS[key]
    return key, repo_exact, entry, fname


def _strip_state_dict_prefix(state: dict, prefix: str) -> dict:
    keys = [k for k in state if isinstance(k, str) and k.startswith(prefix)]
    if not keys:
        return state
    return {k[len(prefix) :]: v for k, v in state.items() if isinstance(k, str) and k.startswith(prefix)}


def load_eupe_vit_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load EUPE .pt from disk; tolerate nested dicts and DDP 'module.' prefix."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unexpected EUPE checkpoint type from {ckpt_path}: {type(ckpt)}")
    ckpt = _strip_state_dict_prefix(ckpt, "module.")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    log(f"  EUPE weights: missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if unexpected:
        log(f"  EUPE unexpected (first 12): {unexpected[:12]}")


def build_eupe_vit_model(hub_entrypoint: str, device: torch.device) -> torch.nn.Module:
    # hub.load forwards kwargs to the entrypoint; pretrained must be False here or
    # eupe_* will try to download official .pth from dl.fbaipublicfiles.com (we use HF .pt).
    model = torch.hub.load(
        EUPE_TORCH_HUB, hub_entrypoint, pretrained=False, trust_repo=True
    )
    return model.eval().to(device)


# ---------------------------------------------------------------------------
# DINOv2 / DINOv3 embedder (Transformers)
# ---------------------------------------------------------------------------

class DinoEmbedder:
    def __init__(self, model_id, device):
        log("Loading DINO…")
        self.processor = load_image_processor_for_model(model_id)
        self.model = load_auto_model_for_model(model_id, device)
        self.device = device
        with torch.no_grad():
            dummy = np.zeros((32, 32, 3), dtype=np.uint8)
            inp = self.processor(images=[dummy], return_tensors="pt")
            out = self.model(**{k: v.to(device) for k, v in inp.items()})
            self.cls_dim = out.last_hidden_state.shape[-1]
        log(f"  CLS dim={self.cls_dim}")

    @torch.no_grad()
    def embed_cls(self, patches_rgb01, batch_size: int | None = None):
        """Embed batch of RGB patches as CLS tokens."""
        bs = int(batch_size) if batch_size is not None else BATCH_SIZE
        bs = max(1, bs)
        imgs = [(p * 255.0).clip(0, 255).astype(np.uint8) for p in patches_rgb01]
        outs = []
        for s in range(0, len(imgs), bs):
            chunk = imgs[s : s + bs]
            inp = {k: v.to(self.device) for k, v in self.processor(images=chunk, return_tensors="pt").items()}
            cls = self.model(**inp).last_hidden_state[:, 0, :]
            outs.append(F.normalize(cls, dim=-1).cpu().numpy())
        return np.concatenate(outs, axis=0)

    @torch.no_grad()
    def embed_patch_tokens(self, patches_rgb01, pooling="mean", batch_size: int | None = None):
        """Spatial patch tokens (exclude CLS). pooling: mean | max | attention."""
        bs = int(batch_size) if batch_size is not None else BATCH_SIZE
        bs = max(1, bs)
        imgs = [(p * 255.0).clip(0, 255).astype(np.uint8) for p in patches_rgb01]
        outs = []
        for s in range(0, len(imgs), bs):
            chunk = imgs[s : s + bs]
            inp = {k: v.to(self.device) for k, v in self.processor(images=chunk, return_tensors="pt").items()}
            hidden = self.model(**inp).last_hidden_state  # (B, 1+P, D)
            patch_tokens = hidden[:, 1:, :]  # (B, P, D)
            if pooling == "mean":
                pooled = patch_tokens.mean(dim=1)
            elif pooling == "max":
                pooled = patch_tokens.max(dim=1)[0]
            elif pooling == "attention":
                attn = F.softmax(patch_tokens.mean(dim=-1, keepdim=True), dim=1)
                pooled = (patch_tokens * attn).sum(dim=1)
            else:
                raise ValueError(f"Unknown patch pooling: {pooling}")
            outs.append(F.normalize(pooled, dim=-1).cpu().numpy())
        return np.concatenate(outs, axis=0)


class EupeVitEmbedder:
    """
    EUPE ViT from Meta (ViT-T/S/B on Hugging Face: single .pt per repo).
    Preprocess: ImageNet norm, square resize (default 256) — matches upstream examples.
    """

    def __init__(self, model_id: str, device, *, resize_size: int = 256):
        from huggingface_hub import hf_hub_download
        from torchvision.transforms import v2 as T

        _key, hf_repo, hub_entry, fname = resolve_eupe_vit_hub_and_file(model_id)
        log(f"Loading EUPE ViT ({hf_repo})…")
        weights_path = hf_hub_download(hf_repo, fname)
        log(f"  weights: {weights_path}")
        self.model = build_eupe_vit_model(hub_entry, device)
        load_eupe_vit_checkpoint(self.model, weights_path)
        self.device = device
        self.resize_size = int(resize_size)
        self._transform = T.Compose(
            [
                T.ToImage(),
                T.Resize((self.resize_size, self.resize_size), antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        with torch.no_grad():
            dummy = np.zeros((32, 32, 3), dtype=np.uint8)
            x = self._tensor_batch([dummy]).to(device)
            out = self.model.forward_features(x)
            if not isinstance(out, dict) or "x_norm_clstoken" not in out:
                raise RuntimeError(f"EUPE forward_features missing x_norm_clstoken; keys={out.keys() if isinstance(out, dict) else type(out)}")
            self.cls_dim = int(out["x_norm_clstoken"].shape[-1])
        log(f"  CLS dim={self.cls_dim}")

    def _tensor_batch(self, patches_rgb01) -> torch.Tensor:
        tensors = []
        for p in patches_rgb01:
            u8 = (np.clip(np.asarray(p, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
            tensors.append(self._transform(u8))
        return torch.stack(tensors, dim=0)

    @torch.no_grad()
    def embed_cls(self, patches_rgb01, batch_size: int | None = None):
        bs = int(batch_size) if batch_size is not None else BATCH_SIZE
        bs = max(1, bs)
        outs = []
        n = len(patches_rgb01)
        for s in range(0, n, bs):
            chunk = patches_rgb01[s : s + bs]
            x = self._tensor_batch(chunk).to(self.device)
            out = self.model.forward_features(x)
            cls = out["x_norm_clstoken"].float()
            outs.append(F.normalize(cls, dim=-1).cpu().numpy())
        return np.concatenate(outs, axis=0)

    @torch.no_grad()
    def embed_patch_tokens(self, patches_rgb01, pooling="mean", batch_size: int | None = None):
        bs = int(batch_size) if batch_size is not None else BATCH_SIZE
        bs = max(1, bs)
        outs = []
        n = len(patches_rgb01)
        for s in range(0, n, bs):
            chunk = patches_rgb01[s : s + bs]
            x = self._tensor_batch(chunk).to(self.device)
            out = self.model.forward_features(x)
            patch_tokens = out["x_norm_patchtokens"].float()
            if pooling == "mean":
                pooled = patch_tokens.mean(dim=1)
            elif pooling == "max":
                pooled = patch_tokens.max(dim=1)[0]
            elif pooling == "attention":
                attn = F.softmax(patch_tokens.mean(dim=-1, keepdim=True), dim=1)
                pooled = (patch_tokens * attn).sum(dim=1)
            else:
                raise ValueError(f"Unknown patch pooling: {pooling}")
            outs.append(F.normalize(pooled, dim=-1).cpu().numpy())
        return np.concatenate(outs, axis=0)


def make_frozen_embedder(dino_version: str, model_id: str, device) -> DinoEmbedder | EupeVitEmbedder:
    """
    Frozen CLS embedder for native pipelines.
    EUPE is selected when --dino-version eupe or when model_id is a known EUPE ViT Hub id.
    """
    ver = (dino_version or "").strip().lower()
    mid = (model_id or "").strip()
    if ver == "eupe" and not mid:
        mid = "facebook/EUPE-ViT-S"
    if ver == "eupe" or is_eupe_vit_model_id(mid):
        return EupeVitEmbedder(mid, device)
    return DinoEmbedder(mid, device)

def extract_all_orbit_mean(lm_list, iv_vol, ex_vol, ev_vox, norm_fn, xy_um, z_um, embedder):
    """Extract orbit-mean features for all landmarks."""
    N = len(lm_list)
    D = embedder.cls_dim
    log(f"Extracting orbit-mean for all {N} landmarks…")
    iv_f = np.zeros((N, D), dtype=np.float32)
    ex_f = np.zeros((N, D), dtype=np.float32)
    for i, lm in enumerate(lm_list):
        if i % 10 == 0:
            log(f"  {i}/{N}…")
        iv_base = slab_to_depth_lut(
            extract_isotropic_slab(
                iv_vol, (lm["invivo_x"], lm["invivo_y"], lm["invivo_z"]),
                INVIVO_VOXEL_DIMS_UM, xy_um, z_um
            ), norm_fn
        )
        ex_base = slab_to_depth_lut(
            extract_isotropic_slab(
                ex_vol, (lm["exvivo_x"], lm["exvivo_y"], lm["exvivo_z"]),
                ev_vox, xy_um, z_um
            ), norm_fn
        )
        iv_cls = embedder.embed_cls(np.stack([rotate_rgb_patch(iv_base, a) for a in ORBIT_ANGLES_DEG]))
        ex_cls = embedder.embed_cls(np.stack([rotate_rgb_patch(ex_base, a) for a in ORBIT_ANGLES_DEG]))
        iv_m = iv_cls.mean(0)
        ex_m = ex_cls.mean(0)
        iv_f[i] = iv_m / max(np.linalg.norm(iv_m), 1e-8)
        ex_f[i] = ex_m / max(np.linalg.norm(ex_m), 1e-8)
    return iv_f, ex_f

# ---------------------------------------------------------------------------
# Alignment methods
# ---------------------------------------------------------------------------

def align_frozen(ex_q, ex_tr, iv_tr):
    """No alignment, just normalize."""
    return ex_q / np.maximum(np.linalg.norm(ex_q, axis=1, keepdims=True), 1e-8)

def align_mean_std(ex_q, ex_tr, iv_tr):
    """Feature mean/std alignment."""
    a = (ex_q - ex_tr.mean(0)) / (ex_tr.std(0) + 1e-8) * (iv_tr.std(0) + 1e-8) + iv_tr.mean(0)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)

def align_ridge(ex_q, ex_tr, iv_tr, lam):
    """Ridge regression alignment."""
    D = ex_tr.shape[1]
    W = np.linalg.solve(
        (ex_tr.T @ ex_tr + lam * np.eye(D)).astype(np.float64),
        (ex_tr.T @ iv_tr).astype(np.float64)
    )
    a = (ex_q.astype(np.float64) @ W).astype(np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)

def pca_fit(X, k):
    """Fit PCA on training data."""
    mean = X.mean(0)
    Xc = X - mean
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]
    evr = float((s[:k] ** 2).sum() / max((s ** 2).sum(), 1e-12))
    return mean, components, evr

def pca_transform(X, mean, components):
    """Project data to PCA space."""
    proj = (X - mean) @ components.T
    return proj / np.maximum(np.linalg.norm(proj, axis=1, keepdims=True), 1e-8)

def align_pca_ridge(ex_q, ex_tr, iv_tr, pca_k, ridge_lambda):
    """PCA-K + ridge regression alignment.  Returns (aligned_query, pca_mean, pca_comps)
    so the caller can project the gallery into the same space."""
    pca_mean, pca_comps, _ = pca_fit(iv_tr, pca_k)
    ex_q_pca = pca_transform(ex_q, pca_mean, pca_comps)
    ex_tr_pca = pca_transform(ex_tr, pca_mean, pca_comps)
    iv_tr_pca = pca_transform(iv_tr, pca_mean, pca_comps)
    aligned = align_ridge(ex_q_pca, ex_tr_pca, iv_tr_pca, ridge_lambda)
    return aligned, pca_mean, pca_comps

def align_procrustes(ex_q, ex_tr, iv_tr):
    """Procrustes alignment."""
    U, _, Vt = np.linalg.svd(ex_tr.T @ iv_tr)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    a = ex_q @ R.T
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)

def align_kernel_ridge(ex_q, ex_tr, iv_tr, alpha=1.0, gamma=None):
    """Kernel Ridge Regression alignment with RBF kernel."""
    from sklearn.kernel_ridge import KernelRidge
    kr = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
    kr.fit(ex_tr, iv_tr)
    a = kr.predict(ex_q).astype(np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8), None, None

def align_gp_regression(ex_q, ex_tr, iv_tr, n_components=64):
    """Gaussian Process regression alignment (PCA-reduced for efficiency)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    # Reduce dim first (GP is O(n^3), 69 samples is fine but 1024 dims needs PCA)
    pca_mean, pca_comps, _ = pca_fit(iv_tr, n_components)
    ex_tr_p = pca_transform(ex_tr, pca_mean, pca_comps)
    iv_tr_p = pca_transform(iv_tr, pca_mean, pca_comps)
    ex_q_p  = pca_transform(ex_q, pca_mean, pca_comps)
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
    gpr.fit(ex_tr_p, iv_tr_p)
    pred = gpr.predict(ex_q_p).astype(np.float32)
    a = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
    return a, pca_mean, pca_comps

def apply_alignment(alignment_type, ex_q, ex_tr, iv_tr, params):
    """Apply specified alignment method.
    Returns (aligned_query, pca_mean, pca_comps) for pca_ridge so the caller
    can project the gallery; returns (aligned_query, None, None) for all others."""
    if alignment_type == "frozen":
        return align_frozen(ex_q, ex_tr, iv_tr), None, None
    elif alignment_type == "mean_std":
        return align_mean_std(ex_q, ex_tr, iv_tr), None, None
    elif alignment_type == "ridge":
        lam = params.get("ridge_lambda", 0.1)
        return align_ridge(ex_q, ex_tr, iv_tr, lam), None, None
    elif alignment_type == "pca_ridge":
        pca_k = params.get("pca_k", 128)
        ridge_lambda = params.get("ridge_lambda", 0.1)
        aligned, pca_mean, pca_comps = align_pca_ridge(ex_q, ex_tr, iv_tr, pca_k, ridge_lambda)
        return aligned, pca_mean, pca_comps
    elif alignment_type == "procrustes":
        return align_procrustes(ex_q, ex_tr, iv_tr), None, None
    elif alignment_type == "kernel_ridge":
        alpha = params.get("kernel_alpha", 1.0)
        gamma = params.get("gamma", None)
        return align_kernel_ridge(ex_q, ex_tr, iv_tr, alpha=alpha, gamma=gamma)
    elif alignment_type == "gp_regression":
        n_components = params.get("n_components", 64)
        return align_gp_regression(ex_q, ex_tr, iv_tr, n_components=n_components)
    elif alignment_type == "tps_fusion":
        raise ValueError(
            "alignment 'tps_fusion' is a meta-alias: use run_config() which rewrites it to "
            "pca_ridge + geometry_fusion with position_predictor='tps'."
        )
    else:
        raise ValueError(f"Unknown alignment type: {alignment_type}")

# ---------------------------------------------------------------------------
# Spatial re-ranking
# ---------------------------------------------------------------------------

def build_knn_graph(coords, k):
    """Build k-NN graph on 3D coordinates."""
    D2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(D2, np.inf)
    return np.argsort(D2, axis=1)[:, :k]

def compute_spatial_scores(
    query_idx, top_k_idx, ex_all_aligned, iv_gal_pca, ex_knn, iv_knn, k
):
    """Compute spatial consistency scores for top-k candidates."""
    spatial_scores = np.zeros(len(top_k_idx), dtype=np.float32)
    ex_nbrs_of_i = ex_knn[query_idx]

    for rank_pos, gal_j in enumerate(top_k_idx):
        iv_nbrs_of_j = set(iv_knn[gal_j])
        hits = 0
        for ex_n in ex_nbrs_of_i:
            best_iv_match = int(np.argmax((ex_all_aligned[[ex_n]] @ iv_gal_pca.T)[0]))
            if best_iv_match in iv_nbrs_of_j:
                hits += 1
        spatial_scores[rank_pos] = hits / k

    return spatial_scores

def rerank_with_spatial(
    feat_sim, top_k_idx, spatial_scores, alpha
):
    """Combine feature similarity and spatial scores."""
    combined = alpha * feat_sim[top_k_idx] + (1.0 - alpha) * spatial_scores
    return np.argsort(-combined)


def predict_query_invivo_affine(
    ex_tr_coords: np.ndarray, iv_tr_coords: np.ndarray, query_ex_coord: np.ndarray
) -> np.ndarray:
    """LOOCV affine map ex→iv (4×3) from training pairs; predict held-out in-vivo xyz (µm)."""
    N_tr = ex_tr_coords.shape[0]
    if N_tr < 4:
        return iv_tr_coords.mean(0).astype(np.float64)
    ones = np.ones((N_tr, 1), dtype=np.float64)
    A = np.hstack([ex_tr_coords.astype(np.float64), ones])
    B = iv_tr_coords.astype(np.float64)
    W, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    q_aug = np.array([*query_ex_coord.astype(np.float64), 1.0], dtype=np.float64)
    return q_aug @ W


def predict_query_invivo_tps(
    ex_tr_coords: np.ndarray,
    iv_tr_coords: np.ndarray,
    query_ex_coord: np.ndarray,
    smoothing: float = 1.0,
) -> np.ndarray:
    """Thin-plate spline RBFInterpolator ex→iv (vector output)."""
    from scipy.interpolate import RBFInterpolator

    x = ex_tr_coords.astype(np.float64)
    y = iv_tr_coords.astype(np.float64)
    n = x.shape[0]
    if n < 6:
        return predict_query_invivo_affine(ex_tr_coords, iv_tr_coords, query_ex_coord)
    sm = max(float(smoothing), 1e-6)
    rbf = RBFInterpolator(x, y, kernel="thin_plate_spline", smoothing=sm)
    return rbf(query_ex_coord.reshape(1, -1).astype(np.float64))[0]


def compute_geometry_fusion_scores(
    query_ex_coord,
    gallery_iv_coords,
    ex_tr_coords,
    iv_tr_coords,
    sigma_um=100.0,
    position_predictor: str = "affine",
    tps_smoothing: float = 1.0,
):
    """Predict query in-vivo position (affine or TPS); Gaussian score gallery in iv space.

    Parameters
    ----------
    position_predictor : str
        ``affine`` (4×3 least squares) or ``tps`` (``RBFInterpolator`` thin_plate_spline).
    tps_smoothing : float
        Passed to ``RBFInterpolator(..., smoothing=...)`` (larger = smoother / less exact).
    """
    N_tr = ex_tr_coords.shape[0]
    if N_tr < 4:
        return np.ones(len(gallery_iv_coords), dtype=np.float32) / max(len(gallery_iv_coords), 1)

    if position_predictor == "tps":
        predicted_iv = predict_query_invivo_tps(
            ex_tr_coords, iv_tr_coords, query_ex_coord, smoothing=tps_smoothing
        )
    else:
        predicted_iv = predict_query_invivo_affine(ex_tr_coords, iv_tr_coords, query_ex_coord)

    dists_sq = np.sum(
        (gallery_iv_coords.astype(np.float64) - predicted_iv[None, :]) ** 2, axis=1
    )
    scores = np.exp(-dists_sq / (2.0 * sigma_um ** 2))
    total = scores.sum()
    if total < 1e-12:
        return np.ones(len(gallery_iv_coords), dtype=np.float32) / len(gallery_iv_coords)
    return (scores / total).astype(np.float32)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_discriminability_gap(ranks, N):
    """
    Compute discriminability gap:
    E[R@1] - P(random rank <= 1) = E[R@1] - 1/N
    """
    r1 = (ranks <= 1).mean()
    random_r1 = 1.0 / N
    return float(r1 - random_r1)

def bootstrap_metrics(ranks_arr, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Compute metrics with bootstrap CI."""
    rng = np.random.default_rng(seed)
    N = len(ranks_arr)
    r1_boot = np.zeros(n_boot)
    r5_boot = np.zeros(n_boot)
    r10_boot = np.zeros(n_boot)
    for b in range(n_boot):
        s = rng.choice(ranks_arr, size=N, replace=True)
        r1_boot[b] = (s <= 1).mean()
        r5_boot[b] = (s <= 5).mean()
        r10_boot[b] = (s <= 10).mean()

    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        "R1": float((ranks_arr <= 1).mean()),
        "R1_ci": ci(r1_boot),
        "R5": float((ranks_arr <= 5).mean()),
        "R5_ci": ci(r5_boot),
        "R10": float((ranks_arr <= 10).mean()),
        "R10_ci": ci(r10_boot),
        "mean_rank": float(ranks_arr.mean()),
        "median_rank": float(np.median(ranks_arr)),
    }

def loocv_eval(iv_all, ex_all, alignment_type, alignment_params, reranking_type, reranking_params,
               iv_coords=None, ex_coords=None, ex_coords_train_only=False, train_idx=None):
    """
    Leave-one-out cross-validation evaluation.

    Returns ranks (N,) and optionally spatial data.

    Parameters
    ----------
    ex_coords_train_only : bool
        When True and reranking_type == "spatial_knn", build the ex-vivo k-NN graph
        using only the training cells' coordinates (those in train_idx), NOT including
        the query cell's own ex-vivo coordinate.  This prevents "spatial cheating".
    train_idx : array-like or None
        Indices of cells whose ex-vivo coordinates are considered "known" for the
        spatial graph when ex_coords_train_only=True.  If None, falls back to all
        N-1 cells (i.e., standard LOOCV behaviour).
    """
    N, D = iv_all.shape
    ranks = np.zeros(N, dtype=int)

    # Build in-vivo spatial graph once (doesn't depend on query)
    if reranking_type == "spatial_knn":
        knn_k = reranking_params.get("knn_k", 5)
        iv_knn = build_knn_graph(iv_coords, knn_k)
        # ex_knn is built per-fold when ex_coords_train_only=True
        if not ex_coords_train_only:
            ex_knn = build_knn_graph(ex_coords, knn_k)

    # Pre-convert coordinate arrays for geometry fusion
    _geo_iv_coords = iv_coords if iv_coords is not None else None
    _geo_ex_coords = ex_coords if ex_coords is not None else None

    log(f"Running LOOCV ({N} iterations)…")
    for i in range(N):
        if i % 10 == 0:
            log(f"  {i}/{N}…")

        # Training set: exclude query
        gallery_idx = [j for j in range(N) if j != i]
        ex_tr = ex_all[gallery_idx]
        iv_tr = iv_all[gallery_idx]

        # Query and gallery
        ex_q = ex_all[[i]]
        iv_gal = iv_all

        # Apply alignment — for pca_ridge also returns the PCA basis used this fold
        ex_q_aligned, pca_mean, pca_comps = apply_alignment(
            alignment_type, ex_q, ex_tr, iv_tr, alignment_params)

        # Project iv gallery into the same space as the aligned query.
        # For pca_ridge: project to PCA space (no ridge — ridge maps ex→iv, gallery IS iv).
        # For all others: gallery stays in full 1024-dim space.
        if pca_mean is not None:
            iv_gal_cmp = pca_transform(iv_gal, pca_mean, pca_comps)
        else:
            iv_gal_cmp = iv_gal

        # For spatial re-ranking: align all ex features using the same fold's alignment
        if reranking_type == "spatial_knn":
            ex_all_aligned, _, _ = apply_alignment(
                alignment_type, ex_all, ex_tr, iv_tr, alignment_params)

            # Build ex_knn for this fold using training-only coords if requested
            if ex_coords_train_only:
                # Use train_idx if provided, otherwise use all N-1 training cells
                if train_idx is not None:
                    coord_idx = [j for j in train_idx if j != i]
                else:
                    coord_idx = gallery_idx
                ex_coords_fold = ex_coords[coord_idx]
                # Build a local k-NN among just these cells; map back to global indices
                knn_k_fold = min(knn_k, len(coord_idx) - 1)
                ex_knn_fold_local = build_knn_graph(ex_coords_fold, knn_k_fold)
                # Remap local indices → global indices
                coord_idx_arr = np.array(coord_idx)
                ex_knn_fold = coord_idx_arr[ex_knn_fold_local]
                # For the current fold, build a per-cell lookup including all N cells
                # (cells not in coord_idx get empty neighbour lists, handled gracefully)
                ex_knn_i = ex_knn_fold[coord_idx.index(i)] if i in coord_idx else np.array([], dtype=int)
                # Use a full-N placeholder and fill in what we have
                ex_knn_full = np.zeros((N, knn_k_fold), dtype=int)
                for local_pos, global_pos in enumerate(coord_idx):
                    ex_knn_full[global_pos] = ex_knn_fold[local_pos]
                fold_ex_knn = ex_knn_full
            else:
                fold_ex_knn = ex_knn

        # Feature similarity
        feat_sim = (ex_q_aligned @ iv_gal_cmp.T)[0]

        # Get top candidates
        order = np.argsort(-feat_sim)

        if reranking_type == "geometry_fusion":
            sigma_um = reranking_params.get("sigma_um", 100.0)
            w = reranking_params.get("w", 0.5)
            pos_pred = reranking_params.get("position_predictor", "affine")
            tps_eps = float(reranking_params.get("tps_smoothing", 1.0))
            ex_tr_coords_fold = _geo_ex_coords[gallery_idx]
            iv_tr_coords_fold = _geo_iv_coords[gallery_idx]
            geo_scores = compute_geometry_fusion_scores(
                _geo_ex_coords[i],
                _geo_iv_coords,
                ex_tr_coords_fold,
                iv_tr_coords_fold,
                sigma_um,
                position_predictor=pos_pred,
                tps_smoothing=tps_eps,
            )
            # Normalize feat_sim to [0,1] before blending
            fsim_min, fsim_max = feat_sim.min(), feat_sim.max()
            fsim_norm = (feat_sim - fsim_min) / max(fsim_max - fsim_min, 1e-8)
            combined = w * fsim_norm + (1.0 - w) * geo_scores
            order = np.argsort(-combined)

        elif reranking_type == "spatial_knn":
            top_k = reranking_params.get("top_k", 10)
            top_k_idx = order[:top_k]

            # Spatial scoring — pass iv_gal_cmp so dimensions match ex_all_aligned
            spatial_scores = compute_spatial_scores(
                i, top_k_idx, ex_all_aligned, iv_gal_cmp, fold_ex_knn, iv_knn,
                knn_k
            )

            # Re-rank
            alpha = reranking_params.get("alpha", 0.5)
            reranked_local = rerank_with_spatial(feat_sim, top_k_idx, spatial_scores, alpha)

            # Rebuild full order
            full_order = list(order)
            for pos, new_local in enumerate(reranked_local):
                full_order[pos] = top_k_idx[new_local]
            order = np.array(full_order)

        # Rank of correct match (i in gallery)
        ranks[i] = int(np.where(order == i)[0][0]) + 1

    return ranks


def loocv_vision_similarity_margins(iv_all, ex_all, alignment_type, alignment_params):
    """
    Feature-only LOOCV margins: for each query i, similarity to full in-vivo gallery after
    the same per-fold alignment as ``loocv_eval`` (but **no** spatial / geometry reranking).

    Returns
    -------
    margins : (N,) float32
        sim(rank-1) - sim(rank-2) in the sorted gallery for each query.
    top1_correct : (N,) bool
        Whether the top-1 gallery index equals the query index i.
    top1_idx : (N,) int32
        Predicted gallery index for each query.
    rank1_sim : (N,) float32
        Similarity score at the predicted top-1 gallery cell.
    """
    N = iv_all.shape[0]
    margins = np.zeros(N, dtype=np.float32)
    top1_correct = np.zeros(N, dtype=bool)
    top1_idx = np.zeros(N, dtype=np.int32)
    rank1_sim = np.zeros(N, dtype=np.float32)

    for i in range(N):
        gallery_idx = [j for j in range(N) if j != i]
        ex_tr = ex_all[gallery_idx]
        iv_tr = iv_all[gallery_idx]
        ex_q = ex_all[[i]]
        iv_gal = iv_all

        ex_q_aligned, pca_mean, pca_comps = apply_alignment(
            alignment_type, ex_q, ex_tr, iv_tr, alignment_params
        )
        iv_gal_cmp = (
            pca_transform(iv_gal, pca_mean, pca_comps) if pca_mean is not None else iv_gal
        )
        feat_sim = (ex_q_aligned @ iv_gal_cmp.T)[0]
        order = np.argsort(-feat_sim)
        j1 = int(order[0])
        j2 = int(order[1]) if N > 1 else j1
        rank1_sim[i] = float(feat_sim[j1])
        margins[i] = float(feat_sim[j1] - feat_sim[j2])
        top1_idx[i] = j1
        top1_correct[i] = j1 == i

    return margins, top1_correct, top1_idx, rank1_sim


def confidence_precision_coverage(margins, top1_correct, thresholds):
    """
    For each margin threshold, precision@1 on accepted queries and coverage.

    Parameters
    ----------
    margins : (N,)
    top1_correct : (N,) bool — whether rank-1 identity matches query index
    thresholds : iterable of float

    Returns
    -------
    dict with lists: thresholds, precisions, coverages, num_accepted
    """
    margins = np.asarray(margins, dtype=np.float64)
    top1_correct = np.asarray(top1_correct, dtype=bool)
    N = len(margins)
    precisions = []
    coverages = []
    num_accepted = []
    for t in thresholds:
        mask = margins >= t
        c = int(mask.sum())
        num_accepted.append(c)
        coverages.append(float(c) / N if N else 0.0)
        precisions.append(float(top1_correct[mask].mean()) if c else 0.0)
    return {
        "thresholds": [float(x) for x in thresholds],
        "precisions": precisions,
        "coverages": coverages,
        "num_accepted": num_accepted,
    }


def loocv_query_gallery_similarity_matrix(iv_all, ex_all, alignment_type, alignment_params):
    """
    Full (N, N) similarity matrix under LOOCV: row ``i`` uses alignment fit without landmark ``i``.

    ``S[i, j]`` is the cosine similarity (after alignment) between ex-vivo query ``i``
    and in-vivo gallery vector ``j``. Used for fusion with geometric scores.
    """
    N = iv_all.shape[0]
    S = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        gallery_idx = [j for j in range(N) if j != i]
        ex_tr = ex_all[gallery_idx]
        iv_tr = iv_all[gallery_idx]
        ex_q = ex_all[[i]]
        iv_gal = iv_all
        ex_q_aligned, pca_mean, pca_comps = apply_alignment(
            alignment_type, ex_q, ex_tr, iv_tr, alignment_params
        )
        iv_gal_cmp = (
            pca_transform(iv_gal, pca_mean, pca_comps) if pca_mean is not None else iv_gal
        )
        S[i] = (ex_q_aligned @ iv_gal_cmp.T)[0]
    return S


def val_test_eval(iv_all, ex_all, lm_list, alignment_type, alignment_params, reranking_type, reranking_params, iv_coords=None, ex_coords=None):
    """
    60/20/20 train/val/test split evaluation (seed 23).
    """
    set_seed(SPLIT_SEED)
    N = len(lm_list)
    indices = np.arange(N)
    np.random.shuffle(indices)

    n_train = int(0.6 * N)
    n_val = int(0.2 * N)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val:]

    log(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Fit alignment on training set
    ex_tr = ex_all[train_idx]
    iv_tr = iv_all[train_idx]

    # Evaluate on val and test
    val_ranks = np.zeros(len(val_idx), dtype=int)
    test_ranks = np.zeros(len(test_idx), dtype=int)

    for split_name, split_idx, ranks_out in [
        ("val", val_idx, val_ranks),
        ("test", test_idx, test_ranks),
    ]:
        log(f"Evaluating on {split_name}…")
        for pos, i in enumerate(split_idx):
            ex_q = ex_all[[i]]
            iv_gal = iv_all

            # Align
            ex_q_aligned = apply_alignment(alignment_type, ex_q, ex_tr, iv_tr, alignment_params)

            # Feature sim
            feat_sim = (ex_q_aligned @ iv_gal.T)[0]
            order = np.argsort(-feat_sim)

            if reranking_type == "spatial_knn":
                # Similar logic as LOOCV spatial reranking
                knn_k = reranking_params.get("knn_k", 5)
                top_k = reranking_params.get("top_k", 10)
                alpha = reranking_params.get("alpha", 0.5)

                iv_knn = build_knn_graph(iv_coords, knn_k)
                ex_knn = build_knn_graph(ex_coords, knn_k)
                ex_all_aligned = apply_alignment(alignment_type, ex_all, ex_tr, iv_tr, alignment_params)

                top_k_idx = order[:top_k]
                spatial_scores = compute_spatial_scores(
                    i, top_k_idx, ex_all_aligned, iv_all, ex_knn, iv_knn, knn_k
                )
                reranked_local = rerank_with_spatial(feat_sim, top_k_idx, spatial_scores, alpha)
                full_order = list(order)
                for p, new_local in enumerate(reranked_local):
                    full_order[p] = top_k_idx[new_local]
                order = np.array(full_order)

            ranks_out[pos] = int(np.where(order == i)[0][0]) + 1

    return val_ranks, test_ranks

def feature_cache_key(norm, patch_xy_um, patch_z_um):
    """Generate cache file key."""
    return CACHE_DIR / f"orbit_mean_{norm}_{patch_xy_um}x{patch_z_um}_um.npy"

def _load_or_extract_single_scale(lm_list, iv_vol, ex_vol, ev_vox, norm_name, xy_um, z_um, device):
    """Load or extract single-scale orbit-mean features, with caching."""
    norm_fn = NORM_FN_MAP.get(norm_name, norm_perc_linear)

    cache_iv = feature_cache_key(norm_name, xy_um, z_um)
    cache_ex = feature_cache_key(norm_name, xy_um, z_um).with_name(
        cache_iv.name.replace("orbit_mean", "orbit_mean_ex")
    )

    if cache_iv.exists() and cache_ex.exists():
        log(f"Loading cached features from {cache_iv}")
        iv_f = np.load(cache_iv)
        ex_f = np.load(cache_ex)
        return iv_f, ex_f

    embedder = DinoEmbedder(DINO_MODEL_ID, device)
    iv_f, ex_f = extract_all_orbit_mean(lm_list, iv_vol, ex_vol, ev_vox, norm_fn, xy_um, z_um, embedder)

    log(f"Caching features to {cache_iv}")
    np.save(cache_iv, iv_f)
    np.save(cache_ex, ex_f)

    return iv_f, ex_f

def _extract_orbit_mean_with_slab_fn(lm_list, iv_vol, ex_vol, ev_vox, norm_fn, xy_um, z_um, embedder, slab_fn):
    """Like extract_all_orbit_mean but uses an arbitrary slab→RGB function."""
    N = len(lm_list)
    D = embedder.cls_dim
    log(f"Extracting orbit-mean ({slab_fn.__name__}) for all {N} landmarks…")
    iv_f = np.zeros((N, D), dtype=np.float32)
    ex_f = np.zeros((N, D), dtype=np.float32)
    for i, lm in enumerate(lm_list):
        if i % 10 == 0:
            log(f"  {i}/{N}…")
        iv_base = slab_fn(
            extract_isotropic_slab(
                iv_vol, (lm["invivo_x"], lm["invivo_y"], lm["invivo_z"]),
                INVIVO_VOXEL_DIMS_UM, xy_um, z_um
            ), norm_fn
        )
        ex_base = slab_fn(
            extract_isotropic_slab(
                ex_vol, (lm["exvivo_x"], lm["exvivo_y"], lm["exvivo_z"]),
                ev_vox, xy_um, z_um
            ), norm_fn
        )
        iv_cls = embedder.embed_cls(np.stack([rotate_rgb_patch(iv_base, a) for a in ORBIT_ANGLES_DEG]))
        ex_cls = embedder.embed_cls(np.stack([rotate_rgb_patch(ex_base, a) for a in ORBIT_ANGLES_DEG]))
        iv_m = iv_cls.mean(0); iv_f[i] = iv_m / max(np.linalg.norm(iv_m), 1e-8)
        ex_m = ex_cls.mean(0); ex_f[i] = ex_m / max(np.linalg.norm(ex_m), 1e-8)
    return iv_f, ex_f


def _extract_orbit_mean_patch_tokens(lm_list, iv_vol, ex_vol, ev_vox, norm_fn, xy_um, z_um, embedder, pooling="mean"):
    """Orbit-mean using spatial patch tokens (pooling: mean | max | attention)."""
    N = len(lm_list)
    D = embedder.cls_dim
    log(f"Extracting patch-token orbit-mean (pooling={pooling}) for all {N} landmarks…")
    iv_f = np.zeros((N, D), dtype=np.float32)
    ex_f = np.zeros((N, D), dtype=np.float32)
    for i, lm in enumerate(lm_list):
        if i % 10 == 0:
            log(f"  {i}/{N}…")
        iv_base = slab_to_depth_lut(
            extract_isotropic_slab(
                iv_vol, (lm["invivo_x"], lm["invivo_y"], lm["invivo_z"]),
                INVIVO_VOXEL_DIMS_UM, xy_um, z_um
            ), norm_fn
        )
        ex_base = slab_to_depth_lut(
            extract_isotropic_slab(
                ex_vol, (lm["exvivo_x"], lm["exvivo_y"], lm["exvivo_z"]),
                ev_vox, xy_um, z_um
            ), norm_fn
        )
        iv_tok = embedder.embed_patch_tokens(
            np.stack([rotate_rgb_patch(iv_base, a) for a in ORBIT_ANGLES_DEG]),
            pooling=pooling,
        )
        ex_tok = embedder.embed_patch_tokens(
            np.stack([rotate_rgb_patch(ex_base, a) for a in ORBIT_ANGLES_DEG]),
            pooling=pooling,
        )
        iv_m = iv_tok.mean(0); iv_f[i] = iv_m / max(np.linalg.norm(iv_m), 1e-8)
        ex_m = ex_tok.mean(0); ex_f[i] = ex_m / max(np.linalg.norm(ex_m), 1e-8)
    return iv_f, ex_f


def _extract_multi_channel(
    lm_list, iv_vol_a, iv_vol_b, ex_vol, ev_vox, norm_fn, xy_um, z_um, embedder
):
    """Orbit-mean on two in-vivo channels; concatenate to 2×D; duplicate ex descriptor to match.

    Ex-vivo has a single physical channel; we stack ``[ex_m, ex_m]`` so query and gallery
    both live in ℝ^{2D} and PCA/ridge alignment is well-posed (same trick as multi_scale_dino).
    """
    N = len(lm_list)
    D = embedder.cls_dim
    log(f"Extracting dual in-vivo channel orbit-mean for all {N} landmarks (iv 2×D, ex 2×D)…")
    iv_f = np.zeros((N, 2 * D), dtype=np.float32)
    ex_f = np.zeros((N, 2 * D), dtype=np.float32)
    for i, lm in enumerate(lm_list):
        if i % 10 == 0:
            log(f"  {i}/{N}…")
        iv_base_a = slab_to_depth_lut(
            extract_isotropic_slab(
                iv_vol_a, (lm["invivo_x"], lm["invivo_y"], lm["invivo_z"]),
                INVIVO_VOXEL_DIMS_UM, xy_um, z_um
            ),
            norm_fn,
        )
        iv_base_b = slab_to_depth_lut(
            extract_isotropic_slab(
                iv_vol_b, (lm["invivo_x"], lm["invivo_y"], lm["invivo_z"]),
                INVIVO_VOXEL_DIMS_UM, xy_um, z_um
            ),
            norm_fn,
        )
        ex_base = slab_to_depth_lut(
            extract_isotropic_slab(
                ex_vol, (lm["exvivo_x"], lm["exvivo_y"], lm["exvivo_z"]),
                ev_vox, xy_um, z_um
            ),
            norm_fn,
        )
        iv_cls_a = embedder.embed_cls(np.stack([rotate_rgb_patch(iv_base_a, a) for a in ORBIT_ANGLES_DEG]))
        iv_cls_b = embedder.embed_cls(np.stack([rotate_rgb_patch(iv_base_b, a) for a in ORBIT_ANGLES_DEG]))
        ex_cls = embedder.embed_cls(np.stack([rotate_rgb_patch(ex_base, a) for a in ORBIT_ANGLES_DEG]))
        iv_ma = iv_cls_a.mean(0)
        iv_ma /= max(np.linalg.norm(iv_ma), 1e-8)
        iv_mb = iv_cls_b.mean(0)
        iv_mb /= max(np.linalg.norm(iv_mb), 1e-8)
        ex_m = ex_cls.mean(0)
        ex_m /= max(np.linalg.norm(ex_m), 1e-8)
        iv_f[i] = np.concatenate([iv_ma, iv_mb])
        ex_f[i] = np.concatenate([ex_m, ex_m])
    return iv_f, ex_f


def load_or_extract_features(lm_list, iv_vol, ex_vol, ev_vox, norm_name, xy_um, z_um, device,
                              feature_type="dino_cls", extra_vols=None):
    """Load cached features or extract and cache.

    feature_type options:
      "dino_cls"          — standard CLS orbit-mean (default)
      "multi_scale_dino"  — CLS at xy_um and 2×xy_um, concat
      "brightest_slice"   — CLS from the max-intensity z-slice only (no depth blur)
      "depth_normalized_lut" — CLS from Gaussian-weighted depth LUT
      "orbit_mip" / "orbit_variance" / "orbit_multichannel_depth" — CLS with alternate slab→RGB
      "mean_patch_tokens" — mean-pooled spatial patch tokens
      "patch_tokens_max" / "patch_tokens_attention" — other patch-token poolings
      "multi_channel"     — two in-vivo channels → concat iv (2D); ex duplicated to 2D.

    extra_vols : dict | None
      ``{"iv_vol_b": ndarray, "cache_tag": "iv2_iv4"}`` for ``multi_channel``.
    """
    norm_fn = NORM_FN_MAP.get(norm_name, norm_perc_linear)

    if feature_type == "multi_scale_dino":
        xy_um_large = xy_um * 2
        log(f"Multi-scale DINOv3: {xy_um}µm + {xy_um_large}µm")
        iv_s, ex_s = _load_or_extract_single_scale(lm_list, iv_vol, ex_vol, ev_vox, norm_name, xy_um, z_um, device)
        iv_l, ex_l = _load_or_extract_single_scale(lm_list, iv_vol, ex_vol, ev_vox, norm_name, xy_um_large, z_um, device)
        for arr in (iv_s, ex_s, iv_l, ex_l):
            arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8)
        iv_f = np.concatenate([iv_s, iv_l], axis=1)
        ex_f = np.concatenate([ex_s, ex_l], axis=1)
        log(f"Multi-scale features: iv={iv_f.shape}, ex={ex_f.shape}")
        return iv_f, ex_f

    if feature_type == "brightest_slice":
        cache_iv = CACHE_DIR / f"orbit_mean_brightest_{norm_name}_{xy_um}x{z_um}_um.npy"
        cache_ex = CACHE_DIR / f"orbit_mean_ex_brightest_{norm_name}_{xy_um}x{z_um}_um.npy"
        if cache_iv.exists() and cache_ex.exists():
            log(f"Loading cached brightest-slice features from {cache_iv}")
            return np.load(cache_iv), np.load(cache_ex)
        embedder = DinoEmbedder(DINO_MODEL_ID, device)
        iv_f, ex_f = _extract_orbit_mean_with_slab_fn(
            lm_list, iv_vol, ex_vol, ev_vox, norm_fn, xy_um, z_um, embedder,
            slab_fn=slab_to_brightest_slice_rgb
        )
        np.save(cache_iv, iv_f); np.save(cache_ex, ex_f)
        return iv_f, ex_f

    if feature_type == "depth_normalized_lut":
        cache_iv = CACHE_DIR / f"orbit_mean_depthnorm_{norm_name}_{xy_um}x{z_um}_um.npy"
        cache_ex = CACHE_DIR / f"orbit_mean_ex_depthnorm_{norm_name}_{xy_um}x{z_um}_um.npy"
        if cache_iv.exists() and cache_ex.exists():
            log(f"Loading cached depth-norm features from {cache_iv}")
            return np.load(cache_iv), np.load(cache_ex)
        embedder = DinoEmbedder(DINO_MODEL_ID, device)
        iv_f, ex_f = _extract_orbit_mean_with_slab_fn(
            lm_list, iv_vol, ex_vol, ev_vox, norm_fn, xy_um, z_um, embedder,
            slab_fn=slab_to_depth_lut_gaussian
        )
        np.save(cache_iv, iv_f); np.save(cache_ex, ex_f)
        return iv_f, ex_f

    if feature_type == "orbit_mip":
        tag = f"mip_{norm_name}_{xy_um}x{z_um}_um"
        cache_iv = CACHE_DIR / f"orbit_mean_{tag}.npy"
        cache_ex = CACHE_DIR / f"orbit_mean_ex_{tag}.npy"
        if cache_iv.exists() and cache_ex.exists():
            log(f"Loading cached MIP features from {cache_iv}")
            return np.load(cache_iv), np.load(cache_ex)
        embedder = DinoEmbedder(DINO_MODEL_ID, device)
        iv_f, ex_f = _extract_orbit_mean_with_slab_fn(
            lm_list, iv_vol, ex_vol, ev_vox, norm_fn, xy_um, z_um, embedder,
            slab_fn=slab_to_mip_rgb,
        )
        np.save(cache_iv, iv_f); np.save(cache_ex, ex_f)
        return iv_f, ex_f

    if feature_type == "orbit_variance":
        tag = f"var_{norm_name}_{xy_um}x{z_um}_um"
        cache_iv = CACHE_DIR / f"orbit_mean_{tag}.npy"
        cache_ex = CACHE_DIR / f"orbit_mean_ex_{tag}.npy"
        if cache_iv.exists() and cache_ex.exists():
            log(f"Loading cached variance-projection features from {cache_iv}")
            return np.load(cache_iv), np.load(cache_ex)
        embedder = DinoEmbedder(DINO_MODEL_ID, device)
        iv_f, ex_f = _extract_orbit_mean_with_slab_fn(
            lm_list, iv_vol, ex_vol, ev_vox, norm_fn, xy_um, z_um, embedder,
            slab_fn=slab_to_variance_rgb,
        )
        np.save(cache_iv, iv_f); np.save(cache_ex, ex_f)
        return iv_f, ex_f

    if feature_type == "orbit_multichannel_depth":
        tag = f"mcdepth_{norm_name}_{xy_um}x{z_um}_um"
        cache_iv = CACHE_DIR / f"orbit_mean_{tag}.npy"
        cache_ex = CACHE_DIR / f"orbit_mean_ex_{tag}.npy"
        if cache_iv.exists() and cache_ex.exists():
            log(f"Loading cached multichannel-depth features from {cache_iv}")
            return np.load(cache_iv), np.load(cache_ex)
        embedder = DinoEmbedder(DINO_MODEL_ID, device)
        iv_f, ex_f = _extract_orbit_mean_with_slab_fn(
            lm_list, iv_vol, ex_vol, ev_vox, norm_fn, xy_um, z_um, embedder,
            slab_fn=slab_to_multichannel_rgb_depth,
        )
        np.save(cache_iv, iv_f); np.save(cache_ex, ex_f)
        return iv_f, ex_f

    if feature_type == "mean_patch_tokens":
        cache_iv = CACHE_DIR / f"patch_tokens_{norm_name}_{xy_um}x{z_um}_um.npy"
        cache_ex = CACHE_DIR / f"patch_tokens_ex_{norm_name}_{xy_um}x{z_um}_um.npy"
        if cache_iv.exists() and cache_ex.exists():
            log(f"Loading cached patch-token features from {cache_iv}")
            return np.load(cache_iv), np.load(cache_ex)
        embedder = DinoEmbedder(DINO_MODEL_ID, device)
        iv_f, ex_f = _extract_orbit_mean_patch_tokens(
            lm_list, iv_vol, ex_vol, ev_vox, norm_fn, xy_um, z_um, embedder, pooling="mean"
        )
        np.save(cache_iv, iv_f); np.save(cache_ex, ex_f)
        return iv_f, ex_f

    if feature_type == "patch_tokens_max":
        cache_iv = CACHE_DIR / f"patch_tokens_max_{norm_name}_{xy_um}x{z_um}_um.npy"
        cache_ex = CACHE_DIR / f"patch_tokens_max_ex_{norm_name}_{xy_um}x{z_um}_um.npy"
        if cache_iv.exists() and cache_ex.exists():
            log(f"Loading cached patch-token (max) features from {cache_iv}")
            return np.load(cache_iv), np.load(cache_ex)
        embedder = DinoEmbedder(DINO_MODEL_ID, device)
        iv_f, ex_f = _extract_orbit_mean_patch_tokens(
            lm_list, iv_vol, ex_vol, ev_vox, norm_fn, xy_um, z_um, embedder, pooling="max"
        )
        np.save(cache_iv, iv_f); np.save(cache_ex, ex_f)
        return iv_f, ex_f

    if feature_type == "patch_tokens_attention":
        cache_iv = CACHE_DIR / f"patch_tokens_attn_{norm_name}_{xy_um}x{z_um}_um.npy"
        cache_ex = CACHE_DIR / f"patch_tokens_attn_ex_{norm_name}_{xy_um}x{z_um}_um.npy"
        if cache_iv.exists() and cache_ex.exists():
            log(f"Loading cached patch-token (attention) features from {cache_iv}")
            return np.load(cache_iv), np.load(cache_ex)
        embedder = DinoEmbedder(DINO_MODEL_ID, device)
        iv_f, ex_f = _extract_orbit_mean_patch_tokens(
            lm_list, iv_vol, ex_vol, ev_vox, norm_fn, xy_um, z_um, embedder, pooling="attention"
        )
        np.save(cache_iv, iv_f); np.save(cache_ex, ex_f)
        return iv_f, ex_f

    if feature_type == "multi_channel":
        iv_b = (extra_vols or {}).get("iv_vol_b")
        if iv_b is None:
            raise ValueError(
                "feature_type='multi_channel' requires extra_vols['iv_vol_b'] "
                "(second in-vivo channel volume (Z,Y,X))"
            )
        tag = (extra_vols or {}).get("cache_tag", "ivAB")
        cache_iv = CACHE_DIR / f"multichan_{tag}_{norm_name}_{xy_um}x{z_um}_um.npy"
        cache_ex = CACHE_DIR / f"multichan_ex_{tag}_{norm_name}_{xy_um}x{z_um}_um.npy"
        if cache_iv.exists() and cache_ex.exists():
            log(f"Loading cached multi-channel features from {cache_iv}")
            return np.load(cache_iv), np.load(cache_ex)
        embedder = DinoEmbedder(DINO_MODEL_ID, device)
        iv_f, ex_f = _extract_multi_channel(
            lm_list, iv_vol, iv_b, ex_vol, ev_vox, norm_fn, xy_um, z_um, embedder
        )
        np.save(cache_iv, iv_f)
        np.save(cache_ex, ex_f)
        return iv_f, ex_f

    if feature_type == "multi_scale_4x":
        scales = [50, 100, 150, 200]
        log(f"Multi-scale 4× DINOv3: {scales}µm (z={z_um}µm)")
        iv_parts, ex_parts = [], []
        for s in scales:
            iv_s, ex_s = _load_or_extract_single_scale(
                lm_list, iv_vol, ex_vol, ev_vox, norm_name, s, z_um, device)
            iv_s = iv_s / np.maximum(np.linalg.norm(iv_s, axis=1, keepdims=True), 1e-8)
            ex_s = ex_s / np.maximum(np.linalg.norm(ex_s, axis=1, keepdims=True), 1e-8)
            iv_parts.append(iv_s)
            ex_parts.append(ex_s)
        iv_f = np.concatenate(iv_parts, axis=1)
        ex_f = np.concatenate(ex_parts, axis=1)
        iv_f = iv_f / np.maximum(np.linalg.norm(iv_f, axis=1, keepdims=True), 1e-8)
        ex_f = ex_f / np.maximum(np.linalg.norm(ex_f, axis=1, keepdims=True), 1e-8)
        log(f"Multi-scale 4× features: iv={iv_f.shape}, ex={ex_f.shape}")
        return iv_f, ex_f

    # Default: single-scale dino_cls
    return _load_or_extract_single_scale(lm_list, iv_vol, ex_vol, ev_vox, norm_name, xy_um, z_um, device)

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_config(config: dict) -> dict:
    """
    Run experiment with given config.

    Config schema:
    {
        "description": str,
        "alignment": "frozen" | "mean_std" | "ridge" | "pca_ridge" | "procrustes",
        "alignment_params": dict,
        "reranking": "none" | "spatial_knn",
        "reranking_params": dict,
        "eval_mode": "loocv" | "val_test",
        "norm": "perc_linear" | "perc_gamma" | "clahe" | "histeq",
        "patch_xy_um": float,
        "patch_z_um": float,
    }
    """
    t0 = time.time()
    set_seed(42)
    device = choose_device()
    log(f"Device: {device}")

    desc = config.get("description", "unnamed")
    log(f"Config: {desc}")

    alignment = config.get("alignment", "frozen")
    alignment_params = dict(config.get("alignment_params", {}))
    reranking = config.get("reranking", "none")
    reranking_params = dict(config.get("reranking_params", {}))

    # TPS position predictor inside geometry fusion (non-rigid ex→iv prior).
    if alignment == "tps_fusion":
        inner = config.get("tps_inner_alignment", "pca_ridge")
        inner_params = config.get("tps_inner_alignment_params", alignment_params)
        alignment = inner
        alignment_params = dict(inner_params)
        reranking = "geometry_fusion"
        reranking_params.setdefault("w", 0.3)
        reranking_params.setdefault("sigma_um", 50.0)
        reranking_params["position_predictor"] = "tps"
        reranking_params["tps_smoothing"] = float(config.get("tps_smoothing", 1.0))
        log(f"tps_fusion → alignment={alignment}, geometry_fusion w={reranking_params.get('w')} "
            f"σ={reranking_params.get('sigma_um')}µm tps_smoothing={reranking_params.get('tps_smoothing', 1.0)}")
    eval_mode = config.get("eval_mode", "loocv")
    norm = config.get("norm", "perc_linear")
    xy_um = config.get("patch_xy_um", 50)
    z_um = config.get("patch_z_um", 28)
    feature_type = config.get("feature", "dino_cls")

    log(f"Alignment: {alignment} {alignment_params}")
    log(f"Reranking: {reranking} {reranking_params}")
    log(f"Eval: {eval_mode}, norm={norm}, patch={xy_um}x{z_um}µm, feature={feature_type}")

    # Load data
    try:
        lm_list = load_landmarks()
        log(f"Landmarks: {len(lm_list)}")

        iv_vol, ex_vol = load_volumes()
        ev_vox = fit_exvivo_voxel_dims(lm_list)
        log(f"Ex-vivo voxel (µm): {tuple(round(v, 3) for v in ev_vox)}")

        # Load extra volumes for multi_channel feature (two in-vivo channels, 1-based indices)
        extra_vols = None
        if feature_type == "multi_channel":
            chs = config.get("invivo_channels_1based", [2, 4])
            if len(chs) != 2:
                raise ValueError("multi_channel requires invivo_channels_1based: [a, b] (two channels)")
            c1, c2 = int(chs[0]), int(chs[1])
            zs_raw = tifffile.imread(str(ZSTACK_PATH)).astype(np.float32)
            if zs_raw.ndim != 4:
                raise ValueError(f"Expected zstack (Z,C,Y,X); got {zs_raw.shape}")
            nch = zs_raw.shape[1]
            if max(c1, c2) > nch or min(c1, c2) < 1:
                raise ValueError(f"Channels {c1},{c2} invalid for zstack with C={nch}")
            log(f"Multi-channel in-vivo: hyperstack channels {c1} and {c2} (1-based)")
            iv_vol = zs_raw[:, c1 - 1, :, :]
            iv_b = zs_raw[:, c2 - 1, :, :]
            extra_vols = {
                "iv_vol_b": iv_b,
                "cache_tag": f"iv{c1}_iv{c2}",
            }
            log(f"  primary iv (ch{c1}): {iv_vol.shape}  second (ch{c2}): {iv_b.shape}")

        # Extract or load features
        iv_all, ex_all = load_or_extract_features(
            lm_list, iv_vol, ex_vol, ev_vox, norm, xy_um, z_um, device,
            feature_type=feature_type, extra_vols=extra_vols,
        )
        log(f"Features: iv={iv_all.shape}, ex={ex_all.shape}")

        # Build spatial coords if needed
        iv_coords = None
        ex_coords = None
        if reranking in ("spatial_knn", "geometry_fusion"):
            dx, dy, dz = INVIVO_VOXEL_DIMS_UM
            iv_coords = np.array(
                [[lm["invivo_x"] * dx, lm["invivo_y"] * dy, lm["invivo_z"] * dz]
                 for lm in lm_list],
                dtype=np.float32
            )
            ex_coords = np.array(
                [[lm["exvivo_x"] * ev_vox[0], lm["exvivo_y"] * ev_vox[1], lm["exvivo_z"] * ev_vox[2]]
                 for lm in lm_list],
                dtype=np.float32
            )

        # Compute 60/20/20 train split indices (seed=23) for coord-blind mode
        set_seed(SPLIT_SEED)
        _split_indices = np.arange(len(lm_list))
        np.random.shuffle(_split_indices)
        _n_train = int(0.6 * len(lm_list))
        train_idx_split = list(_split_indices[:_n_train])

        # Run evaluation
        log(f"=== Starting {eval_mode.upper()} evaluation ===")

        if eval_mode in ("loocv", "loocv_coord_blind"):
            coord_blind = (eval_mode == "loocv_coord_blind")
            ranks = loocv_eval(
                iv_all, ex_all,
                alignment, alignment_params,
                reranking, reranking_params,
                iv_coords, ex_coords,
                ex_coords_train_only=coord_blind,
                train_idx=train_idx_split if coord_blind else None,
            )
            metrics = bootstrap_metrics(ranks)
            result = {
                "loocv_R1": metrics["R1"],
                "loocv_R5": metrics["R5"],
                "loocv_R10": metrics["R10"],
                "loocv_mean_rank": metrics["mean_rank"],
                "loocv_discriminability_gap": compute_discriminability_gap(ranks, len(ranks)),
            }

            # Feature-contribution metrics when spatial re-ranking is active
            if reranking == "spatial_knn":
                log("Computing feature-only R@1 (alpha=1.0)…")
                rp_feat = dict(reranking_params)
                rp_feat["alpha"] = 1.0
                ranks_feat = loocv_eval(
                    iv_all, ex_all,
                    alignment, alignment_params,
                    reranking, rp_feat,
                    iv_coords, ex_coords,
                    ex_coords_train_only=coord_blind,
                    train_idx=train_idx_split if coord_blind else None,
                )
                log("Computing spatial-only R@1 (alpha=0.0)…")
                rp_spatial = dict(reranking_params)
                rp_spatial["alpha"] = 0.0
                ranks_spatial = loocv_eval(
                    iv_all, ex_all,
                    alignment, alignment_params,
                    reranking, rp_spatial,
                    iv_coords, ex_coords,
                    ex_coords_train_only=coord_blind,
                    train_idx=train_idx_split if coord_blind else None,
                )
                result["loocv_feat_only_R1"] = float((ranks_feat <= 1).mean())
                result["loocv_spatial_only_R1"] = float((ranks_spatial <= 1).mean())

            # Vision-only confidence: margin between top-1 and top-2 similarity (per LOOCV fold)
            if config.get("compute_confidence_metrics") and reranking == "none":
                log("Computing LOOCV vision-only similarity margins (top-1 vs top-2)…")
                m, corr, _, _ = loocv_vision_similarity_margins(
                    iv_all, ex_all, alignment, alignment_params
                )
                mmax = float(np.max(m)) if len(m) else 0.0
                th = np.linspace(0.0, mmax + 1e-6, 21).tolist()
                curve = confidence_precision_coverage(m, corr, th)
                curve["mean_margin"] = float(np.mean(m))
                curve["max_margin"] = mmax
                curve["num_landmarks"] = int(len(m))
                result["confidence_margin_curve"] = curve

            # Per-cell failure analysis
            if config.get("save_per_cell", False):
                per_cell = []
                ex_coords_arr = np.array(
                    [[lm["exvivo_x"], lm["exvivo_y"], lm["exvivo_z"]] for lm in lm_list],
                    dtype=np.float32
                )
                ex_dists = squareform(pdist(ex_coords_arr))
                np.fill_diagonal(ex_dists, np.inf)
                nn_dists = ex_dists.min(axis=1)
                median_nn = float(np.median(nn_dists))
                for i, lm in enumerate(lm_list):
                    per_cell.append({
                        "cell_id": lm["id"],
                        "rank": int(ranks[i]),
                        "invivo_x": lm["invivo_x"],
                        "invivo_y": lm["invivo_y"],
                        "invivo_z": lm["invivo_z"],
                        "exvivo_x": lm["exvivo_x"],
                        "exvivo_y": lm["exvivo_y"],
                        "exvivo_z": lm["exvivo_z"],
                        "exvivo_nn_dist": float(nn_dists[i]),
                    })
                # Hard cells: bottom 10% by rank
                rank_arr = np.array([p["rank"] for p in per_cell])
                hard_thresh = np.percentile(rank_arr, 90)
                hard_ranks = rank_arr[rank_arr >= hard_thresh]
                mean_hard_rank = float(hard_ranks.mean()) if len(hard_ranks) else 0.0
                # Spatial outliers: nn_dist > 2x median
                outlier_count = sum(1 for p in per_cell if p["rank"] > 1 and p["exvivo_nn_dist"] > 2 * median_nn)
                failure_count = sum(1 for p in per_cell if p["rank"] > 1)
                frac_outlier_failures = outlier_count / max(failure_count, 1)
                result["loocv_hard10_mean_rank"] = mean_hard_rank
                result["loocv_failure_spatial_outlier_frac"] = frac_outlier_failures
                per_cell_file = OUTPUT_DIR / "loocv_per_cell_analysis.json"
                per_cell_file.write_text(json.dumps(per_cell, indent=2))
                log(f"Per-cell analysis saved → {per_cell_file}")
                log(f"  Hard-10% mean rank: {mean_hard_rank:.1f}")
                log(f"  Failures explained by spatial outlier: {frac_outlier_failures:.3f}")

        elif eval_mode == "val_test":
            val_ranks, test_ranks = val_test_eval(
                iv_all, ex_all, lm_list,
                alignment, alignment_params,
                reranking, reranking_params,
                iv_coords, ex_coords
            )
            val_metrics = bootstrap_metrics(val_ranks)
            test_metrics = bootstrap_metrics(test_ranks)
            result = {
                "val_R1": val_metrics["R1"],
                "val_R5": val_metrics["R5"],
                "val_R10": val_metrics["R10"],
                "val_mean_rank": val_metrics["mean_rank"],
                "val_discriminability_gap": compute_discriminability_gap(val_ranks, len(val_ranks)),
                "test_R1": test_metrics["R1"],
                "test_R5": test_metrics["R5"],
                "test_R10": test_metrics["R10"],
                "test_mean_rank": test_metrics["mean_rank"],
                "test_discriminability_gap": compute_discriminability_gap(test_ranks, len(test_ranks)),
            }
        else:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")

        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        result["meta"] = {
            "description": desc,
            "alignment": alignment,
            "alignment_params": alignment_params,
            "reranking": reranking,
            "reranking_params": reranking_params,
            "eval_mode": eval_mode,
            "norm": norm,
            "patch_xy_um": xy_um,
            "patch_z_um": z_um,
            "feature": feature_type,
        }

        # Log stdout
        for key in sorted(result.keys()):
            if key != "meta":
                log(f"{key}: {result[key]}")
        log(f"Elapsed: {elapsed / 60:.1f} min")

        # Save to JSON
        result_file = RESULTS_DIR / f"result_{int(time.time())}.json"
        result_file.write_text(json.dumps(result, indent=2))
        log(f"Saved → {result_file}")

        return result

    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified native LOOCV experiment runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file"
    )
    args = parser.parse_args()

    config_file = Path(args.config)
    if not config_file.exists():
        print(f"Config file not found: {config_file}")
        return 1

    config = json.loads(config_file.read_text())
    result = run_config(config)
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
