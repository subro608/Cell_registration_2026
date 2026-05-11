#!/usr/bin/env python3
"""
Shared augmentation pipeline for cross-domain cell patch pre-training.

Augmentations simulate in-vivo two-photon → ex-vivo widefield appearance shift.
All functions operate on (H, W, 3) float32 patches in [0.0, 1.0].

Used by Cursor V2 data scripts and Claude Code V2 SimCLR (import this module).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, rotate as nd_rotate
from skimage.transform import AffineTransform, resize, warp


def aug_brightness(p: np.ndarray, rng: np.random.Generator, low: float = 0.3, high: float = 3.0) -> np.ndarray:
    return np.clip(p * rng.uniform(low, high), 0, 1).astype(np.float32)


def aug_gamma(p: np.ndarray, rng: np.random.Generator, low: float = 0.4, high: float = 2.5) -> np.ndarray:
    return np.clip(np.power(np.clip(p, 1e-8, 1.0), rng.uniform(low, high)), 0, 1).astype(np.float32)


def aug_gaussian_noise(p: np.ndarray, rng: np.random.Generator, max_sigma: float = 0.06) -> np.ndarray:
    sigma = rng.uniform(0, max_sigma)
    return np.clip(p + rng.normal(0, sigma, p.shape).astype(np.float32), 0, 1)


def aug_gaussian_blur(p: np.ndarray, rng: np.random.Generator, max_sigma: float = 2.5) -> np.ndarray:
    sigma = rng.uniform(0, max_sigma)
    if sigma < 0.3:
        return p
    out = p.copy()
    for c in range(3):
        out[:, :, c] = gaussian_filter(p[:, :, c], sigma=sigma)
    return out


def aug_rotation(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(0, 360)
    return np.clip(
        nd_rotate(p, angle, axes=(0, 1), reshape=False, order=1, mode="reflect"),
        0,
        1,
    ).astype(np.float32)


def aug_flip(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = p
    if rng.random() > 0.5:
        out = out[::-1].copy()
    if rng.random() > 0.5:
        out = out[:, ::-1].copy()
    return out


def aug_elastic(p: np.ndarray, rng: np.random.Generator, alpha: float = 40.0, sigma: float = 5.0) -> np.ndarray:
    H, W = p.shape[:2]
    dx = gaussian_filter(rng.uniform(-1, 1, (H, W)).astype(np.float32), sigma) * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, (H, W)).astype(np.float32), sigma) * alpha
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.array(
        [
            np.clip(yy.astype(np.float32) + dy, 0, H - 1).ravel(),
            np.clip(xx.astype(np.float32) + dx, 0, W - 1).ravel(),
        ]
    )
    out = np.empty_like(p)
    for c in range(3):
        out[:, :, c] = map_coordinates(
            p[:, :, c], coords, order=1, mode="reflect"
        ).reshape(H, W)
    return np.clip(out, 0, 1).astype(np.float32)


def aug_random_erase(p: np.ndarray, rng: np.random.Generator, max_n: int = 3) -> np.ndarray:
    H, W = p.shape[:2]
    out = p.copy()
    n = int(rng.integers(0, max_n + 1))
    for _ in range(n):
        ey = int(rng.integers(0, max(1, H - H // 8)))
        ex = int(rng.integers(0, max(1, W - W // 8)))
        eh = int(rng.integers(H // 16, H // 8 + 1))
        ew = int(rng.integers(W // 16, W // 8 + 1))
        fill = float(rng.uniform(0, 0.3))
        out[ey : ey + eh, ex : ex + ew] = fill
    return out


def aug_channel_jitter(
    p: np.ndarray, rng: np.random.Generator, scale_range: float = 0.3, shift_range: float = 0.1
) -> np.ndarray:
    out = p.copy()
    for c in range(3):
        out[:, :, c] = np.clip(
            p[:, :, c] * rng.uniform(1 - scale_range, 1 + scale_range)
            + rng.uniform(-shift_range, shift_range),
            0,
            1,
        )
    return out.astype(np.float32)


def aug_random_crop_resize(p: np.ndarray, rng: np.random.Generator, min_frac: float = 0.6) -> np.ndarray:
    H, W = p.shape[:2]
    frac = rng.uniform(min_frac, 1.0)
    ch, cw = max(2, int(H * frac)), max(2, int(W * frac))
    sy = int(rng.integers(0, max(1, H - ch + 1)))
    sx = int(rng.integers(0, max(1, W - cw + 1)))
    crop = p[sy : sy + ch, sx : sx + cw]
    resized = resize(crop, (H, W), order=1, preserve_range=True, anti_aliasing=True)
    return np.clip(resized, 0, 1).astype(np.float32)


def aug_affine_shear_tilt(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    In-plane affine: small shear, scale, rotation, translation (approx. tilt / oblique section).
    Center-fixed composition so the cell stays roughly centered.
    """
    H, W = int(p.shape[0]), int(p.shape[1])
    shift_x, shift_y = rng.uniform(-4.0, 4.0, size=2).astype(np.float64)
    scale = (
        1.0 + float(rng.uniform(-0.07, 0.07)),
        1.0 + float(rng.uniform(-0.07, 0.07)),
    )
    rotation = np.deg2rad(float(rng.uniform(-10.0, 10.0)))
    shear = np.deg2rad(float(rng.uniform(-12.0, 12.0)))
    t_center = AffineTransform(translation=(W / 2.0, H / 2.0))
    t_uncenter = AffineTransform(translation=(-W / 2.0, -H / 2.0))
    t_aff = AffineTransform(
        scale=scale,
        rotation=rotation,
        shear=shear,
        translation=(shift_x, shift_y),
    )
    tform = t_center + t_aff + t_uncenter
    out = np.empty_like(p, dtype=np.float32)
    inv = tform.inverse
    for c in range(3):
        out[:, :, c] = warp(
            p[:, :, c],
            inv,
            order=1,
            mode="reflect",
            preserve_range=True,
        ).astype(np.float32)
    return np.clip(out, 0, 1).astype(np.float32)


AUGMENTATION_PRESETS: dict[str, dict[str, Any]] = {
    "heavy": {
        "brightness": True,
        "gamma": True,
        "noise": True,
        "blur": True,
        "rotation": True,
        "flip": True,
        "elastic": True,
        "erase": True,
        "channel_jitter": True,
        "crop_resize": True,
    },
    "medium": {
        "brightness": True,
        "gamma": True,
        "noise": True,
        "blur": True,
        "rotation": True,
        "flip": True,
        "elastic": False,
        "erase": True,
        "channel_jitter": True,
        "crop_resize": False,
    },
    "rotation_only": {
        "brightness": False,
        "gamma": False,
        "noise": False,
        "blur": False,
        "rotation": True,
        "flip": True,
        "elastic": False,
        "erase": False,
        "channel_jitter": False,
        "crop_resize": False,
    },
    "domain_gap": {
        "brightness": True,
        "gamma": True,
        "noise": True,
        "blur": True,
        "rotation": False,
        "flip": False,
        "elastic": False,
        "erase": False,
        "channel_jitter": True,
        "crop_resize": False,
    },
    "geometry_strong": {
        "brightness": False,
        "gamma": False,
        "noise": False,
        "blur": False,
        "rotation": True,
        "flip": True,
        "elastic": False,
        "erase": False,
        "channel_jitter": False,
        "crop_resize": False,
        "shear_tilt": True,
        "double_rotation": True,
    },
    "geometry_plus_medium": {
        "brightness": True,
        "gamma": True,
        "noise": True,
        "blur": True,
        "rotation": True,
        "flip": True,
        "elastic": False,
        "erase": True,
        "channel_jitter": True,
        "crop_resize": False,
        "shear_tilt": True,
        "double_rotation": True,
    },
    # Stronger spatial diversity than rotation_only; always re-crop (harder within-modality).
    "rotation_crop_heavy": {
        "brightness": False,
        "gamma": False,
        "noise": False,
        "blur": False,
        "rotation": True,
        "flip": True,
        "elastic": False,
        "erase": False,
        "channel_jitter": False,
        "crop_resize": True,
        "crop_resize_always": True,
        "crop_min_frac": 0.45,
    },
    # Photometric + structural noise without full "heavy" elastic/brightness swing.
    "hard_intra": {
        "brightness": False,
        "gamma": False,
        "noise": True,
        "noise_max_sigma": 0.04,
        "blur": True,
        "blur_max_sigma": 1.2,
        "rotation": True,
        "flip": True,
        "elastic": False,
        "erase": True,
        "channel_jitter": True,
        "channel_jitter_scale": 0.12,
        "channel_jitter_shift": 0.06,
        "crop_resize": False,
    },
}


def augment_patch(patch_rgb: np.ndarray, rng: np.random.Generator, preset: str = "heavy") -> np.ndarray:
    cfg = AUGMENTATION_PRESETS[preset]
    p = patch_rgb.copy().astype(np.float32)

    if cfg["brightness"]:
        p = aug_brightness(p, rng)
    if cfg["gamma"]:
        p = aug_gamma(p, rng)
    if cfg["noise"]:
        p = aug_gaussian_noise(p, rng, max_sigma=float(cfg.get("noise_max_sigma", 0.06)))
    if cfg["blur"]:
        p = aug_gaussian_blur(p, rng, max_sigma=float(cfg.get("blur_max_sigma", 2.5)))
    if cfg["rotation"]:
        p = aug_rotation(p, rng)
    if cfg["flip"]:
        p = aug_flip(p, rng)
    if cfg.get("shear_tilt", False):
        p = aug_affine_shear_tilt(p, rng)
    if cfg.get("double_rotation", False) and cfg["rotation"]:
        p = aug_rotation(p, rng)
    if cfg["elastic"] and rng.random() > 0.5:
        p = aug_elastic(p, rng)
    if cfg["erase"]:
        p = aug_random_erase(p, rng)
    if cfg["channel_jitter"]:
        p = aug_channel_jitter(
            p,
            rng,
            scale_range=float(cfg.get("channel_jitter_scale", 0.3)),
            shift_range=float(cfg.get("channel_jitter_shift", 0.1)),
        )
    crop_always = bool(cfg.get("crop_resize_always", False))
    if cfg["crop_resize"] and (crop_always or rng.random() > 0.5):
        p = aug_random_crop_resize(p, rng, min_frac=float(cfg.get("crop_min_frac", 0.6)))

    return np.clip(p, 0, 1).astype(np.float32)


def make_simclr_batch(
    patches: list[np.ndarray], rng: np.random.Generator, preset: str = "heavy"
) -> tuple[np.ndarray, np.ndarray]:
    view1 = np.stack([augment_patch(p, rng, preset) for p in patches])
    view2 = np.stack([augment_patch(p, rng, preset) for p in patches])
    return view1, view2


def compute_domain_stats(patches: np.ndarray) -> dict:
    """Per-channel mean/std over (N, H, W, 3) patches in [0,1]."""
    flat = patches.reshape(-1, 3)
    return {
        "mean": flat.mean(axis=0).tolist(),
        "std": flat.std(axis=0).tolist(),
    }


def style_transfer_patch(patch: np.ndarray, src_stats: dict, tgt_stats: dict) -> np.ndarray:
    """Reinhard-style per-channel mean/std: map patch from src domain stats toward tgt."""
    src_mean = np.array(src_stats["mean"], dtype=np.float32)
    src_std = np.array(src_stats["std"], dtype=np.float32) + 1e-8
    tgt_mean = np.array(tgt_stats["mean"], dtype=np.float32)
    tgt_std = np.array(tgt_stats["std"], dtype=np.float32) + 1e-8
    out = (patch.astype(np.float32) - src_mean) / src_std * tgt_std + tgt_mean
    return np.clip(out, 0, 1).astype(np.float32)


def load_domain_style_stats(path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["invivo"], data["exvivo"]


def measure_aug_diversity(patch: np.ndarray, n_samples: int = 20, preset: str = "heavy", seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    changes = []
    for _ in range(n_samples):
        aug = augment_patch(patch, rng, preset)
        changes.append(float(np.abs(aug - patch).mean()))
    return {
        "mean_pixel_change": float(np.mean(changes)),
        "std_pixel_change": float(np.std(changes)),
        "preset": preset,
        "n_samples": n_samples,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dummy = rng.uniform(0, 1, (224, 224, 3)).astype(np.float32)
    for preset in AUGMENTATION_PRESETS:
        stats = measure_aug_diversity(dummy, n_samples=10, preset=preset)
        print(f"[{preset}] mean_pixel_change={stats['mean_pixel_change']:.4f}")
    v1, v2 = make_simclr_batch([dummy], rng, "heavy")
    assert v1.shape == (1, 224, 224, 3)
    print("Augmentation module OK.")
