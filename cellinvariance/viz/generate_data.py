#!/usr/bin/env python3
"""
Extract all data needed for dino_repr_interactive_v2.html.

Best run: w7_e200_ms3_hi_ap400_j0_xy80z24_skh07_dv2_g0
  - Multiscale 3x (xy=[40,80,120] um, z=[12,24,36] um)
  - Hard-intra augmentation, 400 aug/cell, no jitter
  - Better EX->IV top-5 (proto): 22.97%  vs prev 17.47%

Outputs: figures/report/native_baseline/dino_repr_interactive_v2_data.json
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

VIZ_DIR = Path(__file__).resolve().parent
ROOT    = VIZ_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

import runner as nr
from contrastive import (
    MLPProjector,
    embed_patches_cls,
    extract_patch_ex_at,
    extract_patch_iv_at,
)

# ── Config ─────────────────────────────────────────────────────────────────
TAG         = "w8_e200_ms3_hi_ap800_j0_xy80z24_skh07_dv2_g0"
PROJ_PT     = ROOT / "results" / "w8_projector.pt"
RESULTS_JSON= ROOT / "results" / "w8_results.json"
OUT_JSON    = VIZ_DIR / "dino_repr_data.json"

# Multiscale for DINO embedding (must match training)
MS_XY = (40, 80, 120)   # µm
MS_Z  = (12, 24, 36)    # µm

# Display patch size (used for images)
DISP_XY_UM = 80
DISP_Z_UM  = 24

# Contact-sheet patch size (thumbnails)
THUMB_XY_UM = 80
THUMB_Z_UM  = 24

EMBED_BATCH  = 8
MLP_OUT_DIM  = 64
MLP_HIDDEN   = 256
MODEL_ID     = "facebook/dinov2-small"
N_AUG_EMBED  = 12   # augmented samples per cell to embed for scatter dots


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def array_to_png_b64(arr: np.ndarray) -> str:
    """(H,W,3) float32 [0,1] → base64 PNG string."""
    from PIL import Image
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,D), b: (M,D) → (N,M) cosine similarity."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_n @ b_n.T


def fit_pca2d(emb: np.ndarray):
    """Fit PCA on emb (N, D), return (pca object, (N, 2) projection, mean)."""
    from sklearn.decomposition import PCA
    mean = emb.mean(axis=0)
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(emb - mean).astype(np.float32)
    return pca, proj, mean


def apply_pca2d(pca, mean, emb: np.ndarray) -> np.ndarray:
    """Project emb through fitted pca. emb: (N, D) → (N, 2)."""
    return pca.transform(emb - mean).astype(np.float32)


def main() -> None:
    from native_augmentations import augment_patch

    log(f"Best run: {TAG}")
    log(f"Proj checkpoint: {PROJ_PT}")

    # ── Load results metadata ──────────────────────────────────────────────
    with open(RESULTS_JSON) as f:
        results = json.load(f)
    log(f"Run metrics: top1={results.get('ex_to_iv_top1_acc',0):.4f}  "
        f"proto-top5={results.get('ex_to_iv_top5_acc_cell_proto',0):.4f}  "
        f"mrr={results.get('ex_to_iv_mrr_cell_proto',0):.4f}")

    # ── Load data ──────────────────────────────────────────────────────────
    log("Loading volumes…")
    lm_list        = nr.load_landmarks()
    iv_vol, ex_vol = nr.load_volumes()
    ev_vox         = nr.fit_exvivo_voxel_dims(lm_list)
    N              = len(lm_list)
    log(f"  {N} landmarks  iv={iv_vol.shape}  ex={ex_vol.shape}")

    # ── Extract patches ────────────────────────────────────────────────────
    log(f"Extracting display/thumb patches + {len(MS_XY)} multiscale patches…")
    iv_disp, ex_disp = [], []
    iv_thumb, ex_thumb = [], []
    iv_ms = [[] for _ in MS_XY]   # iv_ms[scale_idx][cell_idx] = patch
    ex_ms = [[] for _ in MS_XY]

    for i, lm in enumerate(lm_list):
        iv_xyz = (lm["invivo_x"],  lm["invivo_y"],  lm["invivo_z"])
        ex_xyz = (lm["exvivo_x"],  lm["exvivo_y"],  lm["exvivo_z"])

        iv_disp.append(extract_patch_iv_at(iv_vol, *iv_xyz, DISP_XY_UM, DISP_Z_UM))
        ex_disp.append(extract_patch_ex_at(ex_vol, *ex_xyz, ev_vox, DISP_XY_UM, DISP_Z_UM))

        iv_thumb.append(extract_patch_iv_at(iv_vol, *iv_xyz, THUMB_XY_UM, THUMB_Z_UM))
        ex_thumb.append(extract_patch_ex_at(ex_vol, *ex_xyz, ev_vox, THUMB_XY_UM, THUMB_Z_UM))

        for s_idx, (xy, z) in enumerate(zip(MS_XY, MS_Z)):
            iv_ms[s_idx].append(extract_patch_iv_at(iv_vol, *iv_xyz, xy, z))
            ex_ms[s_idx].append(extract_patch_ex_at(ex_vol, *ex_xyz, ev_vox, xy, z))

        if (i + 1) % 30 == 0:
            log(f"  {i+1}/{N} cells done")

    log("All patches extracted.")

    # ── Generate aug patches at each scale ────────────────────────────────
    # Augment each scale independently (matches training pipeline).
    # iv_ms_aug[s_idx] = flat list of N*K patches at scale s
    log(f"Generating {N_AUG_EMBED} aug samples × {len(MS_XY)} scales per cell…")
    rng_aug = np.random.default_rng(42)
    aug_preset = "hard_intra"

    iv_ms_aug = [[] for _ in MS_XY]   # [scale][cell*aug_idx] = patch
    ex_ms_aug = [[] for _ in MS_XY]

    for i in range(N):
        for k in range(N_AUG_EMBED):
            for s_idx in range(len(MS_XY)):
                iv_ms_aug[s_idx].append(augment_patch(iv_ms[s_idx][i], rng_aug, aug_preset))
                ex_ms_aug[s_idx].append(augment_patch(ex_ms[s_idx][i], rng_aug, aug_preset))

    log(f"  Aug patches: {len(iv_ms_aug[0])} per scale ({N}×{N_AUG_EMBED})")

    # ── DINO embeddings (multi-scale base) ────────────────────────────────
    log(f"Loading DINO model {MODEL_ID}…")
    device = nr.choose_device()
    embedder = nr.DinoEmbedder(MODEL_ID, device)

    log("Computing multiscale DINO base embeddings…")
    iv_scale_embs, ex_scale_embs = [], []
    for s_idx, (xy, z) in enumerate(zip(MS_XY, MS_Z)):
        log(f"  Base scale {s_idx+1}/3: {xy}µm × {z}µm")
        iv_scale_embs.append(embed_patches_cls(embedder, iv_ms[s_idx], EMBED_BATCH))
        ex_scale_embs.append(embed_patches_cls(embedder, ex_ms[s_idx], EMBED_BATCH))

    # Concatenate → (N, 3×384) = (N, 1152)
    dino_iv = np.concatenate(iv_scale_embs, axis=1).astype(np.float32)
    dino_ex = np.concatenate(ex_scale_embs, axis=1).astype(np.float32)
    log(f"Base DINO: IV={dino_iv.shape}  EX={dino_ex.shape}")

    # ── DINO aug embeddings (proper multi-scale) ──────────────────────────
    log(f"Computing DINO aug embeddings (multi-scale, {N_AUG_EMBED}/cell)…")
    iv_aug_scale_embs, ex_aug_scale_embs = [], []
    for s_idx, (xy, z) in enumerate(zip(MS_XY, MS_Z)):
        log(f"  Aug scale {s_idx+1}/3: {xy}µm × {z}µm  ({N*N_AUG_EMBED} patches)")
        iv_aug_scale_embs.append(
            embed_patches_cls(embedder, iv_ms_aug[s_idx], EMBED_BATCH)
        )  # (N*K, 384)
        ex_aug_scale_embs.append(
            embed_patches_cls(embedder, ex_ms_aug[s_idx], EMBED_BATCH)
        )

    # Concatenate scales → (N*K, 1152)
    dino_iv_aug_flat = np.concatenate(iv_aug_scale_embs, axis=1).astype(np.float32)
    dino_ex_aug_flat = np.concatenate(ex_aug_scale_embs, axis=1).astype(np.float32)
    dino_iv_aug = dino_iv_aug_flat.reshape(N, N_AUG_EMBED, -1)  # (N, K, 1152)
    dino_ex_aug = dino_ex_aug_flat.reshape(N, N_AUG_EMBED, -1)
    log(f"Aug DINO: IV={dino_iv_aug.shape}  EX={dino_ex_aug.shape}")

    # ── MLP projector — base embeddings + mid-layer ───────────────────────
    log(f"Loading MLP projector from {PROJ_PT.name}…")
    proj = MLPProjector(in_dim=dino_iv.shape[1], hidden=MLP_HIDDEN, out_dim=MLP_OUT_DIM)
    proj.load_state_dict(torch.load(PROJ_PT, map_location="cpu"))
    proj.eval()

    with torch.no_grad():
        iv_t  = torch.tensor(dino_iv, dtype=torch.float32)
        ex_t  = torch.tensor(dino_ex, dtype=torch.float32)

        # Final layer (64-d, L2-normalized)
        mlp_iv  = proj(iv_t).numpy().astype(np.float32)
        mlp_ex  = proj(ex_t).numpy().astype(np.float32)

        # Mid layer (256-d, ReLU-activated)
        mlp_mid_iv = F.relu(proj.fc1(iv_t)).numpy().astype(np.float32)
        mlp_mid_ex = F.relu(proj.fc1(ex_t)).numpy().astype(np.float32)

        # Aug — base and mid layer (proper multi-scale 1152-d input)
        iv_aug_t = torch.tensor(dino_iv_aug_flat, dtype=torch.float32)
        ex_aug_t = torch.tensor(dino_ex_aug_flat, dtype=torch.float32)

        mlp_iv_aug_flat     = proj(iv_aug_t).numpy().astype(np.float32)
        mlp_ex_aug_flat     = proj(ex_aug_t).numpy().astype(np.float32)
        mlp_mid_iv_aug_flat = F.relu(proj.fc1(iv_aug_t)).numpy().astype(np.float32)
        mlp_mid_ex_aug_flat = F.relu(proj.fc1(ex_aug_t)).numpy().astype(np.float32)

        mlp_iv_aug     = mlp_iv_aug_flat.reshape(N, N_AUG_EMBED, -1)
        mlp_ex_aug     = mlp_ex_aug_flat.reshape(N, N_AUG_EMBED, -1)
        mlp_mid_iv_aug = mlp_mid_iv_aug_flat.reshape(N, N_AUG_EMBED, -1)
        mlp_mid_ex_aug = mlp_mid_ex_aug_flat.reshape(N, N_AUG_EMBED, -1)

    log(f"MLP final: IV={mlp_iv.shape}  EX={mlp_ex.shape}")
    log(f"MLP mid:   IV={mlp_mid_iv.shape}  EX={mlp_mid_ex.shape}")
    log(f"MLP aug:   IV={mlp_iv_aug.shape}  mid={mlp_mid_iv_aug.shape}")

    # ── PCA 2D ────────────────────────────────────────────────────────────
    log("Computing PCA 2D projections (joint IV+EX)…")

    # DINO PCA: fit on joint 1152-d base embeddings
    joint_dino   = np.vstack([dino_iv, dino_ex])
    pca_dino, jd_pca, mean_dino = fit_pca2d(joint_dino)
    dino_iv_pca  = jd_pca[:N]
    dino_ex_pca  = jd_pca[N:]
    dino_iv_aug_pca = apply_pca2d(pca_dino, mean_dino,
                                  dino_iv_aug.reshape(-1, dino_iv_aug.shape[2])
                                  ).reshape(N, N_AUG_EMBED, 2)
    dino_ex_aug_pca = apply_pca2d(pca_dino, mean_dino,
                                  dino_ex_aug.reshape(-1, dino_ex_aug.shape[2])
                                  ).reshape(N, N_AUG_EMBED, 2)

    # MLP-mid PCA (256-d)
    joint_mid    = np.vstack([mlp_mid_iv, mlp_mid_ex])
    pca_mid, jm_pca, mean_mid = fit_pca2d(joint_mid)
    mid_iv_pca   = jm_pca[:N]
    mid_ex_pca   = jm_pca[N:]
    mid_iv_aug_pca = apply_pca2d(pca_mid, mean_mid,
                                 mlp_mid_iv_aug.reshape(-1, mlp_mid_iv_aug.shape[2])
                                 ).reshape(N, N_AUG_EMBED, 2)
    mid_ex_aug_pca = apply_pca2d(pca_mid, mean_mid,
                                 mlp_mid_ex_aug.reshape(-1, mlp_mid_ex_aug.shape[2])
                                 ).reshape(N, N_AUG_EMBED, 2)

    # MLP-out PCA (64-d)
    joint_mlp    = np.vstack([mlp_iv, mlp_ex])
    pca_mlp, jmlp_pca, mean_mlp = fit_pca2d(joint_mlp)
    mlp_iv_pca   = jmlp_pca[:N]
    mlp_ex_pca   = jmlp_pca[N:]
    mlp_iv_aug_pca = apply_pca2d(pca_mlp, mean_mlp,
                                 mlp_iv_aug.reshape(-1, mlp_iv_aug.shape[2])
                                 ).reshape(N, N_AUG_EMBED, 2)
    mlp_ex_aug_pca = apply_pca2d(pca_mlp, mean_mlp,
                                 mlp_ex_aug.reshape(-1, mlp_ex_aug.shape[2])
                                 ).reshape(N, N_AUG_EMBED, 2)

    log("PCA done.")

    # ── Discriminability matrices ──────────────────────────────────────────
    log("Computing discriminability matrices…")
    iv_iv_mlp_cosim  = cosine_sim_matrix(mlp_iv,  mlp_iv ).astype(np.float32)
    ex_iv_mlp_cosim  = cosine_sim_matrix(mlp_ex,  mlp_iv ).astype(np.float32)
    iv_iv_dino_cosim = cosine_sim_matrix(dino_iv,  dino_iv).astype(np.float32)
    ex_iv_dino_cosim = cosine_sim_matrix(dino_ex,  dino_iv).astype(np.float32)

    # ── Retrieval rankings ────────────────────────────────────────────────
    log("Computing retrieval rankings…")
    # IV→EX: for each IV cell i, top-5 EX by ex_iv_cosim[:, i]
    iv_top5_ex = []
    for i in range(N):
        sims = ex_iv_mlp_cosim[:, i]
        iv_top5_ex.append(np.argsort(sims)[::-1][:5].tolist())

    # EX→IV: for each EX cell j, top-5 IV by ex_iv_cosim[j, :]
    ex_top5_iv = []
    for j in range(N):
        sims = ex_iv_mlp_cosim[j, :]
        ex_top5_iv.append(np.argsort(sims)[::-1][:5].tolist())

    # Correctness based on EX→IV direction
    correct_in_top5 = [int(i in ex_top5_iv[i]) for i in range(N)]
    top5_acc = sum(correct_in_top5) / N
    log(f"MLP EX→IV top-5 acc: {top5_acc:.3f}")

    # ── Encode images ─────────────────────────────────────────────────────
    log("Encoding display images…")
    iv_disp_b64  = [array_to_png_b64(p) for p in iv_disp]
    ex_disp_b64  = [array_to_png_b64(p) for p in ex_disp]
    log("Encoding thumbnail images…")
    iv_thumb_b64 = [array_to_png_b64(p) for p in iv_thumb]
    ex_thumb_b64 = [array_to_png_b64(p) for p in ex_thumb]
    log("Images encoded.")

    # ── Cell IDs ──────────────────────────────────────────────────────────
    cell_ids = [f"L{lm.get('landmark_id', i+1)}" for i, lm in enumerate(lm_list)]

    # ── Assemble output JSON ───────────────────────────────────────────────
    log("Assembling output JSON…")
    out = {
        "meta": {
            "tag":          TAG,
            "model_id":     MODEL_ID,
            "ms_xy_um":     list(MS_XY),
            "ms_z_um":      list(MS_Z),
            "disp_xy_um":   DISP_XY_UM,
            "disp_z_um":    DISP_Z_UM,
            "thumb_xy_um":  THUMB_XY_UM,
            "thumb_z_um":   THUMB_Z_UM,
            "mlp_out_dim":  MLP_OUT_DIM,
            "mlp_mid_dim":  MLP_HIDDEN,
            "dino_dim":     int(dino_iv.shape[1]),
            "n_cells":      N,
            "n_aug_embed":  N_AUG_EMBED,
        },
        "metrics": {
            "ex_to_iv_top1":            results.get("ex_to_iv_top1_acc"),
            "ex_to_iv_top5_cell_proto": results.get("ex_to_iv_top5_acc_cell_proto"),
            "ex_to_iv_mrr":             results.get("ex_to_iv_mrr_cell_proto"),
            "best_balanced_val_knn":    results.get("best_balanced_val_knn_acc"),
            "best_epoch":               results.get("best_epoch_by_balanced_val_knn"),
            "num_cells":                results.get("num_cells_total"),
        },
        "cell_ids": cell_ids,
        "iv": {
            "disp_b64":      iv_disp_b64,
            "thumb_b64":     iv_thumb_b64,
            "dino_pca":      dino_iv_pca.tolist(),
            "mid_pca":       mid_iv_pca.tolist(),
            "mlp_pca":       mlp_iv_pca.tolist(),
            "dino_emb":      dino_iv.tolist(),
            "mlp_mid_emb":   mlp_mid_iv.tolist(),
            "mlp_emb":       mlp_iv.tolist(),
            "aug_dino_pca":  dino_iv_aug_pca.tolist(),
            "aug_mid_pca":   mid_iv_aug_pca.tolist(),
            "aug_mlp_pca":   mlp_iv_aug_pca.tolist(),
        },
        "ex": {
            "disp_b64":      ex_disp_b64,
            "thumb_b64":     ex_thumb_b64,
            "dino_pca":      dino_ex_pca.tolist(),
            "mid_pca":       mid_ex_pca.tolist(),
            "mlp_pca":       mlp_ex_pca.tolist(),
            "dino_emb":      dino_ex.tolist(),
            "mlp_mid_emb":   mlp_mid_ex.tolist(),
            "mlp_emb":       mlp_ex.tolist(),
            "aug_dino_pca":  dino_ex_aug_pca.tolist(),
            "aug_mid_pca":   mid_ex_aug_pca.tolist(),
            "aug_mlp_pca":   mlp_ex_aug_pca.tolist(),
        },
        "matrices": {
            "iv_iv_mlp_cosim":  iv_iv_mlp_cosim.tolist(),
            "ex_iv_mlp_cosim":  ex_iv_mlp_cosim.tolist(),
            "iv_iv_dino_cosim": iv_iv_dino_cosim.tolist(),
            "ex_iv_dino_cosim": ex_iv_dino_cosim.tolist(),
        },
        "retrieval": {
            "iv_top5_ex":       iv_top5_ex,
            "ex_top5_iv":       ex_top5_iv,
            "correct_in_top5":  correct_in_top5,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    size_mb = OUT_JSON.stat().st_size / 1e6
    log(f"Saved {OUT_JSON} ({size_mb:.1f} MB)")
    log("Done.")


if __name__ == "__main__":
    main()
