#!/usr/bin/env python3
"""
Domain alignment losses for cross-domain cell re-identification.

Provides distributional alignment losses that pull IV and EX feature
distributions together WITHOUT requiring paired cross-domain labels.

Usage:
    from native_domain_alignment import get_alignment_loss, DomainDiscriminator

    align_fn = get_alignment_loss("coral")
    L_align = align_fn(z_iv, z_ex)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss 1: CORAL (CORrelation ALignment)
# ---------------------------------------------------------------------------

def coral_loss(z_iv: torch.Tensor, z_ex: torch.Tensor) -> torch.Tensor:
    """
    Match mean and covariance of IV and EX projected embeddings.

    Sun & Saenko, ECCV 2016 — "Deep CORAL".
    Simplest alignment loss. No hyperparameters.

    Args:
        z_iv: (N_iv, D) L2-normalized IV projections
        z_ex: (N_ex, D) L2-normalized EX projections

    Returns:
        Scalar loss (mean alignment + covariance alignment).
    """
    mean_iv = z_iv.mean(dim=0)
    mean_ex = z_ex.mean(dim=0)
    mean_loss = F.mse_loss(mean_iv, mean_ex)

    n_iv, d = z_iv.shape
    n_ex = z_ex.shape[0]
    z_iv_c = z_iv - mean_iv
    z_ex_c = z_ex - mean_ex
    cov_iv = (z_iv_c.T @ z_iv_c) / max(n_iv - 1, 1)
    cov_ex = (z_ex_c.T @ z_ex_c) / max(n_ex - 1, 1)
    # Frobenius norm, scaled by 4d² as in Deep CORAL paper
    cov_loss = ((cov_iv - cov_ex) ** 2).sum() / (4 * d * d)

    return mean_loss + cov_loss


# ---------------------------------------------------------------------------
# Loss 2: MMD (Maximum Mean Discrepancy) with multi-kernel RBF
# ---------------------------------------------------------------------------

def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """RBF kernel k(x,y) = exp(-||x-y||² / (2σ²))."""
    xx = (x * x).sum(dim=1, keepdim=True)
    yy = (y * y).sum(dim=1, keepdim=True)
    dist_sq = xx + yy.T - 2 * (x @ y.T)
    return torch.exp(-dist_sq / (2 * sigma * sigma))


def mmd_loss(
    z_iv: torch.Tensor,
    z_ex: torch.Tensor,
    sigma_list: list[float] | None = None,
) -> torch.Tensor:
    """
    Multi-kernel Maximum Mean Discrepancy (MMD²).

    Long et al., ICML 2015 — "Learning Transferable Features with Deep
    Adaptation Networks".

    Uses a mixture of RBF kernels at multiple bandwidths so that the test
    is sensitive across scales.

    Args:
        z_iv: (N_iv, D) IV projections
        z_ex: (N_ex, D) EX projections
        sigma_list: RBF bandwidths (default: [0.1, 0.5, 1.0, 2.0, 5.0])

    Returns:
        Scalar MMD² loss.
    """
    if sigma_list is None:
        sigma_list = [0.1, 0.5, 1.0, 2.0, 5.0]

    loss = torch.tensor(0.0, device=z_iv.device, dtype=z_iv.dtype)
    for sigma in sigma_list:
        k_ss = _rbf_kernel(z_iv, z_iv, sigma).mean()
        k_tt = _rbf_kernel(z_ex, z_ex, sigma).mean()
        k_st = _rbf_kernel(z_iv, z_ex, sigma).mean()
        loss = loss + k_ss + k_tt - 2 * k_st
    return loss


# ---------------------------------------------------------------------------
# Loss 3: Sinkhorn Divergence (Optimal Transport)
# ---------------------------------------------------------------------------

def _sinkhorn_divergence(
    z_iv: torch.Tensor,
    z_ex: torch.Tensor,
    blur: float = 0.05,
    scaling: float = 0.9,
    max_iter: int = 50,
) -> torch.Tensor:
    """
    Sinkhorn divergence between two point clouds.

    Lightweight implementation (no geomloss dependency).
    Uses log-domain Sinkhorn for numerical stability.

    Courty et al., ECCV 2018 — "Optimal Transport for Domain Adaptation".

    Args:
        z_iv: (N, D) source features
        z_ex: (M, D) target features
        blur: Entropic regularization (epsilon = blur²)
        scaling: Multi-scale descent parameter
        max_iter: Sinkhorn iterations

    Returns:
        Scalar Sinkhorn divergence.
    """
    eps = blur * blur
    n = z_iv.shape[0]
    m = z_ex.shape[0]

    # Cost matrix
    C = torch.cdist(z_iv, z_ex, p=2).pow(2)

    # Uniform marginals in log-domain
    log_a = torch.full((n,), -math.log(n), device=z_iv.device, dtype=z_iv.dtype)
    log_b = torch.full((m,), -math.log(m), device=z_iv.device, dtype=z_iv.dtype)

    # Sinkhorn iterations in log domain
    f = torch.zeros(n, device=z_iv.device, dtype=z_iv.dtype)
    g = torch.zeros(m, device=z_iv.device, dtype=z_iv.dtype)

    for _ in range(max_iter):
        # f = -eps * logsumexp((g - C) / eps + log_b)
        M = (g.unsqueeze(0) - C) / eps + log_b.unsqueeze(0)
        f = -eps * torch.logsumexp(M, dim=1)

        M = (f.unsqueeze(1) - C) / eps + log_a.unsqueeze(1)
        g = -eps * torch.logsumexp(M, dim=0)

    # OT cost: <π, C> via dual potentials
    ot_cost = (f * torch.exp(log_a)).sum() + (g * torch.exp(log_b)).sum()

    # Sinkhorn divergence = OT(a,b) - 0.5*OT(a,a) - 0.5*OT(b,b)
    # For simplicity, we just return the OT cost (debiased version can be added)
    return ot_cost


def sinkhorn_loss(
    z_iv: torch.Tensor,
    z_ex: torch.Tensor,
    blur: float = 0.05,
    max_iter: int = 50,
) -> torch.Tensor:
    """
    Sinkhorn divergence loss for domain alignment.

    Falls back to geomloss if available (better optimized), otherwise
    uses the built-in implementation above.
    """
    try:
        from geomloss import SamplesLoss
        loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur, scaling=0.9)
        return loss_fn(z_iv, z_ex)
    except ImportError:
        return _sinkhorn_divergence(z_iv, z_ex, blur=blur, max_iter=max_iter)


# ---------------------------------------------------------------------------
# Loss 4: Domain Adversarial (DANN with Gradient Reversal)
# ---------------------------------------------------------------------------

class _GradientReversal(torch.autograd.Function):
    """Gradient reversal layer — identity forward, negated gradient backward."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal."""
    return _GradientReversal.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """Small MLP for domain classification (IV vs EX)."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        z_rev = gradient_reversal(z, alpha)
        return self.net(z_rev)


def dann_loss(
    z_iv: torch.Tensor,
    z_ex: torch.Tensor,
    discriminator: DomainDiscriminator,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Domain adversarial loss.

    Ganin et al., JMLR 2016. The gradient reversal in the discriminator
    means that minimizing this loss makes features more domain-invariant.

    Args:
        z_iv: (N_iv, D) IV projections
        z_ex: (N_ex, D) EX projections
        discriminator: Domain classifier with gradient reversal
        alpha: GRL scaling factor (use warm-up: 0→1 over training)

    Returns:
        Scalar domain classification loss (BCE).
    """
    z_all = torch.cat([z_iv, z_ex], dim=0)
    labels = torch.cat([
        torch.zeros(z_iv.shape[0], device=z_iv.device),
        torch.ones(z_ex.shape[0], device=z_iv.device),
    ])
    logits = discriminator(z_all, alpha=alpha).squeeze(-1)
    return F.binary_cross_entropy_with_logits(logits, labels)


def dann_alpha_schedule(epoch: int, max_epochs: int, schedule: str = "warmup") -> float:
    """
    Schedule for DANN gradient reversal strength.

    "warmup": Sigmoid ramp from 0 to 1 (Ganin et al. recommendation)
    "constant_low": Fixed α=0.1
    "constant_high": Fixed α=1.0
    """
    if schedule == "warmup":
        p = epoch / max(max_epochs, 1)
        return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
    elif schedule == "constant_low":
        return 0.1
    elif schedule == "constant_high":
        return 1.0
    return 1.0


# ---------------------------------------------------------------------------
# Loss 5: Sliced Wasserstein Distance (efficient OT alternative)
# ---------------------------------------------------------------------------

def sliced_wasserstein_loss(
    z_iv: torch.Tensor,
    z_ex: torch.Tensor,
    n_projections: int = 50,
) -> torch.Tensor:
    """
    Sliced Wasserstein Distance — fast 1D projection-based OT approximation.

    Kolouri et al., CVPR 2019. Projects high-D distributions onto random 1D
    lines and computes exact 1D Wasserstein distance on each.

    Args:
        z_iv: (N_iv, D)
        z_ex: (N_ex, D)
        n_projections: Number of random projection directions

    Returns:
        Scalar SWD loss.
    """
    d = z_iv.shape[1]
    # Random projection directions (unit vectors on sphere)
    directions = torch.randn(n_projections, d, device=z_iv.device, dtype=z_iv.dtype)
    directions = F.normalize(directions, dim=1)

    # Project both sets
    proj_iv = z_iv @ directions.T  # (N_iv, n_proj)
    proj_ex = z_ex @ directions.T  # (N_ex, n_proj)

    # Sort along sample dimension for each projection
    proj_iv_sorted = torch.sort(proj_iv, dim=0).values
    proj_ex_sorted = torch.sort(proj_ex, dim=0).values

    # If different sizes, interpolate to same length
    n_iv = proj_iv_sorted.shape[0]
    n_ex = proj_ex_sorted.shape[0]
    if n_iv != n_ex:
        n_common = max(n_iv, n_ex)
        proj_iv_sorted = F.interpolate(
            proj_iv_sorted.T.unsqueeze(0), size=n_common, mode="linear", align_corners=True
        ).squeeze(0).T
        proj_ex_sorted = F.interpolate(
            proj_ex_sorted.T.unsqueeze(0), size=n_common, mode="linear", align_corners=True
        ).squeeze(0).T

    # 1D Wasserstein = mean absolute difference of sorted projections
    return (proj_iv_sorted - proj_ex_sorted).pow(2).mean()


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

_LOSS_REGISTRY = {
    "coral": coral_loss,
    "mmd": mmd_loss,
    "sinkhorn": sinkhorn_loss,
    "swd": sliced_wasserstein_loss,
    # "dann" handled separately (needs discriminator state)
}


def get_alignment_loss(name: str):
    """
    Return a callable alignment loss function.

    For "dann", returns None — use dann_loss() directly with a DomainDiscriminator.
    """
    if name == "none":
        return lambda z_iv, z_ex: torch.tensor(0.0, device=z_iv.device)
    if name == "dann":
        return None  # caller must use dann_loss() with discriminator
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown alignment loss: {name}. Choose from: {list(_LOSS_REGISTRY.keys()) + ['dann', 'none']}")
    return _LOSS_REGISTRY[name]
