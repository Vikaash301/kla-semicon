"""Detail-Weighted Charbonnier Loss for SEM Image Super-Resolution.

Implements the FiDeSR-inspired difficulty and detail-weighted Charbonnier loss:
    L_detail = (1 / N) * sum_p w_p * sqrt((pred_p - target_p)^2 + eps^2)
where:
    w_p = 1 + eta * D_SEM(target)_p * E(target)_p
    E(target) = sqrt((grad_x target)^2 + (grad_y target)^2 + eps_e)
    D_SEM(target) = |grad target - GaussBlur(grad target)| / (std(target) + eps_std)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel2d(
    kernel_size: int = 5, sigma: float = 1.2, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Generates a normalized 2D Gaussian kernel."""
    coords = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2.0
    g1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g2d = torch.outer(g1d, g1d)
    return g2d / g2d.sum()


class DetailWeightedCharbonnierLoss(nn.Module):
    """FiDeSR-inspired Detail-Weighted Charbonnier Reconstruction Loss.

    Weights reconstruction error by localized semiconductor edge energy and high-frequency
    difficulty to prevent over-smoothing of fine pitches, line-edge roughness, and faint defects.

    Args:
        eta: Scaling weight for detail difficulty map (default: 1.0).
        eps: Numerical stability epsilon inside Charbonnier penalty (default: 1e-3).
        kernel_size: Kernel size for local Gaussian smoothing filter (default: 5).
        sigma: Standard deviation for Gaussian smoothing (default: 1.2).
        reduction: Reduction method: 'mean' or 'none' (default: 'mean').
    """

    def __init__(
        self,
        eta: float = 1.0,
        eps: float = 1e-3,
        kernel_size: int = 5,
        sigma: float = 1.2,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.eta = float(eta)
        self.eps = float(eps)
        self.eps_sq = float(eps * eps)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)
        self.reduction = reduction

        # Sobel gradient filters
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(
            1, 1, 3, 3
        )
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(
            1, 1, 3, 3
        )
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        # Gaussian smoothing filter
        gauss_k = _gaussian_kernel2d(kernel_size=self.kernel_size, sigma=self.sigma).view(1, 1, self.kernel_size, self.kernel_size)
        self.register_buffer("gauss_kernel", gauss_k, persistent=False)

    def compute_edge_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Computes isotropic edge energy E(x) = sqrt(Gx^2 + Gy^2 + 1e-6)."""
        b, c, h, w = x.shape
        # Pad with reflection to preserve border dimensions
        x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        # Apply Sobel filters channel-wise
        sobel_x_w = self.sobel_x.expand(c, 1, 3, 3).to(dtype=x.dtype, device=x.device)
        sobel_y_w = self.sobel_y.expand(c, 1, 3, 3).to(dtype=x.dtype, device=x.device)

        gx = F.conv2d(x_pad, sobel_x_w, groups=c)
        gy = F.conv2d(x_pad, sobel_y_w, groups=c)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def compute_sem_difficulty(self, x: torch.Tensor, edge_energy: torch.Tensor) -> torch.Tensor:
        """Computes SEM high-frequency difficulty map D_SEM(x)."""
        b, c, h, w = x.shape
        pad_size = self.kernel_size // 2
        ee_pad = F.pad(edge_energy, (pad_size, pad_size, pad_size, pad_size), mode="reflect")
        gauss_w = self.gauss_kernel.expand(c, 1, self.kernel_size, self.kernel_size).to(
            dtype=x.dtype, device=x.device
        )
        blurred_ee = F.conv2d(ee_pad, gauss_w, groups=c)

        high_freq_contrast = torch.abs(edge_energy - blurred_ee)

        # Local or per-image standard deviation
        # Using per-image/channel standard deviation with epsilon
        std_x = torch.std(x, dim=(-2, -1), keepdim=True)
        d_sem = high_freq_contrast / (std_x + 1e-4)
        return d_sem

    def compute_weight_map(self, target: torch.Tensor) -> torch.Tensor:
        """Computes FiDeSR detail weight map w_p = 1 + eta * D_SEM * E."""
        if self.eta == 0.0:
            return torch.ones_like(target)

        with torch.no_grad():
            edge_energy = self.compute_edge_energy(target)
            d_sem = self.compute_sem_difficulty(target, edge_energy)
            # Detail weighting map
            w_p = 1.0 + self.eta * d_sem * edge_energy
            # Ensure non-negative and bounded weights
            w_p = torch.clamp(w_p, min=1.0, max=20.0)
        return w_p

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for Detail-Weighted Charbonnier Loss.

        Args:
            pred: Reconstructed/super-resolved prediction tensor (B, C, H, W).
            target: Ground truth reference tensor (B, C, H, W).
            weight_map: Optional precomputed per-pixel weight map (B, C, H, W).

        Returns:
            Scalar loss tensor (if reduction='mean') or per-pixel loss tensor.
        """
        diff = pred - target
        charbonnier_penalty = torch.sqrt(diff * diff + self.eps_sq)

        if weight_map is None:
            weight_map = self.compute_weight_map(target)

        weighted_loss = weight_map * charbonnier_penalty

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss
