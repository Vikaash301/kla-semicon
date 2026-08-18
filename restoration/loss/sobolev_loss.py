"""Multi-Scale Sobolev Gradient Loss for High-Fidelity Semiconductor SEM-SR.

Supervises weak spatial derivatives (first-order gradients and second-order Laplacians)
across multiple spatial scales in Sobolev spaces H^1 and H^2. This guarantees sharp,
physically accurate edge profiles for Critical Dimension (CD) metrology and suppresses
spurious high-frequency oscillatory artifacts.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSobolevGradientLoss(nn.Module):
    """Multi-Scale Sobolev Gradient and Laplacian Loss in H^1 / H^2 space.

    Args:
        scales: Number of dyadic resolution scales to compute Sobolev gradients (default: 3).
        scale_weights: Weights for each resolution scale (default: [1.0, 0.5, 0.25]).
        beta_laplacian: Relative weight for second-order Laplacian curvature loss (default: 0.5).
        eps: Epsilon parameter for smooth Charbonnier gradient distance (default: 1e-4).
    """

    def __init__(
        self,
        scales: int = 3,
        scale_weights: Optional[List[float]] = None,
        beta_laplacian: float = 0.5,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.scales = int(scales)
        self.scale_weights = (
            [1.0 / (2**s) for s in range(self.scales)]
            if scale_weights is None
            else [float(w) for w in scale_weights]
        )
        self.beta_laplacian = float(beta_laplacian)
        self.eps = float(eps)

        # Sobel-Feldman horizontal and vertical kernels
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3) / 8.0

        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3) / 8.0

        # Discrete 3x3 isotropic 8-connected Laplacian operator
        laplacian = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3) / 8.0

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian", laplacian)

    def _gradients(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes horizontal Sobel, vertical Sobel, and Laplacian spatial derivatives."""
        gx = F.conv2d(img, self.sobel_x, padding=1)
        gy = F.conv2d(img, self.sobel_y, padding=1)
        lap = F.conv2d(img, self.laplacian, padding=1)
        return gx, gy, lap

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Computes multi-scale Sobolev gradient loss between restored prediction and target.

        Args:
            pred: Restored HR image tensor (B, 1, H, W).
            target: Ground truth HR image tensor (B, 1, H, W).

        Returns:
            Scalar Sobolev gradient loss.
        """
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        curr_pred = pred
        curr_target = target

        for s in range(self.scales):
            w = self.scale_weights[s] if s < len(self.scale_weights) else (1.0 / (2**s))

            pred_gx, pred_gy, pred_lap = self._gradients(curr_pred)
            tgt_gx, tgt_gy, tgt_lap = self._gradients(curr_target)

            # First-order H^1 Sobolev gradient Charbonnier distance
            diff_gx = pred_gx - tgt_gx
            diff_gy = pred_gy - tgt_gy
            h1_dist = torch.sqrt(diff_gx * diff_gx + diff_gy * diff_gy + self.eps * self.eps)

            # Second-order H^2 Sobolev Laplacian Charbonnier distance
            diff_lap = pred_lap - tgt_lap
            h2_dist = torch.sqrt(diff_lap * diff_lap + self.eps * self.eps)

            scale_loss = torch.mean(h1_dist) + self.beta_laplacian * torch.mean(h2_dist)
            total_loss = total_loss + w * scale_loss

            # Downsample for next dyadic scale
            if s < self.scales - 1:
                curr_pred = F.avg_pool2d(curr_pred, kernel_size=2, stride=2)
                curr_target = F.avg_pool2d(curr_target, kernel_size=2, stride=2)

        return total_loss


# Convenient alias
SobolevGradientLoss = MultiScaleSobolevGradientLoss
