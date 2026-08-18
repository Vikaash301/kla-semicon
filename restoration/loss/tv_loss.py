"""Flat-Field Total Variation (TV) Regularizer for Semiconductor SEM Images.

Penalizes high-frequency noise and speckle oscillations exclusively in flat-field regions
(substrate, dielectric background) while preserving sharp nanoscale feature edges and
avoiding line-edge blurring.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlatFieldTVLoss(nn.Module):
    """Flat-Field Weighted Total Variation Regularization Loss.

    Computes total variation on the prediction weighted by an edge-attenuated flat-field
    mask derived from the reference ground truth gradient magnitude.

    Args:
        edge_threshold: Gradient magnitude scale sigma_edge for flat-field detection (default: 0.08).
        eps: Epsilon parameter for smooth Charbonnier total variation (default: 1e-4).
        norm: Weighting formulation ('gaussian' or 'rational', default: 'gaussian').
    """

    def __init__(
        self,
        edge_threshold: float = 0.08,
        eps: float = 1e-4,
        norm: str = "gaussian",
    ) -> None:
        super().__init__()
        self.edge_threshold = float(edge_threshold)
        self.eps = float(eps)
        self.norm = str(norm).lower()

        # Sobel gradient filters for edge detection
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3) / 8.0

        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3) / 8.0

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient_magnitude(self, img: torch.Tensor) -> torch.Tensor:
        """Computes gradient magnitude using Sobel operators."""
        gx = F.conv2d(img, self.sobel_x, padding=1)
        gy = F.conv2d(img, self.sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def compute_flat_weight(self, target: torch.Tensor) -> torch.Tensor:
        """Derives smooth continuous flat-field weight map W_flat in [0, 1].

        Pixels near strong edges (high gradient) get W_flat -> 0 (no TV penalty).
        Pixels in flat background (low gradient) get W_flat -> 1 (strong TV penalty).
        """
        g_mag = self._gradient_magnitude(target)
        if self.norm == "gaussian":
            w_flat = torch.exp(-0.5 * (g_mag / (self.edge_threshold + 1e-6)) ** 2)
        elif self.norm == "rational":
            w_flat = 1.0 / (1.0 + (g_mag / (self.edge_threshold + 1e-6)) ** 2)
        else:
            w_flat = torch.clamp(1.0 - (g_mag / (self.edge_threshold + 1e-6)), 0.0, 1.0)
        return w_flat.detach()

    def forward(
        self,
        pred: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes flat-field TV regularization loss.

        Args:
            pred: Restored prediction tensor (B, 1, H, W).
            target: Optional ground truth reference tensor (B, 1, H, W).
            weight_map: Optional precomputed flat weight map (B, 1, H, W).

        Returns:
            Scalar flat-field TV loss.
        """
        # Determine flat-field weighting
        if weight_map is not None:
            w_flat = weight_map
        elif target is not None:
            w_flat = self.compute_flat_weight(target)
        else:
            # Fallback to self-attenuated flat weight
            w_flat = self.compute_flat_weight(pred.detach())

        # Prediction forward differences for TV
        diff_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        diff_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        # Pad differences to match spatial dimensions
        diff_x = F.pad(diff_x, (0, 1, 0, 0), mode="replicate")
        diff_y = F.pad(diff_y, (0, 0, 0, 1), mode="replicate")

        # Local Charbonnier variation
        tv_density = torch.sqrt(diff_x * diff_x + diff_y * diff_y + self.eps * self.eps)

        # Weighted flat-field total variation
        weighted_tv = w_flat * tv_density
        loss = torch.sum(weighted_tv) / (torch.sum(w_flat) + 1e-6)
        return loss


# Convenient alias
FlatFieldTVRegularizer = FlatFieldTVLoss
