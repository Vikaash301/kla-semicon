"""Orthogonal Subspace Rectification Loss (OSR) for Evidence-DAR.

Enforces geometric orthogonality between degradation representation F_d and content
representation F_c to minimize mutual information I(F_d; F_c) -> 0:
    L_OSR = ||X_d X_c^T||_F^2 / (B * H * W)^2
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OrthogonalSubspaceRectificationLoss(nn.Module):
    """Orthogonal Subspace Rectification Loss.

    Penalizes cross-talk and linear dependence between the degradation feature subspace F_d
    and content feature subspace F_c via Frobenius norm penalty on their cross-correlation.

    Args:
        per_sample: If True, computes orthogonality per batch item and averages;
                    if False, pools across the whole batch (default: False).
        normalize: If True, normalizes by (spatial_elements)^2 (default: True).
        eps: Small epsilon for numerical safeguards (default: 1e-8).
    """

    def __init__(
        self,
        per_sample: bool = False,
        normalize: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.per_sample = per_sample
        self.normalize = normalize
        self.eps = eps

    def forward(
        self,
        f_d: torch.Tensor,
        f_c: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for OSR Loss.

        Args:
            f_d: Degradation feature tensor (B, C, H, W) or (C, N).
            f_c: Content feature tensor (B, C, H, W) or (C, N).

        Returns:
            Scalar Frobenius norm penalty tensor.
        """
        if f_d.shape != f_c.shape:
            raise ValueError(f"Shape mismatch in OSR loss: f_d={f_d.shape} vs f_c={f_c.shape}")

        if f_d.ndim == 4:
            b, c, h, w = f_d.shape
            if self.per_sample:
                # Per-sample cross correlation
                x_d = f_d.view(b, c, h * w)  # (B, C, HW)
                x_c = f_c.view(b, c, h * w)  # (B, C, HW)
                cross = torch.bmm(x_d, x_c.transpose(1, 2))  # (B, C, C)
                fro_sq = torch.sum(cross * cross, dim=(-2, -1))  # (B,)
                n_elem = float(h * w)
                norm_factor = (n_elem * n_elem) if self.normalize else 1.0
                return torch.mean(fro_sq / norm_factor)
            else:
                # Global batch flattened: (C, BHW)
                x_d = f_d.permute(1, 0, 2, 3).reshape(c, -1)  # (C, BHW)
                x_c = f_c.permute(1, 0, 2, 3).reshape(c, -1)  # (C, BHW)
                cross = torch.matmul(x_d, x_c.t())  # (C, C)
                fro_sq = torch.sum(cross * cross)
                n_elem = float(b * h * w)
                norm_factor = (n_elem * n_elem) if self.normalize else 1.0
                return fro_sq / norm_factor

        elif f_d.ndim == 2:
            # Assumed shape: (C, N)
            c, n = f_d.shape
            cross = torch.matmul(f_d, f_c.t())  # (C, C)
            fro_sq = torch.sum(cross * cross)
            norm_factor = float(n * n) if self.normalize else 1.0
            return fro_sq / norm_factor

        elif f_d.ndim == 3:
            # Assumed shape: (B, C, N)
            b, c, n = f_d.shape
            cross = torch.bmm(f_d, f_c.transpose(1, 2))  # (B, C, C)
            fro_sq = torch.sum(cross * cross, dim=(-2, -1))  # (B,)
            norm_factor = float(n * n) if self.normalize else 1.0
            return torch.mean(fro_sq / norm_factor)

        else:
            # Arbitrary dimension fallback
            c = f_d.shape[1] if f_d.ndim > 1 else f_d.shape[0]
            x_d = f_d.reshape(c, -1)
            x_c = f_c.reshape(c, -1)
            cross = torch.matmul(x_d, x_c.t())
            fro_sq = torch.sum(cross * cross)
            norm_factor = float(x_d.shape[1] ** 2) if self.normalize else 1.0
            return fro_sq / norm_factor
