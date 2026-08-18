"""Defect Invariance Regularization Loss for Evidence-DAR SEM-SR.

Enforces that degradation representations (simplex archetype descriptors S_deg and
degradation feature maps F_d) remain invariant under localized structural defect variations
(e.g., simulated bridge, void, or line-edge perturbations):
    L_defect = ||S_deg(x) - S_deg(x + delta)||_2^2 + ||F_d(x) - F_d(x + delta)||_1
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DefectInvarianceLoss(nn.Module):
    """Defect-Invariance Regularization Loss.

    Supervises degradation descriptors so they decouple from semiconductor geometry
    and do not overfit to pattern defect variations.

    Args:
        lambda_s: Weight multiplier for S_deg L2 invariance term (default: 1.0).
        lambda_fd: Weight multiplier for F_d L1 invariance term (default: 1.0).
        reduction: Reduction method: 'mean' or 'sum' (default: 'mean').
    """

    def __init__(
        self,
        lambda_s: float = 1.0,
        lambda_fd: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.lambda_s = float(lambda_s)
        self.lambda_fd = float(lambda_fd)
        self.reduction = reduction

    def forward(
        self,
        s_deg: torch.Tensor,
        f_d: torch.Tensor,
        s_deg_perturbed: Optional[torch.Tensor] = None,
        f_d_perturbed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for Defect Invariance Loss.

        Args:
            s_deg: Archetype degradation descriptor vector (B, D) or (B, C).
            f_d: Degradation feature tensor (B, C, H, W).
            s_deg_perturbed: Descriptor from perturbed input (B, D) or (B, C).
            f_d_perturbed: Feature tensor from perturbed input (B, C, H, W).

        Returns:
            Scalar invariance penalty tensor.
        """
        loss_s = torch.tensor(0.0, device=s_deg.device, dtype=s_deg.dtype)
        loss_fd = torch.tensor(0.0, device=f_d.device, dtype=f_d.dtype)

        # 1. Simplex archetype invariance L2 term: ||S_deg(x) - S_deg(x+delta)||_2^2
        if s_deg_perturbed is not None:
            diff_s = s_deg - s_deg_perturbed
            if self.reduction == "mean":
                loss_s = torch.mean(diff_s.pow(2))
            else:
                loss_s = torch.sum(diff_s.pow(2))

        # 2. Degradation feature invariance L1 term: ||F_d(x) - F_d(x+delta)||_1
        if f_d_perturbed is not None:
            diff_fd = f_d - f_d_perturbed
            if self.reduction == "mean":
                loss_fd = torch.mean(torch.abs(diff_fd))
            else:
                loss_fd = torch.sum(torch.abs(diff_fd))

        total_loss = self.lambda_s * loss_s + self.lambda_fd * loss_fd

        # If neither perturbed tensor was provided, maintain gradient connection
        if s_deg_perturbed is None and f_d_perturbed is None:
            total_loss = total_loss + (s_deg * 0.0).sum() + (f_d * 0.0).sum()

        return total_loss
