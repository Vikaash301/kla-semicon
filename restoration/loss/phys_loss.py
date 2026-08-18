"""Physical Re-degradation Consistency Loss for SEM Metrology.

Enforces probabilistic physical measurement consistency comparing the re-degraded
super-resolved prediction with the measured low-resolution electron microscope image:
    L_phys = -log p(y_LR | A(x_hat))
under calibrated heteroscedastic noise:
    sigma(s) = 0.0233 * (s / 0.1)^0.836
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalRedegradationLoss(nn.Module):
    """Heteroscedastic Gaussian NLL Physical Re-degradation Loss.

    Passes the reconstructed high-resolution image through the microscope forward decimation
    operator A(x_hat) and scores measurement log-likelihood under signal-dependent sensor noise.

    Args:
        noise_scale: Base noise standard deviation at reference intensity s=0.1 (default: 0.0233).
        power_exponent: Power-law exponent for signal dependence (default: 0.836).
        s_ref: Reference signal level (default: 0.1).
        s_min: Floor clamping value to prevent singular noise variances (default: 0.02).
        include_constant: Whether to include the constant 0.5 * log(2 * pi) term (default: False).
        reduction: Reduction method: 'mean', 'sum', or 'none' (default: 'mean').
    """

    def __init__(
        self,
        noise_scale: float = 0.0233,
        power_exponent: float = 0.836,
        s_ref: float = 0.1,
        s_min: float = 0.02,
        include_constant: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.noise_scale = float(noise_scale)
        self.power_exponent = float(power_exponent)
        self.s_ref = float(s_ref)
        self.s_min = float(s_min)
        self.include_constant = include_constant
        self.reduction = reduction

    def compute_sigma(self, signal: torch.Tensor) -> torch.Tensor:
        """Computes signal-dependent standard deviation sigma(s) = 0.0233 * (s / 0.1)^0.836."""
        s_clamped = torch.clamp(signal, min=self.s_min)
        sigma = self.noise_scale * (s_clamped / self.s_ref) ** self.power_exponent
        return torch.clamp(sigma, min=1e-4)

    def forward(
        self,
        pred_hr: torch.Tensor,
        lr_input: torch.Tensor,
        forward_op: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass for Physical Re-degradation Loss.

        Args:
            pred_hr: Reconstructed high-resolution prediction tensor (B, 1, 2H, 2W).
            lr_input: Measured low-resolution input tensor (B, 1, H, W).
            forward_op: Optional custom forward decimation operator (default: 2x avg_pool2d).

        Returns:
            Scalar NLL loss tensor.
        """
        # Apply forward decimation operator
        if forward_op is not None:
            re_degraded = forward_op(pred_hr)
        else:
            re_degraded = F.avg_pool2d(pred_hr, kernel_size=2, stride=2)

        # Compute heteroscedastic noise standard deviation
        # Detach variance to avoid gradient feedback on noise estimation
        sigma = self.compute_sigma(re_degraded.detach())

        # Heteroscedastic Gaussian Negative Log-Likelihood
        residual = lr_input - re_degraded
        nll = 0.5 * (residual / sigma).pow(2) + torch.log(sigma)

        if self.include_constant:
            nll = nll + 0.5 * math.log(2.0 * math.pi)

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        return nll
