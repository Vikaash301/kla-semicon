"""Log Focal Frequency Loss (LFFL) for Microscopy Super-Resolution.

Supervises sparse high-frequency semiconductor structures and suppresses spectral
over-smoothing without amplifying stochastic sensor noise:
    L_LFFL = (1 / |Omega_f|) * sum_{u, v} w(u, v) * |log(1 + |F_pred(u, v)|) - log(1 + |F_target(u, v)|)|^p
where:
    w(u, v) = (|F_pred(u, v) - F_target(u, v)| / (max_{u, v}|F_pred - F_target| + eps_w))^alpha
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogFocalFrequencyLoss(nn.Module):
    """Microscopy-tailored Log Focal Frequency Loss.

    Computes 2D Fourier spectra differences in log magnitude domain with dynamic focal
    frequency weighting to penalize unresolved high-frequency harmonics and pitch errors.

    Args:
        alpha: Focal weighting scaling exponent (default: 1.0). When alpha=0, weighting is uniform.
        p: Power exponent on the spectral log difference (default: 2.0 for L2 squared error).
        norm: FFT normalization mode: 'ortho', 'forward', or 'backward' (default: 'ortho').
        reduction: Reduction method: 'mean', 'sum', or 'none' (default: 'mean').
        eps: Epsilon for magnitude stability and max-normalization denominator (default: 1e-6).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        p: float = 2.0,
        norm: str = "ortho",
        reduction: str = "mean",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.p = float(p)
        self.norm = norm
        self.reduction = reduction
        self.eps = float(eps)

    def _fft_log_magnitude(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes 2D real FFT magnitude and log(1 + magnitude).

        Returns:
            mag: Complex magnitude |F(u, v)| = sqrt(real^2 + imag^2 + eps).
            log_mag: Log-scaled magnitude log(1 + |F(u, v)|).
        """
        # Ensure float32 for FFT operations
        fft_c = torch.fft.rfft2(x.float(), norm=self.norm)
        real, imag = fft_c.real, fft_c.imag
        mag = torch.sqrt(real * real + imag * imag + self.eps**2)
        log_mag = torch.log1p(mag)
        return mag, log_mag

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for Log Focal Frequency Loss.

        Args:
            pred: Predicted super-resolved image tensor (B, C, H, W).
            target: Ground truth target image tensor (B, C, H, W).

        Returns:
            Scalar loss tensor (if reduction='mean').
        """
        # Compute spectra
        mag_pred, log_mag_pred = self._fft_log_magnitude(pred)
        with torch.no_grad():
            mag_target, log_mag_target = self._fft_log_magnitude(target)

        # Log spectral difference
        log_diff = log_mag_pred - log_mag_target

        if self.p == 2.0:
            diff_term = log_diff.pow(2)
        elif self.p == 1.0:
            diff_term = torch.abs(log_diff)
        else:
            diff_term = torch.abs(log_diff).pow(self.p)

        # Dynamic focal frequency weighting
        if self.alpha > 0.0:
            with torch.no_grad():
                freq_err = torch.abs(mag_pred - mag_target)
                # Max-normalize per-channel/image
                max_err = freq_err.amax(dim=(-2, -1), keepdim=True)
                weight = (freq_err / (max_err + self.eps)).pow(self.alpha)
                # Keep baseline floor to supervise all frequencies
                weight = weight + 1.0
        else:
            weight = torch.ones_like(diff_term)

        weighted_loss = weight * diff_term

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss
