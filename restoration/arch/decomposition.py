"""Explicit Feature Decomposer for Evidence-DAR SEM-SR.

Disentangles intermediate features into degradation and content components:
    F_mod = (1 + gamma) * F + beta  (when s_deg is provided)
    F_d = P_deg(F_mod) = Conv1x1(GELU(DepthwiseConv3x3(F_mod)))
    F_c = F - F_d  (Exact Conservation Split)
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn


class ExplicitFeatureDecomposer(nn.Module):
    """Explicit Feature Decomposer with Affine Modulation and Conservation Split."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.channels = channels

        # Affine modulation generator mapping S_deg (B, C) -> (gamma, beta) (B, 2C)
        self.mod_generator = nn.Linear(channels, channels * 2)

        # Lightweight Depthwise-Separable Degradation Projector P_deg
        self.proj_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )
        self.act = nn.GELU()
        self.proj_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters to guarantee exact step-0 identity baseline."""
        # Zero-init modulation generator: gamma = 0, beta = 0 -> F_mod = F
        nn.init.zeros_(self.mod_generator.weight)
        nn.init.zeros_(self.mod_generator.bias)

        # Depthwise conv Kaiming normal
        nn.init.kaiming_normal_(self.proj_dw.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.proj_dw.bias)

        # Zero-init pointwise conv: F_d = 0 -> F_c = F (exact identity)
        nn.init.zeros_(self.proj_pw.weight)
        nn.init.zeros_(self.proj_pw.bias)

    def forward(
        self, f: torch.Tensor, s_deg: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform affine degradation modulation and explicit conservation split.

        Args:
            f: Input feature tensor of shape (B, C, H, W).
            s_deg: Optional degradation descriptor of shape (B, C).

        Returns:
            Tuple of:
                - f_d: Degradation feature component of shape (B, C, H, W).
                - f_c: Content feature component of shape (B, C, H, W) where f_c = f - f_d.
        """
        if s_deg is not None:
            b, c, _, _ = f.shape
            # Generate Affine Modulation Parameters (B, 2C, 1, 1)
            mod = self.mod_generator(s_deg).view(b, c * 2, 1, 1)
            gamma, beta = mod.chunk(2, dim=1)
            f_mod = (1.0 + gamma) * f + beta
        else:
            f_mod = f

        # Project into degradation component F_d via P_deg
        f_d = self.proj_pw(self.act(self.proj_dw(f_mod)))

        # Exact Conservation Split: F_c = F - F_d
        f_c = f - f_d

        return f_d, f_c


# Compatibility alias
FeatureDecomposer = ExplicitFeatureDecomposer
