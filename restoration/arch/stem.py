"""Physical Radiometric Stem Encoders for Unclipped SEM Micrographs."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPhysicalStem(nn.Module):
    """Dual-Stream Physical Radiometric Stem Encoder.

    Processes unclipped SEM flux in dynamic range [-0.28, 2.16] through parallel:
    1. Linear radiometric flux stream: conv_lin(x)
    2. Signed-log flux stream: conv_log(sign(x) * log(1 + |x|))
    Fusing both streams via pointwise 1x1 convolution.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 64, kernel_size: int = 3) -> None:
        super().__init__()
        mid_dim = out_channels // 2
        self.conv_lin = nn.Conv2d(
            in_channels, mid_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=True
        )
        self.conv_log = nn.Conv2d(
            in_channels, mid_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=True
        )
        self.fuse = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear radiometric branch
        feat_lin = F.silu(self.conv_lin(x))
        # Signed log branch (linearizes multiplicative speckle)
        x_log = torch.sign(x) * torch.log1p(torch.abs(x))
        feat_log = F.silu(self.conv_log(x_log))
        # Concatenate and project
        return self.fuse(torch.cat([feat_lin, feat_log], dim=1))
