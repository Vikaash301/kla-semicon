"""Compact grayscale 2x restoration network."""

import torch
from torch import nn
from torch.nn import functional as F


class GatedResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.spatial = nn.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels * 2,
        )
        self.project = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        left, right = self.spatial(self.expand(input_tensor)).chunk(2, dim=1)
        gated = (left * right).clamp(-256.0, 256.0)
        return input_tensor + self.scale * self.project(gated)


class CompactRestorationNet(nn.Module):
    """Restore a Bx1xHxW image to Bx1x(2H)x(2W)."""

    def __init__(
        self,
        channels: int = 32,
        num_blocks: int = 6,
        scale: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if channels < 1 or num_blocks < 1 or scale != 2 or kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(
                "channels and num_blocks must be positive, scale must be 2, and kernel_size must be odd and >=3"
            )

        self.config = {
            "channels": channels,
            "num_blocks": num_blocks,
            "scale": scale,
            "kernel_size": kernel_size,
        }
        self.stem = nn.Conv2d(1, channels, 3, padding=1)
        self.trunk = nn.Sequential(
            *(GatedResidualBlock(channels, kernel_size) for _ in range(num_blocks))
        )
        self.head = nn.Sequential(
            nn.Conv2d(channels, scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )
        nn.init.zeros_(self.head[0].weight)
        nn.init.zeros_(self.head[0].bias)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.clamp(-16.0, 16.0)
        skip = F.interpolate(
            input_tensor, scale_factor=2, mode="bicubic", align_corners=False
        )
        residual = self.head(self.trunk(self.stem(input_tensor)))
        return skip + residual
