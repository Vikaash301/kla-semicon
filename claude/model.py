"""Channel-attention residual restoration network for 1x128x128 -> 1x256x256."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, res_scale: float = 0.1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            ChannelAttention(channels, reduction),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.body(x)


class ResidualGroup(nn.Module):
    def __init__(self, channels: int, num_blocks: int, reduction: int, res_scale: float) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *(ResidualChannelAttentionBlock(channels, reduction, res_scale) for _ in range(num_blocks))
        )
        self.tail = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.tail(self.blocks(x))


class RestorationNet(nn.Module):
    """Bicubic-anchored residual network; predicts the 2x residual on top of bicubic."""

    def __init__(
        self,
        channels: int = 96,
        num_groups: int = 6,
        blocks_per_group: int = 6,
        reduction: int = 16,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = {
            "channels": channels,
            "num_groups": num_groups,
            "blocks_per_group": blocks_per_group,
            "reduction": reduction,
            "res_scale": res_scale,
        }
        self.head = nn.Conv2d(1, channels, 3, padding=1)
        self.body = nn.Sequential(
            *(ResidualGroup(channels, blocks_per_group, reduction, res_scale) for _ in range(num_groups))
        )
        self.body_tail = nn.Conv2d(channels, channels, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
        # Start as an exact bicubic identity so early training cannot regress.
        nn.init.zeros_(self.upsample[-1].weight)
        nn.init.zeros_(self.upsample[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        features = self.head(x - 0.5)
        features = features + self.body_tail(self.body(features))
        return skip + self.upsample(features)


def self_ensemble(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Average predictions over the 8 dihedral transforms (test-time augmentation)."""
    outputs = []
    for flip_h in (False, True):
        for flip_v in (False, True):
            for transpose in (False, True):
                sample = x
                if flip_h:
                    sample = sample.flip(-1)
                if flip_v:
                    sample = sample.flip(-2)
                if transpose:
                    sample = sample.transpose(-1, -2)
                out = model(sample)
                if transpose:
                    out = out.transpose(-1, -2)
                if flip_v:
                    out = out.flip(-2)
                if flip_h:
                    out = out.flip(-1)
                outputs.append(out)
    return torch.stack(outputs).mean(0)
