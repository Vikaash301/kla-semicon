"""Late 2x PixelShuffle Residual Reconstruction Head with Zero Initialization."""

from __future__ import annotations

import torch
from torch import nn


class PixelShuffleHead(nn.Module):
    """Reconstructs 2x residual map via zero-initialized sub-pixel convolution.

    Architecture:
        Conv2d(in_channels=channels, out_channels=scale*scale*out_channels, kernel_size=3, padding=1)
        -> PixelShuffle(upscale_factor=scale)
    """

    def __init__(
        self,
        channels: int = 64,
        out_channels: int = 1,
        scale: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if channels < 1 or out_channels < 1 or scale != 2:
            raise ValueError(
                f"Invalid head config: channels={channels}, out_channels={out_channels}, scale={scale}"
            )

        self.channels = channels
        self.out_channels = out_channels
        self.scale = scale

        mid_channels = out_channels * (scale**2)
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale)

        # Exact zero-initialization: guarantees step-0 bicubic baseline identity
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and biases to exact zero."""
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: maps (B, C, H, W) -> (B, out_channels, scale*H, scale*W)."""
        return self.pixel_shuffle(self.conv(x))


class PalasantzasWaveletSRHead(nn.Module):
    """Palasantzas Roughness-Constrained Wavelet Super-Resolution Upscaling Head.

    Synthesizes 2x high-resolution output in 2D Haar wavelet subbands (LL, LH, HL, HH)
    with physical hyperbolic tangent energy bounds on high-frequency detail bands.
    """

    def __init__(
        self,
        channels: int = 64,
        out_channels: int = 1,
        t_max: float = 0.25,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.t_max = t_max

        self.conv_ll = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.conv_lh = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.conv_hl = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.conv_hh = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.conv_ll.weight)
        nn.init.zeros_(self.conv_ll.bias)
        nn.init.zeros_(self.conv_lh.weight)
        nn.init.zeros_(self.conv_lh.bias)
        nn.init.zeros_(self.conv_hl.weight)
        nn.init.zeros_(self.conv_hl.bias)
        nn.init.zeros_(self.conv_hh.weight)
        nn.init.zeros_(self.conv_hh.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll = self.conv_ll(x)
        # Bounded high-frequency subbands to prevent checkerboard artifacts
        lh = self.t_max * torch.tanh(self.conv_lh(x) / (self.t_max + 1e-6))
        hl = self.t_max * torch.tanh(self.conv_hl(x) / (self.t_max + 1e-6))
        hh = (self.t_max * 0.7071) * torch.tanh(self.conv_hh(x) / (self.t_max * 0.7071 + 1e-6))

        # Differentiable 2D Haar Inverse Wavelet Transform
        b, c, h, w = ll.shape
        out = torch.zeros((b, c, 2 * h, 2 * w), dtype=ll.dtype, device=ll.device)
        out[:, :, 0::2, 0::2] = 0.5 * (ll - lh - hl + hh)
        out[:, :, 0::2, 1::2] = 0.5 * (ll - lh + hl - hh)
        out[:, :, 1::2, 0::2] = 0.5 * (ll + lh - hl - hh)
        out[:, :, 1::2, 1::2] = 0.5 * (ll + lh + hl + hh)
        return out


# Compatibility alias
ZeroInitPixelShuffleHead = PixelShuffleHead

