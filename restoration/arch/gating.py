"""Evidence-DAR Gating and Degradation-Suppressed Trunk Modules.

This module implements the core architectural components for the Evidence-DAR restoration trunk:
1. SimpleGate: Channel chunk elementwise multiplication gate.
2. GatedResidualBlock: High-throughput gated residual block with SimpleGate and clamping bounds.
3. MSTSuppressionGate: Modulation Suppression Transformation gate for proactive noise suppression.
4. DegradationModulation: Continuous degradation affine feature modulation.
5. AdaptiveTimescaleDynamics: Velocity scaling factor lambda_dyn in (0, 1) from degradation descriptor.
6. StageDecomposer: Disentangles latent representations into degradation (F_d) and content (F_c).
7. EvidenceDARStage: Unified stage combining modulation, decomposition, gating, dynamics, and blocks.
8. EvidenceDARTrunk: Multi-stage degradation-adaptive restoration trunk.
"""

from __future__ import annotations

from typing import List, Tuple
import torch
from torch import nn


class SimpleGate(nn.Module):
    """Elementwise multiplication gate dividing channels into two equal halves."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class GatedResidualBlock(nn.Module):
    """Lightweight gated residual convolution block with SimpleGate non-linearity.

    Args:
        channels: Number of latent feature channels (default: 64).
        kernel_size: Spatial kernel size for depthwise convolution (default: 3).
        clamp_val: Maximum absolute activation magnitude for numerical stability (default: 256.0).
    """

    def __init__(
        self,
        channels: int = 64,
        kernel_size: int = 3,
        clamp_val: float = 256.0,
    ) -> None:
        super().__init__()
        self.expand = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)
        self.spatial = nn.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels * 2,
            bias=True,
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.clamp_val = clamp_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.expand(x)
        spatial_out = self.spatial(expanded)
        left, right = spatial_out.chunk(2, dim=1)
        gated = (left * right).clamp(-self.clamp_val, self.clamp_val)
        return x + self.scale * self.project(gated)


class MSTSuppressionGate(nn.Module):
    """Modulation Suppression Transformation (MST) Gate.

    Proactively suppresses degradation features from content representations by
    computing a spatial-channel confidence mask G_s in (0, 1).
    """

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, fc: torch.Tensor, fd: torch.Tensor) -> torch.Tensor:
        """Applies suppression mask to content features.

        Args:
            fc: Content features (B, C, H, W).
            fd: Degradation features (B, C, H, W).

        Returns:
            Suppressed content features \\tilde{F}_c (B, C, H, W).
        """
        joint = torch.cat([fc, fd], dim=1)
        mask = self.gate_conv(joint)
        return fc * mask


class DegradationModulation(nn.Module):
    """Degradation-conditioned affine feature modulation.

    Generates channel-wise scaling (gamma) and shifting (beta) parameters from S_deg:
    F_mod = (1 + gamma) * F + beta
    """

    def __init__(self, deg_dim: int = 64, channels: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(deg_dim, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2),
        )
        # Initialize final linear layer to zero for identity mapping at step 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, f: torch.Tensor, S_deg: torch.Tensor) -> torch.Tensor:
        params = self.mlp(S_deg).view(S_deg.size(0), -1, 1, 1)
        gamma, beta = params.chunk(2, dim=1)
        return (1.0 + gamma) * f + beta


class AdaptiveTimescaleDynamics(nn.Module):
    """Predicts dynamic velocity scaling factor lambda_dyn in (0, 1) from S_deg."""

    def __init__(self, deg_dim: int = 64, channels: int = 64) -> None:
        super().__init__()
        self.velocity_net = nn.Sequential(
            nn.Linear(deg_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, S_deg: torch.Tensor) -> torch.Tensor:
        return self.velocity_net(S_deg).view(S_deg.size(0), -1, 1, 1)


class StageDecomposer(nn.Module):
    """Explicit Feature Decomposer: F_d = P_deg(F_mod), F_c = F_orig - F_d."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.proj_deg = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        # Initialize final conv to zero for step-0 identity
        nn.init.zeros_(self.proj_deg[-1].weight)
        nn.init.zeros_(self.proj_deg[-1].bias)

    def forward(
        self, f_mod: torch.Tensor, f_orig: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fd = self.proj_deg(f_mod)
        fc = f_orig - fd
        return fd, fc


class EvidenceDARStage(nn.Module):
    """A single Evidence-DAR stage combining modulation, decomposition, MST gating,
    adaptive timescale dynamics, and lightweight gated residual convolution blocks.
    """

    def __init__(
        self,
        channels: int = 64,
        num_blocks: int = 3,
        deg_dim: int = 64,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.modulate = DegradationModulation(deg_dim=deg_dim, channels=channels)
        self.decomposer = StageDecomposer(channels=channels)
        self.mst_gate = MSTSuppressionGate(channels=channels)
        self.dynamics = AdaptiveTimescaleDynamics(deg_dim=deg_dim, channels=channels)
        self.blocks = nn.Sequential(
            *(
                GatedResidualBlock(channels=channels, kernel_size=kernel_size)
                for _ in range(num_blocks)
            )
        )
        self.tail = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(
        self, f: torch.Tensor, S_deg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_mod = self.modulate(f, S_deg)
        fd, fc = self.decomposer(f_mod, f)
        fc_gated = self.mst_gate(fc, fd)
        lambda_dyn = self.dynamics(S_deg)
        f_stage = self.blocks(fc_gated)
        f_next = f + lambda_dyn * self.tail(f_stage)
        return f_next, fd, fc


class EvidenceDARTrunk(nn.Module):
    """Complete Evidence-DAR Multi-Stage Restoration Trunk.

    Composed of N stages with B blocks per stage, operating entirely at low resolution.
    """

    def __init__(
        self,
        channels: int = 64,
        num_stages: int = 4,
        blocks_per_stage: int = 3,
        deg_dim: int = 64,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList([
            EvidenceDARStage(
                channels=channels,
                num_blocks=blocks_per_stage,
                deg_dim=deg_dim,
                kernel_size=kernel_size,
            )
            for _ in range(num_stages)
        ])
        self.trunk_tail = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=True
        )

    def forward(
        self, f: torch.Tensor, S_deg: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        fds: List[torch.Tensor] = []
        fcs: List[torch.Tensor] = []
        for stage in self.stages:
            f, fd, fc = stage(f, S_deg)
            fds.append(fd)
            fcs.append(fc)
        f_out = f + self.trunk_tail(f)
        return f_out, fds, fcs
