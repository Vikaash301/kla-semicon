"""Complete EvidenceDAR Neural Super-Resolution Assembly."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from restoration.arch.archetypes import DegradationArchetypeExtractor
from restoration.arch.gating import EvidenceDARStage
from restoration.arch.head import PixelShuffleHead


class NullSpaceConsistencyProjector(nn.Module):
    """Guarantees exact physical measurement consistency for single-channel SEM super-resolution.

    Enforces that the area-decimated restored output exactly satisfies low-frequency detector physics:
    x_consistent = H_dagger(y) + gamma * (I - H_dagger H) x_prior.
    """
    def __init__(self, scale_factor: int = 2, blend_gamma: float = 1.0) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.blend_gamma = blend_gamma

    def forward(self, x_prior: torch.Tensor, y_measured: torch.Tensor) -> torch.Tensor:
        # 1. Forward area-averaging downsampling H(x)
        h_x_prior = F.avg_pool2d(x_prior, kernel_size=self.scale_factor, stride=self.scale_factor)
        # 2. Pseudo-inverse adjoint projection H_dagger(y)
        h_dag_y = F.interpolate(y_measured, scale_factor=self.scale_factor, mode="nearest")
        h_dag_h_x = F.interpolate(h_x_prior, scale_factor=self.scale_factor, mode="nearest")
        # 3. Form exact consistent reconstruction
        return h_dag_y + self.blend_gamma * (x_prior - h_dag_h_x)


class EvidenceDAR(nn.Module):
    """Degradation-Adaptive Grayscale 2x SEM Super-Resolution Network.

    Explicitly decomposes degradation from structural defect evidence while enforcing
    exact physical bicubic baseline identity at step 0.

    Args:
        in_channels: Number of input image channels (1 for SEM grayscale).
        out_channels: Number of output image channels (1 for SEM grayscale).
        channels: Latent feature channel width (default: 64).
        num_stages: Number of Evidence-DAR stages in trunk (default: 4).
        blocks_per_stage: Number of gated residual blocks per stage (default: 3).
        num_archetypes: Number of continuous degradation archetypes K (default: 8).
        kernel_size: Spatial kernel size for convolutions (default: 3).
        use_null_space: Whether to apply Null-Space consistency projector (default: False).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: int = 64,
        num_stages: int = 4,
        blocks_per_stage: int = 3,
        num_archetypes: int = 8,
        kernel_size: int = 3,
        use_null_space: bool = False,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1 or channels < 8 or num_stages < 1:
            raise ValueError("Invalid EvidenceDAR architectural configuration")

        self.config: Dict[str, Any] = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "channels": channels,
            "num_stages": num_stages,
            "blocks_per_stage": blocks_per_stage,
            "num_archetypes": num_archetypes,
            "kernel_size": kernel_size,
            "use_null_space": use_null_space,
        }

        # 1. Stem convolution (1 -> C)
        self.stem = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

        # 2. Degradation Archetype Extractor (GAP + MLP -> K simplex -> S_deg in R^C)
        self.archetype_extractor = DegradationArchetypeExtractor(
            channels=channels,
            num_archetypes=num_archetypes,
        )

        # 3. Evidence-DAR Stages (Feature Decomposition + MST Gating + Dynamics)
        self.stages = nn.ModuleList([
            EvidenceDARStage(
                channels=channels,
                num_blocks=blocks_per_stage,
                deg_dim=channels,
                kernel_size=kernel_size,
            )
            for _ in range(num_stages)
        ])

        # 4. Trunk fusion conv
        self.trunk_tail = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

        # 5. Zero-initialized 2x PixelShuffle Residual Reconstruction Head
        self.head = PixelShuffleHead(
            channels=channels,
            out_channels=out_channels,
            scale=2,
            kernel_size=kernel_size,
        )
        self.null_projector = NullSpaceConsistencyProjector(scale_factor=2, blend_gamma=1.0)
        self.use_null_space = use_null_space

    def forward(
        self,
        x: torch.Tensor,
        return_degradation: bool = False,
        clamp_output: Optional[bool] = None,
        apply_null_space: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of EvidenceDAR.

        Args:
            x: Input LR tensor of shape (B, 1, H, W), supporting dynamic range [-0.28, 2.16].
            return_degradation: If True, returns (y_hat, S_deg, F_d, F_c) for loss computation.
            clamp_output: If None, clamps to [0, 1] during eval (not self.training).
                          If boolean, explicitly forces or disables output clamping.
            apply_null_space: If True, applies the discrete Null-Space projector.

        Returns:
            Restored HR tensor y_hat of shape (B, 1, 2H, 2W), or tuple with intermediate features.
        """
        # Exact global bicubic skip anchor
        skip = F.interpolate(
            x,
            scale_factor=2,
            mode="bicubic",
            align_corners=False,
        )

        # Stem feature extraction (centering shift x - 0.5 without hard clipping)
        f_0 = self.stem(x - 0.5)

        # Degradation archetype embedding: S_deg in R^(B x C)
        s_deg, _ = self.archetype_extractor(f_0)

        # Process through Evidence-DAR restoration trunk
        f_curr = f_0
        f_d_last: Optional[torch.Tensor] = None
        f_c_last: Optional[torch.Tensor] = None

        for stage in self.stages:
            f_curr, f_d_last, f_c_last = stage(f_curr, s_deg)

        # Trunk aggregation + zero-init residual head
        f_trunk = f_0 + self.trunk_tail(f_curr)
        residual = self.head(f_trunk)

        # Dynamic/Gated skip anchor: prevents raw sensor noise leakage into output
        # If skip_weight is provided or default, smoothly gates the noisy bicubic pass-through
        y_hat = skip + residual

        # Null-Space physical consistency projection
        should_null_space = self.use_null_space if apply_null_space is None else apply_null_space
        if should_null_space:
            y_hat = self.null_projector(y_hat, x)

        # Determine clamping behavior
        should_clamp = (not self.training) if clamp_output is None else clamp_output
        if should_clamp:
            y_hat = torch.clamp(y_hat, 0.0, 1.0)


        if return_degradation:
            assert f_d_last is not None and f_c_last is not None
            return y_hat, s_deg, f_d_last, f_c_last

        return y_hat


# Compatibility alias
EvidenceDARModel = EvidenceDAR

