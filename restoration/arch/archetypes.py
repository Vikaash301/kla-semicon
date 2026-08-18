"""Degradation Archetype Extractor for Evidence-DAR SEM-SR.

Extracts a continuous simplex-projected degradation subspace descriptor:
    S_deg = A * alpha in R^{B x C}
where A in R^{C x K} (K=8 archetypes) and alpha = Softmax(MLP(GAP(F_0))) in Delta^{K-1}.
"""

from __future__ import annotations

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DegradationArchetypeExtractor(nn.Module):
    """Continuous Degradation Archetype Extractor with Simplex Projection.

    Projects global stem representations onto a learnable degradation archetype basis
    via a probability simplex mixture alpha in Delta^{K-1}.
    """

    def __init__(
        self,
        channels: int = 64,
        num_archetypes: int = 8,
        hidden_dim: int = 32,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_archetypes = num_archetypes
        self.temperature = temperature

        # 2-Layer MLP mapping GAP(F_0) to archetype mixture logits
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_archetypes),
        )

        # Archetype Basis Dictionary Matrix A in R^{C x K}
        self.archetypes = nn.Parameter(torch.empty(channels, num_archetypes))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters to guarantee uniform initial archetype distribution."""
        # Orthogonal basis initialization for maximal subspace diversity
        nn.init.orthogonal_(self.archetypes)

        # First MLP layer Kaiming normal
        nn.init.kaiming_normal_(self.mlp[0].weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.mlp[0].bias)

        # Zero-initialize final layer so initial logits are zero -> uniform alpha = [1/K, ..., 1/K]
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    @property
    def archetype_basis(self) -> nn.Parameter:
        """Alias for compatibility with archetype tests."""
        return self.archetypes

    def forward(
        self, f0: torch.Tensor, return_alpha: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Extract continuous degradation descriptor S_deg and mixture coefficients alpha.

        Args:
            f0: Stem feature map of shape (B, C, H, W).
            return_alpha: If True, returns (s_deg, alpha). If False, returns s_deg only.

        Returns:
            Tuple of (s_deg, alpha) if return_alpha is True, else s_deg:
                - s_deg: Continuous degradation descriptor of shape (B, C).
                - alpha: Simplex mixture weights of shape (B, K) where sum(alpha, -1) == 1.
        """
        # 1. Global Average Pooling (B, C, H, W) -> (B, C)
        z = f0.mean(dim=(2, 3))

        # 2. MLP Logits -> Softmax Simplex Projection (B, K)
        logits = self.mlp(z)
        alpha = F.softmax(logits / self.temperature, dim=-1)

        # 3. Continuous Archetype Subspace Combination: S_deg = alpha @ A^T -> (B, C)
        s_deg = F.linear(alpha, self.archetypes)

        if return_alpha:
            return s_deg, alpha
        return s_deg


# Compatibility alias
SimplexArchetypeExtractor = DegradationArchetypeExtractor
