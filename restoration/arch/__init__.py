"""Evidence-DAR SEM-SR Neural Architecture Package.

Provides degradation-adaptive neural super-resolution components:
- Archetypes: Continuous simplex-projected degradation subspace extractor (S_deg = A * alpha)
- Decomposition: Explicit linear conservation split (F_d = P_deg(F_mod), F_c = F - F_d)
- Gating: MST suppression gate, adaptive timescale velocity dynamics, and gated residual blocks
- Head: Late 2x sub-pixel convolution head with zero initialization
- Assembly: Complete EvidenceDAR network with bicubic skip anchor and intermediate feature hooks
"""

from __future__ import annotations

from restoration.arch.archetypes import (
    DegradationArchetypeExtractor,
    SimplexArchetypeExtractor,
)
from restoration.arch.decomposition import (
    ExplicitFeatureDecomposer,
    FeatureDecomposer,
)
from restoration.arch.evidence_dar import (
    EvidenceDAR,
    EvidenceDARModel,
)
from restoration.arch.gating import (
    AdaptiveTimescaleDynamics,
    DegradationModulation,
    EvidenceDARStage,
    EvidenceDARTrunk,
    GatedResidualBlock,
    MSTSuppressionGate,
    SimpleGate,
    StageDecomposer,
)
from restoration.arch.head import (
    PixelShuffleHead,
    ZeroInitPixelShuffleHead,
)

__all__ = [
    "DegradationArchetypeExtractor",
    "SimplexArchetypeExtractor",
    "ExplicitFeatureDecomposer",
    "FeatureDecomposer",
    "SimpleGate",
    "GatedResidualBlock",
    "MSTSuppressionGate",
    "DegradationModulation",
    "AdaptiveTimescaleDynamics",
    "StageDecomposer",
    "EvidenceDARStage",
    "EvidenceDARTrunk",
    "PixelShuffleHead",
    "ZeroInitPixelShuffleHead",
    "EvidenceDAR",
    "EvidenceDARModel",
]
