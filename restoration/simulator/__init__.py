"""Calibrated SEM Physical Degradation Simulator & Augmentation Package.

Exports:
- SEMForwardOperator: Physics-grounded forward acquisition operator p_SEM(theta).
- MultiViewConsistencyAugmenter: Stochastic multi-view degradation generator.
- NegativeRestorationSampler: Identity-regularizing negative sample generator.
- SpectralHighFrequencyAugmenter: Spectral shift high-frequency line-pitch augmenter.
- SyntheticDefectGenerator: Realistic semiconductor void, bridge, LER, protrusion, and intrusion generator.
"""

from restoration.simulator.operator import (
    CALIBRATED_BX,
    CALIBRATED_BY,
    CALIBRATED_NOISE_EXPONENT,
    CALIBRATED_NOISE_SCALE,
    CALIBRATED_S_FLOOR,
    CALIBRATED_W4,
    SEMForwardOperator,
)
from restoration.simulator.augment import (
    MultiViewConsistencyAugmenter,
    NegativeRestorationSampler,
    SpectralHighFrequencyAugmenter,
)
from restoration.simulator.defect_synth import (
    SyntheticDefectGenerator,
)

__all__ = [
    "SEMForwardOperator",
    "MultiViewConsistencyAugmenter",
    "NegativeRestorationSampler",
    "SpectralHighFrequencyAugmenter",
    "SyntheticDefectGenerator",
    "CALIBRATED_W4",
    "CALIBRATED_BX",
    "CALIBRATED_BY",
    "CALIBRATED_NOISE_SCALE",
    "CALIBRATED_NOISE_EXPONENT",
    "CALIBRATED_S_FLOOR",
]
