"""Unified Loss Stack for Evidence-DAR SEM Super-Resolution.

Combines all physics-grounded and microscopy-native loss components:
    L_total = lambda_detail * L_detail + lambda_lffl * L_lffl + lambda_osr * L_osr
              + lambda_phys * L_phys + lambda_defect * L_defect
and returns total scalar loss alongside a comprehensive telemetry dictionary.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from restoration.loss.defect_loss import DefectInvarianceLoss
from restoration.loss.detail_loss import DetailWeightedCharbonnierLoss
from restoration.loss.lffl_loss import LogFocalFrequencyLoss
from restoration.loss.osr_loss import OrthogonalSubspaceRectificationLoss
from restoration.loss.phys_loss import PhysicalRedegradationLoss
from restoration.loss.sobolev_loss import MultiScaleSobolevGradientLoss
from restoration.loss.tv_loss import FlatFieldTVLoss


class EvidenceDARLoss(nn.Module):
    """Unified Physics-Grounded Loss Manager for Evidence-DAR.

    Orchestrates the microscopy loss stack:
    1. Detail-Weighted Charbonnier (L_detail)
    2. Log Focal Frequency Loss (L_LFFL)
    3. Orthogonal Subspace Rectification (L_OSR)
    4. Physical Re-degradation Consistency (L_phys)
    5. Defect-Invariance Regularization (L_defect)
    6. Multi-Scale Sobolev Gradient Loss (L_sobolev)
    7. Flat-Field TV Regularizer (L_tv)

    Args:
        lambda_detail: Weight for Detail-Weighted Charbonnier loss (default: 1.0).
        lambda_lffl: Weight for Log Focal Frequency loss (default: 0.1).
        lambda_osr: Weight for Orthogonal Subspace Rectification loss (default: 0.05).
        lambda_phys: Weight for Physical Re-degradation consistency loss (default: 0.05).
        lambda_defect: Weight for Defect Invariance regularization loss (default: 0.01).
        lambda_sobolev: Weight for Multi-Scale Sobolev Gradient loss (default: 0.0).
        lambda_tv: Weight for Flat-Field TV regularizer (default: 0.0).
        eps_detail: Epsilon for Charbonnier loss (default: 1e-3).
        eta_detail: Difficulty weighting multiplier for detail loss (default: 1.0).
        alpha_lffl: Focal scaling exponent for frequency loss (default: 1.0).
    """

    def __init__(
        self,
        lambda_detail: float = 1.0,
        lambda_lffl: float = 0.1,
        lambda_osr: float = 0.05,
        lambda_phys: float = 0.05,
        lambda_defect: float = 0.01,
        lambda_sobolev: float = 0.0,
        lambda_tv: float = 0.0,
        eps_detail: float = 1e-3,
        eta_detail: float = 1.0,
        alpha_lffl: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_detail = float(lambda_detail)
        self.lambda_lffl = float(lambda_lffl)
        self.lambda_osr = float(lambda_osr)
        self.lambda_phys = float(lambda_phys)
        self.lambda_defect = float(lambda_defect)
        self.lambda_sobolev = float(lambda_sobolev)
        self.lambda_tv = float(lambda_tv)

        # Initialize individual loss modules
        self.detail_loss = DetailWeightedCharbonnierLoss(eta=eta_detail, eps=eps_detail)
        self.lffl_loss = LogFocalFrequencyLoss(alpha=alpha_lffl)
        self.osr_loss = OrthogonalSubspaceRectificationLoss(normalize=True)
        self.phys_loss = PhysicalRedegradationLoss()
        self.defect_loss = DefectInvarianceLoss()
        self.sobolev_loss = MultiScaleSobolevGradientLoss()
        self.tv_loss = FlatFieldTVLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lr_input: torch.Tensor,
        S_deg: torch.Tensor,
        F_d: torch.Tensor,
        F_c: torch.Tensor,
        S_deg_perturbed: Optional[torch.Tensor] = None,
        F_d_perturbed: Optional[torch.Tensor] = None,
        weight_map: Optional[torch.Tensor] = None,
        forward_op: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes composite loss and individual component telemetry.

        Args:
            pred: High-resolution restored prediction tensor (B, 1, 2H, 2W).
            target: High-resolution ground truth target tensor (B, 1, 2H, 2W).
            lr_input: Low-resolution input measurement tensor (B, 1, H, W).
            S_deg: Degradation archetype descriptor vector (B, D).
            F_d: Degradation feature tensor (B, C, H, W).
            F_c: Content feature tensor (B, C, H, W).
            S_deg_perturbed: Optional descriptor from perturbed input (B, D).
            F_d_perturbed: Optional feature tensor from perturbed input (B, C, H, W).
            weight_map: Optional precomputed detail weight map.
            forward_op: Optional custom microscope forward operator.

        Returns:
            Tuple of (total_loss_scalar, telemetry_dict).
        """
        # 1. Detail-Weighted Charbonnier Reconstruction Loss
        if self.lambda_detail > 0.0:
            l_detail = self.detail_loss(pred, target, weight_map=weight_map)
        else:
            l_detail = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 2. Log Focal Frequency Loss
        if self.lambda_lffl > 0.0:
            l_lffl = self.lffl_loss(pred, target)
        else:
            l_lffl = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 3. Orthogonal Subspace Rectification Loss
        if self.lambda_osr > 0.0:
            l_osr = self.osr_loss(F_d, F_c)
        else:
            l_osr = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 4. Physical Re-degradation Consistency Loss
        if self.lambda_phys > 0.0:
            l_phys = self.phys_loss(pred, lr_input, forward_op=forward_op)
        else:
            l_phys = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 5. Defect Invariance Regularization
        if self.lambda_defect > 0.0:
            l_defect = self.defect_loss(
                S_deg,
                F_d,
                s_deg_perturbed=S_deg_perturbed,
                f_d_perturbed=F_d_perturbed,
            )
        else:
            l_defect = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 6. Multi-Scale Sobolev Gradient Loss
        if self.lambda_sobolev > 0.0:
            l_sobolev = self.sobolev_loss(pred, target)
        else:
            l_sobolev = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 7. Flat-Field TV Regularizer
        if self.lambda_tv > 0.0:
            l_tv = self.tv_loss(pred, target=target)
        else:
            l_tv = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Composite total loss
        total_loss = (
            self.lambda_detail * l_detail
            + self.lambda_lffl * l_lffl
            + self.lambda_osr * l_osr
            + self.lambda_phys * l_phys
            + self.lambda_defect * l_defect
            + self.lambda_sobolev * l_sobolev
            + self.lambda_tv * l_tv
        )

        telemetry = {
            "loss_total": total_loss.detach(),
            "loss_detail": l_detail.detach(),
            "loss_lffl": l_lffl.detach(),
            "loss_osr": l_osr.detach(),
            "loss_phys": l_phys.detach(),
            "loss_defect": l_defect.detach(),
        }
        if self.lambda_sobolev > 0.0:
            telemetry["loss_sobolev"] = l_sobolev.detach()
        if self.lambda_tv > 0.0:
            telemetry["loss_tv"] = l_tv.detach()

        return total_loss, telemetry
