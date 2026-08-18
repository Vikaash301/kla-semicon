"""Adversarial Challenger Stress Tests for Evidence-DAR and Loss Stack.

Empirical verification suite designed by Challenger 1 to stress-test:
1. Exact Step-0 Bicubic Identity across distributions, shapes, and precision modes.
2. Extreme out-of-range inputs (dynamic range [-2.0, 10.0], high magnitude, constants, NaNs/Infs).
3. Arbitrary spatial input dimensions (even, odd, asymmetric, small, large).
4. Batch size scaling and cross-sample batch independence.
5. Backward pass gradient health across all model and loss parameters.
6. Numerical stability, division-by-zero guards, and corner cases in the 5-component loss stack.
7. Multi-step optimization stability under AdamW.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from restoration.arch.evidence_dar import EvidenceDAR, EvidenceDARModel
from restoration.arch.archetypes import DegradationArchetypeExtractor
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
from restoration.arch.head import PixelShuffleHead
from restoration.loss.defect_loss import DefectInvarianceLoss
from restoration.loss.detail_loss import DetailWeightedCharbonnierLoss
from restoration.loss.lffl_loss import LogFocalFrequencyLoss
from restoration.loss.loss_stack import EvidenceDARLoss
from restoration.loss.osr_loss import OrthogonalSubspaceRectificationLoss
from restoration.loss.phys_loss import PhysicalRedegradationLoss


class TestStep0BicubicIdentityStress:
    """Stress tests for step-0 exact bicubic baseline identity."""

    @pytest.mark.parametrize("shape", [
        (1, 1, 64, 64),
        (2, 1, 128, 128),
        (4, 1, 127, 127),
        (1, 1, 33, 79),
        (8, 1, 15, 15),
        (2, 1, 256, 256),
    ])
    def test_step0_bicubic_identity_various_shapes(self, shape: Tuple[int, ...]) -> None:
        """Verify step-0 identity holds across diverse spatial shapes and batch sizes."""
        torch.manual_seed(42)
        model = EvidenceDAR(channels=64, num_stages=4, blocks_per_stage=3)
        model.eval()

        x = torch.randn(*shape, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(x, clamp_output=False)
            y_bicubic = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)

        delta = torch.max(torch.abs(y_pred - y_bicubic)).item()
        assert delta < 1e-6, f"Step 0 delta {delta} exceeded 1e-6 for shape {shape}"

    @pytest.mark.parametrize("val_min,val_max", [
        (-0.28, 2.16),  # SEM physical range
        (-2.0, 10.0),   # Extreme out-of-range
        (-100.0, 100.0), # Extreme dynamic range
        (0.0, 1.0),     # Standard normalized range
        (5.0, 5.0),     # Uniform constant
        (-1.0, -1.0),   # Uniform negative constant
    ])
    def test_step0_bicubic_identity_dynamic_ranges(self, val_min: float, val_max: float) -> None:
        """Verify step-0 identity is independent of input value distribution."""
        torch.manual_seed(123)
        model = EvidenceDAR()
        model.eval()

        if val_min == val_max:
            x = torch.full((2, 1, 64, 64), fill_value=val_min, dtype=torch.float32)
        else:
            x = torch.empty(2, 1, 64, 64).uniform_(val_min, val_max)

        with torch.no_grad():
            y_pred = model(x, clamp_output=False)
            y_bicubic = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)

        delta = torch.max(torch.abs(y_pred - y_bicubic)).item()
        assert delta < 1e-6, f"Step 0 delta {delta} exceeded 1e-6 for range [{val_min}, {val_max}]"

    def test_step0_bicubic_identity_double_precision(self) -> None:
        """Verify step-0 identity holds under float64 double precision."""
        model = EvidenceDAR().to(dtype=torch.float64)
        model.eval()
        x = torch.randn(2, 1, 64, 64, dtype=torch.float64)
        with torch.no_grad():
            y_pred = model(x, clamp_output=False)
            y_bicubic = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)

        delta = torch.max(torch.abs(y_pred - y_bicubic)).item()
        assert delta < 1e-12, f"Double precision delta {delta} exceeded 1e-12"


class TestExtremeInputsAndRobustness:
    """Stress tests with extreme, degraded, and boundary values."""

    def test_extreme_unclamped_forward(self) -> None:
        """Ensure forward pass computes finite outputs for extreme inputs in [-10.0, 20.0]."""
        model = EvidenceDAR()
        x = torch.empty(2, 1, 64, 64).uniform_(-10.0, 20.0)
        y, s_deg, f_d, f_c = model(x, return_degradation=True, clamp_output=False)

        assert torch.isfinite(y).all(), "Non-finite output detected in forward pass"
        assert torch.isfinite(s_deg).all(), "Non-finite S_deg detected"
        assert torch.isfinite(f_d).all(), "Non-finite F_d detected"
        assert torch.isfinite(f_c).all(), "Non-finite F_c detected"

    def test_eval_mode_clamping_enforcement(self) -> None:
        """Ensure eval mode strictly clamps outputs to [0.0, 1.0] when clamp_output=None."""
        model = EvidenceDAR()
        model.eval()
        x = torch.empty(4, 1, 32, 32).uniform_(-5.0, 5.0)
        y = model(x)

        assert y.min().item() >= 0.0, f"Eval output min {y.min().item()} < 0.0"
        assert y.max().item() <= 1.0, f"Eval output max {y.max().item()} > 1.0"

    def test_training_mode_unclamped_default(self) -> None:
        """Ensure training mode leaves outputs unclamped by default so gradients flow freely."""
        model = EvidenceDAR()
        model.train()
        x = torch.empty(2, 1, 32, 32).uniform_(-2.0, 3.0)
        y = model(x)
        # Should preserve negative and >1.0 bicubic values
        assert (y < 0.0).any() or (y > 1.0).any(), "Training mode unexpectedly clamped output"

    def test_nan_inf_propagation_behavior(self) -> None:
        """Verify behavior when input contains NaNs or Infs."""
        model = EvidenceDAR()
        x_nan = torch.randn(2, 1, 32, 32)
        x_nan[0, 0, 5, 5] = float("nan")
        y_nan = model(x_nan, clamp_output=False)
        assert torch.isnan(y_nan).any(), "Expected NaN in output when input contains NaN"

        x_inf = torch.randn(2, 1, 32, 32)
        x_inf[0, 0, 5, 5] = float("inf")
        y_inf = model(x_inf, clamp_output=False)
        assert torch.isinf(y_inf).any() or torch.isnan(y_inf).any(), "Expected Inf/NaN in output"


class TestSpatialShapesAndBatchScaling:
    """Stress tests on arbitrary spatial shapes, aspect ratios, and batch sizes."""

    @pytest.mark.parametrize("b", [1, 2, 8, 16])
    @pytest.mark.parametrize("h,w", [
        (64, 64),
        (128, 128),
        (256, 256),
        (127, 127),
        (63, 129),
        (15, 31),
    ])
    def test_arbitrary_dimensions_and_batch_sizes(self, b: int, h: int, w: int) -> None:
        """Test forward and output dimensions for various batch sizes and resolutions."""
        model = EvidenceDAR(channels=32, num_stages=2, blocks_per_stage=2)
        x = torch.randn(b, 1, h, w)
        y, s_deg, f_d, f_c = model(x, return_degradation=True)

        assert y.shape == (b, 1, 2 * h, 2 * w), f"Unexpected output shape: {y.shape}"
        assert s_deg.shape == (b, 32), f"Unexpected s_deg shape: {s_deg.shape}"
        assert f_d.shape == (b, 32, h, w), f"Unexpected f_d shape: {f_d.shape}"
        assert f_c.shape == (b, 32, h, w), f"Unexpected f_c shape: {f_c.shape}"

    def test_cross_sample_batch_independence(self) -> None:
        """Verify that sample predictions are strictly independent across batch elements."""
        torch.manual_seed(999)
        model = EvidenceDAR()
        model.eval()

        # Randomize model weights slightly to avoid trivial 0-residual pass
        for p in model.parameters():
            p.data.add_(torch.randn_like(p.data) * 0.01)

        x1 = torch.randn(1, 1, 64, 64)
        x2 = torch.randn(1, 1, 64, 64)
        x_batch = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            y_individual_1 = model(x1, clamp_output=False)
            y_individual_2 = model(x2, clamp_output=False)
            y_batch = model(x_batch, clamp_output=False)

        diff1 = torch.max(torch.abs(y_batch[0:1] - y_individual_1)).item()
        diff2 = torch.max(torch.abs(y_batch[1:2] - y_individual_2)).item()

        assert diff1 < 1e-5, f"Batch crosstalk detected on sample 1: diff={diff1}"
        assert diff2 < 1e-5, f"Batch crosstalk detected on sample 2: diff={diff2}"


class TestGradientSanityAndOptimization:
    """Stress tests on gradient propagation across all trainable parameters."""

    def test_full_backward_gradient_health(self) -> None:
        """Verify that every trainable parameter in EvidenceDAR receives a valid, finite gradient."""
        torch.manual_seed(42)
        model = EvidenceDAR(channels=64, num_stages=4, blocks_per_stage=3)
        # Perturb weights slightly away from zero-init saddle point to verify all layers receive active gradients
        for p in model.parameters():
            p.data.add_(torch.randn_like(p.data) * 0.01)
        loss_stack = EvidenceDARLoss()

        x = torch.randn(2, 1, 64, 64, requires_grad=True)
        target = torch.randn(2, 1, 128, 128)

        y_hat, s_deg, f_d, f_c = model(x, return_degradation=True, clamp_output=False)
        total_loss, telemetry = loss_stack(
            pred=y_hat,
            target=target,
            lr_input=x,
            S_deg=s_deg,
            F_d=f_d,
            F_c=f_c,
            S_deg_perturbed=s_deg + 0.05 * torch.randn_like(s_deg),
            F_d_perturbed=f_d + 0.05 * torch.randn_like(f_d),
        )

        total_loss.backward()

        # Check all named parameters
        missing_grads = []
        nan_inf_grads = []
        zero_grads = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    missing_grads.append(name)
                elif torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    nan_inf_grads.append(name)
                elif torch.all(param.grad == 0.0):
                    zero_grads.append(name)

        assert not missing_grads, f"Parameters missing gradients: {missing_grads}"
        assert not nan_inf_grads, f"Parameters with NaN/Inf gradients: {nan_inf_grads}"
        assert not zero_grads, f"Parameters with exactly zero gradients: {zero_grads}"

    def test_multi_step_adamw_convergence(self) -> None:
        """Perform 10 optimization steps to verify stable loss decay without explosion."""
        torch.manual_seed(42)
        model = EvidenceDAR(channels=32, num_stages=2, blocks_per_stage=2)
        loss_stack = EvidenceDARLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        x = torch.randn(2, 1, 64, 64)
        target = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False) + 0.05 * torch.randn(2, 1, 128, 128)

        losses: List[float] = []
        for step in range(10):
            optimizer.zero_grad()
            y_hat, s_deg, f_d, f_c = model(x, return_degradation=True, clamp_output=False)
            loss, telemetry = loss_stack(
                pred=y_hat,
                target=target,
                lr_input=x,
                S_deg=s_deg,
                F_d=f_d,
                F_c=f_c,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss failed to decrease over 10 steps: initial={losses[0]}, final={losses[-1]}"
        assert all(math.isfinite(l) for l in losses), "Non-finite loss encountered during optimization"


class TestLossStackMicroscopyCornerCases:
    """Stress tests on individual loss functions and their numerical edge cases."""

    def test_detail_loss_flat_image(self) -> None:
        """Ensure DetailWeightedCharbonnierLoss behaves gracefully on uniform/flat targets (std=0)."""
        loss_fn = DetailWeightedCharbonnierLoss()
        pred = torch.full((2, 1, 64, 64), 0.5)
        target = torch.full((2, 1, 64, 64), 0.5)
        loss = loss_fn(pred, target)
        # diff = 0 -> sqrt(eps^2) = eps = 1e-3
        assert math.isclose(loss.item(), 1e-3, rel_tol=1e-3)

    def test_detail_loss_extreme_intensity(self) -> None:
        """Verify DetailWeightedCharbonnierLoss handles out-of-range targets [-2.0, 5.0]."""
        loss_fn = DetailWeightedCharbonnierLoss()
        target = torch.empty(2, 1, 64, 64).uniform_(-2.0, 5.0)
        pred = target + 0.1
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss), "Non-finite loss on extreme intensity target"
        assert loss.item() > 0.0

    def test_lffl_loss_dc_and_sparse_frequencies(self) -> None:
        """Verify LFFL loss on pure DC targets vs high-frequency textures."""
        lffl = LogFocalFrequencyLoss()
        target = torch.zeros(2, 1, 64, 64)
        pred_dc = torch.full((2, 1, 64, 64), 0.1)
        pred_hf = torch.zeros(2, 1, 64, 64)
        # Checkerboard pattern for high-frequency
        pred_hf[:, :, ::2, ::2] = 0.1
        pred_hf[:, :, 1::2, 1::2] = 0.1

        loss_dc = lffl(pred_dc, target)
        loss_hf = lffl(pred_hf, target)

        assert torch.isfinite(loss_dc) and torch.isfinite(loss_hf)

    def test_osr_loss_extreme_dimensions(self) -> None:
        """Verify OSR loss works for batch=1, odd dimensions, and 2D/3D inputs."""
        osr = OrthogonalSubspaceRectificationLoss()
        fd = torch.randn(1, 64, 33, 47)
        fc = torch.randn(1, 64, 33, 47)
        loss = osr(fd, fc)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_phys_loss_negative_signals(self) -> None:
        """Verify PhysicalRedegradationLoss clamps negative signals to s_min without blowing up."""
        phys_loss = PhysicalRedegradationLoss()
        pred_hr = torch.full((2, 1, 64, 64), -1.0)
        lr_input = torch.full((2, 1, 32, 32), -1.0)
        loss = phys_loss(pred_hr, lr_input)
        assert torch.isfinite(loss)

    def test_defect_loss_identity_and_perturbation(self) -> None:
        """Verify DefectInvarianceLoss handles unperturbed (None) and perturbed inputs."""
        defect_loss = DefectInvarianceLoss()
        s_deg = torch.randn(2, 64, requires_grad=True)
        f_d = torch.randn(2, 64, 32, 32, requires_grad=True)

        # Unperturbed: should return 0.0 with grad graph intact
        loss_unperturbed = defect_loss(s_deg, f_d)
        assert loss_unperturbed.item() == 0.0
        loss_unperturbed.backward()
        assert s_deg.grad is not None and f_d.grad is not None

        # Perturbed: positive loss
        s_deg.grad = None
        f_d.grad = None
        s_deg_pert = s_deg + 0.1
        f_d_pert = f_d + 0.1
        loss_perturbed = defect_loss(s_deg, f_d, s_deg_pert, f_d_pert)
        assert loss_perturbed.item() > 0.0
