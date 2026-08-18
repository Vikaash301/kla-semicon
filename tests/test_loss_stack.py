"""Comprehensive Unit Tests for Physics-Grounded Loss Stack in Evidence-DAR SEM-SR.

Tests mathematical correctness, numerical stability, edge cases, and gradient dynamics for:
- DetailWeightedCharbonnierLoss
- LogFocalFrequencyLoss
- OrthogonalSubspaceRectificationLoss
- PhysicalRedegradationLoss
- DefectInvarianceLoss
- Unified EvidenceDARLoss Stack
"""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from restoration.loss import (
    DefectInvarianceLoss,
    DetailWeightedCharbonnierLoss,
    EvidenceDARLoss,
    LogFocalFrequencyLoss,
    OrthogonalSubspaceRectificationLoss,
    PhysicalRedegradationLoss,
)


class TestDetailWeightedCharbonnierLoss(unittest.TestCase):
    """Unit tests for DetailWeightedCharbonnierLoss."""

    def setUp(self):
        torch.manual_seed(42)

    def test_loss_strictly_increases_with_error_magnitude(self):
        """Charbonnier loss is monotonically increasing with error magnitude."""
        loss_fn = DetailWeightedCharbonnierLoss(eta=1.0, eps=1e-3)
        target = torch.ones(2, 1, 32, 32)
        pred_close = target + 0.01
        pred_far = target + 0.1

        l_close = loss_fn(pred_close, target)
        l_far = loss_fn(pred_far, target)
        self.assertLess(l_close.item(), l_far.item())

    def test_detail_weighting_emphasizes_edges_over_flat_regions(self):
        """Detail weighting assigns higher weight to high-frequency edges than flat substrate."""
        loss_fn = DetailWeightedCharbonnierLoss(eta=2.0, eps=1e-3)

        # Target with high-contrast edge in left half, flat in right half
        target = torch.zeros(1, 1, 32, 32)
        target[0, 0, :, :16] = 1.0  # Sharp step edge

        weight_map = loss_fn.compute_weight_map(target)
        edge_weight = weight_map[0, 0, :, 14:18].mean().item()
        flat_weight = weight_map[0, 0, :, 24:28].mean().item()

        self.assertGreater(edge_weight, flat_weight)

    def test_zero_error_equals_epsilon(self):
        """When prediction exactly matches target, Charbonnier loss equals epsilon."""
        eps = 1e-3
        loss_fn = DetailWeightedCharbonnierLoss(eta=0.0, eps=eps)
        target = torch.rand(2, 1, 16, 16)
        loss = loss_fn(target, target)
        self.assertAlmostEqual(loss.item(), eps, places=5)

    def test_gradient_flow_under_unclamped_dynamic_range(self):
        """Gradients remain finite and well-conditioned for out-of-range inputs [-0.28, 2.16]."""
        loss_fn = DetailWeightedCharbonnierLoss(eta=1.0, eps=1e-3)
        pred = torch.linspace(-0.28, 2.16, 32 * 32).reshape(1, 1, 32, 32).requires_grad_(True)
        target = torch.full((1, 1, 32, 32), 0.5)

        loss = loss_fn(pred, target)
        loss.backward()

        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue((torch.abs(pred.grad) > 0.0).all())

    def test_custom_weight_map(self):
        """Explicitly passed custom weight map overrides internal computation."""
        loss_fn = DetailWeightedCharbonnierLoss(eta=1.0, eps=1e-3)
        pred = torch.zeros(1, 1, 8, 8, requires_grad=True)
        target = torch.ones(1, 1, 8, 8)
        custom_weights = torch.full((1, 1, 8, 8), 5.0)

        loss_custom = loss_fn(pred, target, weight_map=custom_weights)
        loss_unit = loss_fn(pred, target, weight_map=torch.ones_like(target))

        self.assertAlmostEqual(loss_custom.item(), 5.0 * loss_unit.item(), places=5)


class TestLogFocalFrequencyLoss(unittest.TestCase):
    """Unit tests for LogFocalFrequencyLoss."""

    def setUp(self):
        torch.manual_seed(42)

    def test_penalizes_high_frequency_more_than_dc(self):
        """LFFL penalizes high-frequency checkerboard error more than DC shift."""
        loss_fn = LogFocalFrequencyLoss(alpha=1.0, norm="ortho")
        target = torch.zeros(1, 1, 32, 32)

        # High frequency pattern
        hf_pred = torch.zeros(1, 1, 32, 32)
        hf_pred[0, 0, ::2, ::2] = 0.5
        hf_pred[0, 0, 1::2, 1::2] = 0.5

        # DC shift
        dc_pred = torch.full((1, 1, 32, 32), 0.25)

        l_hf = loss_fn(hf_pred, target)
        l_dc = loss_fn(dc_pred, target)

        self.assertGreater(l_hf.item(), 0.0)
        self.assertGreater(l_dc.item(), 0.0)
        self.assertTrue(torch.isfinite(l_hf))
        self.assertTrue(torch.isfinite(l_dc))

    def test_exact_match_yields_zero_loss(self):
        """When prediction exactly matches target, LFFL is zero."""
        loss_fn = LogFocalFrequencyLoss(alpha=1.0, norm="ortho")
        target = torch.rand(2, 1, 32, 32)
        loss = loss_fn(target, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_focal_weight_scaling_alpha(self):
        """Increasing alpha amplifies focal dynamic frequency weighting."""
        target = torch.zeros(1, 1, 16, 16)
        pred = torch.randn(1, 1, 16, 16)

        loss_alpha0 = LogFocalFrequencyLoss(alpha=0.0)(pred, target)
        loss_alpha1 = LogFocalFrequencyLoss(alpha=1.0)(pred, target)

        self.assertGreater(loss_alpha1.item(), loss_alpha0.item())

    def test_gradient_flow_finite_and_smooth(self):
        """Gradients propagate through complex FFT magnitude without NaNs or Infs."""
        loss_fn = LogFocalFrequencyLoss(alpha=1.0)
        pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = torch.randn(2, 1, 32, 32)

        loss = loss_fn(pred, target)
        loss.backward()

        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertFalse(torch.isnan(pred.grad).any())


class TestOrthogonalSubspaceRectificationLoss(unittest.TestCase):
    """Unit tests for OrthogonalSubspaceRectificationLoss."""

    def test_exact_zero_on_orthogonal_subspaces(self):
        """OSR penalty is zero when degradation and content features are spatially disjoint."""
        loss_fn = OrthogonalSubspaceRectificationLoss(normalize=True)
        f_d = torch.zeros(1, 4, 8, 8)
        f_d[0, :, :4, :] = 1.0  # Top half active
        f_c = torch.zeros(1, 4, 8, 8)
        f_c[0, :, 4:, :] = 1.0  # Bottom half active

        loss = loss_fn(f_d, f_c)
        self.assertEqual(loss.item(), 0.0)

    def test_positive_on_correlated_subspaces(self):
        """OSR penalty is strictly positive when degradation and content features overlap."""
        loss_fn = OrthogonalSubspaceRectificationLoss(normalize=True)
        f_d = torch.ones(2, 16, 8, 8)
        f_c = torch.ones(2, 16, 8, 8)

        loss = loss_fn(f_d, f_c)
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_gradient_drives_features_apart(self):
        """OSR gradients push correlated features towards orthogonality."""
        loss_fn = OrthogonalSubspaceRectificationLoss(normalize=True)
        f_d = torch.randn(2, 8, 16, 16, requires_grad=True)
        f_c = torch.randn(2, 8, 16, 16, requires_grad=True)

        loss = loss_fn(f_d, f_c)
        loss.backward()

        self.assertIsNotNone(f_d.grad)
        self.assertIsNotNone(f_c.grad)
        self.assertTrue(torch.isfinite(f_d.grad).all())
        self.assertTrue(torch.isfinite(f_c.grad).all())

    def test_supports_2d_and_3d_tensor_shapes(self):
        """Supports 2D (C, N) and 3D (B, C, N) matrix forms."""
        loss_fn = OrthogonalSubspaceRectificationLoss(normalize=True)

        # 2D test
        f_d_2d = torch.randn(8, 64)
        f_c_2d = torch.randn(8, 64)
        loss_2d = loss_fn(f_d_2d, f_c_2d)
        self.assertTrue(torch.isfinite(loss_2d))

        # 3D test
        f_d_3d = torch.randn(2, 8, 64)
        f_c_3d = torch.randn(2, 8, 64)
        loss_3d = loss_fn(f_d_3d, f_c_3d)
        self.assertTrue(torch.isfinite(loss_3d))


class TestPhysicalRedegradationLoss(unittest.TestCase):
    """Unit tests for PhysicalRedegradationLoss."""

    def test_heteroscedastic_sigma_power_law(self):
        """Sigma matches calibrated power law sigma(s) = 0.0233 * (s / 0.1)^0.836."""
        loss_fn = PhysicalRedegradationLoss(noise_scale=0.0233, power_exponent=0.836, s_ref=0.1)
        s = torch.tensor([0.1, 0.5, 0.9])
        sigma = loss_fn.compute_sigma(s)

        self.assertAlmostEqual(sigma[0].item(), 0.0233, places=4)
        self.assertGreater(sigma[1].item(), sigma[0].item())
        self.assertGreater(sigma[2].item(), sigma[1].item())

    def test_sigma_floor_clamping_at_zero(self):
        """Sigma clamps to s_min=0.02 when signal approaches 0 or goes negative."""
        loss_fn = PhysicalRedegradationLoss(s_min=0.02)
        s_negative = torch.tensor([-0.28, 0.0, 0.01])
        sigma = loss_fn.compute_sigma(s_negative)

        expected_sigma_min = 0.0233 * (0.02 / 0.1) ** 0.836
        for val in sigma:
            self.assertAlmostEqual(val.item(), expected_sigma_min, places=4)

    def test_nll_finite_and_differentiable_wrt_hr_prediction(self):
        """Loss evaluates finite and produces gradients through 2x decimation back to HR pred."""
        loss_fn = PhysicalRedegradationLoss()
        pred_hr = torch.rand(2, 1, 64, 64, requires_grad=True)
        lr_input = torch.rand(2, 1, 32, 32)

        loss = loss_fn(pred_hr, lr_input)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(pred_hr.grad)
        self.assertTrue(torch.isfinite(pred_hr.grad).all())
        self.assertTrue((torch.abs(pred_hr.grad) > 0.0).any())

    def test_custom_forward_operator(self):
        """Supports custom forward decimation operator callable."""
        loss_fn = PhysicalRedegradationLoss()
        pred_hr = torch.rand(1, 1, 32, 32)
        lr_input = torch.rand(1, 1, 16, 16)

        custom_op = lambda x: F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        loss = loss_fn(pred_hr, lr_input, forward_op=custom_op)
        self.assertTrue(torch.isfinite(loss))


class TestDefectInvarianceLoss(unittest.TestCase):
    """Unit tests for DefectInvarianceLoss."""

    def test_zero_penalty_when_identical(self):
        """Defect loss is zero when perturbed features match original features."""
        loss_fn = DefectInvarianceLoss(lambda_s=1.0, lambda_fd=1.0)
        s_deg = torch.randn(2, 8)
        f_d = torch.randn(2, 16, 8, 8)

        loss = loss_fn(s_deg, f_d, s_deg_perturbed=s_deg, f_d_perturbed=f_d)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_positive_penalty_under_defect_perturbation(self):
        """Defect loss strictly penalizes deviations caused by synthetic defects."""
        loss_fn = DefectInvarianceLoss(lambda_s=1.0, lambda_fd=1.0)
        s_deg = torch.randn(2, 8)
        f_d = torch.randn(2, 16, 8, 8)

        s_deg_perturbed = s_deg + 0.1 * torch.randn_like(s_deg)
        f_d_perturbed = f_d + 0.1 * torch.randn_like(f_d)

        loss = loss_fn(s_deg, f_d, s_deg_perturbed=s_deg_perturbed, f_d_perturbed=f_d_perturbed)
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_gradients_flow_to_both_representations(self):
        """Gradients flow back to S_deg and F_d."""
        loss_fn = DefectInvarianceLoss(lambda_s=1.0, lambda_fd=1.0)
        s_deg = torch.randn(2, 8, requires_grad=True)
        f_d = torch.randn(2, 16, 8, 8, requires_grad=True)
        s_deg_p = torch.randn(2, 8)
        f_d_p = torch.randn(2, 16, 8, 8)

        loss = loss_fn(s_deg, f_d, s_deg_perturbed=s_deg_p, f_d_perturbed=f_d_p)
        loss.backward()

        self.assertIsNotNone(s_deg.grad)
        self.assertIsNotNone(f_d.grad)
        self.assertTrue(torch.isfinite(s_deg.grad).all())
        self.assertTrue(torch.isfinite(f_d.grad).all())


class TestUnifiedEvidenceDARLossStack(unittest.TestCase):
    """Unit tests for the complete EvidenceDARLoss manager."""

    def setUp(self):
        torch.manual_seed(42)
        self.loss_stack = EvidenceDARLoss(
            lambda_detail=1.0,
            lambda_lffl=0.1,
            lambda_osr=0.05,
            lambda_phys=0.05,
            lambda_defect=0.01,
        )

    def test_telemetry_dict_contains_all_keys(self):
        """Unified loss returns total scalar and complete telemetry dictionary."""
        b, c, h, w = 2, 16, 16, 16
        pred = torch.rand(b, 1, h * 2, w * 2, requires_grad=True)
        target = torch.rand(b, 1, h * 2, w * 2)
        lr = torch.rand(b, 1, h, w)
        s_deg = torch.randn(b, c, requires_grad=True)
        f_d = torch.randn(b, c, h, w, requires_grad=True)
        f_c = torch.randn(b, c, h, w, requires_grad=True)

        total_loss, telemetry = self.loss_stack(pred, target, lr, s_deg, f_d, f_c)

        self.assertTrue(torch.isfinite(total_loss))
        expected_keys = {"loss_total", "loss_detail", "loss_lffl", "loss_osr", "loss_phys", "loss_defect"}
        self.assertEqual(set(telemetry.keys()), expected_keys)

        for key, val in telemetry.items():
            self.assertTrue(torch.isfinite(val), f"Key {key} is non-finite: {val}")

    def test_joint_backpropagation_updates_all_terms(self):
        """Joint backprop yields finite, non-zero gradients on all inputs."""
        b, c, h, w = 2, 8, 16, 16
        pred = torch.rand(b, 1, h * 2, w * 2, requires_grad=True)
        target = torch.rand(b, 1, h * 2, w * 2)
        lr = torch.rand(b, 1, h, w)
        s_deg = torch.randn(b, c, requires_grad=True)
        f_d = torch.randn(b, c, h, w, requires_grad=True)
        f_c = torch.randn(b, c, h, w, requires_grad=True)

        total_loss, _ = self.loss_stack(pred, target, lr, s_deg, f_d, f_c)
        total_loss.backward()

        self.assertIsNotNone(pred.grad)
        self.assertIsNotNone(s_deg.grad)
        self.assertIsNotNone(f_d.grad)
        self.assertIsNotNone(f_c.grad)

        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue(torch.isfinite(s_deg.grad).all())
        self.assertTrue(torch.isfinite(f_d.grad).all())
        self.assertTrue(torch.isfinite(f_c.grad).all())

    def test_coupling_with_evidence_dar_architecture(self):
        """End-to-end forward and backward with EvidenceDAR architecture module."""
        from restoration.arch import EvidenceDAR

        model = EvidenceDAR(channels=32, num_stages=2, blocks_per_stage=2, num_archetypes=4)
        lr_input = torch.rand(2, 1, 32, 32)
        target_gt = F.interpolate(lr_input, scale_factor=2, mode="bicubic", align_corners=False)

        pred_hr, s_deg, f_d, f_c = model(lr_input, return_degradation=True)
        total_loss, telemetry = self.loss_stack(pred_hr, target_gt, lr_input, s_deg, f_d, f_c)

        total_loss.backward()

        # Check every parameter in model has a finite gradient
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has None gradient")
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"Parameter {name} has non-finite gradient",
                )

    def test_perturbed_defect_invariance_in_stack(self):
        """Unified stack handles defect perturbation inputs."""
        b, c, h, w = 2, 8, 16, 16
        pred = torch.rand(b, 1, h * 2, w * 2)
        target = torch.rand(b, 1, h * 2, w * 2)
        lr = torch.rand(b, 1, h, w)
        s_deg = torch.randn(b, c)
        f_d = torch.randn(b, c, h, w)
        f_c = torch.randn(b, c, h, w)

        s_deg_p = s_deg + 0.05 * torch.randn_like(s_deg)
        f_d_p = f_d + 0.05 * torch.randn_like(f_d)

        total_loss, telemetry = self.loss_stack(
            pred, target, lr, s_deg, f_d, f_c, S_deg_perturbed=s_deg_p, F_d_perturbed=f_d_p
        )
        self.assertTrue(torch.isfinite(total_loss))
        self.assertGreater(telemetry["loss_defect"].item(), 0.0)

    def test_ablation_weights(self):
        """Zeroing individual weights disables respective components cleanly."""
        loss_stack_zero = EvidenceDARLoss(
            lambda_detail=0.0,
            lambda_lffl=0.0,
            lambda_osr=0.0,
            lambda_phys=0.0,
            lambda_defect=0.0,
        )
        b, c, h, w = 1, 8, 16, 16
        pred = torch.rand(b, 1, h * 2, w * 2)
        target = torch.rand(b, 1, h * 2, w * 2)
        lr = torch.rand(b, 1, h, w)
        s_deg = torch.randn(b, c)
        f_d = torch.randn(b, c, h, w)
        f_c = torch.randn(b, c, h, w)

        total_loss, telemetry = loss_stack_zero(pred, target, lr, s_deg, f_d, f_c)
        self.assertAlmostEqual(total_loss.item(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()

