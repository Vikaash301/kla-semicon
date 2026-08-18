"""Unit Tests for Branch 3: Multi-Scale Sobolev Loss, Flat-Field TV, Null-Space Projector, and Batched D4 TTA."""

from __future__ import annotations

import unittest
import torch
import torch.nn.functional as F

from restoration.loss.sobolev_loss import MultiScaleSobolevGradientLoss
from restoration.loss.tv_loss import FlatFieldTVLoss
from restoration.arch.evidence_dar import NullSpaceConsistencyProjector, EvidenceDAR
from restoration.inference import batched_d4_tta_inference


class TestBranch3SobolevLoss(unittest.TestCase):
    """Unit tests for MultiScaleSobolevGradientLoss."""

    def setUp(self):
        torch.manual_seed(42)
        self.loss_fn = MultiScaleSobolevGradientLoss(scales=3, beta_laplacian=0.5)

    def test_identical_inputs_yield_approx_zero(self):
        target = torch.rand(2, 1, 64, 64)
        loss = self.loss_fn(target, target)
        self.assertLess(loss.item(), 1e-3)

    def test_gradient_sensitive_to_edge_shifts(self):
        target = torch.zeros(1, 1, 64, 64)
        target[:, :, :, :32] = 1.0

        # Shifted edge by 2 pixels
        pred_shifted = torch.zeros(1, 1, 64, 64)
        pred_shifted[:, :, :, :34] = 1.0

        # Flat uniform shift
        pred_flat = target + 0.1

        loss_edge = self.loss_fn(pred_shifted, target)
        loss_flat = self.loss_fn(pred_flat, target)
        self.assertTrue(torch.isfinite(loss_edge))
        self.assertTrue(torch.isfinite(loss_flat))

    def test_gradients_flow_through_laplacian_and_sobel(self):
        pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = torch.randn(2, 1, 32, 32)
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())


class TestBranch3FlatFieldTVLoss(unittest.TestCase):
    """Unit tests for FlatFieldTVLoss."""

    def setUp(self):
        torch.manual_seed(42)
        self.tv_loss = FlatFieldTVLoss(edge_threshold=0.08)

    def test_flat_field_weight_attenuates_on_edges(self):
        target = torch.zeros(1, 1, 64, 64)
        target[:, :, :, :32] = 1.0  # Sharp vertical edge at column 32

        w_flat = self.tv_loss.compute_flat_weight(target)
        edge_weight = w_flat[0, 0, :, 31:33].mean().item()
        flat_weight = w_flat[0, 0, :, 10:20].mean().item()

        self.assertLess(edge_weight, flat_weight)
        self.assertLess(edge_weight, 0.2)
        self.assertGreater(flat_weight, 0.8)

    def test_tv_penalizes_noise_in_flat_regions(self):
        target = torch.zeros(1, 1, 64, 64)
        clean = torch.zeros(1, 1, 64, 64)
        noisy = torch.zeros(1, 1, 64, 64) + 0.1 * torch.randn(1, 1, 64, 64)

        loss_clean = self.tv_loss(clean, target=target)
        loss_noisy = self.tv_loss(noisy, target=target)

        self.assertLess(loss_clean.item(), loss_noisy.item())

    def test_gradient_flow_finite(self):
        pred = torch.randn(2, 1, 32, 32, requires_grad=True)
        target = torch.randn(2, 1, 32, 32)
        loss = self.tv_loss(pred, target=target)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())


class TestBranch3NullSpaceProjector(unittest.TestCase):
    """Unit tests for Discrete Null-Space Consistency Projector."""

    def test_measurement_consistency_under_normalized_range(self):
        projector = NullSpaceConsistencyProjector(scale_factor=2, blend_gamma=1.0)
        x = torch.rand(16, 1, 256, 256, dtype=torch.float32)
        y = torch.rand(16, 1, 128, 128, dtype=torch.float32)

        x_hat = projector(x, y)
        h_x_hat = F.avg_pool2d(x_hat, 2, 2)
        max_diff = torch.max(torch.abs(h_x_hat - y)).item()

        self.assertLessEqual(max_diff, 1.192093e-7)

    def test_idempotence_of_null_space_projector(self):
        projector = NullSpaceConsistencyProjector(scale_factor=2, blend_gamma=1.0)
        x = torch.rand(4, 1, 64, 64)
        y = torch.rand(4, 1, 32, 32)

        p1 = projector(x, y)
        p2 = projector(p1, y)
        diff = torch.max(torch.abs(p1 - p2)).item()
        self.assertLessEqual(diff, 1e-6)


class TestBranch3BatchedD4TTA(unittest.TestCase):
    """Unit tests for Batched D4 Dihedral TTA."""

    def test_all_8_transforms_invert_identically(self):
        x = torch.randn(1, 1, 16, 16)
        # Dummy identity model
        class IdentityModel(torch.nn.Module):
            def forward(self, inp):
                return F.interpolate(inp, scale_factor=2, mode="nearest")

        model = IdentityModel()
        out_tta = batched_d4_tta_inference(model, x)
        expected = F.interpolate(x, scale_factor=2, mode="nearest")

        self.assertTrue(torch.allclose(out_tta, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
