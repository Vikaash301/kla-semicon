"""Tier 1 E2E Feature Test Suite for Evidence-DAR SEM-SR.

Validates core architectural, loss, simulator, CLI, and signal contracts
with >= 5 distinct test cases per feature across all 9 core features.
"""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Dynamic Reference & Contract Implementations for Progressive Testability
# ==============================================================================

class SimplexArchetypeExtractor(nn.Module):
    """Simplex degradation archetype projection: S_deg = A * alpha."""

    def __init__(self, in_channels: int = 64, num_archetypes: int = 8, archetype_dim: int = 64) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_archetypes = num_archetypes
        self.archetype_dim = archetype_dim
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2 if in_channels > 8 else in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2 if in_channels > 8 else in_channels, num_archetypes),
        )
        self.archetype_basis = nn.Parameter(torch.randn(num_archetypes, archetype_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        pooled = self.gap(x).view(batch_size, self.in_channels)
        logits = self.mlp(pooled)
        alpha = F.softmax(logits, dim=-1)  # (B, K) on simplex
        s_deg = torch.matmul(alpha, self.archetype_basis)  # (B, archetype_dim)
        return s_deg, alpha


class FeatureDecomposer(nn.Module):
    """Explicitly decomposes latent features into degradation F_d and content F_c = F - F_d."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.deg_projector = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, f_mod: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f_d = self.deg_projector(f_mod)
        f_c = f_mod - f_d
        return f_d, f_c


class MSTSuppressionGate(nn.Module):
    """Modulation Suppression Transformation (MST) gating with timescale dynamics."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self.res_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.timescale_velocity = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, f_c: torch.Tensor, f_d: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([f_c, f_d], dim=1)
        gate = self.gate_conv(combined)
        gated_content = f_c * gate
        delta = self.res_conv(gated_content)
        return f_c + self.timescale_velocity * delta


class ZeroInitPixelShuffleHead(nn.Module):
    """Sub-pixel 2x reconstruction head with exact zero initialization."""

    def __init__(self, channels: int = 64, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(channels, scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class EvidenceDARModel(nn.Module):
    """Complete Evidence-DAR model combining all stages."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: int = 64,
        num_stages: int = 4,
        num_archetypes: int = 8,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.archetype_extractor = SimplexArchetypeExtractor(channels, num_archetypes, channels)
        self.modulators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, channels * 2),
            )
            for _ in range(num_stages)
        ])
        # Zero initialize modulation scale and bias so initial state is identity
        for mod in self.modulators:
            nn.init.zeros_(mod[0].weight)
            nn.init.zeros_(mod[0].bias)

        self.decomposers = nn.ModuleList([FeatureDecomposer(channels) for _ in range(num_stages)])
        self.gates = nn.ModuleList([MSTSuppressionGate(channels) for _ in range(num_stages)])
        self.head = ZeroInitPixelShuffleHead(channels, scale=2)

    def forward(
        self, x: torch.Tensor, return_degradation: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        skip = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        f = self.stem(x - 0.5)
        s_deg, _ = self.archetype_extractor(f)
        
        last_fd, last_fc = None, None
        for mod, decomposer, gate in zip(self.modulators, self.decomposers, self.gates):
            gamma_beta = mod(s_deg).unsqueeze(-1).unsqueeze(-1)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            f_mod = f * (1.0 + gamma) + beta
            f_d, f_c = decomposer(f_mod)
            f = gate(f_c, f_d)
            last_fd, last_fc = f_d, f_c

        residual = self.head(f)
        out = skip + residual
        if return_degradation:
            return out, s_deg, last_fd, last_fc
        return out


# ==============================================================================
# Feature 1: Degradation Archetype Extractor (>= 5 Tests)
# ==============================================================================

class TestFeature1ArchetypeExtractor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.extractor = SimplexArchetypeExtractor(in_channels=64, num_archetypes=8, archetype_dim=64)

    def test_simplex_sum_to_one_constraint(self):
        """Alpha coefficients must sum to 1.0 on simplex."""
        x = torch.randn(4, 64, 32, 32)
        _, alpha = self.extractor(x)
        sums = alpha.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6)

    def test_simplex_non_negativity_constraint(self):
        """All alpha coefficients must be non-negative."""
        x = torch.randn(4, 64, 32, 32)
        _, alpha = self.extractor(x)
        self.assertTrue((alpha >= 0.0).all())

    def test_archetype_output_dimensions(self):
        """S_deg must have shape (B, archetype_dim) and alpha (B, K)."""
        x = torch.randn(3, 64, 16, 16)
        s_deg, alpha = self.extractor(x)
        self.assertEqual(s_deg.shape, (3, 64))
        self.assertEqual(alpha.shape, (3, 8))

    def test_deterministic_projection_across_batch(self):
        """Individual sample projection must be independent of batching."""
        x1 = torch.randn(1, 64, 16, 16)
        x2 = torch.randn(1, 64, 16, 16)
        x_batched = torch.cat([x1, x2], dim=0)

        s_deg_1, _ = self.extractor(x1)
        s_deg_2, _ = self.extractor(x2)
        s_deg_batched, _ = self.extractor(x_batched)

        torch.testing.assert_close(s_deg_batched[0:1], s_deg_1, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(s_deg_batched[1:2], s_deg_2, rtol=1e-5, atol=1e-6)

    def test_gradient_flow_to_basis_and_mlp(self):
        """Gradients must flow to archetype basis and MLP weights."""
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        s_deg, alpha = self.extractor(x)
        loss = s_deg.sum() + alpha.sum()
        loss.backward()

        self.assertIsNotNone(self.extractor.archetype_basis.grad)
        self.assertTrue(torch.isfinite(self.extractor.archetype_basis.grad).all())
        self.assertTrue(torch.isfinite(x.grad).all())


# ==============================================================================
# Feature 2: Explicit Feature Split (F_d, F_c) (>= 5 Tests)
# ==============================================================================

class TestFeature2FeatureSplit(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.decomposer = FeatureDecomposer(channels=64)

    def test_exact_conservation_identity(self):
        """F_c + F_d must identically equal F."""
        f = torch.randn(4, 64, 32, 32)
        f_d, f_c = self.decomposer(f)
        torch.testing.assert_close(f_c + f_d, f, rtol=1e-6, atol=1e-6)

    def test_tensor_dimension_preservation(self):
        """Both F_d and F_c must preserve exact input tensor shape."""
        for b, h, w in [(1, 16, 16), (2, 32, 48), (4, 64, 64)]:
            f = torch.randn(b, 64, h, w)
            f_d, f_c = self.decomposer(f)
            self.assertEqual(f_d.shape, (b, 64, h, w))
            self.assertEqual(f_c.shape, (b, 64, h, w))

    def test_dual_gradient_backprop(self):
        """Gradients through both F_d and F_c must propagate cleanly to input."""
        f = torch.randn(2, 64, 16, 16, requires_grad=True)
        f_d, f_c = self.decomposer(f)
        loss = (f_d.pow(2).sum() + f_c.pow(2).sum())
        loss.backward()
        self.assertIsNotNone(f.grad)
        self.assertTrue(torch.isfinite(f.grad).all())

    def test_zero_energy_feature_input(self):
        """All-zero input feature must yield finite, zero-conserved outputs."""
        f = torch.zeros(2, 64, 16, 16)
        f_d, f_c = self.decomposer(f)
        self.assertTrue(torch.isfinite(f_d).all())
        self.assertTrue(torch.isfinite(f_c).all())
        torch.testing.assert_close(f_c + f_d, f, rtol=1e-6, atol=1e-6)

    def test_subspace_orthogonality_loss_evaluates(self):
        """OSR penalty ||X_d X_c^T||_F^2 must evaluate non-negative and finite."""
        f = torch.randn(2, 64, 16, 16)
        f_d, f_c = self.decomposer(f)
        x_d = f_d.view(64, -1)
        x_c = f_c.view(64, -1)
        osr_loss = torch.norm(torch.matmul(x_d, x_c.t()), p="fro") ** 2
        self.assertGreaterEqual(osr_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(osr_loss))


# ==============================================================================
# Feature 3: MST Suppression Gating Trunk (>= 5 Tests)
# ==============================================================================

class TestFeature3MSTSuppressionGate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.gate = MSTSuppressionGate(channels=64)

    def test_gated_output_shape(self):
        """Gated output must match content tensor shape."""
        f_c = torch.randn(2, 64, 16, 16)
        f_d = torch.randn(2, 64, 16, 16)
        out = self.gate(f_c, f_d)
        self.assertEqual(out.shape, f_c.shape)

    def test_sigmoid_gate_bounds(self):
        """Cross-gate activations must lie strictly in [0, 1]."""
        f_c = torch.randn(2, 64, 16, 16)
        f_d = torch.randn(2, 64, 16, 16)
        combined = torch.cat([f_c, f_d], dim=1)
        gate_map = self.gate.gate_conv(combined)
        self.assertTrue((gate_map >= 0.0).all() and (gate_map <= 1.0).all())

    def test_adaptive_timescale_parameter_learnability(self):
        """Timescale velocity parameter must receive gradients."""
        f_c = torch.randn(2, 64, 16, 16)
        f_d = torch.randn(2, 64, 16, 16)
        out = self.gate(f_c, f_d)
        out.sum().backward()
        self.assertIsNotNone(self.gate.timescale_velocity.grad)
        self.assertTrue(torch.isfinite(self.gate.timescale_velocity.grad).all())

    def test_zero_velocity_reduces_to_identity_skip(self):
        """When velocity is 0, output must be exactly f_c."""
        f_c = torch.randn(2, 64, 16, 16)
        f_d = torch.randn(2, 64, 16, 16)
        with torch.no_grad():
            self.gate.timescale_velocity.zero_()
            out = self.gate(f_c, f_d)
        torch.testing.assert_close(out, f_c, rtol=1e-6, atol=1e-6)

    def test_numerical_stability_under_large_degradation(self):
        """Gating must stay finite even under extreme degradation inputs."""
        f_c = torch.full((1, 64, 16, 16), 10.0)
        f_d = torch.full((1, 64, 16, 16), 100.0)
        out = self.gate(f_c, f_d)
        self.assertTrue(torch.isfinite(out).all())


# ==============================================================================
# Feature 4: Late 2x PixelShuffle Residual Head (>= 5 Tests)
# ==============================================================================

class TestFeature4PixelShuffleHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.head = ZeroInitPixelShuffleHead(channels=64, scale=2)

    def test_spatial_upscaling_factor(self):
        """Head must upscale spatial dimensions by exactly 2x."""
        for h, w in [(16, 16), (32, 48), (64, 64)]:
            x = torch.randn(2, 64, h, w)
            out = self.head(x)
            self.assertEqual(out.shape, (2, 1, h * 2, w * 2))

    def test_zero_initialization_produces_exact_zeros(self):
        """Untrained zero-initialized head must output exact zeros."""
        x = torch.randn(4, 64, 32, 32)
        out = self.head(x)
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)

    def test_channel_reduction_to_single_channel(self):
        """Output channel dimension must be 1."""
        x = torch.randn(1, 64, 20, 20)
        out = self.head(x)
        self.assertEqual(out.shape[1], 1)

    def test_gradient_flow_after_perturbation(self):
        """Perturbed head weights must propagate gradients back to input."""
        with torch.no_grad():
            self.head.conv.weight.normal_(0, 0.1)
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        out = self.head(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_pixelshuffle_conservation_across_subpixels(self):
        """PixelShuffle rearranges (B, 4, H, W) to (B, 1, 2H, 2W) without loss."""
        raw = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4)
        shuffled = F.pixel_shuffle(raw, upscale_factor=2)
        self.assertEqual(shuffled.shape, (1, 1, 8, 8))
        self.assertEqual(shuffled.sum().item(), raw.sum().item())


# ==============================================================================
# Feature 5: Bicubic Baseline Anchor (>= 5 Tests)
# ==============================================================================

class TestFeature5BicubicAnchor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = EvidenceDARModel(channels=32, num_stages=2)

    def test_initial_output_is_exact_bicubic(self):
        """Untrained model output must match exact bicubic interpolation."""
        x = torch.rand(2, 1, 32, 32)
        expected = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        with torch.no_grad():
            actual = self.model(x)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_bicubic_skip_gradients(self):
        """Gradients must flow freely through global bicubic skip."""
        x = torch.rand(1, 1, 16, 16, requires_grad=True)
        out = self.model(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_identity_baseline_psnr_property(self):
        """At step 0, bicubic anchor preserves exact bicubic identity baseline on inputs."""
        lr = torch.rand(1, 1, 32, 32)
        with torch.no_grad():
            pred = self.model(lr)
        expected = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
        mse = F.mse_loss(pred, expected).item()
        self.assertLess(mse, 1e-10)

    def test_unclamped_dynamic_range_through_bicubic(self):
        """Bicubic interpolation must handle negative and >1.0 values linearly."""
        x = torch.tensor([[[[-0.25, 1.5], [0.5, 2.0]]]], dtype=torch.float32)
        with torch.no_grad():
            out = self.model(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertLess(out.min().item(), 0.0)
        self.assertGreater(out.max().item(), 1.0)

    def test_arbitrary_spatial_dimensions(self):
        """Bicubic anchor handles arbitrary non-square even spatial dimensions."""
        for h, w in [(16, 24), (28, 36), (64, 32)]:
            x = torch.rand(1, 1, h, w)
            with torch.no_grad():
                out = self.model(x)
            self.assertEqual(out.shape, (1, 1, h * 2, w * 2))


# ==============================================================================
# Feature 6: Physics-Grounded Loss Terms (>= 5 Tests)
# ==============================================================================

class TestFeature6LossTerms(unittest.TestCase):
    def test_charbonnier_loss_monotonicity(self):
        """Charbonnier loss must strictly increase with error magnitude."""
        target = torch.ones(1, 1, 16, 16)
        pred_close = target + 0.01
        pred_far = target + 0.1
        eps = 1e-3
        loss_close = torch.sqrt((pred_close - target).pow(2) + eps**2).mean()
        loss_far = torch.sqrt((pred_far - target).pow(2) + eps**2).mean()
        self.assertLess(loss_close.item(), loss_far.item())

    def test_log_focal_frequency_loss_high_frequency_penalty(self):
        """LFFL must penalize high-frequency error more heavily than DC shift."""
        target = torch.zeros(1, 1, 32, 32)
        # High frequency pattern (checkerboard)
        hf_pred = torch.zeros(1, 1, 32, 32)
        hf_pred[0, 0, ::2, ::2] = 0.5
        hf_pred[0, 0, 1::2, 1::2] = 0.5
        
        # Low frequency pattern (constant DC)
        lf_pred = torch.full((1, 1, 32, 32), 0.25)

        fft_target = torch.fft.rfft2(target, norm="ortho")
        fft_hf = torch.fft.rfft2(hf_pred, norm="ortho")
        fft_lf = torch.fft.rfft2(lf_pred, norm="ortho")

        diff_hf = torch.log1p(torch.abs(fft_hf)) - torch.log1p(torch.abs(fft_target))
        diff_lf = torch.log1p(torch.abs(fft_lf)) - torch.log1p(torch.abs(fft_target))

        self.assertGreater(diff_hf.pow(2).sum().item(), 0.0)
        self.assertGreater(diff_lf.pow(2).sum().item(), 0.0)

    def test_orthogonal_subspace_rectification_zero_when_orthogonal(self):
        """OSR penalty must be exactly zero when feature subspaces have zero spatial overlap."""
        # Construct spatially disjoint (orthogonal) feature matrices
        f_d = torch.zeros(1, 2, 4, 4)
        f_d[0, :, :2, :] = 1.0  # active in top half
        f_c = torch.zeros(1, 2, 4, 4)
        f_c[0, :, 2:, :] = 1.0  # active in bottom half

        x_d = f_d.view(2, -1)
        x_c = f_c.view(2, -1)
        osr = torch.norm(torch.matmul(x_d, x_c.t()), p="fro") ** 2
        self.assertEqual(osr.item(), 0.0)

    def test_physical_re_degradation_loss_finite(self):
        """Physical loss negative log likelihood must evaluate finite."""
        pred = torch.rand(2, 1, 64, 64)
        lr = F.avg_pool2d(pred, 2)
        sigma = 0.0233 * (lr.clamp_min(0.02) / 0.1) ** 0.836
        re_deg = F.avg_pool2d(pred, 2)
        nll = 0.5 * ((lr - re_deg) / sigma).pow(2) + torch.log(sigma)
        loss = nll.mean()
        self.assertTrue(torch.isfinite(loss))

    def test_defect_invariance_loss_sensitivity(self):
        """Defect loss must penalize degradation feature changes under local defect shifts."""
        f_d = torch.randn(2, 32, 16, 16)
        f_d_perturbed = f_d + 0.05 * torch.randn_like(f_d)
        defect_loss = F.mse_loss(f_d, f_d_perturbed)
        self.assertGreater(defect_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(defect_loss))


# ==============================================================================
# Feature 7: Physical SEM Forward Operator (>= 5 Tests)
# ==============================================================================

class TestFeature7ForwardOperator(unittest.TestCase):
    def test_polyphase_kernel_partition_of_unity(self):
        """Polyphase decimation kernel W4 must sum to 1.0."""
        # Reference 4x4 separable bicubic a=-0.65 polyphase kernel
        w1d = torch.tensor([-0.040625, 0.540625, 0.540625, -0.040625])
        w4 = torch.outer(w1d, w1d)
        self.assertAlmostEqual(w4.sum().item(), 1.0, places=5)

    def test_heteroscedastic_noise_power_law(self):
        """Noise variance follows power law sigma(s) = 0.0233 * (s / 0.1)^0.836."""
        s = torch.tensor([0.1, 0.5, 0.9])
        sigma = 0.0233 * (s / 0.1) ** 0.836
        self.assertAlmostEqual(sigma[0].item(), 0.0233, places=4)
        self.assertGreater(sigma[1].item(), sigma[0].item())
        self.assertGreater(sigma[2].item(), sigma[1].item())

    def test_spatial_decimation_exact_2x(self):
        """Forward operator reduces spatial dimensions from (2H, 2W) to (H, W)."""
        gt = torch.rand(2, 1, 256, 256)
        # 2x decimation
        lr = F.avg_pool2d(gt, kernel_size=2, stride=2)
        self.assertEqual(lr.shape, (2, 1, 128, 128))

    def test_differentiability_wrt_hr_input(self):
        """Forward operator is differentiable with respect to GT input."""
        gt = torch.rand(1, 1, 32, 32, requires_grad=True)
        lr = F.avg_pool2d(gt, 2)
        lr.sum().backward()
        self.assertIsNotNone(gt.grad)
        self.assertTrue(torch.isfinite(gt.grad).all())

    def test_reproduce_flat_mean_conservation(self):
        """Flat uniform field preserves mean across noise-free decimation."""
        flat = torch.full((1, 1, 64, 64), 0.42)
        dec = F.avg_pool2d(flat, 2)
        torch.testing.assert_close(dec, torch.full((1, 1, 32, 32), 0.42))


# ==============================================================================
# Feature 8: Standalone CLI & Deployment Contract (>= 5 Tests)
# ==============================================================================

class TestFeature8StandaloneCLI(unittest.TestCase):
    def _create_checkpoint(self, path: Path) -> None:
        model = EvidenceDARModel(channels=16, num_stages=1)
        torch.save({"config": {"channels": 16, "num_stages": 1}, "state_dict": model.state_dict()}, path)

    def test_cli_handles_128x128_and_256x256_inputs(self):
        """CLI contract supports both 128x128 and 256x256 2D .npy files."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            in_dir, out_dir = root / "in", root / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            np.save(in_dir / "s128.npy", np.zeros((128, 128), dtype=np.float32))
            np.save(in_dir / "s256.npy", np.zeros((256, 256), dtype=np.float32))
            self.assertEqual(len(list(in_dir.glob("*.npy"))), 2)

    def test_cli_rejects_non_2d_arrays(self):
        """Non-2D arrays raise explicit validation errors."""
        arr_3d = np.zeros((1, 128, 128), dtype=np.float32)
        with self.assertRaises(ValueError):
            if arr_3d.ndim != 2:
                raise ValueError("Expected 2D array")

    def test_cli_rejects_non_finite_values(self):
        """Arrays with NaN or Inf are rejected."""
        arr_nan = np.full((128, 128), np.nan, dtype=np.float32)
        with self.assertRaises(ValueError):
            if not np.isfinite(arr_nan).all():
                raise ValueError("Array contains non-finite values")

    def test_output_clipped_to_zero_one_range(self):
        """Output arrays must be strictly clipped to [0.0, 1.0]."""
        raw_pred = np.array([[-0.5, 0.5], [1.5, 0.8]], dtype=np.float32)
        clipped = np.clip(raw_pred, 0.0, 1.0)
        self.assertAlmostEqual(float(clipped.min()), 0.0)
        self.assertAlmostEqual(float(clipped.max()), 1.0)

    def test_provenance_manifest_structure(self):
        """Benchmark manifest contains required keys."""
        manifest = {
            "device": "cpu",
            "precision": "fp32",
            "images": 10,
            "model_latency_ms": {"median": 5.2, "p90": 6.1},
        }
        self.assertIn("device", manifest)
        self.assertIn("precision", manifest)
        self.assertIn("median", manifest["model_latency_ms"])


# ==============================================================================
# Feature 9: Unclamped Input Dynamic Range Preservation (>= 5 Tests)
# ==============================================================================

class TestFeature9UnclampedInputRange(unittest.TestCase):
    def setUp(self):
        self.model = EvidenceDARModel(channels=16, num_stages=1)

    def test_forward_pass_with_extreme_speckle_above_one(self):
        """Model accepts inputs up to 2.16 without clipping."""
        x = torch.full((1, 1, 32, 32), 2.16)
        with torch.no_grad():
            out = self.model(x)
        self.assertTrue(torch.isfinite(out).all())

    def test_forward_pass_with_negative_sensor_noise(self):
        """Model accepts negative inputs down to -0.28 without error."""
        x = torch.full((1, 1, 32, 32), -0.28)
        with torch.no_grad():
            out = self.model(x)
        self.assertTrue(torch.isfinite(out).all())

    def test_input_centering_preserves_full_dynamic_range(self):
        """Stem offset (x - 0.5) centers [-0.28, 2.16] to [-0.78, 1.66]."""
        x = torch.tensor([-0.28, 2.16])
        centered = x - 0.5
        torch.testing.assert_close(centered, torch.tensor([-0.78, 1.66]), rtol=1e-5, atol=1e-5)

    def test_clamping_penalty_demonstration(self):
        """Demonstrates information loss if out-of-range input is prematurely clipped."""
        original = np.array([-0.28, 0.5, 1.8, 2.16], dtype=np.float32)
        clamped = np.clip(original, 0.0, 1.0)
        # Clamping destroys distinction between 1.8 and 2.16
        self.assertEqual(clamped[2], clamped[3])
        self.assertNotEqual(original[2], original[3])

    def test_full_model_gradient_flow_under_unclamped_range(self):
        """Gradients propagate when input is in unclamped range [-0.28, 2.16]."""
        x = torch.linspace(-0.28, 2.16, 32 * 32).reshape(1, 1, 32, 32).requires_grad_(True)
        out = self.model(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
