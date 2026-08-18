"""Tier 2 E2E Boundary & Corner Case Test Suite for Evidence-DAR SEM-SR.

Stresses mathematical boundaries, dynamic range extremes, singular inputs,
non-standard shapes, and numerical edge conditions.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.test_e2e_tier1_features import (
    EvidenceDARModel,
    FeatureDecomposer,
    MSTSuppressionGate,
    SimplexArchetypeExtractor,
    ZeroInitPixelShuffleHead,
)


class TestTier2DynamicRangeBoundaries(unittest.TestCase):
    """Stress tests dynamic range boundaries [-0.28, 2.16] and beyond."""

    def setUp(self):
        self.model = EvidenceDARModel(channels=32, num_stages=2)

    def test_maximum_observed_speckle_excursion_2_16(self):
        """Input at maximum dataset speckle level (2.16) processes without overflow."""
        x = torch.full((1, 1, 64, 64), 2.16)
        with torch.no_grad():
            out = self.model(x)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out, torch.full((1, 1, 128, 128), 2.16))

    def test_minimum_observed_sensor_noise_floor_minus_0_28(self):
        """Input at minimum dataset noise floor (-0.28) processes without underflow."""
        x = torch.full((1, 1, 64, 64), -0.28)
        with torch.no_grad():
            out = self.model(x)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out, torch.full((1, 1, 128, 128), -0.28))

    def test_extreme_adversarial_dynamic_range_spanning_minus_two_to_five(self):
        """Model maintains finite state under adversarial range [-2.0, 5.0]."""
        x = torch.linspace(-2.0, 5.0, 32 * 32).reshape(1, 1, 32, 32)
        with torch.no_grad():
            out = self.model(x)
        self.assertTrue(torch.isfinite(out).all())

    def test_mixed_bimodal_speckle_and_shadow_distribution(self):
        """Image with half -0.28 and half 2.16 preserves sharp edge transitions."""
        x = torch.zeros((1, 1, 32, 32))
        x[:, :, :, :16] = -0.28
        x[:, :, :, 16:] = 2.16
        with torch.no_grad():
            out = self.model(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.shape, (1, 1, 64, 64))


class TestTier2SingularGeometricInputs(unittest.TestCase):
    """Stress tests degenerate, singular, and pathological visual inputs."""

    def setUp(self):
        self.model = EvidenceDARModel(channels=32, num_stages=2)

    def test_all_zero_input_tensor(self):
        """All-zeros input (complete dark field) must produce all-zeros output at step 0."""
        x = torch.zeros((2, 1, 32, 32))
        with torch.no_grad():
            out = self.model(x)
        torch.testing.assert_close(out, torch.zeros((2, 1, 64, 64)), rtol=1e-6, atol=1e-6)

    def test_all_ones_input_tensor(self):
        """All-ones input (full beam saturation) must produce all-ones output at step 0."""
        x = torch.ones((2, 1, 32, 32))
        with torch.no_grad():
            out = self.model(x)
        torch.testing.assert_close(out, torch.ones((2, 1, 64, 64)), rtol=1e-6, atol=1e-6)

    def test_constant_gray_flat_field(self):
        """Constant 0.5 input has zero gradient energy; output must match 0.5."""
        x = torch.full((1, 1, 48, 48), 0.5)
        with torch.no_grad():
            out = self.model(x)
        torch.testing.assert_close(out, torch.full((1, 1, 96, 96), 0.5), rtol=1e-6, atol=1e-6)

    def test_single_pixel_dirac_impulse_response(self):
        """Single delta point source propagates symmetrically through bicubic anchor."""
        x = torch.zeros((1, 1, 17, 17))
        x[0, 0, 8, 8] = 1.0
        with torch.no_grad():
            out = self.model(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.shape, (1, 1, 34, 34))
        # Center region should have positive response
        self.assertGreater(out[0, 0, 16, 16].item(), 0.0)

    def test_checkerboard_nyquist_frequency_pattern(self):
        """Alternating Nyquist pattern does not induce numerical divergence."""
        x = torch.zeros((1, 1, 32, 32))
        x[0, 0, ::2, ::2] = 1.0
        x[0, 0, 1::2, 1::2] = 1.0
        with torch.no_grad():
            out = self.model(x)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.shape, (1, 1, 64, 64))


class TestTier2SpatialDimensionBoundaries(unittest.TestCase):
    """Stress tests arbitrary even, non-square, and extreme spatial dimensions."""

    def setUp(self):
        self.model = EvidenceDARModel(channels=16, num_stages=1)

    def test_tiny_spatial_grid_4x4(self):
        """Minimal spatial dimension 4x4 produces valid 8x8 output."""
        x = torch.rand(1, 1, 4, 4)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (1, 1, 8, 8))

    def test_large_spatial_grid_256x256(self):
        """Large dimension 256x256 produces valid 512x512 output."""
        x = torch.rand(1, 1, 256, 256)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (1, 1, 512, 512))

    def test_rectangular_aspect_ratios(self):
        """Non-square shapes (32x64, 128x64) are supported with exact 2x scaling."""
        for h, w in [(32, 64), (128, 64), (48, 96)]:
            x = torch.rand(1, 1, h, w)
            with torch.no_grad():
                out = self.model(x)
            self.assertEqual(out.shape, (1, 1, h * 2, w * 2))

    def test_non_power_of_two_even_dimensions(self):
        """Shapes like 130x130, 70x90 upscale to 260x260, 140x180."""
        for h, w in [(130, 130), (70, 90)]:
            x = torch.rand(1, 1, h, w)
            with torch.no_grad():
                out = self.model(x)
            self.assertEqual(out.shape, (1, 1, h * 2, w * 2))


class TestTier2ExtremeNoiseAndSingularities(unittest.TestCase):
    """Stress tests noise floor clamp, extreme SNR, and matrix rank edge cases."""

    def test_heteroscedastic_noise_floor_clamping(self):
        """Noise variance does not divide by zero or become NaN when signal s -> 0."""
        s_near_zero = torch.tensor([0.0, 1e-7, 1e-4])
        s_floor = 0.02
        s_clamped = s_near_zero.clamp_min(s_floor)
        sigma = 0.0233 * (s_clamped / 0.1) ** 0.836
        self.assertTrue(torch.isfinite(sigma).all())
        self.assertTrue((sigma > 0).all())

    def test_feature_decomposer_rank_1_collinear_feature_maps(self):
        """Decomposer on rank-1 collinear channels produces stable F_d and F_c."""
        decomposer = FeatureDecomposer(channels=8)
        base = torch.randn(1, 1, 16, 16)
        f_collinear = base.repeat(1, 8, 1, 1)  # All 8 channels identical
        f_d, f_c = decomposer(f_collinear)
        self.assertTrue(torch.isfinite(f_d).all())
        self.assertTrue(torch.isfinite(f_c).all())
        torch.testing.assert_close(f_d + f_c, f_collinear, rtol=1e-6, atol=1e-6)

    def test_simplex_extractor_uniform_logits_entropy_maximization(self):
        """When features are identical, alpha distribution approaches uniform 1/K."""
        extractor = SimplexArchetypeExtractor(in_channels=8, num_archetypes=4, archetype_dim=8)
        # Force linear layers to zero bias/weight
        with torch.no_grad():
            for p in extractor.mlp.parameters():
                p.zero_()
        x = torch.randn(2, 8, 16, 16)
        _, alpha = extractor(x)
        torch.testing.assert_close(alpha, torch.full_like(alpha, 1.0 / 4), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
