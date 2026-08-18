"""Comprehensive Unit and Integration Test Suite for SEM Simulator & Augmentations.

Tests:
1. SEMForwardOperator (PSF blur in [0.4, 1.4], W4 decimation, power-law noise, speckle, NLL, differentiability)
2. MultiViewConsistencyAugmenter (paired stochastic views, content invariance loss)
3. NegativeRestorationSampler (clean identity pairs, negative sample batch mixing, identity loss)
4. SpectralHighFrequencyAugmenter (1.2x-2.0x downscale crop, radial spectral energy shift)
5. SyntheticDefectGenerator (void, bridge, LER, protrusion, intrusion perturbations delta_defect)
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from restoration.simulator import (
    CALIBRATED_BX,
    CALIBRATED_BY,
    CALIBRATED_NOISE_EXPONENT,
    CALIBRATED_NOISE_SCALE,
    CALIBRATED_S_FLOOR,
    CALIBRATED_W4,
    MultiViewConsistencyAugmenter,
    NegativeRestorationSampler,
    SEMForwardOperator,
    SpectralHighFrequencyAugmenter,
    SyntheticDefectGenerator,
)


class TestSEMForwardOperator(unittest.TestCase):
    """Tests for calibrated SEMForwardOperator."""

    def setUp(self):
        torch.manual_seed(42)
        self.operator = SEMForwardOperator(device="cpu")

    def test_w4_partition_of_unity(self):
        """Polyphase decimation kernel W4 must sum to 1.0 (mean conservation)."""
        w4_sum = float(CALIBRATED_W4.sum())
        self.assertAlmostEqual(w4_sum, 1.0, places=5)
        self.assertEqual(CALIBRATED_W4.shape, (4, 4))

    def test_spatial_2x_decimation_dimensions(self):
        """Operator must reduce spatial dimensions from 2H, 2W to H, W."""
        for h, w in [(64, 64), (128, 128), (256, 256), (32, 64)]:
            hr = torch.rand(2, 1, h, w)
            lr = self.operator.clean(hr)
            self.assertEqual(lr.shape, (2, 1, h // 2, w // 2))

    def test_noise_free_flat_field_conservation(self):
        """Uniform flat field must preserve its exact value through clean decimation."""
        for val in [0.1, 0.42, 0.85]:
            flat = torch.full((1, 1, 64, 64), val)
            lr_clean = self.operator.clean(flat)
            # The W4 kernel has sum=1.0, so flat field is conserved to high precision
            torch.testing.assert_close(lr_clean, torch.full((1, 1, 32, 32), val), rtol=1e-3, atol=1e-3)

    def test_psf_blur_application(self):
        """PSF blur with sigma in [0.4, 1.4] smooths sharp transitions."""
        # Create Dirac delta point in center of black field
        hr = torch.zeros(1, 1, 32, 32)
        hr[0, 0, 16, 16] = 1.0

        for sigma in [0.4, 0.8, 1.4]:
            blurred = self.operator.apply_psf_blur(hr, sigma_x=sigma, sigma_y=sigma)
            self.assertTrue(torch.isfinite(blurred).all())
            # Center peak should spread out as sigma increases
            self.assertLess(blurred[0, 0, 16, 16].item(), 1.0)
            self.assertAlmostEqual(blurred.sum().item(), 1.0, places=3)

    def test_heteroscedastic_power_law_noise_curve(self):
        """Noise standard deviation follows power law sigma(s) = 0.0233 * (s / 0.1)^0.836."""
        s_vals = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        sigma = self.operator.compute_noise_sigma(s_vals)

        # Expected at s=0.1: 0.0233
        self.assertAlmostEqual(sigma[0].item(), 0.0233, places=4)
        # Expected at s=0.5: 0.0233 * (5)^0.836 ~ 0.0890
        self.assertAlmostEqual(sigma[2].item(), 0.0233 * (5.0**0.836), places=3)
        # Expected at s=0.9: 0.0233 * (9)^0.836 ~ 0.1464
        self.assertAlmostEqual(sigma[4].item(), 0.0233 * (9.0**0.836), places=3)

        # Monotonically increasing with signal
        for k in range(len(s_vals) - 1):
            self.assertLess(sigma[k].item(), sigma[k + 1].item())

    def test_noise_floor_clamping(self):
        """Low signal near zero (s < 0.02) is clamped to prevent singularity."""
        s_zero = torch.tensor([0.0, -0.1, 1e-6])
        sigma = self.operator.compute_noise_sigma(s_zero)
        expected_floor_sigma = 0.0233 * (0.02 / 0.1) ** 0.836
        for sig in sigma:
            self.assertAlmostEqual(sig.item(), expected_floor_sigma, places=4)

    def test_multiplicative_speckle_noise(self):
        """Gamma speckle noise with variance <= 0.25 (std <= 0.5) adds signal-proportional variance."""
        hr = torch.full((100, 1, 32, 32), 0.5)
        # With speckle
        degraded = self.operator.degrade(hr, theta={"speckle_std": 0.1, "noise_scale": 0.0})
        # Mean should remain approximately 0.5 (mean 1.0 multiplier)
        self.assertAlmostEqual(degraded.mean().item(), 0.5, delta=0.02)
        # Standard deviation should be positive
        self.assertGreater(degraded.std().item(), 0.02)

    def test_differentiability_with_respect_to_hr(self):
        """Gradients flow back to input HR tensor during degrade() and compute_nll()."""
        hr = torch.rand(2, 1, 64, 64, requires_grad=True)
        degraded = self.operator.degrade(hr, theta={"psf_sigma": 0.8})
        loss = degraded.sum()
        loss.backward()

        self.assertIsNotNone(hr.grad)
        self.assertTrue(torch.isfinite(hr.grad).all())
        self.assertTrue((hr.grad.abs() > 0).any())

    def test_differentiable_physical_consistency_loss_nll(self):
        """compute_nll produces finite gradients for physical loss optimization."""
        pred_hr = torch.rand(2, 1, 64, 64, requires_grad=True)
        lr_obs = torch.rand(2, 1, 32, 32)
        nll = self.operator.compute_nll(pred_hr, lr_obs)
        nll.backward()

        self.assertTrue(torch.isfinite(nll))
        self.assertIsNotNone(pred_hr.grad)
        self.assertTrue(torch.isfinite(pred_hr.grad).all())

    def test_reproducible_random_draws_with_generator(self):
        """Same generator seed yields identical noise realizations."""
        hr = torch.rand(1, 1, 64, 64)
        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)
        g3 = torch.Generator().manual_seed(456)

        deg1 = self.operator.degrade(hr, generator=g1)
        deg2 = self.operator.degrade(hr, generator=g2)
        deg3 = self.operator.degrade(hr, generator=g3)

        torch.testing.assert_close(deg1, deg2)
        self.assertFalse(torch.equal(deg1, deg3))

    def test_unclamped_dynamic_range_output(self):
        """Degraded output is not prematurely clamped and preserves full float range."""
        hr = torch.linspace(-0.2, 1.5, 64 * 64).reshape(1, 1, 64, 64)
        deg = self.operator.degrade(hr)
        self.assertTrue(torch.isfinite(deg).all())
        # Should have values outside [0, 1] due to noise and unclamped inputs
        self.assertLess(deg.min().item(), 0.0)
        self.assertGreater(deg.max().item(), 1.0)

    def test_sample_parameters_dictionary(self):
        """sample_parameters returns valid calibrated ranges."""
        params = self.operator.sample_parameters(randomize=True)
        self.assertGreaterEqual(params["psf_sigma"], 0.4)
        self.assertLessEqual(params["psf_sigma"], 1.4)
        self.assertGreaterEqual(params["noise_scale"], 0.015)
        self.assertLessEqual(params["noise_scale"], 0.035)
        self.assertGreaterEqual(params["speckle_std"], 0.0)
        self.assertLessEqual(params["speckle_std"], 0.25)


class TestMultiViewConsistencyAugmenter(unittest.TestCase):
    """Tests for MultiViewConsistencyAugmenter."""

    def setUp(self):
        torch.manual_seed(42)
        self.augmenter = MultiViewConsistencyAugmenter()

    def test_generate_multiple_views(self):
        """Generates requested number of distinct degraded views."""
        hr = torch.rand(2, 1, 64, 64)
        views = self.augmenter.generate_views(hr, num_views=3)

        self.assertEqual(len(views), 3)
        for v in views:
            self.assertEqual(v.shape, (2, 1, 32, 32))
            self.assertTrue(torch.isfinite(v).all())

        # Check views are non-identical
        self.assertFalse(torch.equal(views[0], views[1]))
        self.assertFalse(torch.equal(views[1], views[2]))

    def test_generate_paired_views(self):
        """generate_paired_views returns mild and severe view pairs."""
        hr = torch.rand(1, 1, 64, 64)
        v1, v2, th1, th2 = self.augmenter.generate_paired_views(hr)

        self.assertEqual(v1.shape, (1, 1, 32, 32))
        self.assertEqual(v2.shape, (1, 1, 32, 32))
        # Mild view has smaller noise/blur than severe view
        self.assertLessEqual(th1["noise_scale"], th2["noise_scale"])
        self.assertLessEqual(th1["speckle_std"], th2["speckle_std"])

    def test_consistency_loss_computation(self):
        """Consistency loss evaluates correctly for MSE, L1, and cosine metrics."""
        f1 = torch.randn(2, 32, 16, 16)
        f2 = torch.randn(2, 32, 16, 16)

        mse_loss = self.augmenter.compute_consistency_loss(f1, f2, loss_type="mse")
        l1_loss = self.augmenter.compute_consistency_loss(f1, f2, loss_type="l1")
        cos_loss = self.augmenter.compute_consistency_loss(f1, f2, loss_type="cosine")

        self.assertTrue(torch.isfinite(mse_loss) and mse_loss > 0)
        self.assertTrue(torch.isfinite(l1_loss) and l1_loss > 0)
        self.assertTrue(torch.isfinite(cos_loss) and cos_loss > 0)

        # Identical features yield 0 loss
        self.assertAlmostEqual(float(self.augmenter.compute_consistency_loss(f1, f1, "mse")), 0.0, places=6)
        self.assertAlmostEqual(float(self.augmenter.compute_consistency_loss(f1, f1, "cosine")), 0.0, places=6)


class TestNegativeRestorationSampler(unittest.TestCase):
    """Tests for NegativeRestorationSampler."""

    def setUp(self):
        torch.manual_seed(42)
        self.sampler = NegativeRestorationSampler(clean_prob=0.5)

    def test_create_negative_sample(self):
        """Creates clean downscaled sample without noise or PSF blur."""
        hr = torch.rand(2, 1, 64, 64)
        neg = self.sampler.create_negative_sample(hr, add_minimal_noise=False)

        self.assertEqual(neg.shape, (2, 1, 32, 32))
        # Matches exact noise-free decimation
        expected = self.sampler.operator.clean(hr, theta={"psf_sigma": 0.0})
        torch.testing.assert_close(neg, expected)

    def test_sample_batch_mixing(self):
        """sample_batch returns mixed batch with boolean mask."""
        hr_batch = torch.rand(8, 1, 64, 64)
        lr_batch, hr_out, mask = self.sampler.sample_batch(hr_batch, negative_prob=0.5)

        self.assertEqual(lr_batch.shape, (8, 1, 32, 32))
        self.assertEqual(hr_out.shape, hr_batch.shape)
        self.assertEqual(mask.shape, (8,))
        self.assertTrue(mask.dtype == torch.bool)

    def test_identity_loss_evaluation(self):
        """compute_identity_loss computes loss exclusively on negative items."""
        pred = torch.rand(4, 1, 64, 64)
        target = torch.rand(4, 1, 64, 64)
        mask = torch.tensor([True, False, True, False])

        loss = self.sampler.compute_identity_loss(pred, target, is_negative_mask=mask)
        self.assertTrue(torch.isfinite(loss) and loss > 0)

        # When mask is all False, loss is 0.0
        no_mask = torch.tensor([False, False, False, False])
        self.assertEqual(self.sampler.compute_identity_loss(pred, target, no_mask).item(), 0.0)


class TestSpectralHighFrequencyAugmenter(unittest.TestCase):
    """Tests for SpectralHighFrequencyAugmenter."""

    def setUp(self):
        torch.manual_seed(42)
        self.augmenter = SpectralHighFrequencyAugmenter(scale_range=(1.2, 2.0), p=1.0)

    def test_spectral_augmentation_output_shape(self):
        """Augmentation preserves or produces specified target shape."""
        hr = torch.rand(2, 1, 128, 128)
        aug = self.augmenter.augment(hr, target_size=(128, 128))
        self.assertEqual(aug.shape, (2, 1, 128, 128))
        self.assertTrue(torch.isfinite(aug).all())

    def test_scale_factor_returned(self):
        """return_scale provides scale factor within specified range."""
        hr = torch.rand(1, 1, 64, 64)
        aug, scale = self.augmenter.augment(hr, target_size=(64, 64), return_scale=True)
        self.assertGreaterEqual(scale, 1.2)
        self.assertLessEqual(scale, 2.0)

    def test_radial_spectral_energy_measurement(self):
        """measure_radial_spectral_energy measures frequency energy distribution."""
        # Create fine grating pattern (high frequency)
        hf_img = torch.zeros(1, 1, 64, 64)
        hf_img[0, 0, :, ::4] = 1.0  # Periodic 4-pixel lines

        # Create flat image (zero high frequency)
        lf_img = torch.full((1, 1, 64, 64), 0.5)

        hf_energy = self.augmenter.measure_radial_spectral_energy(hf_img, num_bands=8)
        lf_energy = self.augmenter.measure_radial_spectral_energy(lf_img, num_bands=8)

        self.assertEqual(hf_energy.shape, (1, 8))
        self.assertEqual(lf_energy.shape, (1, 8))

        # High-frequency pattern should have significant energy in upper bands
        self.assertGreater(hf_energy[0, 2:].sum().item(), lf_energy[0, 2:].sum().item())


class TestSyntheticDefectGenerator(unittest.TestCase):
    """Tests for SyntheticDefectGenerator."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.generator = SyntheticDefectGenerator()

    def test_add_void_defect(self):
        """add_void lowers local intensity at specified center."""
        img = torch.ones(1, 1, 64, 64) * 0.8
        perturbed, meta = self.generator.add_void(img, center=(32, 32), radius=(4, 4), intensity_drop=0.5)

        self.assertEqual(meta["type"], "void")
        self.assertEqual(perturbed.shape, img.shape)
        # Center should be lower intensity
        self.assertLess(perturbed[0, 0, 32, 32].item(), 0.8)
        # Far corner should remain unperturbed
        self.assertAlmostEqual(perturbed[0, 0, 0, 0].item(), 0.8, places=4)

    def test_add_bridge_defect(self):
        """add_bridge increases intensity along line between two points."""
        img = torch.zeros(1, 1, 64, 64)
        perturbed, meta = self.generator.add_bridge(img, start=(16, 16), end=(48, 48), width=2, intensity_boost=0.7)

        self.assertEqual(meta["type"], "bridge")
        self.assertEqual(perturbed.shape, img.shape)
        # Midpoint of bridge (32, 32) should be bright
        self.assertGreater(perturbed[0, 0, 32, 32].item(), 0.3)
        # Point far from line should be near zero
        self.assertAlmostEqual(perturbed[0, 0, 16, 48].item(), 0.0, places=3)

    def test_add_ler_perturbation(self):
        """add_ler creates high-frequency fluctuations along edges."""
        img = torch.zeros(1, 1, 64, 64)
        img[:, :, :, 32:] = 0.8  # Step edge at x=32
        perturbed, meta = self.generator.add_ler(img, amplitude=0.1)

        self.assertEqual(meta["type"], "ler")
        self.assertEqual(perturbed.shape, img.shape)
        # Edge region at x=32 should have non-zero perturbation
        delta = (perturbed - img).abs()
        self.assertGreater(delta[0, 0, :, 30:35].mean().item(), 0.0)
        # Far region from edge should have near zero perturbation
        self.assertAlmostEqual(delta[0, 0, :, 0:10].mean().item(), 0.0, places=4)

    def test_add_protrusion_and_intrusion(self):
        """Protrusion increases intensity and intrusion decreases intensity on edges."""
        img = torch.full((1, 1, 64, 64), 0.5)
        prot, prot_meta = self.generator.add_protrusion(img, center=(20, 20), radius=3, height=0.4)
        intr, intr_meta = self.generator.add_intrusion(img, center=(40, 40), radius=3, depth=0.4)

        self.assertEqual(prot_meta["type"], "protrusion")
        self.assertEqual(intr_meta["type"], "intrusion")
        self.assertGreater(prot[0, 0, 20, 20].item(), 0.5)
        self.assertLess(intr[0, 0, 40, 40].item(), 0.5)

    def test_general_defect_generation_and_delta(self):
        """generate() returns perturbed image, exact delta_defect map, and metadata."""
        img = torch.rand(2, 1, 64, 64)
        perturbed, delta, meta = self.generator.generate(img, num_defects=3)

        self.assertEqual(perturbed.shape, img.shape)
        self.assertEqual(delta.shape, img.shape)
        self.assertEqual(len(meta), 3)
        # Exact arithmetic identity: perturbed == img + delta
        torch.testing.assert_close(perturbed, img + delta, rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.isfinite(perturbed).all())
        self.assertTrue(torch.isfinite(delta).all())


if __name__ == "__main__":
    unittest.main()
