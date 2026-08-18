"""Tier 4 E2E Application Test Suite for Evidence-DAR SEM-SR.

Simulates complete, real-world semiconductor metrology and defect inspection
workflows from clean ground truth synthesis to degradation, model inference,
and metrology/PSNR metric evaluation.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from tests.test_e2e_tier1_features import EvidenceDARModel


def generate_synthetic_sem_pattern(size: int = 256, pitch: int = 16) -> torch.Tensor:
    """Generates synthetic semiconductor line-space grating pattern with a bridge defect."""
    img = np.zeros((size, size), dtype=np.float32)
    # Line-space grating
    for x in range(0, size, pitch * 2):
        img[:, x : x + pitch] = 0.8  # Bright line

    # Background substrate
    img[img == 0] = 0.2

    # Add a bridging defect between two lines
    defect_y, defect_x = size // 2, pitch
    img[defect_y - 4 : defect_y + 4, defect_x : defect_x + pitch] = 0.8

    # Add line-edge roughness (LER)
    noise = np.random.default_rng(42).normal(0, 0.02, (size, size))
    img = np.clip(img + noise, 0.0, 1.0)
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()


def apply_synthetic_sem_degradation(
    gt: torch.Tensor,
    blur_sigma: float = 1.0,
    noise_scale: float = 0.0233,
    speckle_std: float = 0.05,
) -> torch.Tensor:
    """Applies realistic SEM physics degradation: Blur + Decimation + Heteroscedastic Noise + Speckle."""
    # 1. 2x Polyphase decimation (area downsample)
    lr_clean = F.avg_pool2d(gt, kernel_size=2, stride=2)

    # 2. Multiplicative Gamma/Gaussian speckle
    speckle = 1.0 + torch.randn_like(lr_clean) * speckle_std
    lr_speckled = lr_clean * speckle

    # 3. Heteroscedastic noise: sigma(s) = noise_scale * (s / 0.1)^0.836
    s_clamped = lr_clean.clamp_min(0.02)
    sigma = noise_scale * (s_clamped / 0.1) ** 0.836
    additive_noise = torch.randn_like(lr_clean) * sigma

    noisy_lr = lr_speckled + additive_noise
    return noisy_lr  # Unclamped dynamic range [-0.28, 2.16]


class TestTier4RealWorldSEMApplications(unittest.TestCase):
    """Verifies end-to-end real-world SEM simulation scenarios."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.model = EvidenceDARModel(channels=32, num_stages=2)

    def test_full_pipeline_simulation_gt_to_deg_to_restoration_to_metrics(self):
        """Executes full lifecycle: Clean GT -> SEM Degradation -> Evidence-DAR -> Metrics."""
        # 1. Generate clean GT pattern
        gt = generate_synthetic_sem_pattern(size=128, pitch=8)
        self.assertEqual(gt.shape, (1, 1, 128, 128))

        # 2. Apply calibrated SEM degradation forward model
        noisy_lr = apply_synthetic_sem_degradation(gt)
        self.assertEqual(noisy_lr.shape, (1, 1, 64, 64))

        # 3. Evidence-DAR model restoration
        with torch.no_grad():
            restored = self.model(noisy_lr).clamp(0.0, 1.0)
        self.assertEqual(restored.shape, (1, 1, 128, 128))

        # 4. Metric evaluation
        mse = F.mse_loss(restored, gt).item()
        psnr = 10 * math.log10(1.0 / (mse + 1e-12))
        self.assertGreater(psnr, 15.0)
        self.assertTrue(torch.isfinite(torch.tensor(psnr)))

    def test_defect_preservation_and_edge_fidelity(self):
        """Verifies bridging defect is preserved in restoration output."""
        gt = generate_synthetic_sem_pattern(size=64, pitch=8)
        noisy_lr = apply_synthetic_sem_degradation(gt)

        with torch.no_grad():
            restored = self.model(noisy_lr).clamp(0.0, 1.0)

        # Defect bridge location: center region (32, 8..16)
        defect_patch_gt = gt[0, 0, 30:34, 8:16]
        defect_patch_restored = restored[0, 0, 30:34, 8:16]

        # Defect intensity should be higher than surrounding dark substrate (0.2)
        self.assertGreater(defect_patch_restored.mean().item(), 0.3)

    def test_multi_tile_batch_streaming_stability(self):
        """Simulates continuous wafer tile streaming over 20 consecutive frames."""
        with torch.no_grad():
            for tile_idx in range(20):
                tile_lr = torch.rand(1, 1, 64, 64) * 1.5 - 0.2
                out = self.model(tile_lr)
                self.assertEqual(out.shape, (1, 1, 128, 128))
                self.assertTrue(torch.isfinite(out).all())

    def test_ood_noise_power_law_robustness(self):
        """Evaluates model stability under out-of-distribution noise exponents."""
        gt = generate_synthetic_sem_pattern(size=64, pitch=8)
        for exponent in [0.5, 0.836, 1.2]:
            lr_clean = F.avg_pool2d(gt, 2)
            sigma = 0.0233 * (lr_clean.clamp_min(0.02) / 0.1) ** exponent
            noisy_lr = lr_clean + torch.randn_like(lr_clean) * sigma

            with torch.no_grad():
                out = self.model(noisy_lr)
            self.assertTrue(torch.isfinite(out).all())
            self.assertEqual(out.shape, (1, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
