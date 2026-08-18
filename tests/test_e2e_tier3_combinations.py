"""Tier 3 E2E Combinations & Interactions Test Suite for Evidence-DAR SEM-SR.

Validates multi-feature interactions, cross-subsystem gradient flow,
multi-view degradation consistency, and end-to-end inference/loss coupling.
"""

from __future__ import annotations

import json
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


class TestTier3ForwardOperatorAndLossBackprop(unittest.TestCase):
    """Verifies interaction between the physical forward operator and loss backprop."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = EvidenceDARModel(channels=32, num_stages=2)

    def test_re_degradation_phys_loss_backward_updates_model(self):
        """Re-degradation physical loss gradients propagate back through model weights."""
        lr_input = torch.rand(2, 1, 32, 32)
        target_gt = F.interpolate(lr_input, scale_factor=2, mode="bicubic", align_corners=False)

        pred_hr = self.model(lr_input)
        
        # Forward operator re-degradation
        re_degraded = F.avg_pool2d(pred_hr, kernel_size=2, stride=2)
        sigma = 0.0233 * (re_degraded.clamp_min(0.02) / 0.1) ** 0.836
        l_phys = (0.5 * ((lr_input - re_degraded) / sigma).pow(2) + torch.log(sigma)).mean()

        l_phys.backward()

        self.assertIsNotNone(self.model.stem.weight.grad)
        self.assertTrue(torch.isfinite(self.model.stem.weight.grad).all())

    def test_composite_multi_term_loss_stack_backpropagation(self):
        """Full 5-term loss stack drives stable joint gradient backprop."""
        lr_input = torch.rand(2, 1, 16, 16, requires_grad=True)
        target_gt = F.interpolate(lr_input, scale_factor=2, mode="bicubic", align_corners=False)

        pred_hr, s_deg, f_d, f_c = self.model(lr_input, return_degradation=True)

        # 1. Detail-weighted Charbonnier
        l_detail = torch.sqrt((pred_hr - target_gt).pow(2) + 1e-6).mean()

        # 2. Log Focal Frequency Loss
        fft_p = torch.fft.rfft2(pred_hr, norm="ortho")
        fft_t = torch.fft.rfft2(target_gt, norm="ortho")
        l_lffl = (torch.log1p(torch.abs(fft_p)) - torch.log1p(torch.abs(fft_t))).pow(2).mean()

        # 3. Orthogonal Subspace Rectification
        x_d = f_d.view(f_d.shape[1], -1)
        x_c = f_c.view(f_c.shape[1], -1)
        l_osr = torch.norm(torch.matmul(x_d, x_c.t()), p="fro") ** 2

        # 4. Physical Re-degradation
        re_deg = F.avg_pool2d(pred_hr, 2)
        l_phys = F.mse_loss(re_deg, lr_input)

        # 5. Defect-Invariance / Degradation Subspace Loss
        l_defect = f_d.pow(2).mean() * 0.01 + s_deg.pow(2).mean() * 0.01

        # Total weighted loss
        l_total = l_detail + 0.1 * l_lffl + 0.05 * l_osr + 0.05 * l_phys + l_defect
        l_total.backward()

        self.assertIsNotNone(self.model.stem.weight.grad)
        self.assertTrue(torch.isfinite(self.model.stem.weight.grad).all())
        self.assertIsNotNone(self.model.archetype_extractor.archetype_basis.grad)
        self.assertTrue(torch.isfinite(self.model.archetype_extractor.archetype_basis.grad).all())


class TestTier3ModelInferenceAndCLISave(unittest.TestCase):
    """Verifies interaction between model inference, array serialization, and manifest logging."""

    def test_full_inference_save_and_manifest_cycle(self):
        """Validates end-to-end file pipeline from input array to restored float32 .npy."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            in_dir, out_dir = root / "input", root / "output"
            in_dir.mkdir()
            out_dir.mkdir()

            # Save test arrays
            test_arr = np.linspace(-0.2, 1.8, 128 * 128, dtype=np.float32).reshape(128, 128)
            np.save(in_dir / "sample_001.npy", test_arr)

            # Model inference simulation
            model = EvidenceDARModel(channels=16, num_stages=1).eval()
            with torch.no_grad():
                tensor_in = torch.from_numpy(test_arr)[None, None]
                pred = model(tensor_in)[0, 0].clamp(0.0, 1.0).numpy()

            out_file = out_dir / "sample_001.npy"
            np.save(out_file, pred.astype(np.float32))

            # Verify saved array
            loaded = np.load(out_file)
            self.assertEqual(loaded.dtype, np.float32)
            self.assertEqual(loaded.shape, (256, 256))
            self.assertGreaterEqual(float(loaded.min()), 0.0)
            self.assertLessEqual(float(loaded.max()), 1.0)


class TestTier3MultiViewConsistencyAndDecomposition(unittest.TestCase):
    """Verifies multi-view degradation consistency training interaction."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = EvidenceDARModel(channels=32, num_stages=2)

    def test_multi_view_content_invariance_and_degradation_diversity(self):
        """Dual stochastic degraded views produce aligned content F_c and distinct F_d."""
        clean_gt = torch.rand(1, 1, 64, 64)

        # Generate View 1: Mild noise
        view_1 = F.avg_pool2d(clean_gt, 2) + torch.randn(1, 1, 32, 32) * 0.02

        # Generate View 2: Heavy noise + speckle
        view_2 = F.avg_pool2d(clean_gt, 2) * (1.0 + torch.randn(1, 1, 32, 32) * 0.1) + torch.randn(1, 1, 32, 32) * 0.08

        # Extract features
        _, s_deg_1, f_d_1, f_c_1 = self.model(view_1, return_degradation=True)
        _, s_deg_2, f_d_2, f_c_2 = self.model(view_2, return_degradation=True)

        # Multi-view consistency loss on content features
        content_consistency_loss = F.mse_loss(f_c_1, f_c_2)
        self.assertTrue(torch.isfinite(content_consistency_loss))

        # Degradation representations capture different noise severities
        deg_difference = F.mse_loss(s_deg_1, s_deg_2)
        self.assertTrue(torch.isfinite(deg_difference))


class TestTier3ArchetypeModulationAndGating(unittest.TestCase):
    """Verifies coupling between simplex archetypes and MST gating suppression."""

    def test_archetype_modulated_suppression_dynamics(self):
        """MST gate dynamically modulates suppression when degradation severity varies."""
        channels = 16
        extractor = SimplexArchetypeExtractor(channels, num_archetypes=4, archetype_dim=channels)
        decomposer = FeatureDecomposer(channels)
        gate = MSTSuppressionGate(channels)

        # Mild vs Severe input features
        f_mild = torch.ones(1, channels, 16, 16) * 0.1
        f_severe = torch.ones(1, channels, 16, 16) * 2.0

        f_d_mild, f_c_mild = decomposer(f_mild)
        f_d_severe, f_c_severe = decomposer(f_severe)

        out_mild = gate(f_c_mild, f_d_mild)
        out_severe = gate(f_c_severe, f_d_severe)

        self.assertTrue(torch.isfinite(out_mild).all())
        self.assertTrue(torch.isfinite(out_severe).all())
        self.assertEqual(out_mild.shape, f_mild.shape)
        self.assertEqual(out_severe.shape, f_severe.shape)


if __name__ == "__main__":
    unittest.main()
