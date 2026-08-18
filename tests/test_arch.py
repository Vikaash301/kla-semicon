"""Unit and Integration Tests for Evidence-DAR Architecture Package (restoration/arch).

Tests all individual modules and the assembled EvidenceDAR network:
1. DegradationArchetypeExtractor: simplex constraints, uniform init, gradient flow
2. ExplicitFeatureDecomposer: conservation split, affine modulation, step-0 zero F_d
3. GatedResidualBlock, MSTSuppressionGate, AdaptiveTimescaleDynamics, EvidenceDARStage
4. PixelShuffleHead: sub-pixel upscaling, zero initialization
5. EvidenceDAR: step-0 exact bicubic identity, return_degradation hook, dynamic range, latency
"""

from __future__ import annotations

import io
import unittest
import torch
import torch.nn.functional as F

from restoration.arch import (
    AdaptiveTimescaleDynamics,
    DegradationArchetypeExtractor,
    EvidenceDAR,
    EvidenceDARModel,
    EvidenceDARStage,
    EvidenceDARTrunk,
    ExplicitFeatureDecomposer,
    FeatureDecomposer,
    GatedResidualBlock,
    MSTSuppressionGate,
    PixelShuffleHead,
    SimpleGate,
    SimplexArchetypeExtractor,
    StageDecomposer,
    ZeroInitPixelShuffleHead,
)


class TestDegradationArchetypeExtractor(unittest.TestCase):
    """Tests for continuous simplex degradation archetype extractor."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.extractor = DegradationArchetypeExtractor(channels=64, num_archetypes=8)

    def test_simplex_sum_to_one(self) -> None:
        x = torch.randn(4, 64, 32, 32)
        _, alpha = self.extractor(x)
        sums = alpha.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-6)

    def test_simplex_non_negativity(self) -> None:
        x = torch.randn(4, 64, 32, 32)
        _, alpha = self.extractor(x)
        self.assertTrue((alpha >= 0.0).all())

    def test_output_shapes(self) -> None:
        x = torch.randn(3, 64, 16, 16)
        s_deg, alpha = self.extractor(x)
        self.assertEqual(s_deg.shape, (3, 64))
        self.assertEqual(alpha.shape, (3, 8))

    def test_uniform_initialization_mixture(self) -> None:
        """At initialization with zero-initialized final MLP layer, alpha should be uniform 1/K."""
        x = torch.randn(2, 64, 16, 16)
        _, alpha = self.extractor(x)
        expected_uniform = torch.full_like(alpha, 1.0 / 8.0)
        torch.testing.assert_close(alpha, expected_uniform, rtol=1e-5, atol=1e-6)

    def test_gradient_flow(self) -> None:
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        s_deg, alpha = self.extractor(x)
        loss = s_deg.sum() + alpha.sum()
        loss.backward()
        self.assertIsNotNone(self.extractor.archetypes.grad)
        self.assertTrue(torch.isfinite(self.extractor.archetypes.grad).all())
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_alias_compatibility(self) -> None:
        extractor = SimplexArchetypeExtractor(channels=32, num_archetypes=4)
        x = torch.randn(1, 32, 8, 8)
        s_deg, alpha = extractor(x)
        self.assertEqual(s_deg.shape, (1, 32))
        self.assertEqual(alpha.shape, (1, 4))


class TestExplicitFeatureDecomposer(unittest.TestCase):
    """Tests for explicit linear conservation feature decomposer."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.decomposer = ExplicitFeatureDecomposer(channels=64)

    def test_exact_conservation_identity(self) -> None:
        """F_c + F_d must identically equal F."""
        f = torch.randn(4, 64, 32, 32)
        s_deg = torch.randn(4, 64)
        f_d, f_c = self.decomposer(f, s_deg)
        torch.testing.assert_close(f_c + f_d, f, rtol=1e-6, atol=1e-6)

    def test_step0_zero_degradation(self) -> None:
        """At step 0 with zero-init pointwise conv, F_d == 0 and F_c == F."""
        f = torch.randn(2, 64, 16, 16)
        s_deg = torch.randn(2, 64)
        f_d, f_c = self.decomposer(f, s_deg)
        torch.testing.assert_close(f_d, torch.zeros_like(f_d), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(f_c, f, rtol=1e-6, atol=1e-6)

    def test_without_s_deg(self) -> None:
        f = torch.randn(2, 64, 16, 16)
        f_d, f_c = self.decomposer(f)
        torch.testing.assert_close(f_c + f_d, f, rtol=1e-6, atol=1e-6)

    def test_gradient_backprop(self) -> None:
        f = torch.randn(2, 64, 16, 16, requires_grad=True)
        s_deg = torch.randn(2, 64, requires_grad=True)
        f_d, f_c = self.decomposer(f, s_deg)
        loss = f_d.pow(2).sum() + f_c.pow(2).sum()
        loss.backward()
        self.assertIsNotNone(f.grad)
        self.assertTrue(torch.isfinite(f.grad).all())
        self.assertIsNotNone(s_deg.grad)
        self.assertTrue(torch.isfinite(s_deg.grad).all())


class TestGatingAndTrunkModules(unittest.TestCase):
    """Tests for SimpleGate, GatedResidualBlock, MSTSuppressionGate, and EvidenceDARStage."""

    def test_simple_gate(self) -> None:
        gate = SimpleGate()
        x = torch.randn(2, 8, 4, 4)
        out = gate(x)
        self.assertEqual(out.shape, (2, 4, 4, 4))
        expected = x[:, :4] * x[:, 4:]
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)

    def test_gated_residual_block_step0_identity(self) -> None:
        block = GatedResidualBlock(channels=64)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        # Because scale is initialized to 0, out must equal x exactly
        torch.testing.assert_close(out, x, rtol=0, atol=0)

    def test_gated_residual_block_clamping_stability(self) -> None:
        block = GatedResidualBlock(channels=8, clamp_val=256.0)
        with torch.no_grad():
            for p in block.parameters():
                p.fill_(10.0)
        x = torch.full((1, 8, 8, 8), 100.0)
        out = block(x)
        self.assertTrue(torch.isfinite(out).all())

    def test_mst_suppression_gate_range(self) -> None:
        mst = MSTSuppressionGate(channels=32)
        fc = torch.randn(2, 32, 16, 16)
        fd = torch.randn(2, 32, 16, 16)
        out = mst(fc, fd)
        self.assertEqual(out.shape, fc.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_adaptive_timescale_dynamics_range(self) -> None:
        dyn = AdaptiveTimescaleDynamics(deg_dim=64, channels=64)
        s_deg = torch.randn(4, 64)
        v = dyn(s_deg)
        self.assertEqual(v.shape, (4, 64, 1, 1))
        self.assertTrue((v >= 0.0).all() and (v <= 1.0).all())

    def test_evidence_dar_stage(self) -> None:
        stage = EvidenceDARStage(channels=32, num_blocks=2, deg_dim=32)
        f = torch.randn(2, 32, 16, 16)
        s_deg = torch.randn(2, 32)
        f_next, fd, fc = stage(f, s_deg)
        self.assertEqual(f_next.shape, (2, 32, 16, 16))
        self.assertEqual(fd.shape, (2, 32, 16, 16))
        self.assertEqual(fc.shape, (2, 32, 16, 16))
        torch.testing.assert_close(fc + fd, f, rtol=1e-6, atol=1e-6)

    def test_evidence_dar_trunk(self) -> None:
        trunk = EvidenceDARTrunk(channels=32, num_stages=3, blocks_per_stage=2, deg_dim=32)
        f = torch.randn(2, 32, 16, 16)
        s_deg = torch.randn(2, 32)
        f_out, fds, fcs = trunk(f, s_deg)
        self.assertEqual(f_out.shape, (2, 32, 16, 16))
        self.assertEqual(len(fds), 3)
        self.assertEqual(len(fcs), 3)


class TestPixelShuffleHead(unittest.TestCase):
    """Tests for late 2x PixelShuffle residual reconstruction head."""

    def test_subpixel_upscaling_shape(self) -> None:
        head = PixelShuffleHead(channels=64, out_channels=1, scale=2)
        x = torch.randn(2, 64, 16, 16)
        out = head(x)
        self.assertEqual(out.shape, (2, 1, 32, 32))

    def test_exact_zero_initialization(self) -> None:
        head = PixelShuffleHead(channels=64, out_channels=1, scale=2)
        x = torch.randn(3, 64, 16, 16) * 100.0
        out = head(x)
        expected = torch.zeros(3, 1, 32, 32)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_invalid_config_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            PixelShuffleHead(channels=0)
        with self.assertRaises(ValueError):
            PixelShuffleHead(channels=64, scale=3)


class TestEvidenceDARFullModel(unittest.TestCase):
    """Tests for complete EvidenceDAR assembly and interface contracts."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = EvidenceDAR(channels=64, num_stages=4, blocks_per_stage=3, num_archetypes=8)

    def test_challenge_shapes(self) -> None:
        self.model.eval()
        with torch.inference_mode():
            for h, w in ((128, 128), (256, 256)):
                x = torch.rand(1, 1, h, w)
                out = self.model(x)
                self.assertEqual(out.shape, (1, 1, h * 2, w * 2))

    def test_step0_exact_bicubic_identity_raw(self) -> None:
        """Raw model output before clamping is bit-identical to PyTorch bicubic interpolation."""
        for shape in [(1, 1, 17, 23), (2, 1, 128, 128), (1, 1, 64, 64)]:
            x = torch.rand(shape) * 2.44 - 0.28  # Span [-0.28, 2.16]
            expected = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
            with torch.inference_mode():
                actual = self.model(x, clamp_output=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_return_degradation_hook_contract(self) -> None:
        self.model.train()
        x = torch.randn(2, 1, 32, 32)
        y_hat, s_deg, f_d, f_c = self.model(x, return_degradation=True)
        self.assertEqual(y_hat.shape, (2, 1, 64, 64))
        self.assertEqual(s_deg.shape, (2, 64))
        self.assertEqual(f_d.shape, (2, 64, 32, 32))
        self.assertEqual(f_c.shape, (2, 64, 32, 32))

    def test_full_gradient_flow(self) -> None:
        self.model.train()
        x = torch.randn(2, 1, 32, 32, requires_grad=True)
        y_hat, s_deg, f_d, f_c = self.model(x, return_degradation=True)
        loss = y_hat.sum() + s_deg.sum() + f_d.sum() + f_c.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

        missing_grads = []
        nan_grads = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                missing_grads.append(name)
            elif not torch.isfinite(param.grad).all():
                nan_grads.append(name)

        self.assertEqual(missing_grads, [], f"Parameters missing grad: {missing_grads}")
        self.assertEqual(nan_grads, [], f"Parameters with NaN grad: {nan_grads}")

    def test_parameter_count_is_below_two_million(self) -> None:
        param_count = sum(p.numel() for p in self.model.parameters())
        self.assertLess(param_count, 2_000_000)

    def test_unclamped_input_range_stability(self) -> None:
        """Verify inputs from [-0.5, 2.5] produce strictly finite outputs."""
        x = torch.linspace(-0.5, 2.5, 128 * 128).view(1, 1, 128, 128)
        self.model.eval()
        with torch.inference_mode():
            out = self.model(x, clamp_output=False)
        self.assertTrue(torch.isfinite(out).all())

    def test_eval_clamping_behavior(self) -> None:
        """In eval mode, output should be clamped to [0, 1] by default."""
        self.model.eval()
        x = torch.tensor([[[[-1.5, 2.5], [0.5, 1.2]]]])
        with torch.inference_mode():
            out = self.model(x)
        self.assertTrue((out >= 0.0).all())
        self.assertTrue((out <= 1.0).all())

    def test_checkpoint_round_trip_is_deterministic(self) -> None:
        self.model.eval()
        input_tensor = torch.rand(1, 1, 19, 21)
        with torch.inference_mode():
            expected = self.model(input_tensor)

        checkpoint = io.BytesIO()
        torch.save({"config": self.model.config, "state_dict": self.model.state_dict()}, checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored = EvidenceDAR(**saved["config"])
        restored.load_state_dict(saved["state_dict"])
        restored.eval()

        with torch.inference_mode():
            actual = restored(input_tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
