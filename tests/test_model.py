import io
import unittest

import torch

from restoration.model import CompactRestorationNet, GatedResidualBlock


class CompactRestorationNetTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = CompactRestorationNet(channels=24, num_blocks=4)

    def test_challenge_shapes(self):
        self.model.eval()
        with torch.inference_mode():
            for height, width in ((128, 128), (256, 256)):
                output = self.model(torch.rand(1, 1, height, width))
                self.assertEqual(output.shape, (1, 1, height * 2, width * 2))

    def test_initial_output_is_exactly_bicubic(self):
        input_tensor = torch.rand(1, 1, 17, 23)
        expected = torch.nn.functional.interpolate(
            input_tensor, scale_factor=2, mode="bicubic", align_corners=False
        )
        with torch.inference_mode():
            actual = self.model(input_tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_backward_pass_for_arbitrary_supported_shape(self):
        input_tensor = torch.rand(1, 1, 17, 23, requires_grad=True)
        self.model(input_tensor).mean().backward()
        self.assertIsNotNone(input_tensor.grad)
        self.assertTrue(torch.isfinite(input_tensor.grad).all())

    def test_parameter_count_is_below_two_million(self):
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertLess(parameter_count, 2_000_000)

    def test_config_is_plain_serializable_dict(self):
        self.assertIs(type(self.model.config), dict)
        rebuilt = CompactRestorationNet(**self.model.config)
        self.assertEqual(rebuilt.config, self.model.config)

    def test_configurable_large_depthwise_kernel_preserves_contract(self):
        model = CompactRestorationNet(channels=8, num_blocks=2, kernel_size=5)
        self.assertEqual(model.trunk[0].spatial.kernel_size, (5, 5))
        self.assertEqual(model(torch.rand(1, 1, 16, 19)).shape, (1, 1, 32, 38))
        self.assertEqual(model.config["kernel_size"], 5)

    def test_residual_state_stays_finite_under_extreme_gating(self):
        block = GatedResidualBlock(channels=2, kernel_size=5)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.fill_(100.0)

        output = block(torch.full((1, 2, 8, 8), 100.0))

        self.assertTrue(torch.isfinite(output).all())

    def test_extreme_finite_input_cannot_overflow_network(self):
        model = CompactRestorationNet(channels=4, num_blocks=2, kernel_size=5)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(100.0)

        output = model(torch.full((1, 1, 8, 8), 1e30))

        self.assertTrue(torch.isfinite(output).all())

    def test_checkpoint_round_trip_is_deterministic(self):
        self.model.eval()
        input_tensor = torch.rand(1, 1, 19, 21)
        with torch.inference_mode():
            expected = self.model(input_tensor)

        checkpoint = io.BytesIO()
        torch.save({"config": self.model.config, "state_dict": self.model.state_dict()}, checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored = CompactRestorationNet(**saved["config"])
        restored.load_state_dict(saved["state_dict"])
        restored.eval()

        with torch.inference_mode():
            actual = restored(input_tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
