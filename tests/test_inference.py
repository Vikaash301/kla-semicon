import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from restoration import inference as inference_module
from restoration.inference import run_inference
from restoration.model import CompactRestorationNet


class InferenceTest(unittest.TestCase):
    def _checkpoint(self, directory: Path) -> Path:
        model = CompactRestorationNet(channels=4, num_blocks=1)
        torch.nn.init.zeros_(model.head[0].weight)
        torch.nn.init.zeros_(model.head[0].bias)
        path = directory / "model.pt"
        torch.save({"config": model.config, "state_dict": model.state_dict()}, path)
        return path

    def test_writes_float32_prediction_without_preprocessing_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir, output_dir = root / "input", root / "output"
            input_dir.mkdir()
            source = np.linspace(0.2, 0.8, 128 * 128, dtype=np.float32).reshape(128, 128)
            np.save(input_dir / "sample.npy", source)
            (input_dir / "ignored.txt").write_text("ignored", encoding="utf-8")
            benchmark_path = root / "benchmark.json"

            report = run_inference(
                input_dir,
                output_dir,
                self._checkpoint(root),
                device="cpu",
                precision="fp32",
                benchmark_json=benchmark_path,
            )

            prediction = np.load(output_dir / "sample.npy")
            expected = F.interpolate(
                torch.from_numpy(source)[None, None],
                scale_factor=2,
                mode="bicubic",
                align_corners=False,
            )[0, 0].clamp(0, 1).numpy()
            self.assertEqual(prediction.dtype, np.float32)
            np.testing.assert_allclose(prediction, expected, rtol=0, atol=0)
            self.assertEqual(report["images"], 1)
            self.assertEqual(report["precision"], "fp32")
            self.assertIn("median", report["model_latency_ms"])
            self.assertIn("p90", report["end_to_end_ms"])
            self.assertEqual(report["runtime"]["warmup_per_shape"], 1)
            self.assertIn("torch_version", report["runtime"])
            self.assertEqual(json.loads(benchmark_path.read_text()), report)

    def test_discovers_only_top_level_npy_and_rejects_empty_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            (input_dir / "nested").mkdir(parents=True)
            np.save(input_dir / "nested" / "hidden.npy", np.zeros((128, 128), np.float32))

            with self.assertRaisesRegex(ValueError, "no top-level"):
                run_inference(input_dir, root / "output", self._checkpoint(root), device="cpu")

    def test_rejects_invalid_arrays(self):
        invalid = {
            "empty": np.empty((0, 128), dtype=np.float32),
            "non-2D": np.zeros((1, 128, 128), dtype=np.float32),
            "non-finite": np.full((128, 128), np.nan, dtype=np.float32),
            "unsupported shape": np.zeros((64, 64), dtype=np.float32),
        }
        for message, array in invalid.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                input_dir = root / "input"
                input_dir.mkdir()
                np.save(input_dir / "bad.npy", array)
                with self.assertRaisesRegex(ValueError, message):
                    run_inference(
                        input_dir, root / "output", self._checkpoint(root), device="cpu"
                    )

    def test_fp16_falls_back_to_fp32_on_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            input_dir.mkdir()
            np.save(input_dir / "sample.npy", np.zeros((128, 128), np.float32))

            report = run_inference(
                input_dir,
                root / "output",
                self._checkpoint(root),
                device="cpu",
                precision="fp16",
            )
            self.assertEqual(report["precision"], "fp32")

    def test_main_defaults_checkpoint_relative_to_repository(self):
        with patch.object(inference_module, "run_inference", return_value={}) as run:
            with patch("builtins.print"):
                inference_module.main(
                    ["--input-dir", "input", "--output-dir", "output"]
                )

        expected = Path(inference_module.__file__).resolve().parents[1] / "checkpoints" / "best.pt"
        self.assertEqual(run.call_args.args[2], expected)
        self.assertEqual(run.call_args.kwargs["precision"], "fp32")

    def test_rejects_output_directory_equal_to_input_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            input_dir.mkdir()
            np.save(input_dir / "sample.npy", np.zeros((128, 128), np.float32))

            with self.assertRaisesRegex(ValueError, "must differ"):
                run_inference(input_dir, input_dir, self._checkpoint(root), device="cpu")

    def test_resolution_agnostic_256_to_512_and_unclamped_range(self):
        import inference as root_inference
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir, output_dir = root / "input", root / "output"
            input_dir.mkdir()
            source = np.linspace(-0.25, 2.15, 256 * 256, dtype=np.float32).reshape(256, 256)
            np.save(input_dir / "sample_256.npy", source)
            manifest_path = root / "manifest.json"

            report = root_inference.run_inference(
                input_dir,
                output_dir,
                self._checkpoint(root),
                device="cpu",
                precision="fp32",
                benchmark_json=manifest_path,
            )

            prediction = np.load(output_dir / "sample_256.npy")
            self.assertEqual(prediction.dtype, np.float32)
            self.assertEqual(prediction.shape, (512, 512))
            self.assertGreaterEqual(float(prediction.min()), 0.0)
            self.assertLessEqual(float(prediction.max()), 1.0)
            self.assertFalse(np.isnan(prediction).any())
            self.assertFalse(np.isinf(prediction).any())
            self.assertEqual(report["images"], 1)
            self.assertIn("256x256", report["by_shape"])

    def test_root_inference_main_cli_execution(self):
        import inference as root_inference
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir, output_dir = root / "input", root / "output"
            input_dir.mkdir()
            np.save(input_dir / "test.npy", np.zeros((128, 128), dtype=np.float32))
            ckpt_path = self._checkpoint(root)
            manifest_path = root / "manifest.json"

            with patch("builtins.print"):
                root_inference.main([
                    "--input-dir", str(input_dir),
                    "--output-dir", str(output_dir),
                    "--checkpoint", str(ckpt_path),
                    "--device", "cpu",
                    "--manifest", str(manifest_path),
                ])

            self.assertTrue((output_dir / "test.npy").exists())
            self.assertTrue(manifest_path.exists())

    def test_main_still_allows_checkpoint_override(self):
        with patch.object(inference_module, "run_inference", return_value={}) as run:
            with patch("builtins.print"):
                inference_module.main(
                    [
                        "--input-dir",
                        "input",
                        "--output-dir",
                        "output",
                        "--checkpoint",
                        "custom.pt",
                    ]
                )

        self.assertEqual(run.call_args.args[2], "custom.pt")


if __name__ == "__main__":
    unittest.main()
