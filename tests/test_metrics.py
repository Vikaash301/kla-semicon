import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from restoration.metrics import (
    compute_metrics,
    prepare_lpips_input,
    summarize_metrics,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("evaluate", SCRIPT)
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


class MetricsTests(unittest.TestCase):
    def test_metrics_clip_prediction_before_scoring(self):
        target = np.tile(np.linspace(0.0, 1.0, 16, dtype=np.float32), (16, 1))
        prediction = target.copy()
        prediction[:, 0] = -2.0
        prediction[:, -1] = 3.0

        result = compute_metrics(prediction, target, data_range=1.0)

        self.assertTrue(np.isinf(result["psnr"]))
        self.assertAlmostEqual(result["ssim"], 1.0)
        np.testing.assert_array_equal(prediction[:, 0], -2.0)
        np.testing.assert_array_equal(prediction[:, -1], 3.0)

    def test_metrics_reject_bad_shape_rank_and_nonfinite_values(self):
        image = np.zeros((16, 16), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "same shape"):
            compute_metrics(image, image[:, :-1], data_range=1.0)
        with self.assertRaisesRegex(ValueError, "2D"):
            compute_metrics(image[None], image[None], data_range=1.0)
        bad = image.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_metrics(bad, image, data_range=1.0)

    def test_summary_uses_population_mean_and_std(self):
        summary = summarize_metrics(
            [{"psnr": 10.0, "ssim": 0.5}, {"psnr": 14.0, "ssim": 0.9}]
        )
        self.assertEqual(summary["psnr"], {"mean": 12.0, "std": 2.0})
        self.assertAlmostEqual(summary["ssim"]["mean"], 0.7)
        self.assertAlmostEqual(summary["ssim"]["std"], 0.2)

    def test_lpips_input_is_three_channel_minus_one_to_one(self):
        image = np.array([[0.0, 1.0], [0.25, 0.75]], dtype=np.float32)
        tensor = prepare_lpips_input(image)
        self.assertEqual(tuple(tensor.shape), (1, 3, 2, 2))
        self.assertEqual(tensor.dtype, torch.float32)
        torch.testing.assert_close(tensor[0, 0], torch.tensor([[-1.0, 1.0], [-0.5, 0.5]]))
        torch.testing.assert_close(tensor[:, 0], tensor[:, 2])


class EvaluationCliTests(unittest.TestCase):
    def test_pairing_requires_identical_npy_basenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / "left", root / "right"
            left.mkdir()
            right.mkdir()
            np.save(left / "000001.npy", np.zeros((8, 8), dtype=np.float32))
            np.save(right / "000002.npy", np.zeros((8, 8), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "basename mismatch"):
                evaluate.pair_npy_files(left, right)

    def test_bicubic_mode_writes_per_image_and_aggregate_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lr_dir, gt_dir = root / "lr", root / "gt"
            lr_dir.mkdir()
            gt_dir.mkdir()
            lr = np.tile(np.linspace(0, 1, 8, dtype=np.float32), (8, 1))
            gt = evaluate.bicubic_upsample(lr, (16, 16))
            np.save(lr_dir / "sample.npy", lr)
            np.save(gt_dir / "sample.npy", gt)
            output = root / "metrics.json"

            evaluate.main(
                [
                    "--bicubic",
                    "--lr-dir",
                    str(lr_dir),
                    "--gt-dir",
                    str(gt_dir),
                    "--output-json",
                    str(output),
                    "--skip-lpips",
                ]
            )

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["count"], 1)
            self.assertEqual(report["settings"]["data_range"], 1.0)
            self.assertEqual(report["settings"]["lpips"], "skipped")
            self.assertEqual(report["per_image"][0]["name"], "sample.npy")
            self.assertIn("psnr", report["aggregate"])


if __name__ == "__main__":
    unittest.main()
