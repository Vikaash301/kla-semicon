import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "make_comparisons.py"
SPEC = importlib.util.spec_from_file_location("make_comparisons", SCRIPT)
comparisons = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparisons)


class ComparisonTests(unittest.TestCase):
    def test_selection_deterministically_spans_worst_median_best_psnr(self):
        rows = [
            {"name": "a.npy", "psnr": 10.0, "ssim": 0.1},
            {"name": "b.npy", "psnr": 50.0, "ssim": 0.5},
            {"name": "c.npy", "psnr": 30.0, "ssim": 0.3},
            {"name": "d.npy", "psnr": 20.0, "ssim": 0.2},
            {"name": "e.npy", "psnr": 40.0, "ssim": 0.4},
        ]

        selected = comparisons.select_examples(rows, count=3)

        self.assertEqual([row["name"] for row in selected], ["a.npy", "c.npy", "b.npy"])

    def test_cli_creates_comparisons_and_aggregate_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lr_dir, pred_dir, gt_dir, output_dir = (
                root / "lr",
                root / "pred",
                root / "gt",
                root / "visuals",
            )
            for directory in (lr_dir, pred_dir, gt_dir):
                directory.mkdir()

            rows = []
            for index, psnr in enumerate((10.0, 30.0, 20.0)):
                name = f"{index:06d}.npy"
                lr = np.full((4, 4), 0.2 + index * 0.1, dtype=np.float32)
                gt = np.full((8, 8), 0.3 + index * 0.1, dtype=np.float32)
                pred = gt.copy()
                pred[0, 0] += 0.05 * (index + 1)
                np.save(lr_dir / name, lr)
                np.save(pred_dir / name, pred)
                np.save(gt_dir / name, gt)
                rows.append({"name": name, "psnr": psnr, "ssim": 0.8 + index * 0.01})

            metrics_json = root / "model.json"
            metrics_json.write_text(
                json.dumps(
                    {
                        "per_image": rows,
                        "aggregate": {
                            "psnr": {"mean": 20.0, "std": 8.0},
                            "ssim": {"mean": 0.81, "std": 0.01},
                        },
                    }
                ),
                encoding="utf-8",
            )
            baseline_json = root / "baseline.json"
            baseline_json.write_text(
                json.dumps(
                    {
                        "aggregate": {
                            "psnr": {"mean": 15.0, "std": 5.0},
                            "ssim": {"mean": 0.7, "std": 0.02},
                        }
                    }
                ),
                encoding="utf-8",
            )

            comparisons.main(
                [
                    "--lr-dir",
                    str(lr_dir),
                    "--pred-dir",
                    str(pred_dir),
                    "--gt-dir",
                    str(gt_dir),
                    "--metrics-json",
                    str(metrics_json),
                    "--baseline-json",
                    str(baseline_json),
                    "--output-dir",
                    str(output_dir),
                    "--count",
                    "3",
                ]
            )

            comparison_files = sorted(output_dir.glob("*_comparison.png"))
            self.assertEqual(len(comparison_files), 3)
            self.assertTrue((output_dir / "aggregate_before_after.png").is_file())
            for artifact in [*comparison_files, output_dir / "aggregate_before_after.png"]:
                self.assertEqual(artifact.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_cli_rejects_mismatched_array_basenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = [root / name for name in ("lr", "pred", "gt")]
            for directory in dirs:
                directory.mkdir()
            np.save(dirs[0] / "a.npy", np.zeros((4, 4), dtype=np.float32))
            np.save(dirs[1] / "b.npy", np.zeros((8, 8), dtype=np.float32))
            np.save(dirs[2] / "a.npy", np.zeros((8, 8), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "basename mismatch"):
                comparisons.strict_npy_paths(*dirs)


if __name__ == "__main__":
    unittest.main()
