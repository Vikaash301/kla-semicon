import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from restoration.model import CompactRestorationNet


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("validate_checkpoint", SCRIPT)
validate_checkpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validate_checkpoint)


class ValidateCheckpointTest(unittest.TestCase):
    def test_scores_only_checkpoint_validation_stems_and_exports_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "train"
            (data / "GT").mkdir(parents=True)
            (data / "NoisyLR").mkdir()
            for index in range(3):
                lr = np.full((16, 16), 0.2 + index * 0.1, dtype=np.float32)
                gt = torch.nn.functional.interpolate(
                    torch.from_numpy(lr)[None, None],
                    scale_factor=2,
                    mode="bicubic",
                    align_corners=False,
                )[0, 0].numpy()
                np.save(data / "NoisyLR" / f"{index}.npy", lr)
                np.save(data / "GT" / f"{index}.npy", gt)

            model = CompactRestorationNet(channels=4, num_blocks=1)
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "config": model.config,
                    "state_dict": model.state_dict(),
                    "split": {"train": ["0", "2"], "validation": ["1"]},
                    "epoch": 7,
                    "metrics": {"validation_psnr": 99.0},
                },
                checkpoint,
            )

            output = root / "evidence"
            report = validate_checkpoint.validate(
                data, checkpoint, output, device="cpu", include_lpips=False
            )

            self.assertEqual(report["count"], 1)
            self.assertEqual(report["checkpoint"]["epoch"], 7)
            self.assertEqual(report["per_image"][0]["name"], "1.npy")
            self.assertTrue((output / "NoisyLR" / "1.npy").is_file())
            self.assertTrue((output / "GT" / "1.npy").is_file())
            self.assertTrue((output / "predictions" / "1.npy").is_file())
            self.assertFalse((output / "predictions" / "0.npy").exists())
            saved = json.loads((output / "metrics.json").read_text())
            baseline = json.loads((output / "bicubic_metrics.json").read_text())
            self.assertEqual(saved["settings"], baseline["settings"])
            self.assertEqual(saved["aggregate"], baseline["aggregate"])

    def test_rejects_checkpoint_stem_missing_from_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "train"
            (data / "GT").mkdir(parents=True)
            (data / "NoisyLR").mkdir()
            gt = np.zeros((32, 32), dtype=np.float32)
            lr = np.zeros((16, 16), dtype=np.float32)
            np.save(data / "GT" / "0.npy", gt)
            np.save(data / "NoisyLR" / "0.npy", lr)
            model = CompactRestorationNet(channels=4, num_blocks=1)
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "config": model.config,
                    "state_dict": model.state_dict(),
                    "split": {"train": ["0"], "validation": ["missing"]},
                    "epoch": 1,
                    "metrics": {},
                },
                checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_checkpoint.validate(data, checkpoint, root / "out", device="cpu")


if __name__ == "__main__":
    unittest.main()
