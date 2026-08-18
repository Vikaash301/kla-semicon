import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from restoration.training import (
    TrainingConfig,
    charbonnier_loss,
    paired_random_crop,
    sobel_edge_loss,
    train,
)


class TrainingPipelineTest(unittest.TestCase):
    def test_paired_crop_preserves_alignment_and_lr_outliers(self):
        lr = torch.arange(36, dtype=torch.float32).reshape(1, 6, 6) - 10
        gt = lr.repeat_interleave(2, 1).repeat_interleave(2, 2)
        cropped_lr, cropped_gt = paired_random_crop(
            lr, gt, crop_size=4, generator=torch.Generator().manual_seed(4)
        )
        self.assertEqual(cropped_lr.shape, (1, 4, 4))
        self.assertEqual(cropped_gt.shape, (1, 8, 8))
        torch.testing.assert_close(cropped_gt[:, ::2, ::2], cropped_lr)
        self.assertLess(float(cropped_lr.min()), 0.0)

    def test_local_losses_are_finite_and_reward_exact_reconstruction(self):
        target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        exact = charbonnier_loss(target, target)
        wrong = charbonnier_loss(torch.zeros_like(target), target)
        self.assertTrue(torch.isfinite(exact))
        self.assertLess(exact, wrong)
        self.assertEqual(float(sobel_edge_loss(target, target)), 0.0)

    def test_one_epoch_writes_reusable_checkpoint_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "run"
            (root / "GT").mkdir(parents=True)
            (root / "NoisyLR").mkdir()
            for index in range(4):
                rng = np.random.default_rng(index)
                lr = rng.random((64, 64), dtype=np.float32)
                gt = np.repeat(np.repeat(lr, 2, axis=0), 2, axis=1)
                np.save(root / "NoisyLR" / f"{index}.npy", lr)
                np.save(root / "GT" / f"{index}.npy", gt)

            result = train(
                TrainingConfig(
                    data_root=root,
                    output=output,
                    epochs=1,
                    batch_size=2,
                    channels=4,
                    blocks=1,
                    seed=9,
                    workers=0,
                ),
                device=torch.device("cpu"),
            )
            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            self.assertEqual(
                set(checkpoint), {"config", "state_dict", "split", "epoch", "metrics"}
            )
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(len(checkpoint["split"]["validation"]), 1)
            history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
            self.assertEqual(len(history), 1)
            self.assertEqual(result["best_epoch"], 1)

            resumed = train(
                TrainingConfig(
                    data_root=root,
                    output=output / "finetune",
                    epochs=1,
                    batch_size=2,
                    seed=9,
                    workers=0,
                    learning_rate=5e-5,
                    init_checkpoint=output / "best.pt",
                ),
                device=torch.device("cpu"),
            )
            resumed_checkpoint = torch.load(
                output / "finetune" / "last.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(resumed_checkpoint["epoch"], 2)
            self.assertEqual(resumed_checkpoint["split"], checkpoint["split"])
            self.assertGreaterEqual(resumed["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
