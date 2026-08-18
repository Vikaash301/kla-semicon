import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from restoration.data import RestorationDataset, pair_files, split_stems


class DataContractTest(unittest.TestCase):
    def make_pair(self, root: Path, stem: str, gt: np.ndarray, lr: np.ndarray) -> None:
        (root / "GT").mkdir(exist_ok=True)
        (root / "NoisyLR").mkdir(exist_ok=True)
        np.save(root / "GT" / f"{stem}.npy", gt)
        np.save(root / "NoisyLR" / f"{stem}.npy", lr)

    def test_pairs_by_stem_and_rejects_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_pair(root, "b", np.zeros((4, 4)), np.zeros((2, 2)))
            self.make_pair(root, "a", np.zeros((4, 4)), np.zeros((2, 2)))
            self.assertEqual([pair.stem for pair in pair_files(root)], ["a", "b"])
            (root / "NoisyLR" / "b.npy").unlink()
            with self.assertRaisesRegex(ValueError, "missing NoisyLR.*b"):
                pair_files(root)
            np.save(root / "NoisyLR" / "b.npy", np.zeros((2, 2)))
            np.save(root / "NoisyLR" / "c.npy", np.zeros((2, 2)))
            with self.assertRaisesRegex(ValueError, "extra NoisyLR.*c"):
                pair_files(root)

    def test_uses_one_target_range_contract_and_restores_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_pair(
                root,
                "a",
                np.array([[10, 20], [30, 110]], dtype=np.uint16).repeat(2, 0).repeat(2, 1),
                np.array([[0, 120], [50, 80]], dtype=np.uint16),
            )
            dataset = RestorationDataset(root)
            sample = dataset[0]
            self.assertEqual(sample["lr"].dtype, torch.float32)
            self.assertEqual(tuple(sample["lr"].shape), (1, 2, 2))
            self.assertAlmostEqual(float(sample["gt"].min()), 0.0)
            self.assertAlmostEqual(float(sample["gt"].max()), 1.0)
            self.assertLess(float(sample["lr"].min()), 0.0)
            restored = dataset.contract.restore(sample["gt"].numpy()[0])
            np.testing.assert_array_equal(
                restored,
                np.array([[10, 20], [30, 110]], dtype=np.uint16).repeat(2, 0).repeat(2, 1),
            )
            self.assertEqual(sample["metadata"]["original_dtype"], "uint16")

    def test_rejects_non_grayscale_unsupported_dtype_and_wrong_scale(self):
        cases = (
            (np.zeros((4, 4, 1), dtype=np.float32), np.zeros((2, 2)), "2D grayscale"),
            (np.zeros((4, 4), dtype=np.complex64), np.zeros((2, 2)), "integer or float"),
            (np.zeros((5, 4), dtype=np.float32), np.zeros((2, 2)), "exactly 2x"),
        )
        for gt, lr, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.make_pair(root, "a", gt, lr)
                with self.assertRaisesRegex(ValueError, message):
                    RestorationDataset(root)

    def test_split_is_deterministic_and_complete(self):
        stems = [f"{index:03d}" for index in range(10)]
        first = split_stems(stems, validation_fraction=0.2, seed=7)
        second = split_stems(reversed(stems), validation_fraction=0.2, seed=7)
        self.assertEqual(first, second)
        train, validation = first
        self.assertEqual(len(validation), 2)
        self.assertEqual(set(train) | set(validation), set(stems))
        self.assertFalse(set(train) & set(validation))


if __name__ == "__main__":
    unittest.main()
