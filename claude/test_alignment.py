"""Correctness checks for the pieces that would silently corrupt training.

A misaligned LR/GT crop still trains and still reports a plausible loss curve,
so it is exactly the failure that survives to the leaderboard. Run directly:
    python claude/test_alignment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import RestorationNet, self_ensemble
from train import charbonnier, frequency_loss, random_batch

DATA = Path(__file__).resolve().parents[1] / "data" / "train_extracted" / "train"


def test_crop_alignment_is_exact() -> None:
    """GT crop must be the exact 2x-scaled region of the LR crop."""
    # Ramp images make any misalignment a numeric mismatch rather than a visual one.
    lr = torch.arange(64 * 64, dtype=torch.float32).reshape(1, 1, 64, 64)
    gt = F.interpolate(lr, scale_factor=2, mode="nearest")

    for _ in range(50):
        lr_crop, gt_crop = random_batch(lr, gt, 1, 16, torch.device("cpu"))
        assert lr_crop.shape == (1, 1, 16, 16), lr_crop.shape
        assert gt_crop.shape == (1, 1, 32, 32), gt_crop.shape
        expected = F.interpolate(lr_crop, scale_factor=2, mode="nearest")
        assert torch.equal(expected, gt_crop), "LR/GT crops are misaligned"


def test_augmentation_preserves_pairing() -> None:
    """Flips/transposes must be applied identically to LR and GT."""
    lr = torch.randn(4, 1, 32, 32)
    gt = F.interpolate(lr, scale_factor=2, mode="nearest")
    for _ in range(50):
        lr_crop, gt_crop = random_batch(lr, gt, 4, 16, torch.device("cpu"))
        expected = F.interpolate(lr_crop, scale_factor=2, mode="nearest")
        assert torch.allclose(expected, gt_crop, atol=0), "augmentation desynchronised the pair"


def test_model_starts_as_exact_bicubic() -> None:
    """Zero-init head means an untrained model must equal bicubic exactly."""
    torch.manual_seed(0)
    model = RestorationNet(channels=16, num_groups=1, blocks_per_group=1).eval()
    x = torch.rand(2, 1, 32, 32)
    with torch.inference_mode():
        out = model(x)
        bicubic = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
    assert torch.allclose(out, bicubic, atol=1e-6), "model does not start at the bicubic baseline"


def test_shape_and_out_of_range_input() -> None:
    """Real inputs exceed 1.0; the model must accept them and stay finite."""
    model = RestorationNet(channels=16, num_groups=1, blocks_per_group=1).eval()
    x = torch.rand(1, 1, 128, 128) * 1.6
    with torch.inference_mode():
        out = model(x)
    assert out.shape == (1, 1, 256, 256), out.shape
    assert torch.isfinite(out).all(), "non-finite output on out-of-range input"


def test_self_ensemble_is_identity_preserving() -> None:
    """x8 TTA must average correctly-inverted transforms, not scrambled ones."""
    model = RestorationNet(channels=16, num_groups=1, blocks_per_group=1).eval()
    x = torch.rand(1, 1, 32, 32)
    with torch.inference_mode():
        plain = model(x)
        ensembled = self_ensemble(model, x)
    # An untrained (bicubic-identity) model is transform-equivariant, so every
    # branch returns the same image and the mean must equal the plain output.
    assert torch.allclose(plain, ensembled, atol=1e-5), "self_ensemble mis-inverts its transforms"


def test_losses_are_sane() -> None:
    a = torch.rand(2, 1, 16, 16)
    # Charbonnier floors at epsilon (1e-3) by construction, so identical inputs
    # give exactly eps rather than 0.
    assert charbonnier(a, a).item() <= 1.001e-3, "charbonnier not at its epsilon floor"
    assert frequency_loss(a, a).item() < 1e-5, "frequency loss not ~0 for identical inputs"
    b = torch.rand(2, 1, 16, 16)
    assert charbonnier(a, b).item() > charbonnier(a, a).item()


def test_split_has_no_leakage() -> None:
    """The headline comparison is only valid if validation was never trained on."""
    split = json.loads((Path(__file__).resolve().parent / "split.json").read_text())
    train, validation = set(split["train"]), set(split["validation"])
    assert not (train & validation), "train/validation overlap: results are invalid"
    assert len(validation) == 320, len(validation)
    if DATA.exists():
        on_disk = {p.stem for p in (DATA / "GT").glob("*.npy")}
        assert train | validation == on_disk, "split does not match the dataset on disk"


def test_real_pair_is_aligned_on_disk() -> None:
    """Dataset sanity: GT must be the 2x counterpart of its NoisyLR partner."""
    if not DATA.exists():
        return
    stem = sorted(p.stem for p in (DATA / "GT").glob("*.npy"))[0]
    gt = np.load(DATA / "GT" / f"{stem}.npy")
    lr = np.load(DATA / "NoisyLR" / f"{stem}.npy")
    assert gt.shape == (lr.shape[0] * 2, lr.shape[1] * 2), (gt.shape, lr.shape)
    # Downsampled GT should correlate strongly with the degraded LR; near-zero
    # correlation would mean the pairing itself is broken.
    reference = torch.from_numpy(gt)[None, None]
    reduced = F.avg_pool2d(reference, 2)[0, 0].numpy().ravel()
    correlation = np.corrcoef(reduced, lr.ravel())[0, 1]
    assert correlation > 0.5, f"GT/LR pairing looks broken (corr={correlation:.3f})"


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
