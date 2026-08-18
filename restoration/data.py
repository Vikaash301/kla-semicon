from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Pair:
    stem: str
    gt_path: Path
    lr_path: Path


@dataclass(frozen=True)
class NormalizationContract:
    offset: float
    scale: float
    original_dtype: str
    value_min: float
    value_max: float

    def normalize(self, array: np.ndarray) -> np.ndarray:
        return (array.astype(np.float32) - self.offset) / self.scale

    def restore(self, array: np.ndarray, dtype: str | None = None) -> np.ndarray:
        target_dtype = np.dtype(dtype or self.original_dtype)
        restored = np.asarray(array, dtype=np.float64) * self.scale + self.offset
        if np.issubdtype(target_dtype, np.integer):
            limits = np.iinfo(target_dtype)
            restored = np.clip(np.rint(restored), limits.min, limits.max)
        return restored.astype(target_dtype)


def pair_files(root: str | Path) -> list[Pair]:
    root = Path(root)
    gt = {path.stem: path for path in (root / "GT").glob("*.npy")}
    lr = {path.stem: path for path in (root / "NoisyLR").glob("*.npy")}
    missing_lr = sorted(gt.keys() - lr.keys())
    extra_lr = sorted(lr.keys() - gt.keys())
    if missing_lr or extra_lr:
        details = []
        if missing_lr:
            details.append(f"missing NoisyLR stems: {', '.join(missing_lr)}")
        if extra_lr:
            details.append(f"extra NoisyLR stems: {', '.join(extra_lr)}")
        raise ValueError("; ".join(details))
    if not gt:
        raise ValueError(f"no .npy pairs found under {root}")
    return [Pair(stem, gt[stem], lr[stem]) for stem in sorted(gt)]


def load_array(path: str | Path) -> np.ndarray:
    array = np.load(Path(path), allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"{path} must be a 2D grayscale array, got {array.shape}")
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
        raise ValueError(f"{path} dtype must be integer or float, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path} contains non-finite values")
    return array


def infer_normalization(pairs: Sequence[Pair]) -> NormalizationContract:
    value_min = float("inf")
    value_max = float("-inf")
    dtype: np.dtype | None = None
    for pair in pairs:
        gt = load_array(pair.gt_path)
        lr = load_array(pair.lr_path)
        if gt.shape != (lr.shape[0] * 2, lr.shape[1] * 2):
            raise ValueError(
                f"{pair.stem} must have exactly 2x spatial scale; GT={gt.shape}, NoisyLR={lr.shape}"
            )
        if dtype is None:
            dtype = gt.dtype
        elif dtype != gt.dtype:
            raise ValueError(f"GT dtype must be consistent across the dataset: {dtype} vs {gt.dtype}")
        value_min = min(value_min, float(gt.min()))
        value_max = max(value_max, float(gt.max()))
    scale = value_max - value_min
    if scale <= 0:
        raise ValueError("GT dataset value range must be non-zero")
    return NormalizationContract(value_min, scale, str(dtype), value_min, value_max)


class RestorationDataset(Dataset):
    def __init__(self, root: str | Path, stems: Iterable[str] | None = None) -> None:
        pairs = pair_files(root)
        self.contract = infer_normalization(pairs)
        if stems is not None:
            selected = set(stems)
            pairs = [pair for pair in pairs if pair.stem in selected]
            missing = selected - {pair.stem for pair in pairs}
            if missing:
                raise ValueError(f"unknown stems: {', '.join(sorted(missing))}")
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.pairs[index]
        gt = load_array(pair.gt_path)
        lr = load_array(pair.lr_path)
        return {
            "lr": torch.from_numpy(self.contract.normalize(lr)[None, ...]),
            "gt": torch.from_numpy(self.contract.normalize(gt)[None, ...]),
            "stem": pair.stem,
            "metadata": {
                "original_dtype": str(gt.dtype),
                "value_min": self.contract.value_min,
                "value_max": self.contract.value_max,
                "normalization_offset": self.contract.offset,
                "normalization_scale": self.contract.scale,
            },
        }


class VRAMRestorationDataset(Dataset):
    """Restoration dataset that pre-loads all tensors directly into GPU VRAM.

    Bypasses host RAM, DataLoader worker queues, and SSD paging during training.
    """

    def __init__(
        self,
        root: str | Path,
        stems: Iterable[str] | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        pairs = pair_files(root)
        self.contract = infer_normalization(pairs)
        if stems is not None:
            selected = set(stems)
            pairs = [pair for pair in pairs if pair.stem in selected]
            missing = selected - {pair.stem for pair in pairs}
            if missing:
                raise ValueError(f"unknown stems: {', '.join(sorted(missing))}")
        self.pairs = pairs
        self.device = torch.device(device)

        # Pre-load all tensors directly to GPU VRAM
        lr_list = []
        gt_list = []
        self.stems = []
        for pair in pairs:
            gt_arr = self.contract.normalize(load_array(pair.gt_path))
            lr_arr = self.contract.normalize(load_array(pair.lr_path))
            lr_list.append(torch.from_numpy(lr_arr[None, ...]))
            gt_list.append(torch.from_numpy(gt_arr[None, ...]))
            self.stems.append(pair.stem)

        # Stack into contiguous GPU tensors
        self.lr_tensors = torch.stack(lr_list).to(self.device, dtype=torch.float32)
        self.gt_tensors = torch.stack(gt_list).to(self.device, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "lr": self.lr_tensors[index],
            "gt": self.gt_tensors[index],
            "stem": self.stems[index],
        }


def split_stems(
    stems: Iterable[str], validation_fraction: float = 0.2, seed: int = 0
) -> tuple[list[str], list[str]]:
    ordered = sorted(stems)
    if len(ordered) < 2:
        raise ValueError("at least two stems are required")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    random.Random(seed).shuffle(ordered)
    validation_count = max(1, round(len(ordered) * validation_fraction))
    validation = sorted(ordered[:validation_count])
    train = sorted(ordered[validation_count:])
    return train, validation

