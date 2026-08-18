from __future__ import annotations

import os
# Force PyTorch to use ultra-fast async memory allocator to eliminate fragmentation on RTX 5070 Ti (12GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset

from restoration.data import RestorationDataset, VRAMRestorationDataset, split_stems
from restoration.model import CompactRestorationNet


@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    output: Path
    epochs: int = 20
    batch_size: int = 8
    channels: int = 32
    blocks: int = 6
    kernel_size: int = 3
    seed: int = 2026
    workers: int = 0
    max_samples: int | None = None
    validation_fraction: float = 0.1
    crop_size: int = 64
    learning_rate: float = 2e-4
    edge_weight: float = 0.05
    init_checkpoint: Path | None = None
    pin_memory: bool = False
    in_vram: bool = False
    grad_accum_steps: int = 1
    model_type: str = "compact"  # "compact" or "evidence_dar"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def paired_random_crop(
    lr: torch.Tensor,
    gt: torch.Tensor,
    crop_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if lr.ndim != 3 or gt.ndim != 3 or gt.shape[-2:] != (lr.shape[-2] * 2, lr.shape[-1] * 2):
        raise ValueError("expected aligned CHW tensors with exactly 2x GT scale")
    if crop_size > min(lr.shape[-2:]):
        raise ValueError(f"crop size {crop_size} exceeds LR shape {tuple(lr.shape[-2:])}")
    top = int(torch.randint(lr.shape[-2] - crop_size + 1, (), generator=generator))
    left = int(torch.randint(lr.shape[-1] - crop_size + 1, (), generator=generator))
    lr = lr[:, top : top + crop_size, left : left + crop_size]
    gt = gt[:, top * 2 : (top + crop_size) * 2, left * 2 : (left + crop_size) * 2]
    if bool(torch.randint(2, (), generator=generator)):
        lr, gt = lr.flip(-1), gt.flip(-1)
    if bool(torch.randint(2, (), generator=generator)):
        lr, gt = lr.flip(-2), gt.flip(-2)
    if bool(torch.randint(2, (), generator=generator)):
        lr, gt = lr.transpose(-1, -2), gt.transpose(-1, -2)
    return lr.contiguous(), gt.contiguous()


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + epsilon**2).mean()


def _sobel(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kernels = image.new_tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]
    ).unsqueeze(1)
    gradients = functional.conv2d(functional.pad(image, (1, 1, 1, 1), mode="replicate"), kernels)
    return gradients[:, :1], gradients[:, 1:]


def sobel_edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_x, prediction_y = _sobel(prediction)
    target_x, target_y = _sobel(target)
    return functional.l1_loss(prediction_x, target_x) + functional.l1_loss(prediction_y, target_y)


class _SelectedDataset(Dataset):
    def __init__(self, dataset: Union[RestorationDataset, VRAMRestorationDataset], indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.dataset[self.indices[index]]
        return {"lr": sample["lr"], "gt": sample["gt"], "stem": sample["stem"]}


class _CropDataset(_SelectedDataset):
    def __init__(
        self,
        dataset: Union[RestorationDataset, VRAMRestorationDataset],
        indices: list[int],
        crop_size: int,
        seed: int,
    ) -> None:
        super().__init__(dataset, indices)
        self.crop_size = crop_size
        self.seed = seed
        self.epoch = 0

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = super().__getitem__(index)
        generator = torch.Generator().manual_seed(self.seed + self.epoch * len(self) + index)
        lr, gt = paired_random_crop(sample["lr"], sample["gt"], self.crop_size, generator)
        return {"lr": lr, "gt": gt, "stem": sample["stem"]}


def _validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    scores = []
    with torch.inference_mode():
        for sample in loader:
            target = sample["gt"].to(device, non_blocking=True)
            lr = sample["lr"].to(device, non_blocking=True)
            prediction = model(lr)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            prediction = prediction.clamp(0.0, 1.0)
            mse = functional.mse_loss(prediction, target).item()
            scores.append(float("inf") if mse == 0 else -10.0 * math.log10(mse))
    return float(np.mean(scores))


def train(config: TrainingConfig, device: torch.device | None = None) -> dict[str, float | int]:
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    seed_everything(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.output.mkdir(parents=True, exist_ok=True)

    if config.in_vram and device.type == "cuda":
        print(f"Loading complete dataset directly into VRAM on {device}...")
        dataset = VRAMRestorationDataset(config.data_root, device=device)
        stems = dataset.stems
    else:
        dataset = RestorationDataset(config.data_root)
        stems = [pair.stem for pair in dataset.pairs]

    if config.max_samples is not None:
        if config.max_samples < 2:
            raise ValueError("max_samples must be at least 2")
        stems = stems[: config.max_samples]
    initial = None
    if config.init_checkpoint is not None:
        initial = torch.load(config.init_checkpoint, map_location="cpu", weights_only=True)
        train_stems = list(initial["split"]["train"])
        validation_stems = list(initial["split"]["validation"])
        missing = sorted((set(train_stems) | set(validation_stems)) - set(stems))
        if missing:
            raise ValueError(f"checkpoint split stems missing from dataset: {', '.join(missing)}")
    else:
        train_stems, validation_stems = split_stems(
            stems, validation_fraction=config.validation_fraction, seed=config.seed
        )
    index_by_stem = {stem: index for index, stem in enumerate(stems)}
    split = {"train": train_stems, "validation": validation_stems}
    training_dataset = _CropDataset(
        dataset, [index_by_stem[stem] for stem in train_stems], config.crop_size, config.seed
    )
    validation_dataset = _SelectedDataset(
        dataset, [index_by_stem[stem] for stem in validation_stems]
    )
    loader_generator = torch.Generator().manual_seed(config.seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        generator=loader_generator,
        pin_memory=config.pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=config.pin_memory,
    )

    if config.model_type == "evidence_dar":
        from restoration.arch.evidence_dar import EvidenceDAR
        model = EvidenceDAR(
            channels=config.channels,
            num_stages=max(1, config.blocks // 2),
            blocks_per_stage=2,
            kernel_size=config.kernel_size,
        ).to(device)
    else:
        model = CompactRestorationNet(
            **initial["config"] if initial is not None else {
                "channels": config.channels,
                "num_blocks": config.blocks,
                "scale": 2,
                "kernel_size": config.kernel_size,
            }
        ).to(device)

    if initial is not None:
        model.load_state_dict(initial["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history_path = config.output / "history.jsonl"
    history_path.write_text("", encoding="utf-8")
    best_psnr = (
        float(initial["metrics"]["validation_psnr"])
        if initial is not None
        else float("-inf")
    )
    best_epoch = int(initial["epoch"]) if initial is not None else 0
    start_epoch = best_epoch + 1
    if initial is not None:
        torch.save(initial, config.output / "best.pt")

    print(
        f"device={device} train={len(training_dataset)} validation={len(validation_dataset)} "
        f"amp={amp_enabled} in_vram={config.in_vram} pin_memory={config.pin_memory} "
        f"workers={config.workers} grad_accum_steps={config.grad_accum_steps}"
    )
    for epoch in range(start_epoch, start_epoch + config.epochs):
        training_dataset.epoch = epoch
        model.train()
        loss_sum = 0.0
        sample_count = 0
        optimizer.zero_grad(set_to_none=True)

        for step, sample in enumerate(training_loader, 1):
            lr = sample["lr"].to(device, non_blocking=True)
            gt = sample["gt"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(lr)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]
                reconstruction = charbonnier_loss(prediction, gt)
                edge = sobel_edge_loss(prediction, gt)
                loss = (reconstruction + config.edge_weight * edge) / config.grad_accum_steps

            scaler.scale(loss).backward()

            if step % config.grad_accum_steps == 0 or step == len(training_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = lr.shape[0]
            loss_sum += float(loss.detach()) * config.grad_accum_steps * batch_size
            sample_count += batch_size
            if step == 1 or step == len(training_loader) or step % 10 == 0:
                print(
                    f"epoch={epoch}/{start_epoch + config.epochs - 1} step={step}/{len(training_loader)} "
                    f"loss={(float(loss.detach()) * config.grad_accum_steps):.6f}"
                )

        metrics = {
            "train_loss": loss_sum / sample_count,
            "validation_psnr": _validate(model, validation_loader, device),
        }
        checkpoint = {
            "config": getattr(model, "config", {}),
            "state_dict": model.state_dict(),
            "split": split,
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(checkpoint, config.output / "last.pt")
        if metrics["validation_psnr"] > best_psnr:
            best_psnr = metrics["validation_psnr"]
            best_epoch = epoch
            torch.save(checkpoint, config.output / "best.pt")
        with history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps({"epoch": epoch, **metrics}) + "\n")
        print(
            f"epoch={epoch} train_loss={metrics['train_loss']:.6f} "
            f"validation_psnr={metrics['validation_psnr']:.4f} best={best_psnr:.4f}"
        )
    return {"best_epoch": best_epoch, "best_psnr": best_psnr}
