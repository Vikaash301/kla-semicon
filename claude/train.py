"""Train the channel-attention restoration network on the released 128->256 pairs."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import RestorationNet

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train_extracted" / "train"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(path: Path) -> tuple[list[str], list[str]]:
    split = json.loads(path.read_text(encoding="utf-8"))
    return list(split["train"]), list(split["validation"])


def preload(stems: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    lr = np.stack([np.load(DATA / "NoisyLR" / f"{s}.npy") for s in stems]).astype(np.float32)
    gt = np.stack([np.load(DATA / "GT" / f"{s}.npy") for s in stems]).astype(np.float32)
    return torch.from_numpy(lr).unsqueeze(1), torch.from_numpy(gt).unsqueeze(1)


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target).square() + eps * eps).mean()


def frequency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 in the 2D FFT magnitude domain; penalises lost high-frequency structure."""
    pred_spectrum = torch.fft.rfft2(pred.float(), norm="ortho")
    target_spectrum = torch.fft.rfft2(target.float(), norm="ortho")
    return (pred_spectrum - target_spectrum).abs().mean()


def random_batch(
    lr: torch.Tensor,
    gt: torch.Tensor,
    batch_size: int,
    crop: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.randint(0, lr.shape[0], (batch_size,))
    top = torch.randint(0, lr.shape[-2] - crop + 1, (batch_size,))
    left = torch.randint(0, lr.shape[-1] - crop + 1, (batch_size,))
    lr_crops = torch.empty(batch_size, 1, crop, crop)
    gt_crops = torch.empty(batch_size, 1, crop * 2, crop * 2)
    for i in range(batch_size):
        t, l = int(top[i]), int(left[i])
        lr_crops[i] = lr[index[i], :, t : t + crop, l : l + crop]
        gt_crops[i] = gt[index[i], :, t * 2 : (t + crop) * 2, l * 2 : (l + crop) * 2]
    if random.random() < 0.5:
        lr_crops, gt_crops = lr_crops.flip(-1), gt_crops.flip(-1)
    if random.random() < 0.5:
        lr_crops, gt_crops = lr_crops.flip(-2), gt_crops.flip(-2)
    if random.random() < 0.5:
        lr_crops, gt_crops = lr_crops.transpose(-1, -2), gt_crops.transpose(-1, -2)
    return (
        lr_crops.contiguous().to(device, non_blocking=True),
        gt_crops.contiguous().to(device, non_blocking=True),
    )


@torch.inference_mode()
def validate(model: nn.Module, lr: torch.Tensor, gt: torch.Tensor, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for i in range(lr.shape[0]):
        pred = model(lr[i : i + 1].to(device)).clamp(0.0, 1.0)
        mse = F.mse_loss(pred, gt[i : i + 1].to(device)).item()
        total += 99.0 if mse == 0 else -10.0 * math.log10(mse)
    model.train()
    return total / lr.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="rcan")
    parser.add_argument("--iters", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop", type=int, default=64)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--freq-weight", type=float, default=0.05)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(__file__).resolve().parent / "runs" / args.name
    output.mkdir(parents=True, exist_ok=True)

    train_stems, val_stems = load_split(Path(__file__).resolve().parent / "split.json")
    print(f"loading {len(train_stems)} train / {len(val_stems)} val pairs into RAM ...", flush=True)
    train_lr, train_gt = preload(train_stems)
    val_lr, val_gt = preload(val_stems)

    model = RestorationNet(
        channels=args.channels, num_groups=args.groups, blocks_per_group=args.blocks
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    decay = args.ema_decay
    ema = torch.optim.swa_utils.AveragedModel(
        model, avg_fn=lambda averaged, current, _: decay * averaged + (1 - decay) * current
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_iter = 0
    best = float("-inf")
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        start_iter = int(state["iter"])
        best = float(state["best"])
        print(f"resumed from {args.resume} at iter {start_iter} (best {best:.4f})", flush=True)

    history = output / "history.jsonl"
    print(f"device={device} params={params:,} iters={args.iters}", flush=True)

    started = time.perf_counter()
    running = 0.0
    for step in range(start_iter + 1, args.iters + 1):
        lr_batch, gt_batch = random_batch(train_lr, train_gt, args.batch_size, args.crop, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(lr_batch)
            loss = charbonnier(pred, gt_batch) + args.freq_weight * frequency_loss(pred, gt_batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        schedule.step()
        ema.update_parameters(model)
        running += float(loss.detach())

        if step % 200 == 0:
            rate = (step - start_iter) / max(1e-9, time.perf_counter() - started)
            print(
                f"iter {step}/{args.iters} loss={running / 200:.5f} "
                f"lr={schedule.get_last_lr()[0]:.2e} {rate:.1f}it/s",
                flush=True,
            )
            running = 0.0

        if step % args.val_every == 0 or step == args.iters:
            raw_psnr = validate(model, val_lr, val_gt, device)
            ema_psnr = validate(ema.module, val_lr, val_gt, device)
            use_ema = ema_psnr > raw_psnr
            psnr = max(raw_psnr, ema_psnr)
            checkpoint = {
                "config": model.config,
                "state_dict": model.state_dict(),
                "ema": ema.state_dict(),
                "ema_state_dict": ema.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedule": schedule.state_dict(),
                "iter": step,
                "best": max(best, psnr),
                "use_ema": use_ema,
                "val_psnr": psnr,
                "split_validation": val_stems,
            }
            torch.save(checkpoint, output / "last.pt")
            if psnr > best:
                best = psnr
                torch.save(checkpoint, output / "best.pt")
            with history.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"iter": step, "raw_psnr": raw_psnr, "ema_psnr": ema_psnr, "best": best}
                    )
                    + "\n"
                )
            print(
                f"  [val] iter={step} raw={raw_psnr:.4f} ema={ema_psnr:.4f} best={best:.4f}",
                flush=True,
            )

    print(f"done best_val_psnr={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
