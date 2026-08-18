"""Score a restoration checkpoint against bicubic on the held-out validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ms_ssim
from torch.nn import functional as F

MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
MSSSIM_WIN = 11

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from model import RestorationNet, self_ensemble  # noqa: E402
from restoration.metrics import compute_metrics, summarize_metrics  # noqa: E402

DATA = ROOT / "data" / "train_extracted" / "train"


def load_pair(stem: str) -> tuple[np.ndarray, np.ndarray]:
    lr_path = DATA / "NoisyLR" / f"{stem}.npy"
    gt_path = DATA / "GT" / f"{stem}.npy"
    for path in (lr_path, gt_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing validation file: {path}")
    lr = np.load(lr_path).astype(np.float32)
    gt = np.load(gt_path).astype(np.float32)
    if lr.shape != (128, 128):
        raise ValueError(f"{lr_path} must be 128x128, got {lr.shape}")
    if gt.shape != (256, 256):
        raise ValueError(f"{gt_path} must be 256x256, got {gt.shape}")
    return lr, gt


def msssim_levels(size: int) -> int:
    """Largest level count whose smallest scale still exceeds the gaussian window."""
    levels = len(MSSSIM_WEIGHTS)
    while levels > 1 and size <= (MSSSIM_WIN - 1) * 2 ** (levels - 1):
        levels -= 1
    return levels


def build_model(checkpoint: dict, use_ema: bool, device: torch.device) -> RestorationNet:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing a 'config' dict")
    key = "ema_state_dict" if use_ema else "state_dict"
    if key not in checkpoint:
        raise ValueError(f"checkpoint has no '{key}'")
    model = RestorationNet(**config)
    model.load_state_dict(checkpoint[key])
    return model.eval().to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=HERE / "split.json")
    parser.add_argument("--self-ensemble", action="store_true")
    parser.add_argument("--ema", dest="ema", action="store_true", default=None)
    parser.add_argument("--no-ema", dest="ema", action="store_false")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.split.is_file():
        raise FileNotFoundError(f"split not found: {args.split}")

    stems = list(json.loads(args.split.read_text(encoding="utf-8"))["validation"])
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        stems = stems[: args.limit]
    if not stems:
        raise ValueError("validation split is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    use_ema = bool(checkpoint.get("use_ema", False)) if args.ema is None else args.ema
    model = build_model(checkpoint, use_ema, device)
    include_lpips = not args.skip_lpips

    levels = msssim_levels(256)
    weights = torch.tensor(MSSSIM_WEIGHTS[:levels], device=device)
    weights = weights / weights.sum()

    model_rows: list[dict[str, float]] = []
    bicubic_rows: list[dict[str, float]] = []
    with torch.inference_mode():
        for stem in stems:
            lr, gt = load_pair(stem)
            x = torch.from_numpy(lr).view(1, 1, 128, 128).to(device)
            target = torch.from_numpy(gt).view(1, 1, 256, 256).to(device)
            pred = self_ensemble(model, x) if args.self_ensemble else model(x)
            bicubic = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
            for rows, tensor in ((model_rows, pred), (bicubic_rows, bicubic)):
                clipped = tensor.clamp(0.0, 1.0).float()
                image = clipped.squeeze().cpu().numpy()
                rows.append(
                    {
                        "name": stem,
                        **compute_metrics(image, gt, include_lpips=include_lpips),
                        "ms_ssim": float(
                            ms_ssim(
                                clipped,
                                target,
                                data_range=1.0,
                                win_size=MSSSIM_WIN,
                                weights=weights,
                            )
                        ),
                    }
                )

    def block(rows: list[dict[str, float]]) -> dict:
        scored = [{k: v for k, v in row.items() if k != "name"} for row in rows]
        return {"per_image": rows, "aggregate": summarize_metrics(scored)}

    model_block, bicubic_block = block(model_rows), block(bicubic_rows)
    metric_names = ["psnr", "ssim"] + (["lpips"] if include_lpips else []) + ["ms_ssim"]
    delta = {
        name: model_block["aggregate"][name]["mean"] - bicubic_block["aggregate"][name]["mean"]
        for name in metric_names
    }

    report = {
        "checkpoint": str(args.checkpoint),
        "iter": checkpoint.get("iter"),
        "used_ema": use_ema,
        "self_ensemble": args.self_ensemble,
        "count": len(stems),
        "model": model_block,
        "bicubic": bicubic_block,
        "delta": delta,
        "delta_note": "model_mean - bicubic_mean; positive is better for psnr/ssim/ms_ssim, negative is better for lpips",
        "settings": {
            "ms_ssim": {
                "implementation": "pytorch_msssim.ms_ssim",
                "levels": levels,
                "requested_levels": len(MSSSIM_WEIGHTS),
                "weights": [round(float(w), 6) for w in weights],
                "win_size": MSSSIM_WIN,
                "data_range": 1.0,
            }
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\ncheckpoint={args.checkpoint} iter={report['iter']} ema={use_ema} "
          f"self_ensemble={args.self_ensemble} n={len(stems)}")
    print(f"{'metric':<8}{'model':>12}{'bicubic':>12}{'delta':>12}")
    for name in metric_names:
        print(
            f"{name:<8}{model_block['aggregate'][name]['mean']:>12.5f}"
            f"{bicubic_block['aggregate'][name]['mean']:>12.5f}{delta[name]:>+12.5f}"
        )
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
