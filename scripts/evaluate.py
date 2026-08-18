"""Evaluate restored NPY images or a torch bicubic baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restoration.metrics import compute_metrics, summarize_metrics


def pair_npy_files(left_dir: Path, right_dir: Path) -> list[tuple[Path, Path]]:
    left = {path.name: path for path in Path(left_dir).glob("*.npy")}
    right = {path.name: path for path in Path(right_dir).glob("*.npy")}
    if left.keys() != right.keys():
        missing_left = sorted(right.keys() - left.keys())
        missing_right = sorted(left.keys() - right.keys())
        raise ValueError(
            f".npy basename mismatch: missing from first={missing_left}, missing from second={missing_right}"
        )
    if not left:
        raise ValueError("no .npy files found")
    return [(left[name], right[name]) for name in sorted(left)]


def bicubic_upsample(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"bicubic input must be 2D, got {image.ndim}D")
    tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None]
    output = F.interpolate(tensor, size=target_shape, mode="bicubic", align_corners=False)
    return output[0, 0].numpy()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path)
    parser.add_argument("--lr-dir", type=Path)
    parser.add_argument("--bicubic", action="store_true")
    lpips_group = parser.add_mutually_exclusive_group()
    lpips_group.add_argument("--lpips", action="store_true", help="compute LPIPS-Alex (may download weights)")
    lpips_group.add_argument("--skip-lpips", action="store_true", help="skip LPIPS for offline evaluation")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.bicubic:
        if args.lr_dir is None or args.pred_dir is not None:
            raise ValueError("bicubic mode requires --lr-dir and does not accept --pred-dir")
        input_dir = args.lr_dir
    else:
        if args.pred_dir is None or args.lr_dir is not None:
            raise ValueError("prediction mode requires --pred-dir and does not accept --lr-dir")
        input_dir = args.pred_dir

    rows = []
    per_image = []
    for input_path, gt_path in pair_npy_files(input_dir, args.gt_dir):
        target = np.load(gt_path, allow_pickle=False)
        prediction = np.load(input_path, allow_pickle=False)
        if args.bicubic:
            prediction = bicubic_upsample(prediction, target.shape)
        metrics = compute_metrics(
            prediction,
            target,
            data_range=1.0,
            include_lpips=args.lpips and not args.skip_lpips,
        )
        rows.append(metrics)
        per_image.append({"name": input_path.name, **metrics})

    report = {
        "count": len(rows),
        "settings": {
            "data_range": 1.0,
            "prediction_clip": [0.0, 1.0],
            "ssim": {
                "gaussian_weights": True,
                "sigma": 1.5,
                "use_sample_covariance": False,
            },
            "std_ddof": 0,
            "lpips": "alex" if args.lpips and not args.skip_lpips else "skipped",
            "mode": "bicubic" if args.bicubic else "predictions",
        },
        "per_image": per_image,
        "aggregate": summarize_metrics(rows),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
