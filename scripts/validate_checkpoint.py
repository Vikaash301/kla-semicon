"""Score a frozen checkpoint on its untouched validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restoration.data import load_array, pair_files
from restoration.metrics import compute_metrics, summarize_metrics
from restoration.model import CompactRestorationNet
from scripts.evaluate import bicubic_upsample


def validate(
    data_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "auto",
    include_lpips: bool = False,
) -> dict:
    data_root = Path(data_root)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    selected_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validation_stems = list(checkpoint["split"]["validation"])
    pairs = {pair.stem: pair for pair in pair_files(data_root)}
    missing = sorted(set(validation_stems) - set(pairs))
    if missing:
        raise ValueError(f"checkpoint validation stems missing from dataset: {', '.join(missing)}")

    model = CompactRestorationNet(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(selected_device)

    directories = {
        "lr": output_dir / "NoisyLR",
        "gt": output_dir / "GT",
        "pred": output_dir / "predictions",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    model_rows = []
    baseline_rows = []
    with torch.inference_mode():
        for stem in validation_stems:
            pair = pairs[stem]
            lr = load_array(pair.lr_path).astype(np.float32, copy=False)
            gt = load_array(pair.gt_path).astype(np.float32, copy=False)
            tensor = torch.from_numpy(np.ascontiguousarray(lr))[None, None].to(selected_device)
            prediction = model(tensor)[0, 0].clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
            baseline = bicubic_upsample(lr, gt.shape)
            name = f"{stem}.npy"
            np.save(directories["lr"] / name, lr)
            np.save(directories["gt"] / name, gt)
            np.save(directories["pred"] / name, prediction)
            model_rows.append(
                {"name": name, **compute_metrics(prediction, gt, include_lpips=include_lpips, lpips_device=selected_device)}
            )
            baseline_rows.append(
                {"name": name, **compute_metrics(baseline, gt, include_lpips=include_lpips, lpips_device=selected_device)}
            )

    settings = {
        "data_range": 1.0,
        "prediction_clip": [0.0, 1.0],
        "ssim": {"gaussian_weights": True, "sigma": 1.5, "use_sample_covariance": False},
        "std_ddof": 0,
        "lpips": "alex" if include_lpips else "skipped",
    }
    checkpoint_info = {
        "path": str(checkpoint_path),
        "epoch": checkpoint["epoch"],
        "config": checkpoint["config"],
        "training_metrics": checkpoint.get("metrics", {}),
    }

    def report(rows: list[dict]) -> dict:
        metrics_only = [{key: value for key, value in row.items() if key != "name"} for row in rows]
        return {
            "count": len(rows),
            "settings": settings,
            "checkpoint": checkpoint_info,
            "per_image": rows,
            "aggregate": summarize_metrics(metrics_only),
        }

    model_report = report(model_rows)
    baseline_report = report(baseline_rows)
    (output_dir / "metrics.json").write_text(json.dumps(model_report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "bicubic_metrics.json").write_text(
        json.dumps(baseline_report, indent=2) + "\n", encoding="utf-8"
    )
    return model_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    lpips = parser.add_mutually_exclusive_group()
    lpips.add_argument("--lpips", action="store_true")
    lpips.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()
    result = validate(
        args.data_root,
        args.checkpoint,
        args.output_dir,
        device=args.device,
        include_lpips=args.lpips and not args.skip_lpips,
    )
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
