"""Create deterministic restoration comparison evidence from saved arrays/metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def strict_npy_paths(
    lr_dir: Path, pred_dir: Path, gt_dir: Path
) -> dict[str, tuple[Path, Path, Path]]:
    groups = [
        {path.name: path for path in Path(directory).glob("*.npy")}
        for directory in (lr_dir, pred_dir, gt_dir)
    ]
    names = [set(group) for group in groups]
    if not names[0] or names[0] != names[1] or names[0] != names[2]:
        raise ValueError(
            ".npy basename mismatch across LR, prediction, and GT directories: "
            f"LR={sorted(names[0])}, prediction={sorted(names[1])}, GT={sorted(names[2])}"
        )
    return {name: tuple(group[name] for group in groups) for name in sorted(names[0])}


def select_examples(rows: list[dict], count: int = 6) -> list[dict]:
    """Select evenly spaced PSNR ranks, including the endpoints."""
    if count < 1:
        raise ValueError("count must be positive")
    if not rows:
        raise ValueError("metrics JSON has no per-image rows")
    ranked = sorted(rows, key=lambda row: (float(row["psnr"]), str(row["name"])))
    chosen = min(count, len(ranked))
    if chosen == 1:
        return [ranked[(len(ranked) - 1) // 2]]
    indices = np.rint(np.linspace(0, len(ranked) - 1, chosen)).astype(int)
    return [ranked[index] for index in indices]


def _load_2d(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{path} must contain one finite 2D array")
    return array


def _bicubic(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None]
    return F.interpolate(
        tensor, size=shape, mode="bicubic", align_corners=False
    )[0, 0].numpy()


def render_comparison(
    paths: tuple[Path, Path, Path], metrics: dict, output_path: Path
) -> None:
    lr, prediction, target = (_load_2d(path) for path in paths)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and GT must have the same shape for {paths[1].name}: "
            f"{prediction.shape} != {target.shape}"
        )
    degraded = _bicubic(lr, target.shape)
    error = np.abs(np.clip(prediction, 0.0, 1.0) - target)

    figure, axes = plt.subplots(1, 4, figsize=(14, 3.7), constrained_layout=True)
    panels = (
        (degraded, "Degraded input (bicubic)"),
        (prediction, "Restored output"),
        (target, "Ground truth"),
    )
    for axis, (image, title) in zip(axes[:3], panels):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.set_axis_off()
    error_plot = axes[3].imshow(
        error, cmap="magma", vmin=0.0, vmax=max(float(error.max()), 1e-12)
    )
    axes[3].set_title("Absolute error")
    axes[3].set_axis_off()
    figure.colorbar(error_plot, ax=axes[3], label="Absolute error", fraction=0.046, pad=0.04)
    figure.suptitle(
        f'{metrics["name"]} | PSNR {float(metrics["psnr"]):.3f} dB | '
        f'SSIM {float(metrics["ssim"]):.4f}'
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def render_aggregate(model: dict, baseline: dict, output_path: Path) -> None:
    model_aggregate = model.get("aggregate", {})
    baseline_aggregate = baseline.get("aggregate", {})
    preferred = ["psnr", "ssim", "lpips"]
    metrics = [key for key in preferred if key in model_aggregate and key in baseline_aggregate]
    if not metrics:
        raise ValueError("model and baseline JSON have no common aggregate metrics")

    figure, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), squeeze=False)
    for axis, key in zip(axes[0], metrics):
        means = [baseline_aggregate[key]["mean"], model_aggregate[key]["mean"]]
        stds = [baseline_aggregate[key]["std"], model_aggregate[key]["std"]]
        axis.bar(["Bicubic", "Restored"], means, yerr=stds, capsize=4, color=["0.6", "0.25"])
        axis.set_title(f"Aggregate {key.upper()}")
        axis.set_ylabel("Mean ± reported std")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    paths = strict_npy_paths(args.lr_dir, args.pred_dir, args.gt_dir)
    model_report = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    rows = model_report.get("per_image", [])
    rows_by_name = {row["name"]: row for row in rows}
    if len(rows_by_name) != len(rows) or set(rows_by_name) != set(paths):
        raise ValueError("metrics JSON basename mismatch with array directories")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in select_examples(rows, args.count):
        render_comparison(
            paths[row["name"]], row, args.output_dir / f'{Path(row["name"]).stem}_comparison.png'
        )
    if args.baseline_json:
        baseline_report = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        render_aggregate(
            model_report, baseline_report, args.output_dir / "aggregate_before_after.png"
        )


if __name__ == "__main__":
    main()
