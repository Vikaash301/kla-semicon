"""Standalone inference for grayscale restoration checkpoints (Compact & Evidence-DAR)."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from restoration.model import CompactRestorationNet


SUPPORTED_SHAPES = {(128, 128), (256, 256)}
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "best.pt"


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def _load_validated(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.size == 0:
        raise ValueError(f"empty array: {path}")
    if array.ndim != 2:
        raise ValueError(f"non-2D array: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite array: {path}")
    if array.shape not in SUPPORTED_SHAPES:
        raise ValueError(f"unsupported shape {array.shape}: {path}")
    return array.astype(np.float32, copy=False)


def _resolve_runtime(device: str, precision: str) -> tuple[torch.device, str]:
    device_name = device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device_name)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    if precision not in {"auto", "fp16", "fp32", "bf16"}:
        raise ValueError(f"unsupported precision: {precision}")
    if precision == "bf16":
        actual_precision = "bf16" if selected.type == "cuda" else "fp32"
    elif precision in {"auto", "fp16"}:
        actual_precision = "fp16" if selected.type == "cuda" else "fp32"
    else:
        actual_precision = "fp32"
    return selected, actual_precision



def batched_d4_tta_inference(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Executes full 8-fold D4 dihedral TTA in a single GPU batch forward pass.

    Args:
        model: PyTorch restoration model.
        x: Input tensor of shape (1, 1, H, W).
    Returns:
        Averaged HR output of shape (1, 1, 2H, 2W).
    """
    # 1. Generate 8 isometric views: 4 rotations x 2 reflections
    x0 = x
    x1 = torch.rot90(x, 1, dims=(-2, -1))
    x2 = torch.rot90(x, 2, dims=(-2, -1))
    x3 = torch.rot90(x, 3, dims=(-2, -1))
    x_f = torch.flip(x, dims=[-1])
    x4 = x_f
    x5 = torch.rot90(x_f, 1, dims=(-2, -1))
    x6 = torch.rot90(x_f, 2, dims=(-2, -1))
    x7 = torch.rot90(x_f, 3, dims=(-2, -1))

    # 2. Stack into single batch B=8
    x_batch = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=0)

    # 3. Single forward pass
    out_batch = model(x_batch)
    if isinstance(out_batch, tuple):
        out_batch = out_batch[0]

    # 4. Invert isometric transformations
    y0 = out_batch[0:1]
    y1 = torch.rot90(out_batch[1:2], -1, dims=(-2, -1))
    y2 = torch.rot90(out_batch[2:3], -2, dims=(-2, -1))
    y3 = torch.rot90(out_batch[3:4], -3, dims=(-2, -1))
    y4 = torch.flip(out_batch[4:5], dims=[-1])
    y5 = torch.flip(torch.rot90(out_batch[5:6], -1, dims=(-2, -1)), dims=[-1])
    y6 = torch.flip(torch.rot90(out_batch[6:7], -2, dims=(-2, -1)), dims=[-1])
    y7 = torch.flip(torch.rot90(out_batch[7:8], -3, dims=(-2, -1)), dims=[-1])

    # 5. Fast ensemble mean
    return (y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7) / 8.0


def _model_latency(
    model: torch.nn.Module, sample: torch.Tensor, device: torch.device, use_tta: bool = False
) -> tuple[torch.Tensor, float]:
    forward_fn = (lambda s: batched_d4_tta_inference(model, s)) if use_tta else model

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        prediction = forward_fn(sample)
        end.record()
        torch.cuda.synchronize(device)
        return prediction, float(start.elapsed_time(end))

    start_time = time.perf_counter()
    prediction = forward_fn(sample)
    return prediction, (time.perf_counter() - start_time) * 1000


def _compute_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = saved.get("config", {})

    if "num_stages" in config or "num_archetypes" in config:
        from restoration.arch.evidence_dar import EvidenceDAR
        model = EvidenceDAR(**config)
        weights = saved.get("ema_state_dict", saved.get("state_dict", saved))
        model.load_state_dict(weights, strict=False)
    elif "num_groups" in config and "blocks_per_group" in config:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude"))
        from claude.model import RestorationNet
        model = RestorationNet(**config)
        weights = saved.get("ema_state_dict", saved.get("state_dict", saved))
        model.load_state_dict(weights, strict=False)
    else:
        model = CompactRestorationNet(**config)
        weights = saved.get("ema_state_dict", saved.get("state_dict", saved))
        model.load_state_dict(weights, strict=False)

    return model.eval().to(device)


def run_inference(
    input_dir: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "auto",
    precision: str = "fp32",
    use_tta: bool = False,
    benchmark_json: str | Path | None = None,
) -> dict:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    checkpoint_path = Path(checkpoint)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output directories must differ")
    inputs = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".npy"
    )
    if not inputs:
        raise ValueError(f"no top-level .npy files found in {input_path}")
    validated = {path: _load_validated(path) for path in inputs}

    selected_device, actual_precision = _resolve_runtime(device, precision)
    model = load_model_from_checkpoint(checkpoint_path, selected_device)
    if actual_precision == "fp16":
        model.half()
        tensor_dtype = torch.float16
    elif actual_precision == "bf16":
        model.to(dtype=torch.bfloat16)
        tensor_dtype = torch.bfloat16
    else:
        tensor_dtype = torch.float32

    output_path.mkdir(parents=True, exist_ok=True)
    model_times: list[float] = []
    end_to_end_times: list[float] = []
    times_by_shape: dict[str, dict[str, list[float]]] = {}

    with torch.inference_mode():
        for shape in sorted({array.shape for array in validated.values()}):
            warmup = next(array for array in validated.values() if array.shape == shape)
            sample = torch.from_numpy(warmup)[None, None].to(
                selected_device, dtype=tensor_dtype
            )
            if use_tta:
                batched_d4_tta_inference(model, sample)
            else:
                model(sample)
            if selected_device.type == "cuda":
                torch.cuda.synchronize(selected_device)

        for path in inputs:
            wall_start = time.perf_counter()
            array = _load_validated(path)
            sample = torch.from_numpy(array)[None, None].to(
                selected_device, dtype=tensor_dtype
            )
            prediction, model_ms = _model_latency(model, sample, selected_device, use_tta=use_tta)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            prediction = prediction[0, 0].clamp(0, 1).float().cpu().numpy()
            destination = output_path / path.name
            with destination.open("wb") as output_file:
                np.save(output_file, prediction.astype(np.float32, copy=False))
            wall_ms = (time.perf_counter() - wall_start) * 1000

            model_times.append(model_ms)
            end_to_end_times.append(wall_ms)
            shape_key = f"{array.shape[0]}x{array.shape[1]}"
            shape_times = times_by_shape.setdefault(
                shape_key, {"model": [], "end_to_end": []}
            )
            shape_times["model"].append(model_ms)
            shape_times["end_to_end"].append(wall_ms)

    report = {
        "device": str(selected_device),
        "precision": actual_precision,
        "use_tta": use_tta,
        "images": len(inputs),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _compute_sha256(checkpoint_path) if checkpoint_path.is_file() else "unknown",
        "checkpoint_bytes": checkpoint_path.stat().st_size if checkpoint_path.is_file() else 0,
        "runtime": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(selected_device)
                if selected_device.type == "cuda"
                else platform.processor() or "CPU"
            ),
            "warmup_per_shape": 1,
        },
        "model_latency_ms": _summary(model_times),
        "end_to_end_ms": _summary(end_to_end_times),
        "by_shape": {
            shape: {
                "count": len(times["model"]),
                "model_latency_ms": _summary(times["model"]),
                "end_to_end_ms": _summary(times["end_to_end"]),
            }
            for shape, times in times_by_shape.items()
        },
    }
    if benchmark_json is not None:
        benchmark_path = Path(benchmark_json)
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16", "auto"), default="fp32")
    parser.add_argument("--tta", action="store_true", help="Enable batched D4 8-fold test-time augmentation")
    parser.add_argument("--benchmark-json")
    parser.add_argument("--manifest", help="Alias for --benchmark-json")
    arguments = parser.parse_args(argv)
    benchmark_file = arguments.benchmark_json or arguments.manifest
    report = run_inference(
        arguments.input_dir,
        arguments.output_dir,
        arguments.checkpoint,
        device=arguments.device,
        precision=arguments.precision,
        use_tta=arguments.tta,
        benchmark_json=benchmark_file,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
