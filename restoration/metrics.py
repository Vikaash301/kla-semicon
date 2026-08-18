"""Reproducible restoration metrics for normalized grayscale images.

SSIM uses scikit-image with ``gaussian_weights=True``, ``sigma=1.5``,
``use_sample_covariance=False``, and the caller's explicit ``data_range``.
Reported standard deviations are population values (``ddof=0``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


_LPIPS_MODELS: dict[str, torch.nn.Module] = {}


def _validated_pair(
    prediction: np.ndarray, target: np.ndarray, data_range: float
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target must have the same shape, got {prediction.shape} and {target.shape}")
    if prediction.ndim != 2:
        raise ValueError(f"prediction and target must be 2D, got {prediction.ndim}D")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction and target must contain only finite values")
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    return prediction.astype(np.float32, copy=False), target.astype(np.float32, copy=False)


def prepare_lpips_input(image: np.ndarray) -> torch.Tensor:
    """Convert one 2D grayscale image in [0, 1] to LPIPS NCHW in [-1, 1]."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"LPIPS input must be 2D, got {image.ndim}D")
    if not np.isfinite(image).all():
        raise ValueError("LPIPS input must contain only finite values")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).unsqueeze(0)
    return tensor.repeat(1, 3, 1, 1).mul(2.0).sub(1.0)


def _lpips_model(device: torch.device) -> torch.nn.Module:
    key = str(device)
    if key not in _LPIPS_MODELS:
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError("LPIPS requested; install the optional 'lpips' package or use --skip-lpips") from error
        _LPIPS_MODELS[key] = lpips.LPIPS(net="alex").eval().to(device)
    return _LPIPS_MODELS[key]


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    data_range: float = 1.0,
    include_lpips: bool = False,
    lpips_device: str | torch.device | None = None,
) -> dict[str, float]:
    """Score one same-shape 2D pair; prediction is clipped once to [0, 1]."""
    prediction, target = _validated_pair(prediction, target, data_range)
    prediction = np.clip(prediction, 0.0, 1.0)
    result = {
        "psnr": float(peak_signal_noise_ratio(target, prediction, data_range=data_range)),
        "ssim": float(
            structural_similarity(
                target,
                prediction,
                data_range=data_range,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
            )
        ),
    }
    if include_lpips:
        device = torch.device(lpips_device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = _lpips_model(device)
        with torch.inference_mode():
            result["lpips"] = float(
                model(
                    prepare_lpips_input(prediction).to(device),
                    prepare_lpips_input(target).to(device),
                ).item()
            )
    return result


def summarize_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Return population mean/std for every metric in non-empty rows."""
    if not rows:
        raise ValueError("at least one metric row is required")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise ValueError("all metric rows must have identical keys")
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows], ddof=0)),
        }
        for key in keys
    }
